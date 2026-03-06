#if MLX_ENABLED
import Foundation
import MLX
import OpenFoundationModels
@preconcurrency import MLXLMCommon

actor MLXLanguageModelRuntime {
    private struct PreparedRun {
        let metadata: MLXModelMetadata
        let plan: MLXExecutionPlan
        let preparation: MLXExecutionPreparation
        let parameters: GenerateParameters
        let cache: [KVCache]?
        let prefixDebugSummary: String?
    }

    private let modelContainer: ModelContainer
    private let planner = MLXTranscriptPlanner()
    private let tuner = MLXGenerationTuner()
    private let executor = MLXExecutor()
    private let assembler = MLXResponseAssembler()

    private var metadata: MLXModelMetadata?
    private var tuningProfile: MLXGenerationProfile?
    private var prefixCacheStore = MLXPrefixCacheStore()
    private var lastDiagnostics: MLXRunDiagnostics?

    init(modelContainer: ModelContainer) {
        self.modelContainer = modelContainer
    }

    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        let result = try await execute(transcript: transcript, options: options)
        try assembler.validate(plan: result.plan, events: result.events)
        return try assembler.finalEntry(plan: result.plan, events: result.events)
    }

    func stream(
        transcript: Transcript,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let prepared = try await self.prepareRun(
                        transcript: transcript,
                        options: options
                    )

                    await self.logPlan(
                        metadata: prepared.metadata,
                        plan: prepared.plan,
                        promptTokenCount: prepared.preparation.promptTokenCount,
                        parameters: prepared.parameters,
                        cacheOutcome: prepared.preparation.cacheOutcome,
                        prefixDebugSummary: prepared.prefixDebugSummary
                    )

                    let startedAt = Date()
                    let stream = try await self.executor.execute(
                        container: self.modelContainer,
                        input: prepared.preparation.input,
                        cache: prepared.cache,
                        parameters: prepared.parameters,
                        reusedPrefixTokenCount: prepared.preparation.reusedPrefixTokenCount
                    )

                    var events: [MLXGenerationEvent] = []
                    var firstChunkLatency: TimeInterval?
                    var streamingState = MLXStreamingResponseState()
                    for try await event in stream {
                        try Task.checkCancellation()
                        events.append(event)

                        if firstChunkLatency == nil,
                           case .textChunk = event {
                            firstChunkLatency = Date().timeIntervalSince(startedAt)
                        }

                        switch event {
                        case .textChunk(let text):
                            if prepared.plan.toolPolicy == .disabled {
                                let result = self.assembler.streamDelta(state: streamingState, chunk: text)
                                streamingState = result.state
                                if !result.delta.isEmpty {
                                    continuation.yield(self.assembler.streamEntry(for: result.delta))
                                }
                            } else {
                            }
                        case .nativeToolCall, .info:
                            break
                        case .completed:
                            if prepared.plan.toolPolicy != .disabled {
                                try self.assembler.validate(
                                    plan: prepared.plan,
                                    events: events
                                )
                                let finalEntry = try self.assembler.finalEntry(
                                    plan: prepared.plan,
                                    events: events
                                )
                                continuation.yield(finalEntry)
                            } else {
                                let finalText = self.assembler.sanitizeAssistantResponse(streamingState.rawText)
                                if !finalText.isEmpty, finalText != streamingState.emittedVisibleText {
                                    let startIndex = finalText.index(
                                        finalText.startIndex,
                                        offsetBy: streamingState.emittedVisibleText.count
                                    )
                                    let trailingText = String(finalText[startIndex...])
                                    if !trailingText.isEmpty {
                                        continuation.yield(self.assembler.streamEntry(for: trailingText))
                                    }
                                }
                            }
                        }
                    }

                    await self.recordDiagnostics(
                        metadata: prepared.metadata,
                        plan: prepared.plan,
                        promptTokenCount: prepared.preparation.promptTokenCount,
                        cacheOutcome: prepared.preparation.cacheOutcome,
                        parameters: prepared.parameters,
                        startedAt: startedAt,
                        events: events,
                        firstChunkLatency: firstChunkLatency
                    )

                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func execute(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> (plan: MLXExecutionPlan, events: [MLXGenerationEvent], firstChunkLatency: TimeInterval?) {
        let prepared = try await prepareRun(transcript: transcript, options: options)

        await logPlan(
            metadata: prepared.metadata,
            plan: prepared.plan,
            promptTokenCount: prepared.preparation.promptTokenCount,
            parameters: prepared.parameters,
            cacheOutcome: prepared.preparation.cacheOutcome,
            prefixDebugSummary: prepared.prefixDebugSummary
        )

        let startedAt = Date()
        let stream = try await executor.execute(
            container: modelContainer,
            input: prepared.preparation.input,
            cache: prepared.cache,
            parameters: prepared.parameters,
            reusedPrefixTokenCount: prepared.preparation.reusedPrefixTokenCount
        )

        var events: [MLXGenerationEvent] = []
        var firstChunkLatency: TimeInterval?
        for try await event in stream {
            if firstChunkLatency == nil,
               case .textChunk = event {
                firstChunkLatency = Date().timeIntervalSince(startedAt)
            }
            events.append(event)
        }

        await recordDiagnostics(
            metadata: prepared.metadata,
            plan: prepared.plan,
            promptTokenCount: prepared.preparation.promptTokenCount,
            cacheOutcome: prepared.preparation.cacheOutcome,
            parameters: prepared.parameters,
            startedAt: startedAt,
            events: events,
            firstChunkLatency: firstChunkLatency
        )

        return (prepared.plan, events, firstChunkLatency)
    }

    private func resolveMetadata() async -> MLXModelMetadata {
        if let metadata {
            return metadata
        }

        let configuration = await modelContainer.configuration
        let modelID = configuration.name
        let resolved = await MLXModelMetadataRegistry.shared.metadata(for: modelID) ?? .fallback(modelID: modelID)
        metadata = resolved
        return resolved
    }

    private func prepareRun(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> PreparedRun {
        let metadata = await resolveMetadata()
        let plan = try planner.plan(transcript: transcript, options: options, metadata: metadata)
        let preparedInput = try await executor.prepareInput(
            container: modelContainer,
            input: plan.input
        )
        let profile = tuner.makeProfile(
            plan: plan,
            metadata: metadata,
            promptTokenCount: preparedInput.promptTokenCount
        )
        tuningProfile = profile
        let parameters = tuner.makeParameters(options: options, profile: profile)
        let reuseDecision = try await resolveReuseDecision(
            plan: plan,
            metadata: metadata,
            parameters: parameters,
            fullInput: preparedInput.input
        )
        let preparation = executor.makeExecutionPreparation(
            input: preparedInput.input,
            reuseDecision: reuseDecision
        )
        let prefixDebugSummary = await makePrefixDebugSummary(
            plan: plan,
            fullInput: preparation.input,
            reuseDecision: reuseDecision
        )

        return PreparedRun(
            metadata: metadata,
            plan: plan,
            preparation: preparation,
            parameters: parameters,
            cache: reuseDecision.cache,
            prefixDebugSummary: prefixDebugSummary
        )
    }

    private func resolveReuseDecision(
        plan: MLXExecutionPlan,
        metadata: MLXModelMetadata,
        parameters: GenerateParameters,
        fullInput: LMInput
    ) async throws -> MLXCacheReuseDecision {
        let cachedDecision = prefixCacheStore.lookup(plan: plan, metadata: metadata)
        if cachedDecision.cache != nil {
            return try await validatedReuseDecision(
                decision: cachedDecision,
                plan: plan,
                fullInput: fullInput,
                cacheKey: plan.cachePlan.cacheKey
            )
        }

        if plan.cachePlan.reuseScope != .prefixReusable {
            return cachedDecision
        }

        guard let cacheKey = plan.cachePlan.cacheKey,
              let prefixInput = plan.cachePlan.prefixInput
        else {
            return cachedDecision
        }

        do {
            let snapshot = try await executor.buildPrefixSnapshot(
                container: modelContainer,
                input: prefixInput,
                parameters: parameters
            )
            let builtDecision = MLXCacheReuseDecision(
                cache: try materializePromptCache(from: snapshot),
                prefixTokenCount: snapshot.prefixTokenCount,
                outcome: "built"
            )
            let validatedDecision = try await validatedReuseDecision(
                decision: builtDecision,
                plan: plan,
                fullInput: fullInput,
                cacheKey: cacheKey
            )
            if validatedDecision.cache != nil {
                prefixCacheStore.store(
                    cacheKey: cacheKey,
                    snapshot: snapshot,
                    metadata: metadata
                )

                #if DEBUG
                print(
                    "[MLXLanguageModel] prefixCache stored modelID=\(metadata.modelID) tokens=\(snapshot.prefixTokenCount)"
                )
                #endif
            }
            return validatedDecision
        } catch {
            Logger.warning("[MLXLanguageModel] Prefix snapshot build skipped: \(error)")
            return .noReuse(reason: .prefixSnapshotUnsupported)
        }
    }

    private func validatedReuseDecision(
        decision: MLXCacheReuseDecision,
        plan: MLXExecutionPlan,
        fullInput: LMInput,
        cacheKey: MLXPrefixCacheKey?
    ) async throws -> MLXCacheReuseDecision {
        guard let cache = decision.cache,
              let prefixInput = plan.cachePlan.prefixInput
        else {
            return decision
        }

        let prefixPrepared = try await modelContainer.preparePrefix(input: prefixInput)
        let fullTokens = flattenedTokenIDs(from: fullInput.text.tokens)
        let prefixTokens = flattenedTokenIDs(from: prefixPrepared.text.tokens)
        let validation = MLXPrefixReuseValidator.validate(
            fullTokens: fullTokens,
            prefixTokens: prefixTokens,
            requestedPrefixTokenCount: decision.prefixTokenCount,
            successOutcome: decision.outcome
        )

        if validation.shouldInvalidate, let cacheKey {
            prefixCacheStore.invalidate(cacheKey: cacheKey)
        }

        guard let acceptedPrefixTokenCount = validation.acceptedPrefixTokenCount else {
            return MLXCacheReuseDecision(
                cache: nil,
                prefixTokenCount: nil,
                outcome: validation.outcome
            )
        }

        return MLXCacheReuseDecision(
            cache: cache,
            prefixTokenCount: acceptedPrefixTokenCount,
            outcome: validation.outcome
        )
    }

    private func logPlan(
        metadata: MLXModelMetadata,
        plan: MLXExecutionPlan,
        promptTokenCount: Int,
        parameters: GenerateParameters,
        cacheOutcome: String,
        prefixDebugSummary: String?
    ) async {
        #if DEBUG
        print(
            "[MLXLanguageModel] plan modelID=\(metadata.modelID) runtime=\(metadata.runtimeFamily.rawValue) toolPolicy=\(plan.toolPolicy.rawValue) promptTokens=\(promptTokenCount) cache=\(cacheOutcome) tools=\(plan.plannerDiagnostics.toolDefinitionCount) latestUser=\"\(plan.plannerDiagnostics.latestUserPreview)\" prefixMessages=\(plan.cachePlan.prefixMessages.count) suffixMessages=\(plan.cachePlan.suffixMessages.count) prefillStep=\(parameters.prefillStepSize) kvBits=\(String(describing: parameters.kvBits)) maxKVSize=\(String(describing: parameters.maxKVSize))"
        )
        if let prefixDebugSummary {
            print("[MLXLanguageModel] prefixDebug \(prefixDebugSummary)")
        }
        #endif
    }

    private func recordDiagnostics(
        metadata: MLXModelMetadata,
        plan: MLXExecutionPlan,
        promptTokenCount: Int,
        cacheOutcome: String,
        parameters: GenerateParameters,
        startedAt: Date,
        events: [MLXGenerationEvent],
        firstChunkLatency: TimeInterval?
    ) async {
        let collected = assembler.collect(events: events)
        let parametersSummary =
            "prefillStep=\(parameters.prefillStepSize),kvBits=\(String(describing: parameters.kvBits)),maxKVSize=\(String(describing: parameters.maxKVSize))"

        lastDiagnostics = MLXRunDiagnostics(
            modelID: metadata.modelID,
            runtimeFamily: metadata.runtimeFamily,
            toolPolicy: plan.toolPolicy,
            promptTokenCount: promptTokenCount,
            cacheOutcome: cacheOutcome,
            parametersSummary: parametersSummary,
            firstChunkLatency: firstChunkLatency,
            totalLatency: Date().timeIntervalSince(startedAt),
            outputCharacterCount: collected.text.count,
            usedNativeToolCalls: !collected.nativeToolCalls.isEmpty
        )

        #if DEBUG
        let rawPreview = previewText(collected.text)
        let sanitizedPreview = previewText(assembler.sanitizeAssistantResponse(collected.text))
        let textualToolCallDetected = ToolCallDetector.entryIfPresent(
            assembler.sanitizeAssistantResponse(collected.text)
        ) != nil
        print(
            "[MLXLanguageModel] completed runtime=\(metadata.runtimeFamily.rawValue) toolPolicy=\(plan.toolPolicy.rawValue) promptTokens=\(promptTokenCount) cache=\(cacheOutcome) chars=\(collected.text.count) nativeCalls=\(collected.nativeToolCalls.count) textualToolCall=\(textualToolCallDetected) rawPreview=\"\(rawPreview)\" sanitizedPreview=\"\(sanitizedPreview)\""
        )
        #endif
    }

    private func makePrefixDebugSummary(
        plan: MLXExecutionPlan,
        fullInput: LMInput,
        reuseDecision: MLXCacheReuseDecision
    ) async -> String? {
        #if DEBUG
        guard let prefixInput = plan.cachePlan.prefixInput else {
            return nil
        }

        do {
            let prefixPrepared = try await modelContainer.preparePrefix(input: prefixInput)
            let fullTokens = flattenedTokenIDs(from: fullInput.text.tokens)
            let prefixTokens = flattenedTokenIDs(from: prefixPrepared.text.tokens)
            let reusedPrefixTokenCount = reuseDecision.prefixTokenCount ?? 0
            let prefixMatchesFull = fullTokens.count >= prefixTokens.count
                && Array(fullTokens.prefix(prefixTokens.count)) == prefixTokens
            let suffixTokens = reusedPrefixTokenCount <= fullTokens.count
                ? Array(fullTokens.dropFirst(reusedPrefixTokenCount))
                : []
            let prefixTail = Array(prefixTokens.suffix(12))
            let suffixHead = Array(suffixTokens.prefix(12))
            let suffixPreview = await decodedPreview(tokens: suffixTokens)

            return "preparedPrefixTokens=\(prefixTokens.count) reusedPrefixTokens=\(reusedPrefixTokenCount) prefixMatchesFull=\(prefixMatchesFull) suffixTokens=\(suffixTokens.count) prefixTail=\(prefixTail) suffixHead=\(suffixHead) suffixPreview=\"\(suffixPreview)\""
        } catch {
            return "error=\(error)"
        }
        #else
        return nil
        #endif
    }

    private func flattenedTokenIDs(from tokens: MLXArray) -> [Int] {
        tokens.flattened().asArray(Int.self)
    }

    private func decodedPreview(tokens: [Int]) async -> String {
        guard !tokens.isEmpty else {
            return ""
        }
        let previewTokens = Array(tokens.prefix(48))
        let decoded = await modelContainer.decode(tokens: previewTokens)
        return previewText(decoded)
    }

    private func previewText(_ text: String, limit: Int = 160) -> String {
        let normalized = text
            .replacingOccurrences(of: "\n", with: "\\n")
            .replacingOccurrences(of: "\r", with: "\\r")
        if normalized.count <= limit {
            return normalized
        }
        let endIndex = normalized.index(normalized.startIndex, offsetBy: limit)
        return String(normalized[..<endIndex]) + "..."
    }
}
#endif
