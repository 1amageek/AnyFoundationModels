#if MLX_ENABLED
import Foundation
import OpenFoundationModels
@preconcurrency import MLXLMCommon

actor MLXLanguageModelRuntime {
    private let modelContainer: ModelContainer
    private let requestConverter = MLXRequestConverter()
    private let responseConverter = MLXResponseConverter()

    private var metadata: MLXModelMetadata?

    init(modelContainer: ModelContainer) {
        self.modelContainer = modelContainer
    }

    // MARK: - Non-Streaming Generation

    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        let metadata = await resolveMetadata()
        let request = try requestConverter.convert(transcript: transcript, metadata: metadata)
        let hasTools = request.hasTools
        let hasSchema = request.hasSchema
        let lmInput = try await modelContainer.prepare(input: request.input)
        let promptTokenCount = lmInput.text.tokens.size
        let parameters = makeParameters(
            options: options,
            promptTokenCount: promptTokenCount,
            metadata: metadata
        )

        #if DEBUG
        logRequest(
            metadata: metadata,
            hasTools: hasTools,
            hasSchema: hasSchema,
            promptTokenCount: promptTokenCount,
            parameters: parameters
        )
        #endif

        let stream = try await modelContainer.generate(
            input: lmInput,
            parameters: parameters
        )

        var generations: [Generation] = []
        for await generation in stream {
            generations.append(generation)
        }

        return try responseConverter.finalEntry(
            from: generations,
            hasTools: hasTools,
            hasSchema: hasSchema
        )
    }

    // MARK: - Streaming Generation

    func stream(
        transcript: Transcript,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let metadata = await self.resolveMetadata()
                    let request = try self.requestConverter.convert(
                        transcript: transcript,
                        metadata: metadata
                    )
                    let hasTools = request.hasTools
                    let hasSchema = request.hasSchema
                    let lmInput = try await self.modelContainer.prepare(input: request.input)
                    let promptTokenCount = lmInput.text.tokens.size
                    let parameters = self.makeParameters(
                        options: options,
                        promptTokenCount: promptTokenCount,
                        metadata: metadata
                    )

                    #if DEBUG
                    self.logRequest(
                        metadata: metadata,
                        hasTools: hasTools,
                        hasSchema: hasSchema,
                        promptTokenCount: promptTokenCount,
                        parameters: parameters
                    )
                    #endif

                    let stream = try await self.modelContainer.generate(
                        input: lmInput,
                        parameters: parameters
                    )

                    if !hasTools {
                        // Text mode: stream deltas incrementally
                        var streamingState = MLXResponseConverter.StreamingState()
                        var allGenerations: [Generation] = []

                        for await generation in stream {
                            try Task.checkCancellation()
                            allGenerations.append(generation)

                            if case .chunk(let text) = generation {
                                let delta = self.responseConverter.streamDelta(
                                    state: &streamingState,
                                    chunk: text
                                )
                                if !delta.isEmpty {
                                    continuation.yield(
                                        self.responseConverter.streamEntry(for: delta)
                                    )
                                }
                            }
                        }

                        // Flush trailing text after final sanitization
                        let sanitized = self.responseConverter.sanitizeAssistantResponse(
                            streamingState.rawText
                        )
                        if !sanitized.isEmpty,
                           sanitized != streamingState.emittedVisibleText
                        {
                            let startIndex = sanitized.index(
                                sanitized.startIndex,
                                offsetBy: streamingState.emittedVisibleText.count
                            )
                            let trailing = String(sanitized[startIndex...])
                            if !trailing.isEmpty {
                                continuation.yield(
                                    self.responseConverter.streamEntry(for: trailing)
                                )
                            }
                        }
                    } else {
                        // Tool mode: buffer everything, yield final entry at end
                        var allGenerations: [Generation] = []
                        for await generation in stream {
                            try Task.checkCancellation()
                            allGenerations.append(generation)
                        }

                        let entry = try self.responseConverter.finalEntry(
                            from: allGenerations,
                            hasTools: hasTools,
                            hasSchema: hasSchema
                        )
                        continuation.yield(entry)
                    }

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

    // MARK: - Metadata Resolution

    private func resolveMetadata() async -> MLXModelMetadata {
        if let metadata {
            return metadata
        }

        let configuration = await modelContainer.configuration
        let modelID = configuration.name
        let resolved = await MLXModelMetadataRegistry.shared.metadata(for: modelID)
            ?? .fallback(modelID: modelID)
        metadata = resolved
        return resolved
    }

    // MARK: - Parameter Computation

    private func makeParameters(
        options: GenerationOptions?,
        promptTokenCount: Int,
        metadata: MLXModelMetadata
    ) -> GenerateParameters {
        // Prefill step size scales with prompt length
        var prefillStepSize = 512
        if promptTokenCount > 8192 {
            prefillStepSize = 1536
        } else if promptTokenCount > 2048 {
            prefillStepSize = 1024
        }
        // VLM cap
        if metadata.runtimeFamily == .vlm {
            prefillStepSize = min(prefillStepSize, 1024)
        }

        // KV quantization for long prompts
        let kvBits: Int? = promptTokenCount > 8192 ? 4 : nil
        let quantizedKVStart = kvBits == nil ? 0 : 4096

        return GenerateParameters(
            maxTokens: options?.maximumResponseTokens ?? 2048,
            maxKVSize: nil,
            kvBits: kvBits,
            kvGroupSize: 64,
            quantizedKVStart: quantizedKVStart,
            temperature: options?.temperature.map { Float($0) } ?? 0,
            topP: 0.9,
            repetitionPenalty: 1.05,
            repetitionContextSize: 64,
            prefillStepSize: prefillStepSize
        )
    }

    // MARK: - Debug Logging

    #if DEBUG
    private func logRequest(
        metadata: MLXModelMetadata,
        hasTools: Bool,
        hasSchema: Bool,
        promptTokenCount: Int,
        parameters: GenerateParameters
    ) {
        print(
            "[MLXLanguageModel] generate modelID=\(metadata.modelID) runtime=\(metadata.runtimeFamily.rawValue) hasTools=\(hasTools) hasSchema=\(hasSchema) promptTokens=\(promptTokenCount) prefillStep=\(parameters.prefillStepSize) kvBits=\(String(describing: parameters.kvBits))"
        )
    }
    #endif
}
#endif
