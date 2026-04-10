#if MLX_ENABLED
import Foundation
import OpenFoundationModels
@preconcurrency import MLXLMCommon

actor MLXLanguageModelRuntime {
    private let modelContainer: ModelContainer
    private let profile: MLXModelProfile
    private let requestConverter = MLXRequestConverter()
    private let responseConverter = MLXResponseConverter()

    init(loadedModel: MLXLoadedModel) {
        self.modelContainer = loadedModel.container
        self.profile = loadedModel.profile
    }

    // MARK: - Non-Streaming Generation

    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        let request = try requestConverter.convert(transcript: transcript, profile: profile)
        let hasTools = request.hasTools
        let hasSchema = request.hasSchema
        let lmInput = try await modelContainer.prepare(input: request.input)
        let promptTokenCount = lmInput.text.tokens.size
        let parameters = makeParameters(
            options: options,
            promptTokenCount: promptTokenCount,
            profile: profile
        )

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
                    let request = try self.requestConverter.convert(
                        transcript: transcript,
                        profile: self.profile
                    )
                    let hasTools = request.hasTools
                    let hasSchema = request.hasSchema
                    let lmInput = try await self.modelContainer.prepare(input: request.input)
                    let promptTokenCount = lmInput.text.tokens.size
                    let parameters = self.makeParameters(
                        options: options,
                        promptTokenCount: promptTokenCount,
                        profile: self.profile
                    )

                    let stream = try await self.modelContainer.generate(
                        input: lmInput,
                        parameters: parameters
                    )

                    var generations: [Generation] = []
                    var streamingState = MLXResponseConverter.StreamingState()
                    var hasYieldedResponseEntry = false
                    for await generation in stream {
                        try Task.checkCancellation()
                        generations.append(generation)
                        if case .chunk(let text) = generation {
                            let deltas = self.responseConverter.streamDelta(
                                state: &streamingState,
                                chunk: text
                            )
                            if let entry = self.responseConverter.streamEntry(
                                answer: deltas.answer,
                                reasoning: deltas.reasoning
                            ) {
                                continuation.yield(entry)
                                hasYieldedResponseEntry = true
                            }
                        }
                    }

                    let final = try self.responseConverter.finalEntry(
                        from: generations,
                        hasTools: hasTools,
                        hasSchema: hasSchema
                    )
                    switch final {
                    case .toolCalls:
                        continuation.yield(final)
                    case .response where !hasYieldedResponseEntry:
                        continuation.yield(final)
                    default:
                        break
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

    // MARK: - Parameter Computation

    private func makeParameters(
        options: GenerationOptions?,
        promptTokenCount: Int,
        profile: MLXModelProfile
    ) -> GenerateParameters {
        var prefillStepSize = 512
        if promptTokenCount > 8192 {
            prefillStepSize = 1536
        } else if promptTokenCount > 2048 {
            prefillStepSize = 1024
        }
        if profile.runtimeFamily == .vlm {
            prefillStepSize = min(prefillStepSize, 1024)
        }

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
}
#endif
