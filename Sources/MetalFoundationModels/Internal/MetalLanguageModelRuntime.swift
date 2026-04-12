#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM

/// Orchestrates Metal inference using swift-lm.
actor MetalLanguageModelRuntime {
    private let container: LanguageModelContainer
    private let showsThinking: Bool
    private let requestConverter = MetalRequestConverter()
    private let responseConverter = MetalResponseConverter()

    init(container: LanguageModelContainer, showsThinking: Bool) {
        self.container = container
        self.showsThinking = showsThinking
    }

    /// Generate a complete response.
    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        let request = try await requestConverter.convert(
            transcript: transcript,
            configuration: container.configuration,
            showsThinking: showsThinking
        )
        let parameters = Self.makeParameters(options: options, showsThinking: showsThinking)
        let stream = try await container.generate(request.input, parameters: parameters)

        var generations: [GenerationEvent] = []
        for await generation in stream {
            generations.append(generation)
        }

        return try responseConverter.finalEntry(
            from: generations,
            hasTools: request.hasTools,
            hasSchema: request.hasSchema
        )
    }

    /// Stream partial responses.
    func stream(
        transcript: Transcript,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let request = try await self.requestConverter.convert(
                        transcript: transcript,
                        configuration: self.container.configuration,
                        showsThinking: self.showsThinking
                    )
                    let parameters = Self.makeParameters(
                        options: options,
                        showsThinking: self.showsThinking
                    )
                    let generationStream = try await self.container.generate(
                        request.input,
                        parameters: parameters
                    )

                    if request.hasTools {
                        var buffered: [GenerationEvent] = []
                        for await generation in generationStream {
                            try Task.checkCancellation()
                            buffered.append(generation)
                        }

                        let entry = try self.responseConverter.finalEntry(
                            from: buffered,
                            hasTools: request.hasTools,
                            hasSchema: request.hasSchema
                        )
                        continuation.yield(entry)
                    } else {
                        var buffered: [GenerationEvent] = []
                        var streamingState = MetalResponseConverter.StreamingState()
                        var hasYieldedEntry = false

                        for await generation in generationStream {
                            try Task.checkCancellation()
                            buffered.append(generation)
                            let deltas = self.responseConverter.streamDelta(
                                state: &streamingState,
                                event: generation
                            )
                            if let entry = self.responseConverter.streamEntry(
                                answer: deltas.answer,
                                reasoning: deltas.reasoning
                            ) {
                                continuation.yield(entry)
                                hasYieldedEntry = true
                            }
                        }

                        if !hasYieldedEntry {
                            let entry = try self.responseConverter.finalEntry(
                                from: buffered,
                                hasTools: request.hasTools,
                                hasSchema: request.hasSchema
                            )
                            continuation.yield(entry)
                        }
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

    static func makeParameters(
        options: GenerationOptions?,
        showsThinking: Bool
    ) -> GenerationParameters {
        var params = GenerationParameters(
            temperature: 0,
            topP: 0.9,
            repetitionPenalty: 1.05,
            repetitionContextSize: 64,
            reasoning: showsThinking ? .separate : .hidden
        )

        if let options {
            if let temperature = options.temperature {
                params.temperature = Float(temperature)
            }
            if let maxTokens = options.maximumResponseTokens {
                params.maxTokens = maxTokens
            }
        }
        return params
    }
}

#endif
