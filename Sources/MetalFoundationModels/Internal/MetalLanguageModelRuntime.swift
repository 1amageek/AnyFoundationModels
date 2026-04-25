#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM

/// Orchestrates Metal inference using swift-lm.
actor MetalLanguageModelRuntime {
    private let configuration: ModelConfiguration
    private let generateStream:
        @Sendable (ModelInput, GenerationParameters) async throws -> AsyncStream<GenerationEvent>
    private let showsThinking: Bool
    private let requestConverter = MetalRequestConverter()
    private let responseConverter = MetalResponseConverter()

    init(container: LanguageModelContainer, showsThinking: Bool) {
        self.configuration = container.configuration
        self.generateStream = { input, parameters in
            try await container.generate(input, parameters: parameters)
        }
        self.showsThinking = showsThinking
    }

    init(
        configuration: ModelConfiguration,
        showsThinking: Bool,
        generateStream: @escaping
            @Sendable (ModelInput, GenerationParameters) async throws -> AsyncStream<GenerationEvent>
    ) {
        self.configuration = configuration
        self.generateStream = generateStream
        self.showsThinking = showsThinking
    }

    /// Generate a complete response.
    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        let request = try await requestConverter.convert(
            transcript: transcript,
            configuration: configuration,
            showsThinking: showsThinking
        )
        let parameters = Self.makeParameters(options: options, showsThinking: showsThinking)
        let stream = try await generateStream(request.input, parameters)

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
                        configuration: self.configuration,
                        showsThinking: self.showsThinking
                    )
                    let parameters = Self.makeParameters(
                        options: options,
                        showsThinking: self.showsThinking
                    )
                    let generationStream = try await self.generateStream(
                        request.input,
                        parameters
                    )

                    var buffered: [GenerationEvent] = []
                    var streamingState = MetalResponseConverter.StreamingState()
                    var hasYieldedResponseEntry = false

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
                            hasYieldedResponseEntry = true
                        }
                    }

                    let finalEntry = try self.responseConverter.finalEntry(
                        from: buffered,
                        hasTools: request.hasTools,
                        hasSchema: request.hasSchema
                    )

                    switch finalEntry {
                    case .toolCalls:
                        continuation.yield(finalEntry)
                    case .response where !hasYieldedResponseEntry:
                        continuation.yield(finalEntry)
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

    static func makeParameters(
        options: GenerationOptions?,
        showsThinking: Bool
    ) -> GenerationParameters {
        var params = GenerationParameters(
            streamChunkTokenCount: 1,
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
