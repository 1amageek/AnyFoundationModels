#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import SwiftLM

/// Metal-accelerated language model using swift-lm's direct Metal inference.
///
/// No MLX dependency. Uses STAF quantized weight format with hand-written
/// Metal kernels for maximum performance on Apple Silicon.
public struct MetalLanguageModel: OpenFoundationModels.LanguageModel, Sendable {

    private let runtime: MetalLanguageModelRuntime

    public init(container: LanguageModelContainer, showsThinking: Bool = false) {
        self.runtime = MetalLanguageModelRuntime(
            container: container,
            showsThinking: showsThinking
        )
    }

    public init(languageModelContainer: LanguageModelContainer, showsThinking: Bool = false) {
        self.init(container: languageModelContainer, showsThinking: showsThinking)
    }

    public var isAvailable: Bool { true }

    public func supports(locale: Locale) -> Bool { true }

    public func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        try await runtime.generate(transcript: transcript, options: options)
    }

    public func stream(
        transcript: Transcript,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let stream = await runtime.stream(transcript: transcript, options: options)
                    for try await entry in stream {
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
}

#endif
