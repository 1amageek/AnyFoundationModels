#if MLX_ENABLED
import Foundation
import OpenFoundationModels
@preconcurrency import MLXLMCommon

public struct MLXLanguageModel: OpenFoundationModels.LanguageModel, Sendable {

    private let runtime: MLXLanguageModelRuntime

    public init(loadedModel: MLXLoadedModel) {
        self.runtime = MLXLanguageModelRuntime(loadedModel: loadedModel)
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
