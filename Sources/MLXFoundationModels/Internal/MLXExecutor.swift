#if MLX_ENABLED
import Foundation
import MLX
@preconcurrency import MLXLMCommon

enum MLXGenerationEvent: Sendable, Equatable {
    case textChunk(String)
    case nativeToolCall(name: String, argsJSON: String)
    case info
    case completed
}

struct MLXExecutionPreparation {
    let input: LMInput
    let promptTokenCount: Int
    let reusedPrefixTokenCount: Int?
    let cacheOutcome: String
}

struct MLXExecutor {
    func prepareInput(
        container: ModelContainer,
        input: consuming sending UserInput
    ) async throws -> (input: LMInput, promptTokenCount: Int) {
        let fullInput = try await container.prepare(input: input)
        return (fullInput, fullInput.text.tokens.size)
    }

    func makeExecutionPreparation(
        input fullInput: consuming sending LMInput,
        reuseDecision: MLXCacheReuseDecision
    ) -> MLXExecutionPreparation {
        let promptTokenCount = fullInput.text.tokens.size

        guard let prefixTokenCount = reuseDecision.prefixTokenCount,
              let _ = reuseDecision.cache,
              fullInput.image == nil,
              fullInput.video == nil,
              prefixTokenCount > 0,
              prefixTokenCount < promptTokenCount
        else {
            return MLXExecutionPreparation(
                input: fullInput,
                promptTokenCount: promptTokenCount,
                reusedPrefixTokenCount: nil,
                cacheOutcome: reuseDecision.outcome
            )
        }

        return MLXExecutionPreparation(
            input: fullInput,
            promptTokenCount: promptTokenCount,
            reusedPrefixTokenCount: prefixTokenCount,
            cacheOutcome: reuseDecision.outcome
        )
    }

    func execute(
        container: ModelContainer,
        input: consuming sending LMInput,
        cache: [KVCache]?,
        parameters: GenerateParameters,
        reusedPrefixTokenCount: Int?
    ) async throws -> AsyncThrowingStream<MLXGenerationEvent, Error> {
        var effectiveParameters = parameters
        effectiveParameters.reusedPrefixTokenCount = reusedPrefixTokenCount ?? 0

        let stream: AsyncStream<Generation> = try await container.perform(
            values: (MLXTransferBox(input), effectiveParameters)
        ) { context, payload in
            let (boxedInput, generationParameters) = payload
            return try MLXLMCommon.generate(
                input: boxedInput.consume(),
                cache: cache,
                parameters: generationParameters,
                context: context
            )
        }

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let encoder = JSONEncoder()
                    for await generation in stream {
                        try Task.checkCancellation()
                        switch generation {
                        case .chunk(let text):
                            continuation.yield(.textChunk(text))
                        case .toolCall(let call):
                            let data = try encoder.encode(call.function.arguments)
                            let json = String(decoding: data, as: UTF8.self)
                            continuation.yield(.nativeToolCall(name: call.function.name, argsJSON: json))
                        case .info:
                            continuation.yield(.info)
                        }
                    }
                    continuation.yield(.completed)
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

    func buildPrefixSnapshot(
        container: ModelContainer,
        input: consuming sending UserInput,
        parameters: GenerateParameters
    ) async throws -> PromptCacheSnapshot {
        let lmInput = try await container.preparePrefix(input: input)
        return try await container.perform { context in
            var cache = context.model.newCache(parameters: parameters)
            let prefixTokenCount = lmInput.text.tokens.size

            switch try context.model.prepare(lmInput, cache: cache, windowSize: parameters.prefillStepSize) {
            case .tokens(let tokens):
                let output = context.model(tokens[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: nil)
                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: parameters.kvBits,
                    kvGroupSize: parameters.kvGroupSize,
                    quantizedKVStart: parameters.quantizedKVStart
                )
                eval(output.logits)
            case .logits(let output):
                maybeQuantizeKVCache(
                    cache: &cache,
                    kvBits: parameters.kvBits,
                    kvGroupSize: parameters.kvGroupSize,
                    quantizedKVStart: parameters.quantizedKVStart
                )
                eval(output.logits)
            }

            return try capturePromptCache(
                cache: cache,
                prefixTokenCount: prefixTokenCount
            )
        }
    }
}

private final class MLXTransferBox<T>: @unchecked Sendable {
    private var value: T?

    init(_ value: consuming T) {
        self.value = consume value
    }

    consuming func consume() -> T {
        guard let value else {
            fatalError("value already consumed")
        }
        self.value = nil
        return value
    }
}
#endif
