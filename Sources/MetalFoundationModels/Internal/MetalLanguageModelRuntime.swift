#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import SwiftLM

/// Orchestrates Metal inference: converts OpenFoundationModels types to SwiftLM types.
actor MetalLanguageModelRuntime {

    private let modelContainer: ModelContainer

    init(modelContainer: ModelContainer) {
        self.modelContainer = modelContainer
    }

    /// Generate a complete response.
    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        print("[MetalRuntime] generate: converting transcript")
        let messages = convertTranscript(transcript)
        print("[MetalRuntime] generate: \(messages.count) messages")
        let input = try modelContainer.prepare(input: UserInput(chat: messages))
        let parameters = convertOptions(options)
        print("[MetalRuntime] generate: starting inference")

        var fullText = ""
        let stream = modelContainer.generate(input: input, parameters: parameters)
        for await generation in stream {
            if let chunk = generation.chunk {
                fullText += chunk
            }
        }

        print("[MetalRuntime] generate: complete, \(fullText.count) chars")
        return .response(.init(assetIDs: [], segments: [.text(.init(content: fullText))]))
    }

    /// Stream partial responses.
    func stream(
        transcript: Transcript,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let messages = self.convertTranscript(transcript)
                    let input = try self.modelContainer.prepare(input: UserInput(chat: messages))
                    let parameters = self.convertOptions(options)

                    var accumulatedText = ""
                    let generationStream = self.modelContainer.generate(input: input, parameters: parameters)
                    for await generation in generationStream {
                        if let chunk = generation.chunk {
                            accumulatedText += chunk
                            continuation.yield(
                                .response(.init(assetIDs: [], segments: [.text(.init(content: accumulatedText))]))
                            )
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    // MARK: - Conversion

    private func convertTranscript(_ transcript: Transcript) -> [ChatMessage] {
        let resolved = transcript.resolved()
        var messages: [ChatMessage] = []

        for entry in resolved {
            switch entry {
            case .prompt(let prompt):
                let text = prompt.segments.compactMap { segment -> String? in
                    if case .text(let textSegment) = segment {
                        return textSegment.content
                    }
                    return nil
                }.joined()

                if !text.isEmpty {
                    messages.append(ChatMessage(role: .user, content: text))
                }

            case .response(let response):
                let text = response.segments.compactMap { segment -> String? in
                    if case .text(let textSegment) = segment {
                        return textSegment.content
                    }
                    return nil
                }.joined()

                if !text.isEmpty {
                    messages.append(ChatMessage(role: .assistant, content: text))
                }

            case .instructions(let instructions):
                let text = instructions.segments.compactMap { segment -> String? in
                    if case .text(let textSegment) = segment {
                        return textSegment.content
                    }
                    return nil
                }.joined()
                if !text.isEmpty {
                    messages.append(ChatMessage(role: .system, content: text))
                }

            default:
                break
            }
        }

        return messages
    }

    private func convertOptions(_ options: GenerationOptions?) -> GenerateParameters {
        // GenerationOptions from OpenFoundationModels may not expose
        // temperature/topP directly. Use defaults for now.
        GenerateParameters()
    }
}

#endif
