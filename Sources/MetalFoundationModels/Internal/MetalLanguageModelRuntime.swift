#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import SwiftLM

/// Orchestrates Metal inference: converts OpenFoundationModels types to SwiftLM types.
actor MetalLanguageModelRuntime {

    enum ConversionError: Error, CustomStringConvertible {
        case invalidBase64ImageData

        var description: String {
            switch self {
            case .invalidBase64ImageData:
                return "Failed to decode base64 image data from transcript"
            }
        }
    }

    private let inferenceSession: InferenceSession
    private let showsThinking: Bool

    init(inferenceSession: InferenceSession, showsThinking: Bool) {
        self.inferenceSession = inferenceSession
        self.showsThinking = showsThinking
    }

    /// Generate a complete response.
    func generate(
        transcript: Transcript,
        options: GenerationOptions?
    ) async throws -> Transcript.Entry {
        let messages = try convertTranscript(transcript)
        let parameters = convertOptions(options)
        let input = ModelInput(
            chat: messages,
            promptOptions: PromptPreparationOptions(isThinkingEnabled: showsThinking)
        )

        var answerText = ""
        var reasoningText = ""
        let stream = try await inferenceSession.generate(input, parameters: parameters)
        for await generation in stream {
            switch generation {
            case .text(let chunk):
                answerText += chunk
            case .reasoning(let chunk):
                reasoningText += chunk
            case .completed:
                break
            }
        }

        return .response(.init(
            assetIDs: [],
            segments: makeResponseSegments(answer: answerText, reasoning: reasoningText)
        ))
    }

    /// Stream partial responses.
    func stream(
        transcript: Transcript,
        options: GenerationOptions?
    ) -> AsyncThrowingStream<Transcript.Entry, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    let messages = try self.convertTranscript(transcript)
                    let parameters = self.convertOptions(options)
                    let input = ModelInput(
                        chat: messages,
                        promptOptions: PromptPreparationOptions(isThinkingEnabled: self.showsThinking)
                    )

                    let generationStream = try await self.inferenceSession.generate(
                        input,
                        parameters: parameters
                    )
                    for await generation in generationStream {
                        switch generation {
                        case .text(let chunk):
                            continuation.yield(
                                .response(.init(assetIDs: [], segments: [.text(.init(content: chunk))]))
                            )
                        case .reasoning(let chunk):
                            continuation.yield(
                                .response(.init(assetIDs: [], segments: [.reasoning(.init(content: chunk))]))
                            )
                        case .completed:
                            break
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

    private func convertTranscript(_ transcript: Transcript) throws -> [InputMessage] {
        let resolved = transcript.resolved()
        var messages: [InputMessage] = []

        for entry in resolved {
            switch entry {
            case .prompt(let prompt):
                let content = try convertSegments(prompt.segments)
                if !content.isEmpty {
                    messages.append(.user(content))
                }

            case .response(let response):
                let text = textFromSegments(response.segments)
                if !text.isEmpty {
                    messages.append(.assistant(text))
                }

            case .instructions(let instructions):
                let text = textFromSegments(instructions.segments)
                if !text.isEmpty {
                    messages.append(.system(text))
                }

            default:
                break
            }
        }

        return messages
    }

    private func convertSegments(_ segments: [Transcript.Segment]) throws -> [InputMessage.Content] {
        try segments.compactMap { segment -> InputMessage.Content? in
            switch segment {
            case .text(let textSegment):
                return .text(textSegment.content)
            case .reasoning:
                return nil
            case .image(let imageSegment):
                return try convertImageSegment(imageSegment)
            case .structure(let structure):
                return .text(structure.content.jsonString)
            }
        }
    }

    private func convertImageSegment(_ segment: Transcript.ImageSegment) throws -> InputMessage.Content {
        switch segment.source {
        case .url(let url):
            return .image(InputImage(fileURL: url))
        case .base64(let base64String, let mediaType):
            guard let data = Data(base64Encoded: base64String) else {
                throw ConversionError.invalidBase64ImageData
            }
            return .image(InputImage(data: data, mimeType: mediaType))
        }
    }

    private func textFromSegments(_ segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment -> String? in
            if case .text(let textSegment) = segment {
                return textSegment.content
            }
            return nil
        }.joined()
    }

    private func convertOptions(_ options: GenerationOptions?) -> GenerationParameters {
        var params = GenerationParameters(temperature: 0)
        params.reasoning = showsThinking ? .separate : .hidden
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

    private func makeResponseSegments(answer: String, reasoning: String) -> [Transcript.Segment] {
        var segments: [Transcript.Segment] = []
        if !reasoning.isEmpty {
            segments.append(.reasoning(.init(content: reasoning)))
        }
        if !answer.isEmpty {
            segments.append(.text(.init(content: answer)))
        }
        return segments
    }
}

#endif
