#if RESPONSE_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Builds ResponsesRequest from Transcript for the OpenAI Responses API.
internal struct ResponseRequestBuilder: OpenFoundationModelsExtra.RequestBuilder {

    // MARK: - Build Result

    struct BuildResult: Sendable {
        let request: ResponsesRequest
    }

    // MARK: - Properties

    let modelName: String

    // MARK: - RequestBuilder Protocol

    func build(transcript: Transcript, options: GenerationOptions?, stream: Bool) -> BuildResult {
        let resolved = transcript.resolved()
        let inputItems = buildInputItems(from: resolved)
        let tools = buildTools(from: resolved)
        let textFormat = buildFormat(from: resolved)
        let resolvedOptions = options ?? resolved.latestOptions

        // Extract system instructions from the first system message
        var instructions: String?
        var filteredItems = inputItems
        if let first = inputItems.first,
           case .message(let msg) = first, msg.role == "system" {
            switch msg.content {
            case .text(let text):
                instructions = text
            case .parts:
                instructions = nil
            }
            filteredItems = Array(inputItems.dropFirst())
        }

        var request = ResponsesRequest(
            model: modelName,
            input: filteredItems,
            instructions: instructions,
            tools: tools,
            stream: stream
        )

        if let opts = resolvedOptions {
            request.maxOutputTokens = opts.maximumResponseTokens
            if let temperature = opts.temperature {
                request.temperature = temperature
            }
        }

        if let format = textFormat {
            request.text = format
        }

        return BuildResult(request: request)
    }

    // MARK: - Input Items

    private func buildInputItems(from resolved: ResolvedTranscript) -> [InputItem] {
        var items: [InputItem] = []

        for entry in resolved {
            switch entry {
            case .instructions(let i):
                let text = segmentsToText(i.segments)
                if !text.isEmpty {
                    items.append(.message(MessageItem(role: "system", content: text)))
                }

            case .prompt(let p):
                let hasImages = p.segments.contains { if case .image = $0 { return true }; return false }
                if hasImages {
                    let parts = buildContentParts(from: p.segments)
                    items.append(.message(MessageItem(role: "user", content: parts)))
                } else {
                    items.append(.message(MessageItem(role: "user", content: segmentsToText(p.segments))))
                }

            case .response(let r):
                items.append(.message(MessageItem(role: "assistant", content: segmentsToText(r.segments))))

            case .tool(let interaction):
                var callIdQueue = interaction.calls.map { $0.id }
                for call in interaction.calls {
                    items.append(.functionCall(FunctionCallItem(
                        id: call.id,
                        callId: call.id,
                        name: call.toolName,
                        arguments: call.arguments.jsonString,
                        status: nil
                    )))
                }
                for output in interaction.outputs {
                    let callId = callIdQueue.isEmpty ? output.id : callIdQueue.removeFirst()
                    items.append(.functionCallOutput(FunctionCallOutputItem(
                        callId: callId,
                        output: segmentsToText(output.segments)
                    )))
                }
            }
        }

        return items
    }

    private func buildContentParts(from segments: [Transcript.Segment]) -> [InputContentPart] {
        segments.compactMap { segment in
            switch segment {
            case .text(let t): return .text(t.content)
            case .reasoning: return nil
            case .structure(let s): return .text(s.content.jsonString)
            case .image(let img):
                switch img.source {
                case .base64(let data, let mediaType):
                    return .image(url: "data:\(mediaType);base64,\(data)")
                case .url(let url):
                    return .image(url: url.absoluteString)
                }
            }
        }
    }

    // MARK: - Tool Building

    private func buildTools(from resolved: ResolvedTranscript) -> [ToolDefinition]? {
        guard !resolved.toolDefinitions.isEmpty else { return nil }
        return resolved.toolDefinitions.map { def in
            ToolDefinition(
                name: def.name,
                description: def.description,
                parameters: def.parameters._jsonSchema,
                strict: nil
            )
        }
    }

    // MARK: - Response Format Building

    private func buildFormat(from resolved: ResolvedTranscript) -> TextFormat? {
        guard let latestFormat = resolved.latestResponseFormat else { return nil }
        if let schema = latestFormat._schema {
            do {
                let jsonValue = try JSONValue(schema._jsonSchema)
                let name = String(describing: type(of: schema))
                return TextFormat(format: .jsonSchema(name: name, schema: jsonValue, strict: true))
            } catch {
                return TextFormat(format: .jsonObject)
            }
        }
        if latestFormat._type != nil {
            return TextFormat(format: .jsonObject)
        }
        return nil
    }
}

#endif
