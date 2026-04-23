#if OLLAMA_ENABLED
import AnyFoundationModelsSupport
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Builds ChatRequest from Transcript for the Ollama API.
struct OllamaRequestBuilder: OpenFoundationModelsExtra.RequestBuilder {

    // MARK: - Build Result

    struct BuildResult: Sendable {
        let request: ChatRequest
    }

    // MARK: - Properties

    let configuration: OllamaConfiguration
    let modelName: String
    let thinkingMode: ThinkingMode?

    // MARK: - Structured Output Instructions Template

    private static let structuredOutputInstructionsTemplate = """
        # JSON Response Format

        You MUST respond with a valid JSON object wrapped in a markdown code block.

        ## Schema
        ```json
        {{schema}}
        ```

        ## Required Properties
        {{properties}}

        ## Output Format
        Your response MUST be exactly in this format:
        ```json
        {your JSON here}
        ```

        ## Rules
        1. Output ONLY a single markdown code block with ```json and ``` markers
        2. The JSON must be valid and match the schema exactly
        3. Use double quotes for all strings
        4. Do not include trailing commas
        5. No text before or after the code block
        """

    // MARK: - RequestBuilder Protocol

    func build(transcript: Transcript, options: GenerationOptions?, stream: Bool) -> BuildResult {
        let resolved = transcript.resolved()
        var messages = buildMessages(from: resolved)

        #if DEBUG
        print("[OllamaRequestBuilder] === Transcript ===")
        for entry in transcript {
            print("[OllamaRequestBuilder] \(entry)")
        }
        #endif

        let tools = buildTools(from: resolved)

        #if DEBUG
        if let tools = tools {
            print("[OllamaRequestBuilder] buildTools: \(tools.count) tools")
            for tool in tools {
                print("[OllamaRequestBuilder]   \(tool.function.name)")
            }
        } else {
            print("[OllamaRequestBuilder] buildTools: nil")
        }
        #endif

        let responseFormat = buildFormat(from: resolved)
        let transcriptOptions = options ?? resolved.latestOptions
        let finalOptions = resolveGenerationOptions(transcriptOptions)

        let resolvedThinkingMode: ThinkingMode?
        if responseFormat != nil && responseFormat != .text {
            addStructuredOutputInstructions(to: &messages, format: responseFormat!)
            resolvedThinkingMode = .disabled

            #if DEBUG
            print("[OllamaRequestBuilder] Structured output mode enabled")
            print("[OllamaRequestBuilder] === FINAL SYSTEM MESSAGE ===")
            if let sysMsg = messages.first(where: { $0.role == .system }) {
                print(sysMsg.content)
            }
            print("[OllamaRequestBuilder] === END SYSTEM MESSAGE ===")
            #endif
        } else {
            resolvedThinkingMode = thinkingMode
        }

        let request = ChatRequest(
            model: modelName,
            messages: messages,
            stream: stream,
            options: finalOptions?.toOllamaOptions(),
            format: responseFormat,
            keepAlive: configuration.keepAlive,
            tools: tools,
            think: resolvedThinkingMode
        )

        #if DEBUG
        do {
            let data = try JSONEncoder().encode(request)
            if let json = String(data: data, encoding: .utf8) {
                print("[OllamaRequestBuilder] === HTTP Request Body ===")
                print(json)
                print("[OllamaRequestBuilder] === END ===")
            }
        } catch {
            print("[OllamaRequestBuilder] Failed to encode request for debug logging: \(error)")
        }
        #endif

        return BuildResult(request: request)
    }

    // MARK: - Message Building

    private func buildMessages(from resolved: ResolvedTranscript) -> [Message] {
        var messages: [Message] = []
        for entry in resolved {
            switch entry {
            case .instructions(let i):
                let content = segmentsToText(
                    SessionToolInstructions.userAuthoredSegments(from: i.segments)
                )
                if !content.isEmpty {
                    messages.append(Message(role: .system, content: content))
                }
            case .prompt(let p):
                let content = segmentsToText(p.segments)
                let images = extractImages(from: p.segments)
                messages.append(Message(role: .user, content: content, images: images))
            case .response(let r):
                messages.append(Message(role: .assistant, content: segmentsToText(r.segments)))
            case .tool(let interaction):
                let ollamaToolCalls = interaction.calls.map { call in
                    ToolCall(function: ToolCall.FunctionCall(
                        name: call.toolName,
                        arguments: call.arguments.toJSONValue()
                    ))
                }
                messages.append(Message(role: .assistant, content: "", toolCalls: ollamaToolCalls))
                for output in interaction.outputs {
                    messages.append(Message(
                        role: .tool,
                        content: segmentsToText(output.segments),
                        toolName: output.toolName
                    ))
                }
            }
        }
        return messages
    }

    // MARK: - Tool Building

    private func buildTools(from resolved: ResolvedTranscript) -> [Tool]? {
        guard !resolved.toolDefinitions.isEmpty else { return nil }
        return resolved.toolDefinitions.map { def in
            Tool(
                type: "function",
                function: Tool.Function(
                    name: def.name,
                    description: def.description,
                    parameters: def.parameters._jsonSchema
                )
            )
        }
    }

    // MARK: - Response Format Building

    private func buildFormat(from resolved: ResolvedTranscript) -> ResponseFormat? {
        guard let latestFormat = resolved.latestResponseFormat else { return nil }
        if let schema = latestFormat._schema {
            return .jsonSchema(schema._jsonSchema)
        }
        if latestFormat._type != nil {
            return .json
        }
        return nil
    }

    // MARK: - Image Extraction

    private func extractImages(from segments: [Transcript.Segment]) -> [String]? {
        let images = segments.compactMap { seg -> String? in
            guard case .image(let img) = seg else { return nil }
            if case .base64(let data, _) = img.source { return data }
            return nil
        }
        return images.isEmpty ? nil : images
    }

    // MARK: - Private Helpers

    private func resolveGenerationOptions(_ options: GenerationOptions?) -> GenerationOptions? {
        guard let forcedTemperature = configuration.temperature else {
            return options
        }
        var resolved = options ?? GenerationOptions()
        resolved.temperature = forcedTemperature
        return resolved
    }

    private func addStructuredOutputInstructions(to messages: inout [Message], format: ResponseFormat) {
        let instructions = generateInstructions(for: format)
        guard !instructions.isEmpty else { return }
        for i in 0..<messages.count {
            if messages[i].role == .system {
                messages[i] = Message(
                    role: .system,
                    content: messages[i].content + "\n\n" + instructions
                )
                return
            }
        }
        messages.insert(Message(role: .system, content: instructions), at: 0)
    }

    private func generateInstructions(for format: ResponseFormat) -> String {
        switch format {
        case .jsonSchema(let schema):
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let schemaString: String
            do {
                let jsonData = try encoder.encode(schema)
                schemaString = String(data: jsonData, encoding: .utf8) ?? "{}"
            } catch {
                schemaString = "{}"
            }
            let properties = extractPropertyDescriptions(from: schema)
            return Self.structuredOutputInstructionsTemplate
                .replacingOccurrences(of: "{{schema}}", with: schemaString)
                .replacingOccurrences(of: "{{properties}}", with: properties)
        case .json:
            return Self.structuredOutputInstructionsTemplate
                .replacingOccurrences(of: "{{schema}}", with: #"{"type":"object"}"#)
                .replacingOccurrences(of: "{{properties}}", with: "(dynamic structure)")
        case .text:
            return ""
        }
    }

    private func extractPropertyDescriptions(from schema: JSONSchema) -> String {
        guard case let .object(_, _, _, _, _, _, properties, required, _) = schema,
              !properties.isEmpty else {
            return "(no properties defined)"
        }
        var descriptions: [String] = []
        for (name, propSchema) in properties {
            let type = schemaTypeName(propSchema)
            let desc = schemaDescription(propSchema)
            let isRequired = required.contains(name)
            var line = "- `\(name)` (\(type))"
            if isRequired { line += " [required]" }
            if let desc, !desc.isEmpty { line += ": \(desc)" }
            descriptions.append(line)
        }
        return descriptions.isEmpty ? "(no properties defined)" : descriptions.joined(separator: "\n")
    }

    private func schemaTypeName(_ schema: JSONSchema) -> String {
        switch schema {
        case .object: return "object"
        case .array: return "array"
        case .string: return "string"
        case .number: return "number"
        case .integer: return "integer"
        case .boolean: return "boolean"
        case .null: return "null"
        case .anyOf: return "anyOf"
        case .allOf: return "allOf"
        case .oneOf: return "oneOf"
        case .not: return "not"
        case .reference: return "ref"
        case .any: return "any"
        case .empty: return "any"
        }
    }

    private func schemaDescription(_ schema: JSONSchema) -> String? {
        switch schema {
        case .object(_, let desc, _, _, _, _, _, _, _): return desc
        case .array(_, let desc, _, _, _, _, _, _, _, _): return desc
        case .string(_, let desc, _, _, _, _, _, _, _, _): return desc
        case .number(_, let desc, _, _, _, _, _, _, _, _, _): return desc
        case .integer(_, let desc, _, _, _, _, _, _, _, _, _): return desc
        case .boolean(_, let desc, _): return desc
        default: return nil
        }
    }
}

#endif
