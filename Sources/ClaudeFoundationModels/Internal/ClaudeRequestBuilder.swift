#if CLAUDE_ENABLED
import AnyFoundationModelsSupport
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Builds MessagesRequest from Transcript for the Claude Messages API.
internal struct ClaudeRequestBuilder: OpenFoundationModelsExtra.RequestBuilder {

    // MARK: - Build Result

    struct BuildResult: Sendable {
        let request: MessagesRequest
        let betaHeaders: [String]?
        let responseSchema: GenerationSchema?
    }

    // MARK: - Properties

    let modelName: String
    let defaultMaxTokens: Int
    let thinkingBudgetTokens: Int?
    let thinkingBlockManager: ThinkingBlockManager

    // MARK: - Initialization

    init(modelName: String, defaultMaxTokens: Int, thinkingBudgetTokens: Int?) {
        self.modelName = modelName
        self.defaultMaxTokens = defaultMaxTokens
        self.thinkingBudgetTokens = thinkingBudgetTokens
        self.thinkingBlockManager = ThinkingBlockManager()
    }

    // MARK: - RequestBuilder Protocol

    func build(transcript: Transcript, options: GenerationOptions?, stream: Bool) throws -> BuildResult {
        let resolved = transcript.resolved()

        var (messages, systemPrompt) = buildMessages(from: resolved)

        let pendingThinkingBlocks = thinkingBlockManager.take()
        if !pendingThinkingBlocks.isEmpty {
            messages = ThinkingBlockManager.inject(pendingThinkingBlocks, into: messages)
        }

        let tools: [Tool]? = resolved.toolDefinitions.isEmpty ? nil :
            try resolved.toolDefinitions.map { try convertToolDefinition($0) }

        let finalOptions = options ?? resolved.latestOptions
        let claudeOptions = finalOptions?.toClaudeOptions() ?? ClaudeOptions(temperature: nil, topK: nil, topP: nil)

        let (thinking, maxTokens, temperature, topK) = resolveThinkingParameters(
            claudeOptions: claudeOptions,
            requestedMaxTokens: finalOptions?.maximumResponseTokens,
            defaultMaxTokens: defaultMaxTokens,
            thinkingBudgetTokens: thinkingBudgetTokens
        )

        let responseSchema = resolved.latestResponseFormat?._schema
        let outputFormat = try responseSchema.map { try OutputFormat(schema: $0) }
        let betaHeaders: [String]? = outputFormat != nil ? ["structured-outputs-2025-11-13"] : nil

        let request = MessagesRequest(
            model: modelName,
            messages: messages,
            maxTokens: maxTokens,
            system: systemPrompt,
            tools: tools,
            toolChoice: tools != nil ? .auto() : nil,
            stream: stream,
            temperature: temperature,
            topK: topK,
            topP: claudeOptions.topP,
            thinking: thinking,
            outputFormat: outputFormat
        )

        return BuildResult(request: request, betaHeaders: betaHeaders, responseSchema: responseSchema)
    }

    // MARK: - Message Building

    private func buildMessages(from resolved: ResolvedTranscript) -> ([Message], String?) {
        var messages: [Message] = []
        var systemPrompt: String? = nil

        for entry in resolved {
            switch entry {
            case .instructions(let i):
                let content = segmentsToText(
                    SessionToolInstructions.userAuthoredSegments(from: i.segments)
                )
                if !content.isEmpty { systemPrompt = content }

            case .prompt(let p):
                let hasImages = p.segments.contains { if case .image = $0 { return true }; return false }
                if hasImages {
                    messages.append(Message(role: .user, content: convertSegmentsToContentBlocks(p.segments)))
                } else {
                    messages.append(Message(role: .user, content: segmentsToText(p.segments)))
                }

            case .response(let r):
                messages.append(Message(role: .assistant, content: segmentsToText(r.segments)))

            case .tool(let interaction):
                messages.append(Message(role: .assistant, content: convertToolCallsToBlocks(interaction.calls)))
                if !interaction.outputs.isEmpty {
                    var callIdQueue = interaction.calls.map { $0.id }
                    let resultBlocks: [ContentBlock] = interaction.outputs.map { output in
                        let toolUseId = callIdQueue.isEmpty ? output.id : callIdQueue.removeFirst()
                        return .toolResult(ToolResultBlock(
                            toolUseId: toolUseId,
                            content: segmentsToText(output.segments)
                        ))
                    }
                    messages.append(Message(role: .user, content: resultBlocks))
                }
            }
        }

        return (messages, systemPrompt)
    }

    // MARK: - Segment Helpers

    private func convertSegmentsToContentBlocks(_ segments: [Transcript.Segment]) -> [ContentBlock] {
        segments.compactMap { segment in
            switch segment {
            case .text(let t): return .text(TextBlock(text: t.content))
            case .reasoning: return nil
            case .structure(let s): return .text(TextBlock(text: s.content.jsonString))
            case .image(let img):
                switch img.source {
                case .base64(let data, let mediaType):
                    return .image(ImageBlock(source: .init(type: "base64", mediaType: mediaType, data: data, url: nil)))
                case .url(let url):
                    return .image(ImageBlock(source: .init(type: "url", mediaType: nil, data: nil, url: url.absoluteString)))
                }
            }
        }
    }

    private func convertToolCallsToBlocks(_ toolCalls: Transcript.ToolCalls) -> [ContentBlock] {
        toolCalls.map { toolCall in
            .toolUse(ToolUseBlock(
                id: toolCall.id,
                name: toolCall.toolName,
                input: toolCall.arguments.toJSONValue()
            ))
        }
    }

    // MARK: - Tool Definition

    private func convertToolDefinition(_ definition: Transcript.ToolDefinition) throws -> Tool {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let jsonData = try encoder.encode(definition.parameters)
        var schemaValue = try JSONDecoder().decode(JSONValue.self, from: jsonData)
        schemaValue = setAdditionalPropertiesFalse(schemaValue)
        schemaValue = stripNonStandardSchemaKeys(schemaValue)
        if case .object(var dict) = schemaValue {
            dict.removeValue(forKey: "description")
            schemaValue = .object(dict)
        }
        return Tool(name: definition.name, description: definition.description, inputSchema: schemaValue)
    }

    // MARK: - Thinking Parameter Resolution

    private func resolveThinkingParameters(
        claudeOptions: ClaudeOptions,
        requestedMaxTokens: Int?,
        defaultMaxTokens: Int,
        thinkingBudgetTokens: Int?
    ) -> (ThinkingConfig?, Int, Double?, Int?) {
        guard let budget = thinkingBudgetTokens else {
            return (nil, requestedMaxTokens ?? defaultMaxTokens, claudeOptions.temperature, claudeOptions.topK)
        }
        return (.enabled(budgetTokens: budget), budget + (requestedMaxTokens ?? defaultMaxTokens), nil, nil)
    }
}

// MARK: - Claude API Schema Fixup

/// Recursively strips non-standard JSON Schema keys that Claude API rejects.
internal func stripNonStandardSchemaKeys(_ value: JSONValue) -> JSONValue {
    guard case .object(var dict) = value else { return value }
    dict.removeValue(forKey: "title")
    dict.removeValue(forKey: "x-order")
    if let properties = dict["properties"], case .object(var propDict) = properties {
        for (key, propValue) in propDict { propDict[key] = stripNonStandardSchemaKeys(propValue) }
        dict["properties"] = .object(propDict)
    }
    if let items = dict["items"] { dict["items"] = stripNonStandardSchemaKeys(items) }
    if let anyOf = dict["anyOf"], case .array(let schemas) = anyOf {
        dict["anyOf"] = .array(schemas.map { stripNonStandardSchemaKeys($0) })
    }
    return .object(dict)
}

/// Recursively sets `"additionalProperties": false` on all object schemas.
internal func setAdditionalPropertiesFalse(_ value: JSONValue) -> JSONValue {
    guard case .object(var dict) = value else { return value }
    let isObjectType: Bool
    if case .string("object") = dict["type"] { isObjectType = true }
    else if case .array(let types) = dict["type"], types.contains(.string("object")) { isObjectType = true }
    else { isObjectType = false }
    if isObjectType, dict["properties"] != nil { dict["additionalProperties"] = .bool(false) }
    if let properties = dict["properties"], case .object(var propDict) = properties {
        for (key, propValue) in propDict { propDict[key] = setAdditionalPropertiesFalse(propValue) }
        dict["properties"] = .object(propDict)
    }
    if let items = dict["items"] { dict["items"] = setAdditionalPropertiesFalse(items) }
    if let anyOf = dict["anyOf"], case .array(let schemas) = anyOf {
        dict["anyOf"] = .array(schemas.map { setAdditionalPropertiesFalse($0) })
    }
    return .object(dict)
}

// MARK: - Claude Options

internal struct ClaudeOptions: Sendable {
    let temperature: Double?
    let topK: Int?
    let topP: Double?
}

internal extension GenerationOptions {
    func toClaudeOptions() -> ClaudeOptions {
        ClaudeOptions(temperature: temperature, topK: nil, topP: nil)
    }
}

#endif
