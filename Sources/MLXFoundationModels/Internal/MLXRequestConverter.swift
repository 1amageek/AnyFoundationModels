#if MLX_ENABLED
import Foundation
import MLXLMCommon
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Stateless converter: Transcript → UserInput for MLX on-device inference.
struct MLXRequestConverter {

    // MARK: - Conversion Result

    struct ConvertedRequest {
        let input: MLXLMCommon.UserInput
        let hasTools: Bool
        let hasSchema: Bool
    }

    // MARK: - Primary Entry Point

    func convert(
        transcript: Transcript,
        metadata: MLXModelMetadata
    ) throws -> ConvertedRequest {
        let resolved = transcript.resolved()

        // Extract images from prompt segments
        var imageSegments: [Transcript.ImageSegment] = []
        for entry in resolved {
            if case .prompt(let prompt) = entry {
                for segment in prompt.segments {
                    if case .image(let image) = segment {
                        imageSegments.append(image)
                    }
                }
            }
        }
        let images = try ImageSourceConverter.convert(imageSegments)

        // Extract schema JSON from response format
        let schemaJSON: String? = encodeSchema(resolved.latestResponseFormat?._schema)

        // Build chat messages
        let messages = buildMessages(
            from: resolved,
            images: images,
            schemaJSON: schemaJSON
        )

        // Build tool specs
        let hasTools = !resolved.toolDefinitions.isEmpty
        let toolSpecs = hasTools ? buildToolSpecs(from: resolved.toolDefinitions) : nil

        // Build additional context
        let additionalContext = buildAdditionalContext(metadata: metadata)

        let input = UserInput(
            chat: messages,
            tools: toolSpecs,
            additionalContext: additionalContext
        )

        return ConvertedRequest(
            input: input,
            hasTools: hasTools,
            hasSchema: schemaJSON != nil
        )
    }

    // MARK: - Message Building

    private func buildMessages(
        from resolved: ResolvedTranscript,
        images: [UserInput.Image],
        schemaJSON: String?
    ) -> [Chat.Message] {

        enum MessageType { case system, user, assistant, tool }
        var rawMessages: [(type: MessageType, content: String)] = []

        for entry in resolved {
            switch entry {
            case .instructions(let instructions):
                let text = segmentsToText(instructions.segments)
                if !text.isEmpty {
                    rawMessages.append((.system, text))
                }
            case .prompt(let prompt):
                rawMessages.append((.user, segmentsToText(prompt.segments)))
            case .response(let response):
                rawMessages.append((.assistant, segmentsToText(response.segments)))
            case .tool(let interaction):
                let callText = interaction.calls
                    .map { "\($0.toolName)(\($0.arguments.jsonString))" }
                    .joined(separator: "\n")
                if !callText.isEmpty {
                    rawMessages.append((.assistant, callText))
                }
                for output in interaction.outputs {
                    rawMessages.append((.tool, segmentsToText(output.segments)))
                }
            }
        }

        var messages: [Chat.Message] = []

        // System message: join all instructions, append schema if present
        let systemTexts = rawMessages
            .filter { $0.type == .system }
            .map(\.content)
            .filter { !$0.isEmpty }

        if !systemTexts.isEmpty || schemaJSON != nil {
            var content = systemTexts.joined(separator: "\n\n")
            if let schemaJSON {
                if !content.isEmpty { content += "\n\n" }
                content += "Respond with JSON matching this schema:\n\(schemaJSON)"
            }
            messages.append(.system(content))
        }

        // Non-system messages: attach images to last user message only
        let nonSystem = rawMessages.filter { $0.type != .system }
        let lastUserIndex = nonSystem.lastIndex(where: { $0.type == .user })

        for (index, message) in nonSystem.enumerated() {
            switch message.type {
            case .user:
                if index == lastUserIndex && !images.isEmpty {
                    messages.append(.user(message.content, images: images))
                } else {
                    messages.append(.user(message.content))
                }
            case .assistant:
                messages.append(.assistant(message.content))
            case .tool:
                messages.append(.tool(message.content))
            case .system:
                break
            }
        }

        return messages
    }

    // MARK: - Tool Specs

    private func buildToolSpecs(
        from toolDefinitions: [Transcript.ToolDefinition]
    ) -> [[String: any Sendable]]? {
        guard !toolDefinitions.isEmpty else { return nil }

        let specs: [[String: any Sendable]] = toolDefinitions.compactMap { definition in
            var function: [String: any Sendable] = [
                "name": definition.name,
                "description": definition.description,
            ]
            do {
                function["parameters"] = try encodeToolParameters(definition.parameters)
            } catch {
                Logger.warning(
                    "[MLXRequestConverter] Failed to convert tool schema for \(definition.name): \(error)"
                )
            }
            return [
                "type": "function" as any Sendable,
                "function": function as any Sendable,
            ]
        }

        return specs.isEmpty ? nil : specs
    }

    // MARK: - Schema Encoding

    private func encodeToolParameters(
        _ parameters: GenerationSchema
    ) throws -> any Sendable {
        let data = try JSONEncoder().encode(parameters)
        let jsonValue = try JSONDecoder().decode(MLXLMCommon.JSONValue.self, from: data)
        return sendableValue(for: jsonValue)
    }

    private func sendableValue(for jsonValue: MLXLMCommon.JSONValue) -> any Sendable {
        switch jsonValue {
        case .null:
            return Optional<String>.none as any Sendable
        case .bool(let value):
            return value
        case .int(let value):
            return value
        case .double(let value):
            return value
        case .string(let value):
            return value
        case .array(let values):
            return values.map(sendableValue(for:)) as [any Sendable]
        case .object(let values):
            return values.mapValues(sendableValue(for:)) as [String: any Sendable]
        }
    }

    private func encodeSchema(_ schema: GenerationSchema?) -> String? {
        guard let schema else { return nil }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        do {
            let data = try encoder.encode(schema)
            return String(decoding: data, as: UTF8.self)
        } catch {
            Logger.warning("[MLXRequestConverter] Failed to encode schema: \(error)")
            return nil
        }
    }

    // MARK: - Additional Context

    private func buildAdditionalContext(
        metadata: MLXModelMetadata
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = [:]
        switch metadata.thinkingPreference {
        case .enabled:
            context["enable_thinking"] = true
        case .disabled:
            context["enable_thinking"] = false
        case .unspecified:
            break
        }
        return context
    }

    // MARK: - Segment → Text

    private func segmentsToText(_ segments: [Transcript.Segment]) -> String {
        var texts: [String] = []
        var imageIndex = 1
        for segment in segments {
            switch segment {
            case .text(let text):
                texts.append(text.content)
            case .structure(let structure):
                texts.append(structure.content.jsonString)
            case .image:
                texts.append("[Image #\(imageIndex)]")
                imageIndex += 1
            }
        }
        return texts.joined(separator: " ")
    }
}
#endif
