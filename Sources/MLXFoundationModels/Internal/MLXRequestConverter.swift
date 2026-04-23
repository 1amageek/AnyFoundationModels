#if MLX_ENABLED
import AnyFoundationModelsSupport
import Foundation
import MLXLMCommon
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Stateless converter: Transcript → UserInput for MLX on-device inference.
struct MLXRequestConverter {
    enum ConversionError: LocalizedError {
        case unsupportedImageInput(modelID: String)

        var errorDescription: String? {
            switch self {
            case .unsupportedImageInput(let modelID):
                return "Model '\(modelID)' does not support image input."
            }
        }
    }

    // MARK: - Conversion Result

    struct ConvertedRequest {
        let input: MLXLMCommon.UserInput
        let hasTools: Bool
        let hasSchema: Bool
    }

    // MARK: - Primary Entry Point

    func convert(
        transcript: Transcript,
        profile: MLXModelProfile
    ) throws -> ConvertedRequest {
        let resolved = transcript.resolved()
        let responseSchema = resolved.latestResponseFormat?._schema

        let messages = try buildMessages(from: resolved, profile: profile)
        let toolSpecs = buildToolSpecs(from: resolved.toolDefinitions)
        let additionalContext = buildAdditionalContext(
            profile: profile,
            responseSchema: responseSchema
        )

        let input = UserInput(
            chat: messages,
            tools: toolSpecs,
            additionalContext: additionalContext.isEmpty ? nil : additionalContext
        )

        return ConvertedRequest(
            input: input,
            hasTools: !resolved.toolDefinitions.isEmpty,
            hasSchema: responseSchema != nil
        )
    }

    // MARK: - Message Building

    private func buildMessages(
        from resolved: ResolvedTranscript,
        profile: MLXModelProfile
    ) throws -> [Chat.Message] {
        var messages: [Chat.Message] = []

        for entry in resolved {
            switch entry {
            case .instructions(let instructions):
                let text = segmentsToText(
                    SessionToolInstructions.userAuthoredSegments(from: instructions.segments)
                )
                if !text.isEmpty {
                    messages.append(.system(text))
                }
            case .prompt(let prompt):
                let content = segmentsToText(prompt.segments)
                let imgs = try images(from: prompt.segments, profile: profile)
                if imgs.isEmpty {
                    messages.append(.user(content))
                } else {
                    messages.append(.user(content, images: imgs))
                }
            case .response(let response):
                messages.append(.assistant(segmentsToText(response.segments)))
            case .tool(let interaction):
                let callText = interaction.calls
                    .map { "\($0.toolName)(\($0.arguments.jsonString))" }
                    .joined(separator: "\n")
                if !callText.isEmpty {
                    messages.append(.assistant(callText))
                }
                for output in interaction.outputs {
                    messages.append(.tool(segmentsToText(output.segments)))
                }
            }
        }

        return messages
    }

    // MARK: - Additional Context

    private func buildAdditionalContext(
        profile: MLXModelProfile,
        responseSchema: GenerationSchema?
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = [:]

        switch profile.thinkingPreference {
        case .enabled:
            context["enable_thinking"] = true
        case .disabled:
            context["enable_thinking"] = false
        case .unspecified:
            break
        }

        if let schema = responseSchema, let schemaJSON = encodeSchemaString(schema) {
            context["response_schema"] = schemaJSON
        }

        return context
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

    private func encodeSchemaString(_ schema: GenerationSchema) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        do {
            let data = try encoder.encode(schema)
            return String(decoding: data, as: UTF8.self)
        } catch {
            Logger.warning("[MLXRequestConverter] Failed to encode response schema: \(error)")
            return nil
        }
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

    // MARK: - Segment → Text

    private func segmentsToText(_ segments: [Transcript.Segment]) -> String {
        var texts: [String] = []
        var imageIndex = 1
        for segment in segments {
            switch segment {
            case .text(let text):
                texts.append(text.content)
            case .reasoning:
                continue
            case .structure(let structure):
                texts.append(structure.content.jsonString)
            case .image:
                texts.append("[Image #\(imageIndex)]")
                imageIndex += 1
            }
        }
        return texts.joined(separator: " ")
    }

    private func images(
        from segments: [Transcript.Segment],
        profile: MLXModelProfile
    ) throws -> [UserInput.Image] {
        let imageSegments = segments.compactMap { segment -> Transcript.ImageSegment? in
            if case .image(let image) = segment {
                return image
            }
            return nil
        }

        guard !imageSegments.isEmpty else {
            return []
        }

        guard profile.supportsImages else {
            throw ConversionError.unsupportedImageInput(modelID: profile.modelID)
        }

        return try ImageSourceConverter.convert(imageSegments)
    }
}
#endif
