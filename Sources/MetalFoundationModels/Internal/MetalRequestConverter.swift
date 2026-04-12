#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import SwiftLM

/// Stateless converter: Transcript → ModelInput for swift-lm backed Metal inference.
struct MetalRequestConverter {
    enum ConversionError: LocalizedError {
        case invalidBase64ImageData
        case remoteImageLoadFailed(url: URL, underlyingError: Error)

        var errorDescription: String? {
            switch self {
            case .invalidBase64ImageData:
                return "Failed to decode base64 image data from transcript."
            case .remoteImageLoadFailed(let url, let underlyingError):
                return "Failed to load remote image '\(url.absoluteString)': \(underlyingError.localizedDescription)"
            }
        }
    }

    struct ConvertedRequest {
        let input: ModelInput
        let hasTools: Bool
        let hasSchema: Bool
    }

    func convert(
        transcript: Transcript,
        configuration: ModelConfiguration,
        showsThinking: Bool
    ) async throws -> ConvertedRequest {
        let resolved = transcript.resolved()
        let responseSchema = resolved.latestResponseFormat?._schema

        var messages: [InputMessage] = []
        if let systemMessage = try buildSystemMessage(
            from: resolved,
            toolDefinitions: resolved.toolDefinitions,
            responseSchema: responseSchema
        ) {
            messages.append(.system(systemMessage))
        }

        for entry in resolved {
            switch entry {
            case .instructions:
                continue
            case .prompt(let prompt):
                let content = try await convertPromptSegments(
                    prompt.segments,
                    configuration: configuration
                )
                if !content.isEmpty {
                    messages.append(.user(content))
                }
            case .response(let response):
                let text = segmentsToText(response.segments)
                if !text.isEmpty {
                    messages.append(.assistant(text))
                }
            case .tool(let interaction):
                let callText = interaction.calls
                    .map { "\($0.toolName)(\($0.arguments.jsonString))" }
                    .joined(separator: "\n")
                if !callText.isEmpty {
                    messages.append(.assistant(callText))
                }
                for output in interaction.outputs {
                    let text = segmentsToText(output.segments)
                    if !text.isEmpty {
                        messages.append(.tool(text))
                    }
                }
            }
        }

        return ConvertedRequest(
            input: ModelInput(
                chat: messages,
                promptOptions: PromptPreparationOptions(isThinkingEnabled: showsThinking)
            ),
            hasTools: !resolved.toolDefinitions.isEmpty,
            hasSchema: responseSchema != nil
        )
    }

    private func buildSystemMessage(
        from resolved: ResolvedTranscript,
        toolDefinitions: [Transcript.ToolDefinition],
        responseSchema: GenerationSchema?
    ) throws -> String? {
        var sections: [String] = []

        let instructionText = resolved.compactMap { entry -> String? in
            guard case .instructions(let instructions) = entry else {
                return nil
            }
            let text = segmentsToText(instructions.segments)
            return text.isEmpty ? nil : text
        }

        if !instructionText.isEmpty {
            sections.append(instructionText.joined(separator: "\n\n"))
        }

        if let toolInstructions = try buildToolInstructions(toolDefinitions) {
            sections.append(toolInstructions)
        }

        if let responseSchema {
            sections.append(
                """
                Respond with JSON that matches this schema exactly:
                \(try encodeSchemaString(responseSchema))
                """
            )
        }

        guard !sections.isEmpty else {
            return nil
        }
        return sections.joined(separator: "\n\n")
    }

    private func buildToolInstructions(
        _ toolDefinitions: [Transcript.ToolDefinition]
    ) throws -> String? {
        guard !toolDefinitions.isEmpty else {
            return nil
        }

        var lines: [String] = [
            """
            If a tool is required, respond with JSON containing a top-level "tool_calls" array. \
            Each item must include "name" and "arguments". Use only the tools defined below.
            """
        ]

        for definition in toolDefinitions {
            lines.append("Tool name: \(definition.name)")
            if !definition.description.isEmpty {
                lines.append("Description: \(definition.description)")
            }
            lines.append("Parameters schema: \(try encodeSchemaString(definition.parameters))")
        }

        return lines.joined(separator: "\n")
    }

    private func convertPromptSegments(
        _ segments: [Transcript.Segment],
        configuration: ModelConfiguration
    ) async throws -> [InputMessage.Content] {
        var contents: [InputMessage.Content] = []

        for segment in segments {
            switch segment {
            case .text(let textSegment):
                contents.append(.text(textSegment.content))
            case .reasoning:
                continue
            case .structure(let structure):
                contents.append(.text(structure.content.jsonString))
            case .image(let imageSegment):
                try validateImageSupport(configuration: configuration)
                contents.append(try await convertImageSegment(imageSegment))
            }
        }

        return contents
    }

    private func validateImageSupport(configuration: ModelConfiguration) throws {
        guard configuration.inputCapabilities.supportsImages else {
            throw MetalLanguageModelError.unsupportedImageInput(modelName: configuration.name)
        }

        let execution = configuration.executionCapabilities
        guard execution.supportsImagePromptPreparation, execution.supportsImageExecution else {
            throw MetalLanguageModelError.unsupportedImageInput(modelName: configuration.name)
        }
    }

    private func convertImageSegment(
        _ segment: Transcript.ImageSegment
    ) async throws -> InputMessage.Content {
        switch segment.source {
        case .url(let url):
            if url.isFileURL {
                return .image(InputImage(fileURL: url))
            }

            do {
                let (data, response) = try await URLSession.shared.data(from: url)
                let mimeType = (response as? HTTPURLResponse)?
                    .value(forHTTPHeaderField: "Content-Type")
                return .image(InputImage(data: data, mimeType: mimeType))
            } catch {
                throw ConversionError.remoteImageLoadFailed(url: url, underlyingError: error)
            }
        case .base64(let base64String, let mediaType):
            guard let data = Data(base64Encoded: base64String) else {
                throw ConversionError.invalidBase64ImageData
            }
            return .image(InputImage(data: data, mimeType: mediaType))
        }
    }

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

    private func encodeSchemaString(_ schema: GenerationSchema) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(schema)
        return String(decoding: data, as: UTF8.self)
    }
}

#endif
