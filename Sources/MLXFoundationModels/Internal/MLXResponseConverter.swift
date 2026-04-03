#if MLX_ENABLED
import Foundation
import MLXLMCommon
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Stateless converter: Generation events → Transcript.Entry.
struct MLXResponseConverter {

    // MARK: - Streaming State

    struct StreamingState: Sendable {
        var rawText: String = ""
        var emittedVisibleText: String = ""
    }

    // MARK: - Final Entry Assembly

    /// Assembles a final Transcript.Entry from collected generation events.
    func finalEntry(
        from generations: [Generation],
        hasTools: Bool,
        hasSchema: Bool
    ) throws -> Transcript.Entry {
        // Collect text and native tool calls
        var text = ""
        var nativeToolCalls: [(name: String, arguments: [String: MLXLMCommon.JSONValue])] = []

        for generation in generations {
            switch generation {
            case .chunk(let chunk):
                text += chunk
            case .toolCall(let toolCall):
                nativeToolCalls.append((
                    name: toolCall.function.name,
                    arguments: toolCall.function.arguments
                ))
            case .info:
                break
            }
        }

        // Priority 1: Native tool calls
        if !nativeToolCalls.isEmpty {
            return try nativeToolCallEntry(from: nativeToolCalls)
        }

        let sanitized = sanitizeAssistantResponse(text)

        // Priority 2: Textual tool call detection
        if hasTools, let toolEntry = ToolCallDetector.entryIfPresent(sanitized) {
            return toolEntry
        }

        // Priority 3: Structured response
        if hasSchema {
            let content = sanitized.isEmpty ? "" : sanitized
            do {
                let generatedContent = try GeneratedContent(json: content)
                return .response(
                    .init(
                        assetIDs: [],
                        segments: [.structure(.init(source: "mlx", content: generatedContent))]
                    )
                )
            } catch {
                Logger.warning("[MLXResponseConverter] Failed to parse structured response: \(error)")
            }
        }

        // Priority 4: Plain text response
        let content = sanitized.isEmpty ? text : sanitized
        return .response(.init(assetIDs: [], segments: [.text(.init(content: content))]))
    }

    // MARK: - Streaming

    /// Processes a text chunk incrementally, stripping think blocks.
    /// Returns the visible text delta to emit.
    func streamDelta(state: inout StreamingState, chunk: String) -> String {
        state.rawText += chunk

        let visible = visibleAssistantText(
            from: state.rawText,
            allowIncompleteTrailingTag: true,
            trimWhitespace: false
        )

        let delta: String
        if visible.hasPrefix(state.emittedVisibleText) {
            let startIndex = visible.index(
                visible.startIndex,
                offsetBy: state.emittedVisibleText.count
            )
            delta = String(visible[startIndex...])
        } else {
            delta = visible
        }

        state.emittedVisibleText = visible
        return delta
    }

    /// Wraps a text chunk in a Transcript.Entry for streaming.
    func streamEntry(for text: String) -> Transcript.Entry {
        .response(.init(assetIDs: [], segments: [.text(.init(content: text))]))
    }

    // MARK: - Sanitization

    /// Strips think blocks, code fences, and trims whitespace from model output.
    func sanitizeAssistantResponse(_ text: String) -> String {
        var result = visibleAssistantText(
            from: text,
            allowIncompleteTrailingTag: false,
            trimWhitespace: false
        )
        result = stripCodeFenceIfJSON(result)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Private: Think Block Stripping

    private func visibleAssistantText(
        from text: String,
        allowIncompleteTrailingTag: Bool,
        trimWhitespace: Bool
    ) -> String {
        let openTag = "<think>"
        let closeTag = "</think>"

        var output = ""
        var index = text.startIndex
        var insideThink = false

        while index < text.endIndex {
            if text[index...].hasPrefix(openTag) {
                insideThink = true
                index = text.index(index, offsetBy: openTag.count)
                continue
            }

            if text[index...].hasPrefix(closeTag) {
                insideThink = false
                index = text.index(index, offsetBy: closeTag.count)
                continue
            }

            if allowIncompleteTrailingTag {
                let remaining = String(text[index...])
                if openTag.hasPrefix(remaining) || closeTag.hasPrefix(remaining) {
                    break
                }
            }

            if !insideThink {
                output.append(text[index])
            }
            index = text.index(after: index)
        }

        if trimWhitespace {
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return output
    }

    // MARK: - Private: Code Fence Stripping

    private func stripCodeFenceIfJSON(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("```"), trimmed.hasSuffix("```") else {
            return text
        }

        let lines = trimmed.components(separatedBy: .newlines)
        guard lines.count >= 3 else {
            return text
        }
        return lines.dropFirst().dropLast().joined(separator: "\n")
    }

    // MARK: - Private: Native Tool Call Conversion

    private func nativeToolCallEntry(
        from infos: [(name: String, arguments: [String: MLXLMCommon.JSONValue])]
    ) throws -> Transcript.Entry {
        let calls: [Transcript.ToolCall] = try infos.map { info in
            let data = try JSONEncoder().encode(info.arguments)
            let argumentsJSON = String(decoding: data, as: UTF8.self)
            let content = try GeneratedContent(json: argumentsJSON)
            return Transcript.ToolCall(
                id: UUID().uuidString,
                toolName: info.name,
                arguments: content
            )
        }
        return .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, calls))
    }
}
#endif
