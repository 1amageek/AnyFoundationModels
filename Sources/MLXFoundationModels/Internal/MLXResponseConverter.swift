#if MLX_ENABLED
import Foundation
import MLXLMCommon
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Stateless converter: Generation events → Transcript.Entry.
struct MLXResponseConverter {

    struct ResponseChannels: Sendable {
        let answer: String
        let reasoning: String
    }

    // MARK: - Streaming State

    struct StreamingState: Sendable {
        var rawText: String = ""
        var emittedVisibleText: String = ""
        var emittedReasoningText: String = ""
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

        let channels = splitResponseChannels(
            from: text,
            allowIncompleteTrailingTag: false,
            trimWhitespace: false
        )
        let sanitizedAnswer = sanitizeAnswer(channels.answer)
        let sanitizedReasoning = sanitizeReasoning(channels.reasoning)

        // Priority 2: Textual tool call detection
        if hasTools, let toolEntry = ToolCallDetector.entryIfPresent(sanitizedAnswer) {
            return toolEntry
        }

        // Priority 3: Structured response
        if hasSchema {
            let content = sanitizedAnswer.isEmpty ? "" : sanitizedAnswer
            do {
                let generatedContent = try GeneratedContent(json: content)
                return .response(
                    .init(
                        assetIDs: [],
                        segments: makeResponseSegments(
                            answer: [.structure(.init(source: "mlx", content: generatedContent))],
                            reasoning: sanitizedReasoning
                        )
                    )
                )
            } catch {
                Logger.warning("[MLXResponseConverter] Failed to parse structured response: \(error)")
            }
        }

        // Priority 4: Plain text response
        return .response(
            .init(
                assetIDs: [],
                segments: makeResponseSegments(
                    answer: sanitizedAnswer.isEmpty ? [] : [.text(.init(content: sanitizedAnswer))],
                    reasoning: sanitizedReasoning
                )
            )
        )
    }

    // MARK: - Streaming

    /// Processes a text chunk incrementally, stripping think blocks.
    /// Returns the visible text delta to emit.
    func streamDelta(state: inout StreamingState, chunk: String) -> ResponseChannels {
        state.rawText += chunk

        let channels = splitResponseChannels(
            from: state.rawText,
            allowIncompleteTrailingTag: true,
            trimWhitespace: false
        )

        let answerDelta = computeDelta(
            current: channels.answer,
            emitted: &state.emittedVisibleText
        )
        let reasoningDelta = computeDelta(
            current: channels.reasoning,
            emitted: &state.emittedReasoningText
        )

        return ResponseChannels(answer: answerDelta, reasoning: reasoningDelta)
    }

    /// Wraps text and reasoning deltas in a Transcript.Entry for streaming.
    func streamEntry(answer: String, reasoning: String) -> Transcript.Entry? {
        guard !answer.isEmpty || !reasoning.isEmpty else {
            return nil
        }
        let segments = makeResponseSegments(
            answer: answer.isEmpty ? [] : [.text(.init(content: answer))],
            reasoning: reasoning
        )
        return .response(.init(assetIDs: [], segments: segments))
    }

    // MARK: - Sanitization

    /// Strips think blocks, code fences, and trims whitespace from model output.
    func sanitizeAssistantResponse(_ text: String) -> String {
        let channels = splitResponseChannels(
            from: text,
            allowIncompleteTrailingTag: false,
            trimWhitespace: false
        )
        return sanitizeAnswer(channels.answer)
    }

    // MARK: - Private: Response Channel Parsing

    private func computeDelta(
        current: String,
        emitted: inout String
    ) -> String {
        let delta: String
        if current.hasPrefix(emitted) {
            let startIndex = current.index(
                current.startIndex,
                offsetBy: emitted.count
            )
            delta = String(current[startIndex...])
        } else {
            delta = current
        }
        emitted = current
        return delta
    }

    private func splitResponseChannels(
        from text: String,
        allowIncompleteTrailingTag: Bool,
        trimWhitespace: Bool
    ) -> ResponseChannels {
        let openTag = "<think>"
        let closeTag = "</think>"

        if !text.localizedCaseInsensitiveContains(openTag),
           let closeRange = text.range(of: closeTag, options: [.caseInsensitive]) {
            let reasoning = String(text[..<closeRange.lowerBound])
            let answer = String(text[closeRange.upperBound...])
            return ResponseChannels(
                answer: trimWhitespace ? answer.trimmingCharacters(in: .whitespacesAndNewlines) : answer,
                reasoning: trimWhitespace ? reasoning.trimmingCharacters(in: .whitespacesAndNewlines) : reasoning
            )
        }

        var answer = ""
        var reasoning = ""
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

            if insideThink {
                reasoning.append(text[index])
            } else {
                answer.append(text[index])
            }
            index = text.index(after: index)
        }

        return ResponseChannels(
            answer: trimWhitespace ? answer.trimmingCharacters(in: .whitespacesAndNewlines) : answer,
            reasoning: trimWhitespace ? reasoning.trimmingCharacters(in: .whitespacesAndNewlines) : reasoning
        )
    }

    private func sanitizeAnswer(_ answer: String) -> String {
        var result = answer
        result = stripCodeFenceIfJSON(result)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func sanitizeReasoning(_ reasoning: String) -> String {
        reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
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

    private func makeResponseSegments(
        answer: [Transcript.Segment],
        reasoning: String
    ) -> [Transcript.Segment] {
        var segments: [Transcript.Segment] = []
        if !reasoning.isEmpty {
            segments.append(.reasoning(.init(content: reasoning)))
        }
        segments.append(contentsOf: answer)
        if segments.isEmpty {
            return [.text(.init(content: ""))]
        }
        return segments
    }
}
#endif
