#if MLX_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

enum MLXResponseAssemblyError: LocalizedError, Equatable {
    case missingRequiredToolResponse

    var errorDescription: String? {
        switch self {
        case .missingRequiredToolResponse:
            return "Model completed without returning text or a tool call for a required-tool turn."
        }
    }
}

struct MLXCollectedOutput: Sendable {
    var text: String
    var nativeToolCalls: [(name: String, argsJSON: String)]
    var sawInfo: Bool
}

struct MLXStreamingResponseState: Sendable, Equatable {
    var rawText = ""
    var emittedVisibleText = ""
}

struct MLXResponseAssembler {
    func collect(events: [MLXGenerationEvent]) -> MLXCollectedOutput {
        var text = ""
        var toolCalls: [(name: String, argsJSON: String)] = []
        var sawInfo = false

        for event in events {
            switch event {
            case .textChunk(let chunk):
                text += chunk
            case .nativeToolCall(let name, let argsJSON):
                toolCalls.append((name: name, argsJSON: argsJSON))
            case .info:
                sawInfo = true
            case .completed:
                break
            }
        }

        return MLXCollectedOutput(text: text, nativeToolCalls: toolCalls, sawInfo: sawInfo)
    }

    func finalEntry(
        plan: MLXExecutionPlan,
        events: [MLXGenerationEvent]
    ) throws -> Transcript.Entry {
        try validate(plan: plan, events: events)
        let collected = collect(events: events)
        if !collected.nativeToolCalls.isEmpty,
           let toolEntry = try nativeToolCallEntry(from: collected.nativeToolCalls) {
            return toolEntry
        }

        let sanitized = sanitizeAssistantResponse(collected.text)
        if plan.toolPolicy != .disabled,
           let toolEntry = ToolCallDetector.entryIfPresent(sanitized) {
            return toolEntry
        }

        return responseEntry(
            text: sanitized,
            responseMode: plan.responseMode,
            fallbackText: collected.text.isEmpty ? "" : sanitized
        )
    }

    func validate(
        plan: MLXExecutionPlan,
        events: [MLXGenerationEvent]
    ) throws {
        let collected = collect(events: events)
        let sanitized = sanitizeAssistantResponse(collected.text)

        if plan.toolPolicy == .required,
           collected.nativeToolCalls.isEmpty,
           ToolCallDetector.entryIfPresent(sanitized) == nil,
           sanitized.isEmpty {
            throw MLXResponseAssemblyError.missingRequiredToolResponse
        }
    }

    func streamEntry(for chunk: String) -> Transcript.Entry {
        .response(.init(assetIDs: [], segments: [.text(.init(content: chunk))]))
    }

    func streamDelta(
        state: MLXStreamingResponseState,
        chunk: String
    ) -> (delta: String, state: MLXStreamingResponseState) {
        var nextState = state
        nextState.rawText += chunk

        let visible = visibleAssistantText(
            from: nextState.rawText,
            allowIncompleteTrailingTag: true,
            trimWhitespace: false
        )

        let delta: String
        if visible.hasPrefix(nextState.emittedVisibleText) {
            let startIndex = visible.index(
                visible.startIndex,
                offsetBy: nextState.emittedVisibleText.count
            )
            delta = String(visible[startIndex...])
        } else {
            delta = visible
        }

        nextState.emittedVisibleText = visible
        return (delta, nextState)
    }

    func sanitizeAssistantResponse(_ text: String) -> String {
        var result = visibleAssistantText(
            from: text,
            allowIncompleteTrailingTag: false,
            trimWhitespace: false
        )
        result = stripCodeFenceIfJSON(result)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

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

    private func responseEntry(
        text: String,
        responseMode: MLXResponseMode,
        fallbackText: String
    ) -> Transcript.Entry {
        let content = text.isEmpty ? fallbackText : text

        if responseMode == .structured {
            do {
                let generatedContent = try GeneratedContent(json: content)
                return .response(
                    .init(
                        assetIDs: [],
                        segments: [.structure(.init(source: "mlx", content: generatedContent))]
                    )
                )
            } catch {
                Logger.warning("[MLXResponseAssembler] Failed to parse structured response: \(error)")
            }
        }

        return .response(.init(assetIDs: [], segments: [.text(.init(content: content))]))
    }

    private func nativeToolCallEntry(
        from infos: [(name: String, argsJSON: String)]
    ) throws -> Transcript.Entry? {
        guard !infos.isEmpty else {
            return nil
        }

        let calls: [Transcript.ToolCall] = try infos.map { info in
            let content = try GeneratedContent(json: info.argsJSON)
            return Transcript.ToolCall(id: UUID().uuidString, toolName: info.name, arguments: content)
        }

        return .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, calls))
    }

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
}
#endif
