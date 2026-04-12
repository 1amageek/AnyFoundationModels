#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import SwiftLM

/// Stateless converter: swift-lm generation events → Transcript.Entry.
struct MetalResponseConverter {
    struct ResponseChannels: Sendable {
        let answer: String
        let reasoning: String
    }

    struct StreamingState: Sendable {
        var rawText: String = ""
        var rawReasoning: String = ""
        var emittedVisibleText: String = ""
        var emittedReasoningText: String = ""
    }

    func finalEntry(
        from generations: [GenerationEvent],
        hasTools: Bool,
        hasSchema: Bool
    ) throws -> Transcript.Entry {
        var text = ""
        var reasoning = ""

        for generation in generations {
            switch generation {
            case .text(let chunk):
                text += chunk
            case .reasoning(let chunk):
                reasoning += chunk
            case .completed:
                break
            }
        }

        let channels = splitResponseChannels(
            from: text,
            allowIncompleteTrailingTag: false,
            trimWhitespace: false
        )
        let sanitizedAnswer = sanitizeAnswer(channels.answer)
        let sanitizedReasoning = sanitizeReasoning(
            mergeReasoning(explicitReasoning: reasoning, taggedReasoning: channels.reasoning)
        )

        if hasTools, let toolEntry = MetalToolCallDetector.entryIfPresent(sanitizedAnswer) {
            return toolEntry
        }

        if hasSchema, !sanitizedAnswer.isEmpty {
            let generatedContent = try GeneratedContent(json: sanitizedAnswer)
            return .response(
                .init(
                    assetIDs: [],
                    segments: makeResponseSegments(
                        answer: [.structure(.init(source: "metal", content: generatedContent))],
                        reasoning: sanitizedReasoning
                    )
                )
            )
        }

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

    func streamDelta(
        state: inout StreamingState,
        event: GenerationEvent
    ) -> ResponseChannels {
        switch event {
        case .text(let chunk):
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
            let reasoningDeltaFromTaggedText = computeDelta(
                current: channels.reasoning,
                emitted: &state.emittedReasoningText
            )
            return ResponseChannels(
                answer: answerDelta,
                reasoning: reasoningDeltaFromTaggedText
            )

        case .reasoning(let chunk):
            state.rawReasoning += chunk
            let reasoningDelta = computeDelta(
                current: state.rawReasoning,
                emitted: &state.emittedReasoningText
            )
            return ResponseChannels(answer: "", reasoning: reasoningDelta)

        case .completed:
            return ResponseChannels(answer: "", reasoning: "")
        }
    }

    func streamEntry(answer: String, reasoning: String) -> Transcript.Entry? {
        guard !answer.isEmpty || !reasoning.isEmpty else {
            return nil
        }

        return .response(
            .init(
                assetIDs: [],
                segments: makeResponseSegments(
                    answer: answer.isEmpty ? [] : [.text(.init(content: answer))],
                    reasoning: reasoning
                )
            )
        )
    }

    private func mergeReasoning(
        explicitReasoning: String,
        taggedReasoning: String
    ) -> String {
        switch (explicitReasoning.isEmpty, taggedReasoning.isEmpty) {
        case (true, true):
            return ""
        case (false, true):
            return explicitReasoning
        case (true, false):
            return taggedReasoning
        case (false, false):
            return "\(explicitReasoning)\n\(taggedReasoning)"
        }
    }

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
        stripCodeFenceIfPresent(answer)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func sanitizeReasoning(_ reasoning: String) -> String {
        reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func stripCodeFenceIfPresent(_ text: String) -> String {
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

    private func makeResponseSegments(
        answer: [Transcript.Segment],
        reasoning: String
    ) -> [Transcript.Segment] {
        var segments: [Transcript.Segment] = []
        if !reasoning.isEmpty {
            segments.append(.reasoning(.init(content: reasoning)))
        }
        segments.append(contentsOf: answer)
        return segments.isEmpty ? [.text(.init(content: ""))] : segments
    }
}

#endif
