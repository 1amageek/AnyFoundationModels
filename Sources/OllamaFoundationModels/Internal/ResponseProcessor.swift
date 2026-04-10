#if OLLAMA_ENABLED
import Foundation

/// Result of processing a Message
enum ProcessedResponse: Sendable {
    case toolCalls([ToolCall])
    case response(ResponseProcessor.ResponseChannels)
    case empty
}

/// Unified response processor for generate() and stream()
///
/// This struct centralizes all response normalization logic:
/// - Native tool_calls extraction
/// - Text-based tool call parsing from content
/// - Text-based tool call parsing from thinking field
/// - Thinking tag stripping (for models that output <think> tags in content)
/// - JSON extraction from mixed content
/// - Content/thinking fallback handling
struct ResponseProcessor: Sendable {

    struct ResponseChannels: Sendable, Equatable {
        let content: String
        let reasoning: String
    }

    // MARK: - Patterns

    /// Pattern to match <think>...</think> tags (including unclosed ones)
    private static let thinkTagPattern = try! NSRegularExpression(
        pattern: #"<think>[\s\S]*?</think>|<think>[\s\S]*$"#,
        options: [.caseInsensitive]
    )

    /// Pattern to match orphaned </think> tag and everything before it
    /// This handles cases where the model outputs thinking without opening <think> tag
    /// Example: "Okay, let me think... </think>\n{\"key\": \"value\"}"
    private static let orphanedThinkClosePattern = try! NSRegularExpression(
        pattern: #"^[\s\S]*</think>"#,
        options: [.caseInsensitive]
    )

    /// Pattern to match content wrapper tags
    private static let contentTagPattern = try! NSRegularExpression(
        pattern: #"</?content>"#,
        options: [.caseInsensitive]
    )

    // Note: Markdown code block and JSON object patterns are now in JSONExtractor

    // MARK: - Process

    /// Process a Message and extract the appropriate response
    ///
    /// Priority order:
    /// 1. Native tool_calls (highest priority)
    /// 2. Text-based tool calls in content
    /// 3. Text-based tool calls in thinking field
    /// 4. Content with <think> tags stripped (for thinking models with think: false)
    /// 5. JSON extracted from content (fallback for mixed content)
    /// 6. Content (if non-empty)
    /// 7. Thinking (fallback for thinking models)
    ///
    /// - Parameter message: The Message from Ollama API
    /// - Returns: ProcessedResponse indicating what type of content was found
    func process(_ message: Message) -> ProcessedResponse {
        // 1. Native tool_calls (highest priority)
        if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            return .toolCalls(toolCalls)
        }

        // 2. Text-based tool calls in content
        if TextToolCallParser.containsToolCallPatterns(message.content) {
            let result = TextToolCallParser.parse(message.content)
            if !result.toolCalls.isEmpty {
                return .toolCalls(result.toolCalls)
            }
        }

        // 3. Text-based tool calls in thinking field
        if let thinking = message.thinking,
           TextToolCallParser.containsToolCallPatterns(thinking) {
            let result = TextToolCallParser.parse(thinking)
            if !result.toolCalls.isEmpty {
                return .toolCalls(result.toolCalls)
            }
        }

        // 4. Normalize content and reasoning into separate channels
        let normalized = normalize(content: message.content, thinking: message.thinking)
        if !normalized.content.isEmpty || !normalized.reasoning.isEmpty {
            return .response(normalized)
        }

        return .empty
    }

    func normalize(content: String, thinking: String?) -> ResponseChannels {
        let inlineChannels = splitInlineThinking(from: content)
        let normalizedContent = processContent(inlineChannels.content)

        let normalizedReasoning: String
        if let thinking, !thinking.isEmpty {
            normalizedReasoning = processReasoning(thinking)
        } else {
            normalizedReasoning = processReasoning(inlineChannels.reasoning)
        }

        return ResponseChannels(
            content: normalizedContent,
            reasoning: normalizedReasoning
        )
    }

    // MARK: - Private Helpers

    /// Process content to extract clean response
    /// Step 1: Strip content wrapper tags
    /// Step 2: Extract JSON from the content
    private func processContent(_ content: String) -> String {
        guard !content.isEmpty else { return "" }

        #if DEBUG
        print("[ResponseProcessor] Input content length: \(content.count)")
        print("[ResponseProcessor] Input preview: \(String(content.prefix(300)))")
        #endif

        // Step 1: Strip non-semantic wrapper tags
        let strippedContent = stripContentTags(from: content)

        #if DEBUG
        print("[ResponseProcessor] After stripThinkTags length: \(strippedContent.count)")
        print("[ResponseProcessor] Stripped preview: \(String(strippedContent.prefix(300)))")
        #endif

        // Step 2: Extract JSON from the stripped content
        if let json = extractJSONFromContent(from: strippedContent) {
            #if DEBUG
            print("[ResponseProcessor] JSON extracted from stripped content: \(String(json.prefix(200)))")
            #endif
            return json
        }

        // If no JSON found in stripped content, try original content
        if let json = extractJSONFromContent(from: content) {
            #if DEBUG
            print("[ResponseProcessor] JSON extracted from original content: \(String(json.prefix(200)))")
            #endif
            return json
        }

        #if DEBUG
        print("[ResponseProcessor] No JSON found, returning stripped content")
        #endif
        // Return stripped content as-is (for non-JSON responses)
        return strippedContent.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func processReasoning(_ reasoning: String) -> String {
        guard !reasoning.isEmpty else { return "" }
        return stripContentTags(from: stripThinkTags(from: reasoning))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Tag Stripping

    /// Strip all think-related tags from content
    private func stripThinkTags(from content: String) -> String {
        var processed = content

        // Strip orphaned </think> and everything before it
        if processed.contains("</think>") && !processed.contains("<think") {
            processed = stripPattern(Self.orphanedThinkClosePattern, from: processed)
        }

        // Strip <think>...</think> tags
        if processed.contains("<think") {
            processed = stripPattern(Self.thinkTagPattern, from: processed)
        }

        return processed
    }

    private func stripContentTags(from content: String) -> String {
        guard content.localizedCaseInsensitiveContains("<content") else {
            return content
        }
        return stripPattern(Self.contentTagPattern, from: content)
    }

    /// Strip pattern from content
    private func stripPattern(_ pattern: NSRegularExpression, from content: String) -> String {
        let range = NSRange(content.startIndex..<content.endIndex, in: content)
        return pattern.stringByReplacingMatches(in: content, options: [], range: range, withTemplate: "")
    }

    private func splitInlineThinking(from content: String) -> ResponseChannels {
        let openTag = "<think>"
        let closeTag = "</think>"

        if !content.localizedCaseInsensitiveContains(openTag),
           let closeRange = content.range(of: closeTag, options: [.caseInsensitive]) {
            let reasoning = String(content[..<closeRange.lowerBound])
            let visible = String(content[closeRange.upperBound...])
            return ResponseChannels(content: visible, reasoning: reasoning)
        }

        var visible = ""
        var reasoning = ""
        var index = content.startIndex
        var insideThink = false

        while index < content.endIndex {
            if content[index...].hasPrefix(openTag) {
                insideThink = true
                index = content.index(index, offsetBy: openTag.count)
                continue
            }

            if content[index...].hasPrefix(closeTag) {
                insideThink = false
                index = content.index(index, offsetBy: closeTag.count)
                continue
            }

            if insideThink {
                reasoning.append(content[index])
            } else {
                visible.append(content[index])
            }
            index = content.index(after: index)
        }

        return ResponseChannels(content: visible, reasoning: reasoning)
    }

    // MARK: - JSON Extraction

    /// Extract JSON from content using shared JSONExtractor
    private func extractJSONFromContent(from content: String) -> String? {
        return JSONExtractor.extract(from: content)
    }
}

#endif
