#if MLX_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

// MARK: - JSONUtils (moved from separate file)

enum JSONUtils {
    // Returns the first complete top-level JSON object found in text, or nil.
    static func firstTopLevelObject(in text: String) -> JSONValue? {
        guard let start = text.firstIndex(of: "{") else { return nil }
        var depth = 0
        var endIndex: String.Index?
        var inString = false
        var escaped = false
        for i in text[start...].indices {
            let ch = text[i]
            if ch == "\\" && inString { escaped.toggle(); continue }
            if ch == "\"" && !escaped { inString.toggle() }
            if !inString {
                if ch == "{" { depth += 1 }
                if ch == "}" { depth -= 1; if depth == 0 { endIndex = i; break } }
            }
            escaped = false
        }
        guard let end = endIndex else { return nil }
        let jsonSlice = text[start...end]
        guard let data = String(jsonSlice).data(using: .utf8) else { return nil }
        do {
            let value = try JSONDecoder().decode(JSONValue.self, from: data)
            guard case .object = value else { return nil }
            return value
        } catch {
            return nil
        }
    }

    // Returns ALL complete top-level JSON objects found in text
    static func allTopLevelObjects(in text: String) -> [JSONValue] {
        var objects: [JSONValue] = []
        var searchIndex = text.startIndex

        while searchIndex < text.endIndex {
            // Find next opening brace
            guard let start = text[searchIndex...].firstIndex(of: "{") else { break }

            // Find the complete object starting from this brace
            var depth = 0
            var endIndex: String.Index?
            var inString = false
            var escaped = false

            for i in text[start...].indices {
                let ch = text[i]
                if ch == "\\" && inString { escaped.toggle(); continue }
                if ch == "\"" && !escaped { inString.toggle() }
                if !inString {
                    if ch == "{" { depth += 1 }
                    if ch == "}" {
                        depth -= 1
                        if depth == 0 {
                            endIndex = i
                            break
                        }
                    }
                }
                escaped = false
            }

            if let end = endIndex {
                let jsonSlice = text[start...end]
                if let data = String(jsonSlice).data(using: .utf8) {
                    do {
                        let value = try JSONDecoder().decode(JSONValue.self, from: data)
                        if case .object = value {
                            objects.append(value)
                        }
                    } catch {
                        // Invalid JSON, skip
                    }
                }
                // Continue searching after this object
                searchIndex = text.index(after: end)
            } else {
                // No valid object found, stop searching
                break
            }
        }

        return objects
    }
}

// MARK: - ToolCallDetector

enum ToolCallDetector {
    private static let toolCallsKeyPattern = #""tool_calls"\s*:\s*\["#

    private static let singleToolCallPattern = #"""
        \{\s*"(?:name|id|function|arguments|parameters)"[^{}]+?\}
        """#

    static func entryIfPresent(_ text: String) -> Transcript.Entry? {
        if let entry = detectWithJSONParsing(text) {
            return entry
        }

        if let entry = detectToolCallsWithRegex(text) {
            return entry
        }

        if let entry = detectXMLFunctionCalls(text) {
            return entry
        }

        return detectBracketToolCalls(text)
    }

    private static func detectWithJSONParsing(_ text: String) -> Transcript.Entry? {
        let cleaned = cleanText(text)
        let objects = JSONUtils.allTopLevelObjects(in: cleaned)

        for obj in objects {
            guard case .object(let dict) = obj,
                  case .array(let arr) = dict["tool_calls"],
                  !arr.isEmpty else { continue }
            return buildToolCallsEntry(from: arr)
        }

        return nil
    }

    private static func detectToolCallsWithRegex(_ text: String) -> Transcript.Entry? {
        do {
            let keyRegex = try NSRegularExpression(pattern: toolCallsKeyPattern, options: [.caseInsensitive])
            let cleaned = cleanText(text)
            let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)

            guard keyRegex.firstMatch(in: cleaned, options: [], range: range) != nil else {
                return nil
            }

            if let toolCallsRange = cleaned.range(of: "\"tool_calls\"") {
                var startIndex = cleaned.startIndex
                var openBraceCount = 0

                // Guard against boundary condition when tool_calls is at the start
                guard toolCallsRange.lowerBound > cleaned.startIndex else {
                    return nil
                }

                var searchIndex = cleaned.index(before: toolCallsRange.lowerBound)
                while searchIndex >= cleaned.startIndex {
                    let char = cleaned[searchIndex]
                    if char == "{" {
                        openBraceCount += 1
                        if openBraceCount == 1 {
                            startIndex = searchIndex
                            break
                        }
                    } else if char == "}" {
                        openBraceCount -= 1
                    }
                    if searchIndex == cleaned.startIndex { break }
                    searchIndex = cleaned.index(before: searchIndex)
                }

                openBraceCount = 0
                var endIndex = cleaned.endIndex
                searchIndex = startIndex
                while searchIndex < cleaned.endIndex {
                    let char = cleaned[searchIndex]
                    if char == "{" {
                        openBraceCount += 1
                    } else if char == "}" {
                        openBraceCount -= 1
                        if openBraceCount == 0 {
                            endIndex = cleaned.index(after: searchIndex)
                            break
                        }
                    }
                    searchIndex = cleaned.index(after: searchIndex)
                }

                let jsonString = String(cleaned[startIndex..<endIndex])
                return parseToolCallsJSON(jsonString)
            }

            return detectIndividualToolCalls(cleaned)

        } catch {
            Logger.warning("[ToolCallDetector] Regex compilation failed: \(error)")
            return detectSimpleToolCalls(text)
        }
    }

    private static func cleanText(_ text: String) -> String {
        return text
            .replacingOccurrences(of: "\u{FEFF}", with: "")
            .replacingOccurrences(of: "\u{200B}", with: "")
            .replacingOccurrences(of: "\u{200C}", with: "")
            .replacingOccurrences(of: "\u{200D}", with: "")
    }

    private static func detectIndividualToolCalls(_ text: String) -> Transcript.Entry? {
        do {
            let regex = try NSRegularExpression(pattern: singleToolCallPattern, options: [.caseInsensitive])
            let range = NSRange(text.startIndex..<text.endIndex, in: text)
            let matches = regex.matches(in: text, options: [], range: range)

            var calls: [Transcript.ToolCall] = []

            for match in matches {
                if let matchRange = Range(match.range, in: text) {
                    let callJSON = String(text[matchRange])
                    if let call = parseIndividualToolCall(callJSON) {
                        calls.append(call)
                    }
                }
            }

            guard !calls.isEmpty else { return nil }
            let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
            return .toolCalls(toolCalls)

        } catch {
            Logger.warning("[ToolCallDetector] Individual tool call regex failed: \(error)")
            return nil
        }
    }

    private static func detectXMLFunctionCalls(_ text: String) -> Transcript.Entry? {
        let cleaned = cleanText(text)
        guard let regex = try? NSRegularExpression(
            pattern: #"<function=([\s\S]*?)</function>"#,
            options: [.caseInsensitive]
        ) else {
            return nil
        }

        let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
        let matches = regex.matches(in: cleaned, options: [], range: range)
        guard !matches.isEmpty else {
            return nil
        }

        let calls: [Transcript.ToolCall] = matches.compactMap { match in
            guard let matchRange = Range(match.range, in: cleaned) else {
                return nil
            }
            return parseXMLFunctionCall(String(cleaned[matchRange]))
        }

        guard !calls.isEmpty else {
            return nil
        }

        let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
        return .toolCalls(toolCalls)
    }

    private static func parseXMLFunctionCall(_ content: String) -> Transcript.ToolCall? {
        guard let nameStart = content.range(of: "<function="),
              let nameEnd = content.range(
                of: ">",
                range: nameStart.upperBound..<content.endIndex
              )
        else {
            return nil
        }

        let functionName = String(content[nameStart.upperBound..<nameEnd.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !functionName.isEmpty else {
            return nil
        }

        var arguments: [String: JSONValue] = [:]
        let parameterSection = String(content[nameEnd.upperBound...])
        var searchRange = parameterSection.startIndex..<parameterSection.endIndex

        while let parameterStart = parameterSection.range(
            of: "<parameter=",
            range: searchRange
        ) {
            guard let parameterNameEnd = parameterSection.range(
                of: ">",
                range: parameterStart.upperBound..<parameterSection.endIndex
            ),
            let parameterEnd = parameterSection.range(
                of: "</parameter>",
                range: parameterNameEnd.upperBound..<parameterSection.endIndex
            ) else {
                break
            }

            let name = String(
                parameterSection[parameterStart.upperBound..<parameterNameEnd.lowerBound]
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            let rawValue = String(
                parameterSection[parameterNameEnd.upperBound..<parameterEnd.lowerBound]
            ).trimmingCharacters(in: .whitespacesAndNewlines)

            if !name.isEmpty {
                arguments[name] = parseXMLParameterValue(rawValue)
            }

            searchRange = parameterEnd.upperBound..<parameterSection.endIndex
        }

        do {
            let generatedContent = try GeneratedContent(jsonValue: .object(arguments))
            return Transcript.ToolCall(
                id: UUID().uuidString,
                toolName: functionName,
                arguments: generatedContent
            )
        } catch {
            return nil
        }
    }

    private static func parseXMLParameterValue(_ rawValue: String) -> JSONValue {
        let trimmed = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return .string("")
        }

        if trimmed == "true" {
            return .bool(true)
        }
        if trimmed == "false" {
            return .bool(false)
        }
        if trimmed == "null" {
            return .null
        }
        if let intValue = Int(trimmed) {
            return .int(intValue)
        }
        if let doubleValue = Double(trimmed) {
            return .double(doubleValue)
        }
        if let data = trimmed.data(using: .utf8),
           let jsonValue = try? JSONDecoder().decode(JSONValue.self, from: data) {
            return jsonValue
        }
        return .string(trimmed)
    }

    private static func parseIndividualToolCall(_ json: String) -> Transcript.ToolCall? {
        guard let data = json.data(using: .utf8) else { return nil }
        let value: JSONValue
        do {
            value = try JSONDecoder().decode(JSONValue.self, from: data)
        } catch {
            return nil
        }
        guard case .object(let dict) = value else { return nil }

        let name: String
        if case .string(let n) = dict["name"], !n.isEmpty { name = n }
        else if case .string(let n) = dict["function"], !n.isEmpty { name = n }
        else { return nil }

        let callID: String
        if case .string(let id) = dict["id"] { callID = id }
        else { callID = UUID().uuidString }

        let argsValue = dict["arguments"] ?? dict["parameters"] ?? .object([:])
        do {
            let gen = try GeneratedContent(jsonValue: argsValue)
            return Transcript.ToolCall(id: callID, toolName: name, arguments: gen)
        } catch {
            return nil
        }
    }

    // MARK: - Bracket-style tool calls: [func_name(k1=v1, k2=v2)]<|tool_call_end|>

    /// Detects bracket-style tool calls emitted by models like LFM2.
    ///
    /// Format: `[function_name(key1=value1, key2="value2")]<|tool_call_end|>`
    private static func detectBracketToolCalls(_ text: String) -> Transcript.Entry? {
        let cleaned = cleanText(text)
        var calls: [Transcript.ToolCall] = []

        var searchStart = cleaned.startIndex
        while searchStart < cleaned.endIndex {
            guard let openBracket = cleaned[searchStart...].firstIndex(of: "[") else { break }
            guard let parsed = parseBracketCall(cleaned, from: openBracket) else {
                searchStart = cleaned.index(after: openBracket)
                continue
            }
            if let call = parsed.call {
                calls.append(call)
            }
            searchStart = parsed.endIndex
        }

        guard !calls.isEmpty else { return nil }
        let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
        return .toolCalls(toolCalls)
    }

    private static func parseBracketCall(
        _ text: String,
        from openBracket: String.Index
    ) -> (call: Transcript.ToolCall?, endIndex: String.Index)? {
        let afterBracket = text.index(after: openBracket)
        guard afterBracket < text.endIndex else { return nil }

        // Extract function name: letters, digits, underscore
        var nameEnd = afterBracket
        while nameEnd < text.endIndex {
            let ch = text[nameEnd]
            guard ch.isLetter || ch.isNumber || ch == "_" else { break }
            nameEnd = text.index(after: nameEnd)
        }
        guard nameEnd > afterBracket, nameEnd < text.endIndex, text[nameEnd] == "(" else {
            return nil
        }
        let funcName = String(text[afterBracket..<nameEnd])

        // Find matching closing paren, tracking nesting
        let argsStart = text.index(after: nameEnd)
        guard let closeParen = findMatchingClose(text, from: nameEnd, open: "(", close: ")") else {
            return nil
        }
        let argsString = String(text[argsStart..<closeParen])

        // Expect ']' after ')'
        let afterParen = text.index(after: closeParen)
        guard afterParen < text.endIndex, text[afterParen] == "]" else { return nil }

        let endIndex = text.index(after: afterParen)

        // Parse kwargs into JSONValue
        let arguments = parseKwargs(argsString)

        do {
            let gen = try GeneratedContent(jsonValue: arguments)
            let call = Transcript.ToolCall(id: UUID().uuidString, toolName: funcName, arguments: gen)
            return (call, endIndex)
        } catch {
            return (nil, endIndex)
        }
    }

    /// Finds the matching closing character, handling nested pairs and strings.
    private static func findMatchingClose(
        _ text: String,
        from start: String.Index,
        open: Character,
        close: Character
    ) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false
        for i in text[start...].indices {
            let ch = text[i]
            if escaped { escaped = false; continue }
            if ch == "\\" && inString { escaped = true; continue }
            if ch == "\"" { inString.toggle(); continue }
            guard !inString else { continue }
            if ch == open { depth += 1 }
            if ch == close {
                depth -= 1
                if depth == 0 { return i }
            }
        }
        return nil
    }

    /// Parses Python-style keyword arguments: `key1=value1, key2="value2"`.
    private static func parseKwargs(_ argsString: String) -> JSONValue {
        let trimmed = argsString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return .object([:]) }

        // Split on top-level commas (not inside strings, brackets, or braces)
        let parts = splitTopLevel(trimmed, separator: ",")
        var result: [String: JSONValue] = [:]

        for part in parts {
            let kv = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let eqIndex = kv.firstIndex(of: "=") else { continue }
            let key = String(kv[kv.startIndex..<eqIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            let valueStr = String(kv[kv.index(after: eqIndex)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty else { continue }
            result[key] = parseValue(valueStr)
        }
        return .object(result)
    }

    /// Splits a string on the given separator, respecting string literals, brackets, and braces.
    private static func splitTopLevel(_ text: String, separator: Character) -> [String] {
        var parts: [String] = []
        var current = ""
        var depth = 0       // tracks [] and {}
        var inString = false
        var escaped = false

        for ch in text {
            if escaped { current.append(ch); escaped = false; continue }
            if ch == "\\" && inString { current.append(ch); escaped = true; continue }
            if ch == "\"" { inString.toggle(); current.append(ch); continue }
            if inString { current.append(ch); continue }
            if ch == "[" || ch == "{" || ch == "(" { depth += 1 }
            if ch == "]" || ch == "}" || ch == ")" { depth -= 1 }
            if ch == separator && depth == 0 {
                parts.append(current)
                current = ""
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { parts.append(current) }
        return parts
    }

    /// Parses a Python-style value literal into a JSONValue.
    private static func parseValue(_ str: String) -> JSONValue {
        // String literal
        if str.hasPrefix("\"") && str.hasSuffix("\"") && str.count >= 2 {
            let inner = String(str.dropFirst().dropLast())
            return .string(inner.replacingOccurrences(of: "\\\"", with: "\""))
        }
        // Boolean
        if str == "true" || str == "True" { return .bool(true) }
        if str == "false" || str == "False" { return .bool(false) }
        // None / null
        if str == "None" || str == "null" || str == "nil" { return .null }
        // Number (int)
        if let intVal = Int(str) { return .int(intVal) }
        // Number (double)
        if let doubleVal = Double(str) { return .double(doubleVal) }
        // Array
        if str.hasPrefix("[") && str.hasSuffix("]") {
            let inner = String(str.dropFirst().dropLast()).trimmingCharacters(in: .whitespacesAndNewlines)
            if inner.isEmpty { return .array([]) }
            let elements = splitTopLevel(inner, separator: ",")
            return .array(elements.map { parseValue($0.trimmingCharacters(in: .whitespacesAndNewlines)) })
        }
        // Fallback: treat as string
        return .string(str)
    }

    private static func detectSimpleToolCalls(_ text: String) -> Transcript.Entry? {
        let objects = JSONUtils.allTopLevelObjects(in: text)

        for obj in objects {
            guard case .object(let dict) = obj,
                  case .array(let arr) = dict["tool_calls"],
                  !arr.isEmpty else { continue }
            return buildToolCallsEntry(from: arr)
        }

        return nil
    }

    private static func parseToolCallsJSON(_ json: String) -> Transcript.Entry? {
        guard let data = json.data(using: .utf8) else { return nil }
        do {
            let value = try JSONDecoder().decode(JSONValue.self, from: data)
            guard case .object(let dict) = value,
                  case .array(let arr) = dict["tool_calls"],
                  !arr.isEmpty else { return nil }
            return buildToolCallsEntry(from: arr)
        } catch {
            return nil
        }
    }

    private static func buildToolCallsEntry(from toolCallsArray: [JSONValue]) -> Transcript.Entry? {
        var calls: [Transcript.ToolCall] = []

        for item in toolCallsArray {
            guard case .object(let dict) = item else { continue }

            let name: String
            if case .string(let n) = dict["name"], !n.isEmpty { name = n }
            else if case .string(let n) = dict["function"], !n.isEmpty { name = n }
            else { continue }

            let callID: String
            if case .string(let id) = dict["id"] { callID = id }
            else { callID = UUID().uuidString }

            let argsValue = dict["arguments"] ?? dict["parameters"] ?? .object([:])
            do {
                let gen = try GeneratedContent(jsonValue: argsValue)
                calls.append(Transcript.ToolCall(id: callID, toolName: name, arguments: gen))
            } catch {
                continue
            }
        }

        guard !calls.isEmpty else { return nil }
        let toolCalls = Transcript.ToolCalls(id: UUID().uuidString, calls)
        return .toolCalls(toolCalls)
    }
}

#endif
