#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra

/// Detects textual tool call payloads in model output.
enum MetalToolCallDetector {
    static func entryIfPresent(_ text: String) -> Transcript.Entry? {
        if let entry = detectJSONToolCalls(in: text) {
            return entry
        }

        if let entry = detectXMLFunctionCalls(in: text) {
            return entry
        }

        return detectBracketToolCalls(in: text)
    }

    private static func detectJSONToolCalls(in text: String) -> Transcript.Entry? {
        let objects = JSONScanner.allTopLevelObjects(in: cleanText(text))

        for object in objects {
            guard case .object(let dictionary) = object else {
                continue
            }

            if case .array(let array) = dictionary["tool_calls"], !array.isEmpty {
                return buildToolCallsEntry(from: array)
            }

            if let toolCall = buildToolCall(from: dictionary) {
                return .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, [toolCall]))
            }
        }

        return nil
    }

    private static func detectXMLFunctionCalls(in text: String) -> Transcript.Entry? {
        let cleaned = cleanText(text)
        let regex: NSRegularExpression
        do {
            regex = try NSRegularExpression(
                pattern: #"<function=([\s\S]*?)</function>"#,
                options: [.caseInsensitive]
            )
        } catch {
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

        return .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, calls))
    }

    private static func detectBracketToolCalls(in text: String) -> Transcript.Entry? {
        let cleaned = cleanText(text)
        var calls: [Transcript.ToolCall] = []
        var searchStart = cleaned.startIndex

        while searchStart < cleaned.endIndex {
            guard let openBracket = cleaned[searchStart...].firstIndex(of: "[") else {
                break
            }
            guard let parsed = parseBracketCall(cleaned, from: openBracket) else {
                searchStart = cleaned.index(after: openBracket)
                continue
            }
            if let call = parsed.call {
                calls.append(call)
            }
            searchStart = parsed.endIndex
        }

        guard !calls.isEmpty else {
            return nil
        }

        return .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, calls))
    }

    private static func buildToolCallsEntry(from array: [JSONValue]) -> Transcript.Entry? {
        let calls = array.compactMap(toolCallValue(from:))
        guard !calls.isEmpty else {
            return nil
        }
        return .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, calls))
    }

    private static func buildToolCall(
        from dictionary: [String: JSONValue]
    ) -> Transcript.ToolCall? {
        toolCallValue(from: .object(dictionary))
    }

    private static func toolCallValue(from value: JSONValue) -> Transcript.ToolCall? {
        guard case .object(let dictionary) = value else {
            return nil
        }

        let name: String
        if case .string(let string) = dictionary["name"], !string.isEmpty {
            name = string
        } else if case .string(let string) = dictionary["function"], !string.isEmpty {
            name = string
        } else {
            return nil
        }

        let callID: String
        if case .string(let string) = dictionary["id"] {
            callID = string
        } else {
            callID = UUID().uuidString
        }

        let arguments = dictionary["arguments"] ?? dictionary["parameters"] ?? .object([:])
        do {
            let generatedContent = try GeneratedContent(jsonValue: arguments)
            return Transcript.ToolCall(
                id: callID,
                toolName: name,
                arguments: generatedContent
            )
        } catch {
            return nil
        }
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
        if let data = trimmed.data(using: .utf8) {
            do {
                return try JSONDecoder().decode(JSONValue.self, from: data)
            } catch {
                return .string(trimmed)
            }
        }
        return .string(trimmed)
    }

    private static func parseBracketCall(
        _ text: String,
        from openBracket: String.Index
    ) -> (call: Transcript.ToolCall?, endIndex: String.Index)? {
        let afterBracket = text.index(after: openBracket)
        guard afterBracket < text.endIndex else {
            return nil
        }

        var nameEnd = afterBracket
        while nameEnd < text.endIndex {
            let character = text[nameEnd]
            guard character.isLetter || character.isNumber || character == "_" else {
                break
            }
            nameEnd = text.index(after: nameEnd)
        }

        guard nameEnd > afterBracket, nameEnd < text.endIndex, text[nameEnd] == "(" else {
            return nil
        }

        let functionName = String(text[afterBracket..<nameEnd])
        let argumentsStart = text.index(after: nameEnd)
        guard let closeParen = findMatchingClose(text, from: nameEnd, open: "(", close: ")") else {
            return nil
        }

        let argumentsString = String(text[argumentsStart..<closeParen])
        let afterParen = text.index(after: closeParen)
        guard afterParen < text.endIndex, text[afterParen] == "]" else {
            return nil
        }

        let arguments = parseKeywordArguments(argumentsString)
        let endIndex = text.index(after: afterParen)

        do {
            let generatedContent = try GeneratedContent(jsonValue: arguments)
            let call = Transcript.ToolCall(
                id: UUID().uuidString,
                toolName: functionName,
                arguments: generatedContent
            )
            return (call, endIndex)
        } catch {
            return (nil, endIndex)
        }
    }

    private static func parseKeywordArguments(_ arguments: String) -> JSONValue {
        let trimmed = arguments.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return .object([:])
        }

        let parts = splitTopLevel(trimmed, separator: ",")
        var result: [String: JSONValue] = [:]

        for part in parts {
            let pair = part.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let equalIndex = pair.firstIndex(of: "=") else {
                continue
            }

            let key = String(pair[..<equalIndex]).trimmingCharacters(in: .whitespacesAndNewlines)
            let value = String(pair[pair.index(after: equalIndex)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if !key.isEmpty {
                result[key] = parseBracketValue(value)
            }
        }

        return .object(result)
    }

    private static func parseBracketValue(_ string: String) -> JSONValue {
        if string.hasPrefix("\""), string.hasSuffix("\""), string.count >= 2 {
            let inner = String(string.dropFirst().dropLast())
            return .string(inner.replacingOccurrences(of: "\\\"", with: "\""))
        }
        if string == "true" || string == "True" {
            return .bool(true)
        }
        if string == "false" || string == "False" {
            return .bool(false)
        }
        if string == "None" || string == "null" || string == "nil" {
            return .null
        }
        if let intValue = Int(string) {
            return .int(intValue)
        }
        if let doubleValue = Double(string) {
            return .double(doubleValue)
        }
        if string.hasPrefix("[") && string.hasSuffix("]") {
            let inner = String(string.dropFirst().dropLast())
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if inner.isEmpty {
                return .array([])
            }
            let elements = splitTopLevel(inner, separator: ",")
            return .array(
                elements.map {
                    parseBracketValue($0.trimmingCharacters(in: .whitespacesAndNewlines))
                }
            )
        }
        return .string(string)
    }

    private static func splitTopLevel(
        _ text: String,
        separator: Character
    ) -> [String] {
        var parts: [String] = []
        var current = ""
        var depth = 0
        var inString = false
        var escaped = false

        for character in text {
            if escaped {
                current.append(character)
                escaped = false
                continue
            }
            if character == "\\" && inString {
                current.append(character)
                escaped = true
                continue
            }
            if character == "\"" {
                inString.toggle()
                current.append(character)
                continue
            }
            if inString {
                current.append(character)
                continue
            }
            if character == "[" || character == "{" || character == "(" {
                depth += 1
            }
            if character == "]" || character == "}" || character == ")" {
                depth -= 1
            }
            if character == separator && depth == 0 {
                parts.append(current)
                current = ""
            } else {
                current.append(character)
            }
        }

        if !current.isEmpty {
            parts.append(current)
        }
        return parts
    }

    private static func findMatchingClose(
        _ text: String,
        from start: String.Index,
        open: Character,
        close: Character
    ) -> String.Index? {
        var depth = 0
        var inString = false
        var escaped = false

        for index in text[start...].indices {
            let character = text[index]
            if escaped {
                escaped = false
                continue
            }
            if character == "\\" && inString {
                escaped = true
                continue
            }
            if character == "\"" {
                inString.toggle()
                continue
            }
            if inString {
                continue
            }
            if character == open {
                depth += 1
            }
            if character == close {
                depth -= 1
                if depth == 0 {
                    return index
                }
            }
        }

        return nil
    }

    private static func cleanText(_ text: String) -> String {
        text
            .replacingOccurrences(of: "\u{FEFF}", with: "")
            .replacingOccurrences(of: "\u{200B}", with: "")
            .replacingOccurrences(of: "\u{200C}", with: "")
            .replacingOccurrences(of: "\u{200D}", with: "")
    }
}

private extension MetalToolCallDetector {
    enum JSONScanner {
        static func allTopLevelObjects(in text: String) -> [JSONValue] {
            var objects: [JSONValue] = []
            var searchIndex = text.startIndex

            while searchIndex < text.endIndex {
                guard let start = text[searchIndex...].firstIndex(of: "{") else {
                    break
                }

                var depth = 0
                var endIndex: String.Index?
                var inString = false
                var escaped = false

                for index in text[start...].indices {
                    let character = text[index]
                    if character == "\\" && inString {
                        escaped.toggle()
                        continue
                    }
                    if character == "\"" && !escaped {
                        inString.toggle()
                    }
                    if !inString {
                        if character == "{" {
                            depth += 1
                        }
                        if character == "}" {
                            depth -= 1
                            if depth == 0 {
                                endIndex = index
                                break
                            }
                        }
                    }
                    escaped = false
                }

                guard let endIndex else {
                    break
                }

                let jsonSlice = text[start...endIndex]
                if let data = String(jsonSlice).data(using: .utf8) {
                    do {
                        let value = try JSONDecoder().decode(JSONValue.self, from: data)
                        if case .object = value {
                            objects.append(value)
                        }
                    } catch {
                        // Ignore malformed candidate and keep scanning.
                    }
                }

                searchIndex = text.index(after: endIndex)
            }

            return objects
        }
    }
}

#endif
