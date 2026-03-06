#if MLX_ENABLED
import CryptoKit
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
@preconcurrency import MLXLMCommon

enum MLXResponseMode: Sendable, Equatable {
    case text
    case toolCapable
    case structured
    case multimodal
}

enum MLXToolPolicy: String, Sendable, Equatable {
    case disabled
    case enabled
    case required
}

enum MLXCacheInvalidationReason: String, Sendable, Equatable {
    case noReusablePrefix
    case noStoredCache
    case cacheKeyChanged
    case prefixSnapshotUnsupported
    case prefixSnapshotMismatch
}

enum MLXCacheReuseScope: Sendable, Equatable {
    case none
    case prefixReusable
    case invalidate(reason: MLXCacheInvalidationReason)
}

struct MLXPrefixCacheKey: Hashable, Sendable, Equatable {
    let rawValue: String
}

struct MLXPlannerDiagnostics: Sendable, Equatable {
    let systemMessageCount: Int
    let userMessageCount: Int
    let assistantMessageCount: Int
    let toolMessageCount: Int
    let imageCount: Int
    let toolDefinitionCount: Int
    let latestUserPreview: String
}

struct MLXCachePlan {
    let reuseScope: MLXCacheReuseScope
    let cacheKey: MLXPrefixCacheKey?
    let prefixMessages: [Chat.Message]
    let suffixMessages: [Chat.Message]
    let prefixInput: UserInput?
}

struct MLXExecutionPlan {
    let input: UserInput
    let responseMode: MLXResponseMode
    let toolPolicy: MLXToolPolicy
    let cachePlan: MLXCachePlan
    let promptTokenEstimate: Int?
    let schemaFingerprint: String?
    let additionalContext: [String: any Sendable]
    let plannerDiagnostics: MLXPlannerDiagnostics
}

struct MLXWarmupPlan {
    let input: UserInput
    let cacheKey: MLXPrefixCacheKey
}

internal struct MLXTranscriptPlanner: OpenFoundationModelsExtra.RequestBuilder {

    struct BuildResult: Sendable {
        let input: UserInput
        let expectsTool: Bool
    }

    func build(transcript: Transcript, options: GenerationOptions?, stream: Bool) throws -> BuildResult {
        let plan = try plan(transcript: transcript, options: options, metadata: .fallback(modelID: "unknown"))
        return BuildResult(input: plan.input, expectsTool: plan.toolPolicy != .disabled)
    }

    func plan(
        transcript: Transcript,
        options: GenerationOptions?,
        metadata: MLXModelMetadata
    ) throws -> MLXExecutionPlan {
        let resolved = transcript.resolved()
        let latestPromptText = latestPromptText(from: resolved)
        let toolPolicy = decideToolPolicy(resolved: resolved, metadata: metadata)
        let toolDirective = toolDirective(
            latestPromptText: latestPromptText,
            toolPolicy: toolPolicy,
            availableTools: resolved.toolDefinitions
        )
        let selectedToolDefinitions = selectToolDefinitions(
            from: resolved.toolDefinitions,
            latestPromptText: latestPromptText,
            toolPolicy: toolPolicy,
            hasToolHistory: resolved.contains { entry in
                if case .tool = entry {
                    return true
                }
                return false
            }
        )
        let messages = try canonicalMessages(from: resolved, toolDirective: toolDirective)
        let schemaFingerprint = fingerprint(for: resolved.latestResponseFormat?._schema)
        let additionalContext = makeAdditionalContext(
            from: resolved,
            metadata: metadata,
            schemaFingerprint: schemaFingerprint
        )
        let toolSpecs = toolPolicy == .disabled ? nil : buildToolSpecs(from: selectedToolDefinitions)
        let input = UserInput(chat: messages, tools: toolSpecs, additionalContext: additionalContext)
        let imageCount = messages.reduce(into: 0) { $0 += $1.images.count }

        let responseMode: MLXResponseMode
        if imageCount > 0 {
            responseMode = .multimodal
        } else if schemaFingerprint != nil {
            responseMode = .structured
        } else if toolPolicy != .disabled {
            responseMode = .toolCapable
        } else {
            responseMode = .text
        }

        let cachePlan = try makeCachePlan(
            messages: messages,
            toolPolicy: toolPolicy,
            toolSpecs: toolSpecs,
            additionalContext: additionalContext,
            metadata: metadata,
            schemaFingerprint: schemaFingerprint,
            imageCount: imageCount
        )

        let diagnostics = MLXPlannerDiagnostics(
            systemMessageCount: messages.filter { $0.role == .system }.count,
            userMessageCount: messages.filter { $0.role == .user }.count,
            assistantMessageCount: messages.filter { $0.role == .assistant }.count,
            toolMessageCount: messages.filter { $0.role == .tool }.count,
            imageCount: imageCount,
            toolDefinitionCount: toolSpecs?.count ?? 0,
            latestUserPreview: previewText(latestPromptText)
        )

        return MLXExecutionPlan(
            input: input,
            responseMode: responseMode,
            toolPolicy: toolPolicy,
            cachePlan: cachePlan,
            promptTokenEstimate: nil,
            schemaFingerprint: schemaFingerprint,
            additionalContext: additionalContext,
            plannerDiagnostics: diagnostics
        )
    }

    func warmupPlan(
        transcript: Transcript,
        options _: GenerationOptions?,
        metadata: MLXModelMetadata
    ) throws -> MLXWarmupPlan? {
        let resolved = transcript.resolved()
        let messages = try canonicalMessages(from: resolved, toolDirective: nil)
        let imageCount = messages.reduce(into: 0) { $0 += $1.images.count }
        guard imageCount == 0, !messages.isEmpty else {
            return nil
        }

        let toolPolicy = decideToolPolicy(resolved: resolved, metadata: metadata)
        guard toolPolicy == .disabled else {
            return nil
        }

        let schemaFingerprint = fingerprint(for: resolved.latestResponseFormat?._schema)
        let additionalContext = makeAdditionalContext(
            from: resolved,
            metadata: metadata,
            schemaFingerprint: schemaFingerprint
        )
        let cacheKey = try makeCacheKey(
            modelID: metadata.modelID,
            runtimeFamily: metadata.runtimeFamily,
            modalityFamily: metadata.modalityFamily,
            messages: messages,
            toolPolicy: toolPolicy,
            toolSpecs: nil,
            schemaFingerprint: schemaFingerprint,
            additionalContext: additionalContext
        )

        return MLXWarmupPlan(
            input: UserInput(chat: messages, tools: nil, additionalContext: additionalContext),
            cacheKey: cacheKey
        )
    }

    private func canonicalMessages(
        from resolved: ResolvedTranscript,
        toolDirective: String?
    ) throws -> [Chat.Message] {
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
        var schemaJSON: String? = nil
        if let schema = resolved.latestResponseFormat?._schema {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let data = try encoder.encode(schema)
            schemaJSON = String(decoding: data, as: UTF8.self)
        }

        enum MessageType {
            case system
            case user
            case assistant
            case tool
        }

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
                let toolCallText = interaction.calls
                    .map { "\($0.toolName)(\($0.arguments.jsonString))" }
                    .joined(separator: "\n")
                if !toolCallText.isEmpty {
                    rawMessages.append((.assistant, toolCallText))
                }
                for output in interaction.outputs {
                    rawMessages.append((.tool, segmentsToText(output.segments)))
                }
            }
        }

        var messages: [Chat.Message] = []
        let systemMessages = rawMessages
            .filter { $0.type == .system }
            .map(\.content)
            .filter { !$0.isEmpty }

        if !systemMessages.isEmpty {
            var content = systemMessages.joined(separator: "\n\n")
            if let toolDirective, !toolDirective.isEmpty {
                content += "\n\n\(toolDirective)"
            }
            if let schemaJSON {
                content += "\n\nRespond with JSON matching this schema:\n\(schemaJSON)"
            }
            messages.append(.system(content))
        } else if toolDirective != nil || schemaJSON != nil {
            var content = ""
            if let toolDirective, !toolDirective.isEmpty {
                content += toolDirective
            }
            if let schemaJSON {
                if !content.isEmpty {
                    content += "\n\n"
                }
                content += "Respond with JSON matching this schema:\n\(schemaJSON)"
            }
            messages.append(.system(content))
        }

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

    private func decideToolPolicy(
        resolved: ResolvedTranscript,
        metadata: MLXModelMetadata
    ) -> MLXToolPolicy {
        guard !resolved.toolDefinitions.isEmpty else {
            return .disabled
        }

        let toolInteractions = resolved.compactMap { entry -> ResolvedTranscript.ToolInteraction? in
            guard case .tool(let interaction) = entry else {
                return nil
            }
            return interaction
        }

        if let lastToolInteraction = toolInteractions.last, lastToolInteraction.outputs.isEmpty {
            return .required
        }

        let latestPromptText = latestPromptText(from: resolved)

        if toolInteractions.isEmpty, isPlainConversation(text: latestPromptText) {
            return .disabled
        }

        if metadata.prefersConservativeToolUse && isPlainConversation(text: latestPromptText) {
            return .disabled
        }

        if requiresToolBeforeAnswer(text: latestPromptText, availableTools: resolved.toolDefinitions) {
            return .required
        }

        if isToolOrientedRequest(text: latestPromptText) || !toolInteractions.isEmpty {
            return .enabled
        }

        return .enabled
    }

    private func makeAdditionalContext(
        from _: ResolvedTranscript,
        metadata: MLXModelMetadata,
        schemaFingerprint _: String?
    ) -> [String: any Sendable] {
        var context: [String: any Sendable] = [:]
        if metadata.prefersThinkingDisabled {
            context["enable_thinking"] = false
        }
        return context
    }

    private func makeCachePlan(
        messages: [Chat.Message],
        toolPolicy: MLXToolPolicy,
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable],
        metadata: MLXModelMetadata,
        schemaFingerprint: String?,
        imageCount: Int
    ) throws -> MLXCachePlan {
        guard imageCount == 0,
              let lastUserIndex = messages.lastIndex(where: { $0.role == .user }),
              lastUserIndex > 0
        else {
            return MLXCachePlan(
                reuseScope: .none,
                cacheKey: nil,
                prefixMessages: [],
                suffixMessages: messages,
                prefixInput: nil
            )
        }

        let prefixMessages = Array(messages[..<lastUserIndex])
        let suffixMessages = Array(messages[lastUserIndex...])
        guard !prefixMessages.isEmpty else {
            return MLXCachePlan(
                reuseScope: .none,
                cacheKey: nil,
                prefixMessages: prefixMessages,
                suffixMessages: suffixMessages,
                prefixInput: nil
            )
        }

        guard supportsStandalonePrefixInput(prefixMessages) else {
            return MLXCachePlan(
                reuseScope: .none,
                cacheKey: nil,
                prefixMessages: prefixMessages,
                suffixMessages: suffixMessages,
                prefixInput: nil
            )
        }

        let key = try makeCacheKey(
            modelID: metadata.modelID,
            runtimeFamily: metadata.runtimeFamily,
            modalityFamily: metadata.modalityFamily,
            messages: prefixMessages,
            toolPolicy: toolPolicy,
            toolSpecs: toolSpecs,
            schemaFingerprint: schemaFingerprint,
            additionalContext: additionalContext
        )

        return MLXCachePlan(
            reuseScope: .prefixReusable,
            cacheKey: key,
            prefixMessages: prefixMessages,
            suffixMessages: suffixMessages,
            prefixInput: UserInput(chat: prefixMessages, tools: toolSpecs, additionalContext: additionalContext)
        )
    }

    private func makeCacheKey(
        modelID: String,
        runtimeFamily: MLXRuntimeFamily,
        modalityFamily: MLXModalityFamily,
        messages: [Chat.Message],
        toolPolicy: MLXToolPolicy,
        toolSpecs: [[String: any Sendable]]?,
        schemaFingerprint: String?,
        additionalContext: [String: any Sendable]
    ) throws -> MLXPrefixCacheKey {
        struct CanonicalMessage: Encodable {
            let role: String
            let content: String
            let imageCount: Int
            let videoCount: Int
        }

        struct CanonicalTool: Encodable {
            let name: String
            let description: String
            let parameters: String?
        }

        struct CanonicalPayload: Encodable {
            let modelID: String
            let runtimeFamily: String
            let modalityFamily: String
            let toolPolicy: String
            let schemaFingerprint: String?
            let messages: [CanonicalMessage]
            let tools: [CanonicalTool]
            let additionalContext: [String: String]
        }

        let canonicalMessages = messages.map {
            CanonicalMessage(
                role: $0.role.rawValue,
                content: $0.content,
                imageCount: $0.images.count,
                videoCount: $0.videos.count
            )
        }

        let canonicalTools: [CanonicalTool] = toolSpecs?.compactMap { spec in
            guard let function = spec["function"] as? [String: any Sendable] else {
                return nil
            }
            return CanonicalTool(
                name: function["name"] as? String ?? "",
                description: function["description"] as? String ?? "",
                parameters: function["parameters"].map { String(describing: $0) }
            )
        } ?? []

        let payload = CanonicalPayload(
            modelID: modelID,
            runtimeFamily: runtimeFamily.rawValue,
            modalityFamily: modalityFamily.rawValue,
            toolPolicy: toolPolicy.rawValue,
            schemaFingerprint: schemaFingerprint,
            messages: canonicalMessages,
            tools: canonicalTools,
            additionalContext: canonicalizeAdditionalContext(additionalContext)
        )

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(payload)
        let digest = SHA256.hash(data: data)
        let rawValue = digest.map { String(format: "%02x", $0) }.joined()
        return MLXPrefixCacheKey(rawValue: rawValue)
    }

    private func buildToolSpecs(
        from toolDefinitions: [Transcript.ToolDefinition]
    ) -> [[String: any Sendable]]? {
        guard !toolDefinitions.isEmpty else {
            return nil
        }

        let specs: [[String: any Sendable]] = toolDefinitions.map { definition in
            var function: [String: any Sendable] = [
                "name": definition.name,
                "description": definition.description,
            ]
            do {
                let jsonValue = try JSONValue(definition.parameters)
                function["parameters"] = jsonValue.sendableValue
            } catch {
                Logger.warning("[MLXTranscriptPlanner] Failed to convert tool schema for \(definition.name): \(error)")
            }
            return [
                "type": "function" as any Sendable,
                "function": function as any Sendable,
            ]
        }

        return specs.isEmpty ? nil : specs
    }

    private func fingerprint(for schema: GenerationSchema?) -> String? {
        guard let schema else {
            return nil
        }

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data: Data
        do {
            data = try encoder.encode(schema)
        } catch {
            Logger.warning("[MLXTranscriptPlanner] Failed to encode schema fingerprint: \(error)")
            return nil
        }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private func canonicalizeAdditionalContext(
        _ context: [String: any Sendable]
    ) -> [String: String] {
        var normalized: [String: String] = [:]
        for key in context.keys.sorted() {
            guard let value = context[key] else {
                continue
            }
            normalized[key] = String(describing: value)
        }
        return normalized
    }

    private func isPlainConversation(text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return true
        }

        let normalized = trimmed.lowercased()
        let exactSmallTalk = [
            "hello", "hi", "hey", "thanks", "thank you", "what's up", "how are you",
            "こんにちは", "こんばんは", "ありがとう", "おはよう", "元気?", "元気？", "調子どう",
        ]
        let prefixSmallTalk = [
            "hello ", "hi ", "hey ", "thanks ", "thank you ",
            "こんにちは", "こんばんは", "ありがとう", "おはよう",
        ]

        if exactSmallTalk.contains(normalized) {
            return true
        }

        if normalized.count <= 24, prefixSmallTalk.contains(where: { normalized.hasPrefix($0) }) {
            return true
        }

        return false
    }

    private func isToolOrientedRequest(text: String) -> Bool {
        let keywords = [
            "file", "files", "code", "repo", "repository", "git", "commit", "search", "find",
            "run", "execute", "edit", "patch", "write", "open", "fetch", "inspect",
            "location", "weather", "route", "map", "nearby",
            "実行", "編集", "修正", "検索", "調査", "確認", "ファイル", "コード", "コミット",
            "現在地", "位置", "場所", "天気", "地図", "近く", "周辺", "経路",
        ]
        return keywords.contains(where: { text.contains($0) })
    }

    private func requiresToolBeforeAnswer(
        text: String,
        availableTools _: [Transcript.ToolDefinition]
    ) -> Bool {
        let requiredKeywords = [
            "current location", "where am i", "my location", "near me", "nearby",
            "weather", "route", "eta", "contact", "notification",
            "現在地", "今いる場所", "自分の位置", "近く", "周辺", "天気", "経路", "到着", "連絡先", "通知",
        ]
        return requiredKeywords.contains(where: { text.contains($0) })
    }

    private func toolDirective(
        latestPromptText: String,
        toolPolicy: MLXToolPolicy,
        availableTools: [Transcript.ToolDefinition]
    ) -> String? {
        guard toolPolicy == .required
            || requiresToolBeforeAnswer(text: latestPromptText, availableTools: availableTools)
        else {
            return nil
        }

        return """
        This request requires a tool call before answering. Do not answer from memory or with a generic greeting. If a matching tool is available, call it first and then answer using the tool result.
        """
    }

    private func selectToolDefinitions(
        from toolDefinitions: [Transcript.ToolDefinition],
        latestPromptText _: String,
        toolPolicy _: MLXToolPolicy,
        hasToolHistory _: Bool
    ) -> [Transcript.ToolDefinition] {
        toolDefinitions
    }

    private func latestPromptText(from resolved: ResolvedTranscript) -> String {
        resolved.reversed().compactMap { entry -> String? in
            guard case .prompt(let prompt) = entry else {
                return nil
            }
            return segmentsToText(prompt.segments).lowercased()
        }.first ?? ""
    }

    private func previewText(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > 80 else {
            return trimmed
        }
        let index = trimmed.index(trimmed.startIndex, offsetBy: 80)
        return String(trimmed[..<index]) + "..."
    }

    private func supportsStandalonePrefixInput(_ messages: [Chat.Message]) -> Bool {
        var sawUserQuery = false

        for message in messages {
            guard message.role == .user else {
                continue
            }

            let trimmed = message.content.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                continue
            }

            if trimmed.hasPrefix("<tool_response>"), trimmed.hasSuffix("</tool_response>") {
                continue
            }

            sawUserQuery = true
            break
        }

        return sawUserQuery
    }
}

private extension MLXTranscriptPlanner {
    func segmentsToText(_ segments: [Transcript.Segment]) -> String {
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
