#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import OpenFoundationModelsExtra
import SwiftLM
import Testing
@testable import MetalFoundationModels

@Suite("MetalRequestConverter Tests")
struct MetalRequestConverterTests {
    private let converter = MetalRequestConverter()

    @Test("Plain text produces system + user messages")
    func plainTextConversation() async throws {
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "You are helpful."))], toolDefinitions: [])),
            .prompt(.init(segments: [.text(.init(content: "Hello"))])),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: textModelConfiguration(),
            showsThinking: false
        )

        #expect(!result.hasTools)
        #expect(!result.hasSchema)
        #expect(!result.input.promptOptions.isThinkingEnabled)

        let chat = extractChat(from: result.input)
        #expect(chat.count == 2)
        #expect(chat[0].role == .system)
        #expect(extractText(from: chat[0].content) == "You are helpful.")
        #expect(chat[1].role == .user)
        #expect(extractText(from: chat[1].content) == "Hello")
    }

    @Test("Multi-turn with tool interaction preserves message order")
    func multiTurnWithToolInteraction() async throws {
        let callArguments = try GeneratedContent(json: #"{"query":"swift"}"#)
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "You are helpful."))], toolDefinitions: [searchToolDefinition()])),
            .prompt(.init(segments: [.text(.init(content: "Search for swift"))])),
            .toolCalls(.init([
                Transcript.ToolCall(id: "call-1", toolName: "search_repo", arguments: callArguments),
            ])),
            .toolOutput(.init(id: "call-1", toolName: "search_repo", segments: [.text(.init(content: "Found 3 results"))])),
            .prompt(.init(segments: [.text(.init(content: "Tell me more"))])),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: textModelConfiguration(),
            showsThinking: true
        )

        #expect(result.hasTools)
        #expect(result.input.promptOptions.isThinkingEnabled)

        let chat = extractChat(from: result.input)
        #expect(chat.count == 5)
        #expect(chat[0].role == .system)
        #expect(chat[1].role == .user)
        #expect(chat[2].role == .assistant)
        let toolCalls = try #require(chat[2].toolCalls)
        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].id == "call-1")
        #expect(toolCalls[0].function.name == "search_repo")
        #expect(toolCalls[0].function.arguments.contains(#""query""#))
        #expect(toolCalls[0].function.arguments.contains(#""swift""#))
        #expect(chat[3].role == .tool)
        #expect(chat[3].toolCallID == "call-1")
        #expect(extractText(from: chat[3].content) == "Found 3 results")
        #expect(chat[4].role == .user)
        #expect(extractText(from: chat[4].content) == "Tell me more")
    }

    @Test("Schema is appended to the system message")
    func schemaInSystemMessage() async throws {
        let schema = GenerationSchema(
            type: String.self,
            description: "A structured response",
            properties: [
                GenerationSchema.Property(name: "summary", description: "Summary", type: String.self, guides: []),
            ]
        )
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "Return data."))], toolDefinitions: [])),
            .prompt(.init(
                segments: [.text(.init(content: "Summarize"))],
                responseFormat: .init(schema: schema)
            )),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: textModelConfiguration(),
            showsThinking: false
        )

        #expect(result.hasSchema)

        let chat = extractChat(from: result.input)
        #expect(chat[0].role == .system)

        let systemText = extractText(from: chat[0].content)
        #expect(systemText.contains("Return data."))
        #expect(systemText.contains("Respond with JSON"))
        #expect(systemText.contains("\"summary\""))
    }

    @Test("Tool definitions are carried via ModelInput.tools, not embedded in system text")
    func toolDefinitionsInModelInputTools() async throws {
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "Use tools."))], toolDefinitions: [searchToolDefinition()])),
            .prompt(.init(segments: [.text(.init(content: "Search"))])),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: textModelConfiguration(),
            showsThinking: false
        )

        let tools = try #require(result.input.tools)
        #expect(tools.count == 1)
        #expect(tools[0].function.name == "search_repo")
        #expect(tools[0].function.description == "Search the repository")

        let chat = extractChat(from: result.input)
        let systemText = extractText(from: chat[0].content)
        #expect(!systemText.contains("tool_calls"))
        #expect(!systemText.contains("search_repo"))
        #expect(!systemText.contains("Parameters schema"))
    }

    @Test("Metal strips session-generated tool appendix from system text")
    func stripsGeneratedToolInstructions() async throws {
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [
                    .text(.init(content: "Use tools.")),
                    .text(.init(content: generatedToolInstructions())),
                ],
                toolDefinitions: [searchToolDefinition()]
            )),
            .prompt(.init(segments: [.text(.init(content: "Search"))])),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: textModelConfiguration(),
            showsThinking: false
        )

        let chat = extractChat(from: result.input)
        let systemText = extractText(from: chat[0].content)
        #expect(systemText.contains("Use tools."))
        #expect(!systemText.contains("# Tools"))
        #expect(!systemText.contains("In this environment you have access"))

        let tools = try #require(result.input.tools)
        #expect(tools.contains { $0.function.name == "search_repo" })
    }

    @Test("All tool definitions are carried via ModelInput.tools")
    func allToolDefinitionsRendered() async throws {
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Use tools when needed."))],
                toolDefinitions: [
                    searchToolDefinition(),
                    fetchDocsToolDefinition(),
                    openIssueToolDefinition(),
                ]
            )),
            .prompt(.init(segments: [.text(.init(content: "Search the repository for Metal stream code"))])),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: textModelConfiguration(),
            showsThinking: false
        )

        let tools = try #require(result.input.tools)
        let names = tools.map(\.function.name)
        #expect(names.contains("search_repo"))
        #expect(names.contains("fetch_docs"))
        #expect(names.contains("open_issue"))
    }

    @Test("Base64 image is converted to InputImage data")
    func base64ImagePrompt() async throws {
        let png = samplePNGData()
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "Describe this image")),
                .image(.init(source: .base64(data: png.base64EncodedString(), mediaType: "image/png"))),
            ])),
        ])

        let result = try await converter.convert(
            transcript: transcript,
            configuration: visionModelConfiguration(),
            showsThinking: false
        )

        let chat = extractChat(from: result.input)
        #expect(chat.count == 1)
        #expect(chat[0].role == .user)
        #expect(chat[0].content.count == 2)

        if case .image(let image) = chat[0].content[1] {
            switch image.source {
            case .data(let data, let mimeType):
                #expect(data == png)
                #expect(mimeType == "image/png")
            case .fileURL:
                Issue.record("Expected in-memory image data")
            }
        } else {
            Issue.record("Expected image content")
        }
    }

    @Test("Image input fails for text-only models")
    func imageInputFailsForTextOnlyModels() async throws {
        let png = samplePNGData()
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "Describe this image")),
                .image(.init(source: .base64(data: png.base64EncodedString(), mediaType: "image/png"))),
            ])),
        ])

        await #expect(throws: MetalLanguageModelError.self) {
            _ = try await converter.convert(
                transcript: transcript,
                configuration: textModelConfiguration(),
                showsThinking: false
            )
        }
    }
}

@Suite("MetalResponseConverter Tests")
struct MetalResponseConverterTests {
    private let converter = MetalResponseConverter()

    @Test("Plain text response")
    func plainTextResponse() throws {
        let entry = try converter.finalEntry(
            from: [.text("Hello world"), .completed(dummyCompletionInfo())],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractResponseText(from: entry) == "Hello world")
    }

    @Test("Think block stripping in final entry")
    func thinkBlockStripping() throws {
        let entry = try converter.finalEntry(
            from: [
                .text("<think>reasoning"),
                .text("</think>Hello"),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractResponseText(from: entry) == "Hello")
        #expect(extractReasoning(from: entry) == "reasoning")
    }

    @Test("Orphaned closing think tag treats prefix as reasoning")
    func orphanedClosingThinkTag() throws {
        let entry = try converter.finalEntry(
            from: [
                .text("prompt-opened reasoning</think>Visible answer"),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractResponseText(from: entry) == "Visible answer")
        #expect(extractReasoning(from: entry) == "prompt-opened reasoning")
    }

    @Test("Leading empty close then closing think tag treats middle as reasoning")
    func leadingEmptyCloseThenReasoning() throws {
        let entry = try converter.finalEntry(
            from: [
                .text("</think>\nprompt-opened reasoning</think>Visible answer"),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractResponseText(from: entry) == "Visible answer")
        #expect(extractReasoning(from: entry) == "prompt-opened reasoning")
    }

    @Test("Separate reasoning channel is preserved")
    func separateReasoningChannel() throws {
        let entry = try converter.finalEntry(
            from: [
                .reasoning("internal chain"),
                .text("Visible answer"),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractResponseText(from: entry) == "Visible answer")
        #expect(extractReasoning(from: entry) == "internal chain")
    }

    @Test("Structured response parses into structure segment")
    func structuredResponse() throws {
        let entry = try converter.finalEntry(
            from: [
                .text("```json\n{\"summary\":\"ok\"}\n```"),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: false,
            hasSchema: true
        )

        guard case .response(let response) = entry else {
            Issue.record("Expected response entry")
            return
        }

        guard case .structure(let structure) = response.segments.first else {
            Issue.record("Expected structure segment")
            return
        }

        let summary: String = try structure.content.value(String.self, forProperty: "summary")
        #expect(summary == "ok")
    }

    @Test("Textual JSON tool call fallback")
    func textualJSONToolCallFallback() throws {
        let entry = try converter.finalEntry(
            from: [
                .text(#"{"tool_calls":[{"name":"search_repo","arguments":{"query":"swift"}}]}"#),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: true,
            hasSchema: false
        )

        guard case .toolCalls(let calls) = entry else {
            Issue.record("Expected tool calls entry")
            return
        }

        #expect(calls.count == 1)
        #expect(calls.first?.toolName == "search_repo")
        let query: String? = try calls.first?.arguments.value(String.self, forProperty: "query")
        #expect(query == "swift")
    }

    @Test("Textual XML tool call fallback")
    func textualXMLToolCallFallback() throws {
        let entry = try converter.finalEntry(
            from: [
                .text("""
                    <tool_call>
                    <function=location_get_current>
                    <parameter=city>Tokyo</parameter>
                    </function>
                    </tool_call>
                    """),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: true,
            hasSchema: false
        )

        guard case .toolCalls(let calls) = entry else {
            Issue.record("Expected tool calls entry")
            return
        }

        #expect(calls.count == 1)
        #expect(calls.first?.toolName == "location_get_current")
        let city: String? = try calls.first?.arguments.value(String.self, forProperty: "city")
        #expect(city == "Tokyo")
    }

    @Test("Textual tool call is ignored when tools are disabled")
    func noToolDetectionWithoutTools() throws {
        let entry = try converter.finalEntry(
            from: [
                .text(#"{"tool_calls":[{"name":"search_repo","arguments":{"query":"test"}}]}"#),
                .completed(dummyCompletionInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        guard case .response = entry else {
            Issue.record("Expected response entry")
            return
        }
    }

    @Test("Streaming returns deltas instead of accumulated text")
    func streamingDelta() {
        var state = MetalResponseConverter.StreamingState()

        let first = converter.streamDelta(state: &state, event: .text("Hel"))
        let second = converter.streamDelta(state: &state, event: .text("lo"))

        #expect(first.answer == "Hel")
        #expect(second.answer == "lo")
        #expect(first.reasoning.isEmpty)
        #expect(second.reasoning.isEmpty)
    }

    @Test("Streaming suppresses think-tagged text")
    func streamingThinkSuppression() {
        var state = MetalResponseConverter.StreamingState()

        let first = converter.streamDelta(state: &state, event: .text("<think>secret"))
        let second = converter.streamDelta(state: &state, event: .text("</think>Hello"))

        #expect(first.answer.isEmpty)
        #expect(first.reasoning == "secret")
        #expect(second.answer == "Hello")
    }
}

@Suite("Metal Parameter Tests")
struct MetalParameterTests {
    @Test("Generation options map to swift-lm parameters")
    func optionMapping() {
        var options = GenerationOptions()
        options.temperature = 0.25
        options.maximumResponseTokens = 128

        let parameters = MetalLanguageModelRuntime.makeParameters(
            options: options,
            showsThinking: true
        )

        #expect(parameters.temperature == 0.25)
        #expect(parameters.maxTokens == 128)
        #expect(parameters.topP == 1.0)
        #expect(parameters.repetitionPenalty == nil)
        #expect(parameters.repetitionContextSize == 64)
        #expect(parameters.reasoning == .separate)
    }

    @Test("Thinking is hidden unless explicitly requested")
    func thinkingHiddenByDefault() {
        let parameters = MetalLanguageModelRuntime.makeParameters(
            options: nil,
            showsThinking: false
        )

        #expect(parameters.reasoning == .hidden)
    }
}

@Suite("Metal Runtime Streaming Tests")
struct MetalRuntimeStreamingTests {
    @Test("Tool-enabled text responses still stream incrementally")
    func toolEnabledTextResponseStreamsIncrementally() async throws {
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [.text(.init(content: "Use tools when needed."))],
                toolDefinitions: [searchToolDefinition()]
            )),
            .prompt(.init(segments: [.text(.init(content: "Say hello"))])),
        ])

        let runtime = MetalLanguageModelRuntime(
            configuration: textModelConfiguration(),
            showsThinking: false,
            generateStream: { _, _ in
                AsyncStream { continuation in
                    continuation.yield(.text("Hel"))
                    continuation.yield(.text("lo"))
                    continuation.yield(.completed(dummyCompletionInfo()))
                    continuation.finish()
                }
            }
        )

        let stream = await runtime.stream(transcript: transcript, options: nil)
        var entries: [Transcript.Entry] = []
        for try await entry in stream {
            entries.append(entry)
        }

        #expect(entries.count == 2)
        #expect(extractResponseText(from: entries[0]) == "Hel")
        #expect(extractResponseText(from: entries[1]) == "lo")
    }
}

private func textModelConfiguration() -> ModelConfiguration {
    ModelConfiguration(
        name: "text-model",
        inputCapabilities: .textOnly,
        executionCapabilities: .init()
    )
}

private func visionModelConfiguration() -> ModelConfiguration {
    ModelConfiguration(
        name: "vision-model",
        inputCapabilities: .init(supportsText: true, supportsImages: true, supportsVideo: false),
        executionCapabilities: .init(
            supportsTextGeneration: true,
            supportsTextEmbeddings: false,
            supportsPromptStateReuse: true,
            supportsImagePromptPreparation: true,
            supportsImageExecution: true,
            supportsVideoPromptPreparation: false,
            supportsVideoExecution: false
        )
    )
}

private func searchToolDefinition() -> Transcript.ToolDefinition {
    Transcript.ToolDefinition(
        name: "search_repo",
        description: "Search the repository",
        parameters: GenerationSchema(type: String.self, description: "Search query", properties: [])
    )
}

private func fetchDocsToolDefinition() -> Transcript.ToolDefinition {
    Transcript.ToolDefinition(
        name: "fetch_docs",
        description: "Fetch API documentation",
        parameters: GenerationSchema(type: String.self, description: "Documentation path", properties: [])
    )
}

private func openIssueToolDefinition() -> Transcript.ToolDefinition {
    Transcript.ToolDefinition(
        name: "open_issue",
        description: "Open a GitHub issue",
        parameters: GenerationSchema(type: String.self, description: "Issue title", properties: [])
    )
}

private func generatedToolInstructions() -> String {
    """
    # Tools

    In this environment you have access to a set of tools you can use to answer the user's question.
    """
}

private func extractChat(from input: ModelInput) -> [InputMessage] {
    switch input.prompt {
    case .chat(let chat):
        return chat
    case .text(let text):
        return [.user(text)]
    }
}

private func extractText(from content: [InputMessage.Content]) -> String {
    content.compactMap { item in
        if case .text(let text) = item {
            return text
        }
        return nil
    }
    .joined()
}

private func extractResponseText(from entry: Transcript.Entry) -> String? {
    guard case .response(let response) = entry else {
        return nil
    }
    return response.segments.compactMap { segment in
        if case .text(let text) = segment {
            return text.content
        }
        return nil
    }.joined()
}

private func extractReasoning(from entry: Transcript.Entry) -> String? {
    guard case .response(let response) = entry else {
        return nil
    }
    let reasoning = response.segments.compactMap { segment in
        if case .reasoning(let text) = segment {
            return text.content
        }
        return nil
    }.joined()
    return reasoning.isEmpty ? nil : reasoning
}

private func dummyCompletionInfo() -> CompletionInfo {
    CompletionInfo(tokenCount: 5, tokensPerSecond: 10, totalTime: 0.5)
}

private func samplePNGData() -> Data {
    guard let data = Data(
        base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+b9FEAAAAASUVORK5CYII="
    ) else {
        fatalError("Failed to decode test PNG fixture")
    }
    return data
}

#endif
