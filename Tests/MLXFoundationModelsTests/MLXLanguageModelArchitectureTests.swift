#if MLX_ENABLED
import Foundation
import MLXLMCommon
import OpenFoundationModels
import OpenFoundationModelsCore
import Testing
@testable import MLXFoundationModels

// MARK: - MLXRequestConverter Tests

@Suite("MLXRequestConverter Tests")
struct MLXRequestConverterTests {
    private let converter = MLXRequestConverter()

    @Test("Plain text produces system + user messages")
    func plainTextConversation() throws {
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "You are helpful."))], toolDefinitions: [])),
            .prompt(.init(segments: [.text(.init(content: "Hello"))])),
        ])

        let result = try converter.convert(transcript: transcript, metadata: llmMetadata())

        #expect(!result.hasTools)
        #expect(!result.hasSchema)
        let chat = extractChat(from: result.input)
        #expect(chat.count == 2)
        #expect(chat[0].role == .system)
        #expect(chat[0].content == "You are helpful.")
        #expect(chat[1].role == .user)
        #expect(chat[1].content == "Hello")
        #expect(result.input.tools == nil)
    }

    @Test("Tool definitions produce tool specs and hasTools=true")
    func toolDefinitionsPresent() throws {
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "Use tools."))], toolDefinitions: [searchToolDefinition()])),
            .prompt(.init(segments: [.text(.init(content: "Search for TODOs"))])),
        ])

        let result = try converter.convert(transcript: transcript, metadata: llmMetadata())

        #expect(result.hasTools)
        #expect(!result.hasSchema)
        #expect(result.input.tools != nil)
        #expect(result.input.tools?.count == 1)
    }

    @Test("Multi-turn with tool interaction produces correct message order")
    func multiTurnWithToolInteraction() throws {
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

        let result = try converter.convert(transcript: transcript, metadata: llmMetadata())

        // system, user, assistant (tool call text), tool (output), user
        let chat = extractChat(from: result.input)
        #expect(chat.count == 5)
        #expect(chat[0].role == .system)
        #expect(chat[1].role == .user)
        #expect(chat[2].role == .assistant)
        #expect(chat[3].role == .tool)
        #expect(chat[3].content == "Found 3 results")
        #expect(chat[4].role == .user)
        #expect(chat[4].content == "Tell me more")
    }

    @Test("Schema appended to system message with hasSchema=true")
    func schemaInResponseFormat() throws {
        let transcript = Transcript(entries: [
            .instructions(.init(segments: [.text(.init(content: "Return data."))], toolDefinitions: [])),
            .prompt(
                .init(
                    segments: [.text(.init(content: "Summarize"))],
                    responseFormat: .init(schema: GenerationSchema(type: String.self, description: "A summary", properties: []))
                )
            ),
        ])

        let result = try converter.convert(transcript: transcript, metadata: llmMetadata())

        #expect(!result.hasTools)
        #expect(result.hasSchema)
        let chat = extractChat(from: result.input)
        #expect(chat[0].role == .system)
        #expect(chat[0].content.contains("Respond with JSON matching this schema:"))
    }

    @Test("Qwen metadata sets enable_thinking=false in additionalContext")
    func qwenThinkingDisabled() throws {
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Hello"))])),
        ])

        let metadata = MLXModelMetadata(modelID: "mlx-community/Qwen3.5-4B", runtimeFamily: .llm)
        let result = try converter.convert(transcript: transcript, metadata: metadata)

        #expect(result.input.additionalContext?["enable_thinking"] as? Bool == false)
    }

    @Test("Non-Qwen metadata has no enable_thinking key")
    func nonQwenNoThinkingKey() throws {
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: "Hello"))])),
        ])

        let metadata = MLXModelMetadata(modelID: "mlx-community/Llama-3B", runtimeFamily: .llm)
        let result = try converter.convert(transcript: transcript, metadata: metadata)

        #expect(result.input.additionalContext?["enable_thinking"] == nil)
    }
}

// MARK: - MLXResponseConverter Tests

@Suite("MLXResponseConverter Tests")
struct MLXResponseConverterTests {
    private let converter = MLXResponseConverter()

    @Test("Plain text response")
    func plainTextResponse() throws {
        let entry = try converter.finalEntry(
            from: [.chunk("Hello world"), .info(dummyInfo())],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractText(from: entry) == "Hello world")
    }

    @Test("Think block stripping in final entry")
    func thinkBlockStripping() throws {
        let entry = try converter.finalEntry(
            from: [
                .chunk("<think>reasoning</think>\nHello"),
                .chunk("</think>\nWorld"),
                .info(dummyInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractText(from: entry) == "Hello\nWorld")
    }

    @Test("Native tool calls take priority over textual detection")
    func nativeToolCallPriority() throws {
        let entry = try converter.finalEntry(
            from: [
                .chunk(#"{"tool_calls":[{"name":"search_repo","arguments":{"query":"fallback"}}]}"#),
                .toolCall(ToolCall(function: .init(name: "search_repo", arguments: ["query": "native"]))),
                .info(dummyInfo()),
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
        #expect(query == "native")
    }

    @Test("Textual JSON tool call fallback")
    func textualJSONToolCallFallback() throws {
        let entry = try converter.finalEntry(
            from: [
                .chunk(#"{"tool_calls":[{"name":"search_repo","arguments":{"query":"swift"}}]}"#),
                .info(dummyInfo()),
            ],
            hasTools: true,
            hasSchema: false
        )

        guard case .toolCalls(let calls) = entry else {
            Issue.record("Expected textual tool call detection")
            return
        }

        #expect(calls.count == 1)
        let query: String? = try calls.first?.arguments.value(String.self, forProperty: "query")
        #expect(query == "swift")
    }

    @Test("Textual XML tool call fallback (Qwen style)")
    func textualXMLToolCallFallback() throws {
        let entry = try converter.finalEntry(
            from: [
                .chunk("""
                    <tool_call>
                    <function=location_get_current>
                    </function>
                    </tool_call>
                    """),
                .info(dummyInfo()),
            ],
            hasTools: true,
            hasSchema: false
        )

        guard case .toolCalls(let calls) = entry else {
            Issue.record("Expected XML textual tool call detection")
            return
        }

        #expect(calls.count == 1)
        #expect(calls.first?.toolName == "location_get_current")
    }

    @Test("Textual tool call not detected when hasTools=false")
    func noToolDetectionWithoutTools() throws {
        let entry = try converter.finalEntry(
            from: [
                .chunk(#"{"tool_calls":[{"name":"search_repo","arguments":{"query":"test"}}]}"#),
                .info(dummyInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        guard case .response = entry else {
            Issue.record("Expected text response when hasTools=false")
            return
        }
    }

    @Test("Streaming delta suppresses think blocks")
    func streamingThinkBlockSuppression() {
        var state = MLXResponseConverter.StreamingState()
        let firstDelta = converter.streamDelta(state: &state, chunk: "<think>secret")
        let secondDelta = converter.streamDelta(state: &state, chunk: "</think>Hello")

        #expect(firstDelta.isEmpty)
        #expect(secondDelta == "Hello")
        #expect(!state.emittedVisibleText.contains("secret"))
    }

    @Test("Code fence stripping")
    func codeFenceStripping() throws {
        let entry = try converter.finalEntry(
            from: [
                .chunk("```json\n{\"key\": \"value\"}\n```"),
                .info(dummyInfo()),
            ],
            hasTools: false,
            hasSchema: false
        )

        #expect(extractText(from: entry) == "{\"key\": \"value\"}")
    }
}

// MARK: - Parameter Computation Tests

@Suite("Parameter Computation Tests")
struct ParameterComputationTests {

    @Test("Short prompt uses prefillStepSize=512, no KV quantization")
    func shortPromptParameters() {
        let params = makeTestParameters(promptTokenCount: 1024, runtimeFamily: .llm)

        #expect(params.prefillStepSize == 512)
        #expect(params.kvBits == nil)
        #expect(params.maxKVSize == nil)
    }

    @Test("Medium prompt uses prefillStepSize=1024")
    func mediumPromptParameters() {
        let params = makeTestParameters(promptTokenCount: 4096, runtimeFamily: .llm)

        #expect(params.prefillStepSize == 1024)
        #expect(params.kvBits == nil)
    }

    @Test("Long prompt uses prefillStepSize=1536 with KV quantization")
    func longPromptParameters() {
        let params = makeTestParameters(promptTokenCount: 9000, runtimeFamily: .llm)

        #expect(params.prefillStepSize == 1536)
        #expect(params.kvBits == 4)
        #expect(params.quantizedKVStart == 4096)
    }

    @Test("VLM caps prefillStepSize at 1024")
    func vlmPrefillCap() {
        let params = makeTestParameters(promptTokenCount: 9000, runtimeFamily: .vlm)

        #expect(params.prefillStepSize == 1024)
        #expect(params.kvBits == 4)
    }
}

@Suite("MLX Metadata Registry Tests")
struct MLXMetadataRegistryTests {

    @Test("Metadata can be looked up by alias")
    func metadataAliasLookup() async {
        let modelID = "file:///tmp/\(UUID().uuidString)"
        let alias = "mlx-community/\(UUID().uuidString)"
        let metadata = MLXModelMetadata(modelID: modelID, runtimeFamily: .vlm)

        await MLXModelMetadataRegistry.shared.register(metadata, aliases: [alias])

        let resolved = await MLXModelMetadataRegistry.shared.metadata(for: alias)
        #expect(resolved?.modelID == modelID)
        #expect(resolved?.runtimeFamily == .vlm)
    }
}

// MARK: - Test Helpers

private func llmMetadata() -> MLXModelMetadata {
    MLXModelMetadata(modelID: "mlx-community/Llama-3B", runtimeFamily: .llm)
}

private func searchToolDefinition() -> Transcript.ToolDefinition {
    Transcript.ToolDefinition(
        name: "search_repo",
        description: "Search the repository",
        parameters: GenerationSchema(type: String.self, description: "Search query", properties: [])
    )
}

private func extractText(from entry: Transcript.Entry) -> String? {
    guard case .response(let response) = entry else { return nil }
    guard response.segments.count == 1 else { return nil }
    guard case .text(let text) = response.segments[0] else { return nil }
    return text.content
}

private func extractChat(from input: UserInput) -> [Chat.Message] {
    switch input.prompt {
    case .chat(let chat):
        return chat
    case .text(let text):
        return [.user(text)]
    case .messages:
        Issue.record("Expected chat-form user input")
        return []
    }
}

private func dummyInfo() -> GenerateCompletionInfo {
    GenerateCompletionInfo(
        promptTokenCount: 10,
        generationTokenCount: 5,
        promptTime: 0.1,
        generationTime: 0.2,
        stopReason: .stop
    )
}

/// Replicates the parameter computation logic from MLXLanguageModelRuntime
/// for isolated testing without ModelContainer.
private func makeTestParameters(
    promptTokenCount: Int,
    runtimeFamily: MLXRuntimeFamily,
    options: GenerationOptions? = nil
) -> GenerateParameters {
    var prefillStepSize = 512
    if promptTokenCount > 8192 {
        prefillStepSize = 1536
    } else if promptTokenCount > 2048 {
        prefillStepSize = 1024
    }
    if runtimeFamily == .vlm {
        prefillStepSize = min(prefillStepSize, 1024)
    }

    let kvBits: Int? = promptTokenCount > 8192 ? 4 : nil
    let quantizedKVStart = kvBits == nil ? 0 : 4096

    return GenerateParameters(
        maxTokens: options?.maximumResponseTokens ?? 2048,
        maxKVSize: nil,
        kvBits: kvBits,
        kvGroupSize: 64,
        quantizedKVStart: quantizedKVStart,
        temperature: options?.temperature.map { Float($0) } ?? 0,
        topP: 0.9,
        repetitionPenalty: 1.05,
        repetitionContextSize: 64,
        prefillStepSize: prefillStepSize
    )
}

#endif
