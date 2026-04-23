#if RESPONSE_ENABLED
import Testing
import Foundation
@testable import ResponseFoundationModels
import OpenFoundationModels
import OpenFoundationModelsExtra

@Suite("ResponseRequestBuilder Tests")
struct TranscriptConverterTests {

    private func makeBuilder() -> ResponseRequestBuilder {
        ResponseRequestBuilder(modelName: "gpt-4.1")
    }

    // MARK: - buildInputItems: Basic Entry Conversion

    @Test("Instructions entry converts to system instructions")
    func buildInputItems_instructions() throws {
        let transcript = Transcript(entries: [
            .instructions(Transcript.Instructions(
                segments: [.text(Transcript.TextSegment(content: "You are helpful"))],
                toolDefinitions: []
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        // System message is extracted to `instructions`, not `input`
        #expect(result.request.instructions == "You are helpful")
        #expect(result.request.input.isEmpty)
    }

    @Test("Instructions drop session-generated tool appendix and keep native tools")
    func buildInputItems_stripsGeneratedToolInstructions() throws {
        let toolDef = Transcript.ToolDefinition(
            name: "search",
            description: "Search the web",
            parameters: GenerationSchema(type: String.self, description: "query", properties: [])
        )
        let transcript = Transcript(entries: [
            .instructions(Transcript.Instructions(
                segments: [
                    .text(Transcript.TextSegment(content: "You are helpful")),
                    .text(Transcript.TextSegment(content: generatedToolInstructions())),
                ],
                toolDefinitions: [toolDef]
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)

        #expect(result.request.instructions == "You are helpful")
        let tools = try #require(result.request.tools)
        #expect(tools.count == 1)
        #expect(tools[0].name == "search")
    }

    @Test("Prompt entry converts to user message")
    func buildInputItems_prompt() throws {
        let transcript = Transcript(entries: [
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "Hello"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.input.count == 1)
        guard case .message(let msg) = result.request.input[0] else {
            Issue.record("Expected message item")
            return
        }
        #expect(msg.role == "user")
        #expect(msg.content == .text("Hello"))
    }

    @Test("Response entry converts to assistant message")
    func buildInputItems_response() throws {
        let transcript = Transcript(entries: [
            .response(Transcript.Response(
                assetIDs: [],
                segments: [.text(Transcript.TextSegment(content: "Hi there"))]
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.input.count == 1)
        guard case .message(let msg) = result.request.input[0] else {
            Issue.record("Expected message item")
            return
        }
        #expect(msg.role == "assistant")
        #expect(msg.content == .text("Hi there"))
    }

    @Test("Empty transcript produces empty items")
    func buildInputItems_empty() {
        let transcript = Transcript(entries: [])
        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.input.isEmpty)
        #expect(result.request.instructions == nil)
    }

    @Test("Full conversation produces correct sequence")
    func buildInputItems_fullConversation() throws {
        let transcript = Transcript(entries: [
            .instructions(Transcript.Instructions(
                segments: [.text(Transcript.TextSegment(content: "Be helpful"))],
                toolDefinitions: []
            )),
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "Hi"))],
                options: GenerationOptions(),
                responseFormat: nil
            )),
            .response(Transcript.Response(
                assetIDs: [],
                segments: [.text(Transcript.TextSegment(content: "Hello!"))]
            )),
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "How are you?"))],
                options: GenerationOptions(),
                responseFormat: nil
            )),
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        // System message is extracted to `instructions`
        #expect(result.request.instructions == "Be helpful")
        // Remaining: user, assistant, user
        #expect(result.request.input.count == 3)

        guard case .message(let m0) = result.request.input[0] else { Issue.record("Expected message at 0"); return }
        #expect(m0.role == "user")
        #expect(m0.content == .text("Hi"))

        guard case .message(let m1) = result.request.input[1] else { Issue.record("Expected message at 1"); return }
        #expect(m1.role == "assistant")
        #expect(m1.content == .text("Hello!"))

        guard case .message(let m2) = result.request.input[2] else { Issue.record("Expected message at 2"); return }
        #expect(m2.role == "user")
        #expect(m2.content == .text("How are you?"))
    }

    @Test("Multiple text segments are joined with space")
    func buildInputItems_multipleTextSegments() throws {
        let transcript = Transcript(entries: [
            .prompt(Transcript.Prompt(
                segments: [
                    .text(Transcript.TextSegment(content: "Hello")),
                    .text(Transcript.TextSegment(content: "World")),
                ],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.input.count == 1)
        guard case .message(let msg) = result.request.input[0] else { Issue.record("Expected message"); return }
        #expect(msg.content == .text("Hello World"))
    }

    // MARK: - buildInputItems: Tool Call Conversion

    @Test("ToolCalls entry converts to function call items")
    func buildInputItems_toolCalls() throws {
        let toolCall = Transcript.ToolCall(
            id: "tc-001",
            toolName: "search",
            arguments: GeneratedContent(properties: ["query": "swift"])
        )
        let transcript = Transcript(entries: [
            .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, [toolCall]))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.input.count == 1)
        guard case .functionCall(let fc) = result.request.input[0] else {
            Issue.record("Expected functionCall item")
            return
        }
        #expect(fc.name == "search")
        #expect(fc.callId == "tc-001")
    }

    @Test("ToolOutput uses pending tool call ID (FIFO matching)")
    func buildInputItems_toolOutputMatchesPendingCallId() throws {
        let toolCall = Transcript.ToolCall(
            id: "tc-original",
            toolName: "search",
            arguments: GeneratedContent(properties: ["q": "test"])
        )
        let toolOutput = Transcript.ToolOutput(
            id: "random-uuid-from-session",
            toolName: "search",
            segments: [.text(Transcript.TextSegment(content: "result"))]
        )

        let transcript = Transcript(entries: [
            .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, [toolCall])),
            .toolOutput(toolOutput),
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        // 1 function call + 1 function call output = 2
        #expect(result.request.input.count == 2)

        guard case .functionCallOutput(let output) = result.request.input[1] else {
            Issue.record("Expected functionCallOutput at index 1")
            return
        }
        // The output should use the original tool call ID, not the random UUID
        #expect(output.callId == "tc-original")
        #expect(output.output == "result")
    }

    @Test("Multiple tool outputs match pending call IDs in FIFO order")
    func buildInputItems_multipleToolOutputsFIFO() throws {
        let call1 = Transcript.ToolCall(
            id: "call-A",
            toolName: "tool1",
            arguments: GeneratedContent(properties: ["x": "1"])
        )
        let call2 = Transcript.ToolCall(
            id: "call-B",
            toolName: "tool2",
            arguments: GeneratedContent(properties: ["x": "2"])
        )
        let output1 = Transcript.ToolOutput(
            id: "random-1",
            toolName: "tool1",
            segments: [.text(Transcript.TextSegment(content: "result1"))]
        )
        let output2 = Transcript.ToolOutput(
            id: "random-2",
            toolName: "tool2",
            segments: [.text(Transcript.TextSegment(content: "result2"))]
        )

        let transcript = Transcript(entries: [
            .toolCalls(Transcript.ToolCalls(id: UUID().uuidString, [call1, call2])),
            .toolOutput(output1),
            .toolOutput(output2),
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        // 2 function calls + 2 function call outputs = 4
        #expect(result.request.input.count == 4)

        guard case .functionCallOutput(let out1) = result.request.input[2] else {
            Issue.record("Expected functionCallOutput at index 2")
            return
        }
        #expect(out1.callId == "call-A")

        guard case .functionCallOutput(let out2) = result.request.input[3] else {
            Issue.record("Expected functionCallOutput at index 3")
            return
        }
        #expect(out2.callId == "call-B")
    }

    @Test("Orphan ToolOutput without preceding ToolCalls is ignored")
    func buildInputItems_orphanToolOutputIsIgnored() throws {
        let toolOutput = Transcript.ToolOutput(
            id: "own-id",
            toolName: "search",
            segments: [.text(Transcript.TextSegment(content: "result"))]
        )

        let transcript = Transcript(entries: [
            .toolOutput(toolOutput),
        ])

        // ResolvedTranscript ignores toolOutput entries with no preceding toolCalls
        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.input.isEmpty)
    }

    // MARK: - Tool Definitions

    @Test("Extract tool definitions from instructions")
    func extractToolDefinitions_fromInstructions() throws {
        let schema = GenerationSchema(
            type: String.self,
            description: "A query string",
            properties: []
        )
        let toolDef = Transcript.ToolDefinition(
            name: "search",
            description: "Search the web",
            parameters: schema
        )
        let transcript = Transcript(entries: [
            .instructions(Transcript.Instructions(
                segments: [.text(Transcript.TextSegment(content: "instructions"))],
                toolDefinitions: [toolDef]
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        let defs = try #require(result.request.tools)
        #expect(defs.count == 1)
        #expect(defs[0].name == "search")
        #expect(defs[0].description == "Search the web")
    }

    @Test("Extract tool definitions returns nil for empty transcript")
    func extractToolDefinitions_empty() {
        let transcript = Transcript(entries: [])
        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.tools == nil)
    }

    @Test("Extract tool definitions returns nil when no tools defined")
    func extractToolDefinitions_noTools() {
        let transcript = Transcript(entries: [
            .instructions(Transcript.Instructions(
                segments: [.text(Transcript.TextSegment(content: "instructions"))],
                toolDefinitions: []
            ))
        ])
        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.tools == nil)
    }

    // MARK: - Generation Options

    @Test("Apply options from last prompt")
    func extractOptions_fromLastPrompt() throws {
        let options1 = GenerationOptions(temperature: 0.5, maximumResponseTokens: 100)
        let options2 = GenerationOptions(temperature: 0.9, maximumResponseTokens: 500)

        let transcript = Transcript(entries: [
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "First"))],
                options: options1,
                responseFormat: nil
            )),
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "Second"))],
                options: options2,
                responseFormat: nil
            )),
        ])

        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.temperature == 0.9)
        #expect(result.request.maxOutputTokens == 500)
    }

    @Test("Explicit options override transcript options")
    func extractOptions_explicitOverridesTranscript() throws {
        let transcriptOptions = GenerationOptions(temperature: 0.9, maximumResponseTokens: 500)
        let explicitOptions = GenerationOptions(temperature: 0.1, maximumResponseTokens: 100)

        let transcript = Transcript(entries: [
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "text"))],
                options: transcriptOptions,
                responseFormat: nil
            ))
        ])

        let result = makeBuilder().build(transcript: transcript, options: explicitOptions, stream: false)
        #expect(result.request.temperature == 0.1)
        #expect(result.request.maxOutputTokens == 100)
    }

    // MARK: - Response Format

    @Test("Extract response format returns nil when no format set")
    func extractResponseFormat_noFormat() {
        let transcript = Transcript(entries: [
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "text"))],
                options: GenerationOptions(),
                responseFormat: nil
            ))
        ])
        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        #expect(result.request.text == nil)
    }

    @Test("Extract response format from prompt with type")
    func extractResponseFormat_withType() throws {
        let responseFormat = Transcript.ResponseFormat(type: String.self)
        let transcript = Transcript(entries: [
            .prompt(Transcript.Prompt(
                segments: [.text(Transcript.TextSegment(content: "text"))],
                options: GenerationOptions(),
                responseFormat: responseFormat
            ))
        ])
        let result = makeBuilder().build(transcript: transcript, options: nil, stream: false)
        // Should have a TextFormat (either jsonSchema or jsonObject)
        #expect(result.request.text != nil)
    }
}

private func generatedToolInstructions() -> String {
    """
    # Tools

    In this environment you have access to a set of tools you can use to answer the user's question.
    """
}

#endif
