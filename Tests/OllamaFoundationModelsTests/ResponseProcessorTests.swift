#if OLLAMA_ENABLED
import Foundation
import Testing
@testable import OllamaFoundationModels

@Suite("ResponseProcessor Tests")
struct ResponseProcessorTests {
    private let processor = ResponseProcessor()

    @Test("Explicit thinking field becomes reasoning channel")
    func explicitThinkingSeparated() {
        let message = Message(
            role: .assistant,
            content: "Hello",
            thinking: "Let me think about this."
        )

        guard case .response(let response) = processor.process(message) else {
            Issue.record("Expected normalized response")
            return
        }

        #expect(response.content == "Hello")
        #expect(response.reasoning == "Let me think about this.")
    }

    @Test("Inline think tags are split from visible content")
    func inlineThinkTagsSeparated() {
        let message = Message(
            role: .assistant,
            content: "<think>secret reasoning</think>\nVisible answer"
        )

        guard case .response(let response) = processor.process(message) else {
            Issue.record("Expected normalized response")
            return
        }

        #expect(response.content == "Visible answer")
        #expect(response.reasoning == "secret reasoning")
    }

    @Test("Orphaned closing think tag treats prefix as reasoning")
    func orphanedThinkCloseSeparated() {
        let message = Message(
            role: .assistant,
            content: "draft reasoning</think>\n{\"value\":1}"
        )

        guard case .response(let response) = processor.process(message) else {
            Issue.record("Expected normalized response")
            return
        }

        #expect(response.content == #"{"value":1}"#)
        #expect(response.reasoning == "draft reasoning")
    }

    @Test("Tool calls inside thinking field still take priority")
    func toolCallsInThinkingDetected() {
        let message = Message(
            role: .assistant,
            content: "",
            thinking: #"{"name":"search_repo","arguments":{"query":"swift"}}"#
        )

        guard case .toolCalls(let toolCalls) = processor.process(message) else {
            Issue.record("Expected tool call extraction")
            return
        }

        #expect(toolCalls.count == 1)
        #expect(toolCalls[0].function.name == "search_repo")
    }
}

#endif
