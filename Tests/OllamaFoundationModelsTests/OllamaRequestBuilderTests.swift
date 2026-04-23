#if OLLAMA_ENABLED
import Foundation
import OpenFoundationModels
import Testing
@testable import OllamaFoundationModels

@Suite("OllamaRequestBuilder Tests")
struct OllamaRequestBuilderTests {
    @Test("Session-generated tool appendix is removed from system message")
    func stripsGeneratedToolInstructions() {
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [
                    .text(.init(content: "You are helpful.")),
                    .text(.init(content: generatedToolInstructions())),
                ],
                toolDefinitions: [searchToolDefinition()]
            ))
        ])

        let builder = OllamaRequestBuilder(
            configuration: OllamaConfiguration(),
            modelName: "qwen3",
            thinkingMode: nil
        )
        let result = builder.build(transcript: transcript, options: nil, stream: false)

        #expect(result.request.messages.count == 1)
        #expect(result.request.messages[0].role == .system)
        #expect(result.request.messages[0].content == "You are helpful.")
        let tools = result.request.tools
        #expect(tools?.count == 1)
        #expect(tools?.first?.function.name == "search_repo")
    }
}

private func searchToolDefinition() -> Transcript.ToolDefinition {
    Transcript.ToolDefinition(
        name: "search_repo",
        description: "Search the repository",
        parameters: GenerationSchema(type: String.self, description: "query", properties: [])
    )
}

private func generatedToolInstructions() -> String {
    """
    # Tools

    In this environment you have access to a set of tools you can use to answer the user's question.
    """
}

#endif
