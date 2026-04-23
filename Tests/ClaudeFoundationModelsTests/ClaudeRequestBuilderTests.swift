#if CLAUDE_ENABLED
import Foundation
import OpenFoundationModels
import Testing
@testable import ClaudeFoundationModels

@Suite("ClaudeRequestBuilder Tests")
struct ClaudeRequestBuilderTests {
    @Test("Session-generated tool appendix is removed from system prompt")
    func stripsGeneratedToolInstructions() throws {
        let transcript = Transcript(entries: [
            .instructions(.init(
                segments: [
                    .text(.init(content: "You are helpful.")),
                    .text(.init(content: generatedToolInstructions())),
                ],
                toolDefinitions: [searchToolDefinition()]
            ))
        ])

        let builder = ClaudeRequestBuilder(
            modelName: "claude-sonnet-4-5",
            defaultMaxTokens: 1024,
            thinkingBudgetTokens: nil
        )
        let result = try builder.build(transcript: transcript, options: nil, stream: false)

        #expect(result.request.system == "You are helpful.")
        let tools = try #require(result.request.tools)
        #expect(tools.count == 1)
        #expect(tools[0].name == "search_repo")
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
