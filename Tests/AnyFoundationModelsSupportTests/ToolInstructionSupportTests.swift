import AnyFoundationModelsSupport
import OpenFoundationModels
import Testing

@Suite("Tool instruction support")
struct ToolInstructionSupportTests {
    @Test("Session-generated tool appendix is filtered from instructions")
    func filtersGeneratedToolInstructions() {
        let segments: [Transcript.Segment] = [
            .text(.init(content: "You are helpful.")),
            .text(.init(content: generatedToolInstructions())),
        ]

        let filtered = SessionToolInstructions.userAuthoredSegments(from: segments)

        #expect(filtered.count == 1)
        guard case .text(let text) = filtered[0] else {
            Issue.record("Expected text segment")
            return
        }
        #expect(text.content == "You are helpful.")
    }
}

private func generatedToolInstructions() -> String {
    """
    # Tools

    In this environment you have access to a set of tools you can use to answer the user's question.
    """
}
