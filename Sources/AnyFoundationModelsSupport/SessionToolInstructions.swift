import Foundation
import OpenFoundationModels

public enum SessionToolInstructions {
    public static let header = "# Tools"
    public static let preamble =
        "In this environment you have access to a set of tools you can use to answer the user's question."

    public static func userAuthoredSegments(
        from segments: [Transcript.Segment]
    ) -> [Transcript.Segment] {
        segments.filter { !isGeneratedToolInstructionSegment($0) }
    }

    public static func isGeneratedToolInstructionSegment(
        _ segment: Transcript.Segment
    ) -> Bool {
        guard case .text(let text) = segment else {
            return false
        }

        let trimmed = text.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix(header) else {
            return false
        }

        return trimmed.contains(preamble)
    }
}
