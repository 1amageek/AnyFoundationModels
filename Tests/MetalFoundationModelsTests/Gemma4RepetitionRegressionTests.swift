#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM
import Testing
@testable import MetalFoundationModels

/// Real-model regression tests for the Gemma 4 repetition-loop symptom
/// observed in jardis. The default Metal runtime applies no repetition
/// penalty, which lets some Gemma 4 generations enter a "Plan: ... Plan: ..."
/// style loop until `maxTokens`. ``MetalConfiguration.repetitionPenalty`` is
/// the construction-time hook that fixes this end-to-end without changing
/// `OpenFoundationModels.GenerationOptions`.
///
/// These tests are gated by `ANYFM_GEMMA4_REPRO=1` and require a Gemma 4
/// model bundle on disk. They exist to:
/// 1. Document the symptom (control: no penalty), and
/// 2. Pin the fix (treatment: penalty applied via MetalConfiguration).
@Suite("Gemma 4 Repetition Regression")
struct Gemma4RepetitionRegressionTests {

    @Test(
        "MetalConfiguration.repetitionPenalty resolves Gemma 4 'Plan:' loop",
        .timeLimit(.minutes(10))
    )
    func repetitionPenaltyResolvesGemma4Loop() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ANYFM_GEMMA4_REPRO"] == "1" else {
            print("[Skip] Set ANYFM_GEMMA4_REPRO=1 to run Gemma 4 repetition regression.")
            return
        }

        guard let container = try await loadGemma4Container(environment: environment) else {
            return
        }

        let prompt = environment["ANYFM_GEMMA4_PROMPT"] ?? "hi"
        let maxTokens = Int(environment["ANYFM_GEMMA4_MAX_TOKENS"] ?? "512") ?? 512
        let systemInstructions = environment["ANYFM_GEMMA4_SYSTEM"] ?? defaultJardisLikeSystem

        // Control: default configuration mirrors the historical runtime.
        let control = try await generate(
            container: container,
            configuration: .default,
            systemInstructions: systemInstructions,
            prompt: prompt,
            maxTokens: maxTokens
        )

        // Treatment: apply a moderate repetition penalty + recent-token window.
        let treatmentConfig = MetalConfiguration(
            repetitionPenalty: 1.1,
            repetitionContextSize: 64
        )
        let treatment = try await generate(
            container: container,
            configuration: treatmentConfig,
            systemInstructions: systemInstructions,
            prompt: prompt,
            maxTokens: maxTokens
        )

        print("[Gemma4Repro] control answer (\(control.answer.count) chars):")
        print(control.answer)
        print("[Gemma4Repro] control reasoning (\(control.reasoning.count) chars):")
        print(control.reasoning)
        print("[Gemma4Repro] ---")
        print("[Gemma4Repro] treatment answer (\(treatment.answer.count) chars):")
        print(treatment.answer)
        print("[Gemma4Repro] treatment reasoning (\(treatment.reasoning.count) chars):")
        print(treatment.reasoning)

        let controlReasoningScore = repetitionScore(control.reasoning)
        let treatmentReasoningScore = repetitionScore(treatment.reasoning)
        let controlAnswerScore = repetitionScore(control.answer)
        let treatmentAnswerScore = repetitionScore(treatment.answer)
        print(String(
            format: "[Gemma4Repro] repetition score answer:   control=%.3f treatment=%.3f",
            controlAnswerScore,
            treatmentAnswerScore
        ))
        print(String(
            format: "[Gemma4Repro] repetition score reasoning: control=%.3f treatment=%.3f",
            controlReasoningScore,
            treatmentReasoningScore
        ))

        // Treatment must not be obviously degenerate.
        let treatmentCombined = treatment.answer + treatment.reasoning
        #expect(!treatmentCombined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!treatmentCombined.contains("\u{FFFD}"))

        // Treatment's repetition score should not be worse than control on
        // either channel. Lower = better (lower share of repeated 4-grams).
        #expect(treatmentAnswerScore <= controlAnswerScore + 0.05)
        #expect(treatmentReasoningScore <= controlReasoningScore + 0.05)
    }

    // MARK: - Helpers

    private struct GeneratedChannels {
        let answer: String
        let reasoning: String
    }

    private func generate(
        container: SwiftLM.LanguageModelContainer,
        configuration: MetalConfiguration,
        systemInstructions: String,
        prompt: String,
        maxTokens: Int
    ) async throws -> GeneratedChannels {
        let model = MetalLanguageModel(
            container: container,
            showsThinking: true,
            configuration: configuration
        )
        var entries: [Transcript.Entry] = []
        if !systemInstructions.isEmpty {
            entries.append(
                .instructions(.init(
                    segments: [.text(.init(content: systemInstructions))],
                    toolDefinitions: []
                ))
            )
        }
        entries.append(
            .prompt(.init(segments: [.text(.init(content: prompt))]))
        )
        let transcript = Transcript(entries: entries)
        let entry = try await model.generate(
            transcript: transcript,
            options: GenerationOptions(maximumResponseTokens: maxTokens)
        )
        return GeneratedChannels(
            answer: answerText(from: entry),
            reasoning: reasoningText(from: entry)
        )
    }

    private let defaultJardisLikeSystem = """
        You are a helpful AI assistant. Before answering, think step by step
        about how to respond. Use a planning style: outline what you will say,
        then write the final answer. Be concise.
        """

    /// Share of 4-grams that are duplicates. 0.0 = every 4-gram unique;
    /// 1.0 = every 4-gram is a repeat of an earlier one.
    private func repetitionScore(_ text: String) -> Double {
        let chars = Array(text)
        guard chars.count >= 4 else { return 0 }
        var seen: Set<String> = []
        var duplicates = 0
        var total = 0
        for i in 0...(chars.count - 4) {
            let ngram = String(chars[i..<(i + 4)])
            total += 1
            if !seen.insert(ngram).inserted {
                duplicates += 1
            }
        }
        guard total > 0 else { return 0 }
        return Double(duplicates) / Double(total)
    }

    private func loadGemma4Container(
        environment: [String: String]
    ) async throws -> SwiftLM.LanguageModelContainer? {
        let loader = MetalModelLoader()
        if let modelPath = environment["ANYFM_GEMMA4_PATH"], !modelPath.isEmpty {
            let url = URL(fileURLWithPath: modelPath)
            print("[Gemma4Repro] loading local model: \(url.path)")
            return try await loader.load(directory: url)
        }
        for url in defaultGemma4Candidates() where FileManager.default.fileExists(atPath: url.path) {
            print("[Gemma4Repro] loading local model: \(url.path)")
            return try await loader.load(directory: url)
        }
        print("[Skip] Set ANYFM_GEMMA4_PATH to a local Gemma 4 bundle to run.")
        return nil
    }

    private func defaultGemma4Candidates() -> [URL] {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return [
            home.appendingPathComponent(
                ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
            ),
        ]
    }

    private func answerText(from entry: Transcript.Entry) -> String {
        guard case .response(let response) = entry else { return "" }
        return response.segments.compactMap { segment -> String? in
            if case .text(let text) = segment { return text.content }
            return nil
        }.joined()
    }

    private func reasoningText(from entry: Transcript.Entry) -> String {
        guard case .response(let response) = entry else { return "" }
        return response.segments.compactMap { segment -> String? in
            if case .reasoning(let text) = segment { return text.content }
            return nil
        }.joined()
    }
}
#endif
