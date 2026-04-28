#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM
import Testing
@testable import MetalFoundationModels

/// Real-model regression tests for the reasoning-channel repetition symptom
/// observed in jardis on multiple Metal models. The default Metal runtime
/// applies no repetition penalty, which lets some models enter a degenerate
/// "Plan: ... Plan: ..." style loop until `maxTokens`. Setting
/// ``MetalConfiguration/repetitionPenalty`` is the construction-time hook
/// that addresses this end-to-end without changing
/// `OpenFoundationModels.GenerationOptions`.
///
/// Each test is gated by its own environment variable and requires a model
/// bundle on disk. They exist to:
/// 1. Document the symptom on each model (control: no penalty), and
/// 2. Pin the fix on each model (treatment: penalty applied via
///    ``MetalConfiguration``).
@Suite("Metal Repetition Regression")
struct MetalRepetitionRegressionTests {

    @Test(
        "Gemma 4 E2B-it: repetitionPenalty reduces reasoning repetition",
        .timeLimit(.minutes(5))
    )
    func gemma4ReasoningRepetition() async throws {
        try await runRepetitionComparison(
            envFlag: "ANYFM_GEMMA4_REPRO",
            pathOverrideEnv: "ANYFM_GEMMA4_PATH",
            promptEnv: "ANYFM_GEMMA4_PROMPT",
            maxTokensEnv: "ANYFM_GEMMA4_MAX_TOKENS",
            maxReasoningTokensEnv: "ANYFM_GEMMA4_MAX_REASONING_TOKENS",
            systemEnv: "ANYFM_GEMMA4_SYSTEM",
            label: "Gemma4",
            defaultModelPath:
                ".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf"
        )
    }

    @Test(
        "Qwen 3.5 0.8B 3bit: repetitionPenalty reduces reasoning repetition",
        .timeLimit(.minutes(5))
    )
    func qwen35ReasoningRepetition() async throws {
        try await runRepetitionComparison(
            envFlag: "ANYFM_QWEN_REPRO",
            pathOverrideEnv: "ANYFM_QWEN_PATH",
            promptEnv: "ANYFM_QWEN_PROMPT",
            maxTokensEnv: "ANYFM_QWEN_MAX_TOKENS",
            maxReasoningTokensEnv: "ANYFM_QWEN_MAX_REASONING_TOKENS",
            systemEnv: "ANYFM_QWEN_SYSTEM",
            label: "Qwen3.5",
            defaultModelPath:
                ".cache/huggingface/hub/models--mlx-community--Qwen3.5-0.8B-3bit/snapshots/63755e255fea6273fd31ec1e2a27accb81d95c2d"
        )
    }

    // MARK: - Shared comparison logic

    private func runRepetitionComparison(
        envFlag: String,
        pathOverrideEnv: String,
        promptEnv: String,
        maxTokensEnv: String,
        maxReasoningTokensEnv: String,
        systemEnv: String,
        label: String,
        defaultModelPath: String
    ) async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment[envFlag] == "1" else {
            print("[Skip] Set \(envFlag)=1 to run \(label) repetition regression.")
            return
        }

        guard let container = try await loadContainer(
            environment: environment,
            pathOverrideEnv: pathOverrideEnv,
            label: label,
            defaultModelPath: defaultModelPath
        ) else {
            return
        }

        let prompt = environment[promptEnv] ?? "hi"
        // Visible answer cap. Kept small so a runaway loop still leaves time
        // budget for both control and treatment generations to finish inside
        // the suite's `.timeLimit`.
        let maxTokens = Int(environment[maxTokensEnv] ?? "64") ?? 64
        // Hard cap on reasoning tokens. Without this, swift-lm uses its
        // historical generous multiplier (`max(maxTokens * 16, maxTokens + 256)`),
        // which lets a degenerate reasoning loop run for thousands of tokens
        // — long enough to blow past any reasonable test time limit on slow
        // 3-bit Metal models. Bounding it explicitly is what makes this test
        // both fast *and* able to detect a runaway loop.
        let maxReasoningTokens = Int(environment[maxReasoningTokensEnv] ?? "128") ?? 128
        let systemInstructions = environment[systemEnv] ?? defaultJardisLikeSystem

        // Control: only bound the runaway, no penalty applied. This is the
        // configuration that should reproduce the loop.
        let controlConfig = MetalConfiguration(maxReasoningTokens: maxReasoningTokens)
        let control = try await generate(
            container: container,
            configuration: controlConfig,
            systemInstructions: systemInstructions,
            prompt: prompt,
            maxTokens: maxTokens
        )

        // Treatment: same bound, plus a moderate repetition penalty over the
        // recent-token window. Differs from control only in the penalty knobs,
        // so the comparison isolates the fix.
        let treatmentConfig = MetalConfiguration(
            repetitionPenalty: 1.1,
            repetitionContextSize: 64,
            maxReasoningTokens: maxReasoningTokens
        )
        let treatment = try await generate(
            container: container,
            configuration: treatmentConfig,
            systemInstructions: systemInstructions,
            prompt: prompt,
            maxTokens: maxTokens
        )

        print("[\(label)Repro] control answer (\(control.answer.count) chars):")
        print(control.answer)
        print("[\(label)Repro] control reasoning (\(control.reasoning.count) chars):")
        print(control.reasoning)
        print("[\(label)Repro] ---")
        print("[\(label)Repro] treatment answer (\(treatment.answer.count) chars):")
        print(treatment.answer)
        print("[\(label)Repro] treatment reasoning (\(treatment.reasoning.count) chars):")
        print(treatment.reasoning)

        let controlReasoningScore = repetitionScore(control.reasoning)
        let treatmentReasoningScore = repetitionScore(treatment.reasoning)
        let controlAnswerScore = repetitionScore(control.answer)
        let treatmentAnswerScore = repetitionScore(treatment.answer)
        print(String(
            format: "[\(label)Repro] repetition score answer:   control=%.3f treatment=%.3f",
            controlAnswerScore,
            treatmentAnswerScore
        ))
        print(String(
            format: "[\(label)Repro] repetition score reasoning: control=%.3f treatment=%.3f",
            controlReasoningScore,
            treatmentReasoningScore
        ))

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

    private func loadContainer(
        environment: [String: String],
        pathOverrideEnv: String,
        label: String,
        defaultModelPath: String
    ) async throws -> SwiftLM.LanguageModelContainer? {
        let loader = MetalModelLoader()
        if let modelPath = environment[pathOverrideEnv], !modelPath.isEmpty {
            let url = URL(fileURLWithPath: modelPath)
            print("[\(label)Repro] loading local model: \(url.path)")
            return try await loader.load(directory: url)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidate = home.appendingPathComponent(defaultModelPath)
        if FileManager.default.fileExists(atPath: candidate.path) {
            print("[\(label)Repro] loading local model: \(candidate.path)")
            return try await loader.load(directory: candidate)
        }
        print("[Skip] Set \(pathOverrideEnv) to a local \(label) bundle to run.")
        return nil
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
