#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM
import Testing
@testable import MetalFoundationModels

@Suite("Metal Real Model Smoke Tests")
struct MetalRealModelSmokeTests {
    @Test("Real swift-lm model output through MetalLanguageModel", .timeLimit(.minutes(10)))
    func realModelOutputThroughAdapter() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ANYFM_REAL_MODEL_SMOKE"] == "1" else {
            print("[Skip] Set ANYFM_REAL_MODEL_SMOKE=1 to run real Metal model smoke.")
            return
        }

        let container = try await loadContainer(environment: environment)
        let model = MetalLanguageModel(container: container)
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "What is the capital of Japan? Answer with exactly one word."))
            ]))
        ])

        let response = try await model.generate(
            transcript: transcript,
            options: GenerationOptions(temperature: 0, maximumResponseTokens: 8)
        )
        let output = responseText(from: response)

        print("[MetalRealModelSmoke] output: \(output)")
        #expect(!output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(output.localizedCaseInsensitiveContains("Tokyo"))
    }

    @Test("Real swift-lm thinking output through MetalLanguageModel", .timeLimit(.minutes(10)))
    func realModelThinkingOutputThroughAdapter() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ANYFM_REAL_MODEL_THINK_SMOKE"] == "1" else {
            print("[Skip] Set ANYFM_REAL_MODEL_THINK_SMOKE=1 to run real Metal thinking smoke.")
            return
        }

        let container = try await loadContainer(environment: environment)
        let model = MetalLanguageModel(container: container, showsThinking: true)
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "Briefly reason, then answer exactly one word: what is the capital of Japan?"))
            ]))
        ])

        let response = try await model.generate(
            transcript: transcript,
            options: GenerationOptions(temperature: 0, maximumResponseTokens: 128)
        )
        let output = responseText(from: response)
        let reasoning = reasoningText(from: response)

        print("[MetalRealModelSmoke] thinking output: \(output)")
        print("[MetalRealModelSmoke] thinking reasoning: \(reasoning)")
        #expect(!output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(containsTokyo(output))
        #expect(!reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    private func loadContainer(
        environment: [String: String]
    ) async throws -> SwiftLM.LanguageModelContainer {
        let loader = MetalModelLoader()
        if let modelPath = environment["ANYFM_REAL_MODEL_PATH"], !modelPath.isEmpty {
            let url = URL(fileURLWithPath: modelPath)
            print("[MetalRealModelSmoke] loading local model: \(url.path)")
            return try await loader.load(directory: url)
        }

        let modelID = environment["ANYFM_REAL_MODEL_ID"] ?? "Qwen/Qwen3.5-0.8B"
        print("[MetalRealModelSmoke] loading repo: \(modelID)")
        return try await loader.load(repo: modelID)
    }

    private func responseText(from entry: Transcript.Entry) -> String {
        guard case .response(let response) = entry else {
            return ""
        }
        return response.segments.compactMap { segment in
            if case .text(let text) = segment {
                return text.content
            }
            return nil
        }.joined()
    }

    private func reasoningText(from entry: Transcript.Entry) -> String {
        guard case .response(let response) = entry else {
            return ""
        }
        return response.segments.compactMap { segment in
            if case .reasoning(let text) = segment {
                return text.content
            }
            return nil
        }.joined()
    }

    private func containsTokyo(_ text: String) -> Bool {
        text.localizedCaseInsensitiveContains("Tokyo")
            || text.contains("東京")
            || text.contains("东京")
    }
}
#endif
