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

        guard let container = try await loadContainer(environment: environment) else {
            return
        }
        let model = MetalLanguageModel(container: container)
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "What is the capital of Japan? Answer with exactly one word."))
            ]))
        ])

        let response = try await model.generate(
            transcript: transcript,
            options: GenerationOptions(maximumResponseTokens: 8)
        )
        let output = responseText(from: response)

        print("[MetalRealModelSmoke] output: \(output)")
        #expect(!output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(output.localizedCaseInsensitiveContains("Tokyo"))
        #expect(!output.contains("�"))
    }

    @Test("Real swift-lm thinking output through MetalLanguageModel", .timeLimit(.minutes(10)))
    func realModelThinkingOutputThroughAdapter() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ANYFM_REAL_MODEL_THINK_SMOKE"] == "1" else {
            print("[Skip] Set ANYFM_REAL_MODEL_THINK_SMOKE=1 to run real Metal thinking smoke.")
            return
        }

        guard let container = try await loadContainer(environment: environment) else {
            return
        }
        let model = MetalLanguageModel(container: container, showsThinking: true)
        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "hi"))
            ]))
        ])

        let response = try await model.generate(
            transcript: transcript,
            options: GenerationOptions(maximumResponseTokens: 256)
        )
        let output = responseText(from: response)
        let reasoning = reasoningText(from: response)

        print("[MetalRealModelSmoke] thinking output: \(output)")
        print("[MetalRealModelSmoke] thinking reasoning: \(reasoning)")
        #expect(!output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(output.localizedCaseInsensitiveContains("hello"))
        #expect(!output.contains("�"))
        #expect(!reasoning.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(!reasoning.contains("�"))
    }

    @Test("Dump rendered prompt for LFM thinking 'hi'", .timeLimit(.minutes(10)))
    func dumpRenderedThinkingPrompt() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ANYFM_DUMP_PROMPT"] == "1" else {
            print("[Skip] Set ANYFM_DUMP_PROMPT=1 to dump rendered prompts.")
            return
        }

        guard let container = try await loadContainer(environment: environment) else {
            return
        }

        let transcript = Transcript(entries: [
            .prompt(.init(segments: [
                .text(.init(content: "hi"))
            ]))
        ])

        let converter = MetalRequestConverter()
        let converted = try await converter.convert(
            transcript: transcript,
            configuration: container.configuration,
            showsThinking: true
        )

        switch converted.input.prompt {
        case .text(let text):
            print("[PromptDump] ModelInput.prompt = .text(\(text))")
        case .chat(let chat):
            print("[PromptDump] ModelInput.prompt = .chat count=\(chat.count)")
            for (index, message) in chat.enumerated() {
                print("[PromptDump]  [\(index)] \(message)")
            }
        }
        print("[PromptDump] hasTools=\(converted.hasTools) hasSchema=\(converted.hasSchema)")
        print("[PromptDump] promptOptions.isThinkingEnabled=\(converted.input.promptOptions.isThinkingEnabled)")

        let context = try LanguageModelContext(container)
        let prepared = try await context.prepare(converted.input)

        print("[PromptDump] === renderedText (begin) ===")
        print(prepared.renderedText)
        print("[PromptDump] === renderedText (end) ===")
        print("[PromptDump] tokenIDs.count=\(prepared.tokenIDs.count)")

        #expect(!prepared.renderedText.isEmpty)
    }

    private func loadContainer(
        environment: [String: String]
    ) async throws -> SwiftLM.LanguageModelContainer? {
        let loader = MetalModelLoader()
        if let modelPath = environment["ANYFM_REAL_MODEL_PATH"], !modelPath.isEmpty {
            let url = URL(fileURLWithPath: modelPath)
            print("[MetalRealModelSmoke] loading local model: \(url.path)")
            return try await loader.load(directory: url)
        }

        for url in localLFMThinkingCandidates() where FileManager.default.fileExists(atPath: url.path) {
            print("[MetalRealModelSmoke] loading local model: \(url.path)")
            return try await loader.load(directory: url)
        }

        if let modelID = environment["ANYFM_REAL_MODEL_ID"], !modelID.isEmpty {
            print("[MetalRealModelSmoke] loading repo: \(modelID)")
            return try await loader.load(repo: modelID)
        }

        print("[Skip] Set ANYFM_REAL_MODEL_PATH to a local LFM thinking bundle to run real Metal model smoke.")
        return nil
    }

    private func localLFMThinkingCandidates() -> [URL] {
        let currentDirectory = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        return [
            URL(
                fileURLWithPath: "../swift-lm/TestData/LFM2.5-1.2B-Thinking",
                relativeTo: currentDirectory
            ).standardizedFileURL,
            URL(fileURLWithPath: "/Users/1amageek/Desktop/Robot/swift-lm/TestData/LFM2.5-1.2B-Thinking"),
        ]
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
