#if MLX_ENABLED
import Foundation
import Testing
@testable import MLXFoundationModels

@Suite("ModelLoader Tests")
struct ModelLoaderTests {
    @Test("Local file URL artifacts are recognized as cached after download")
    func localFileURLCaching() async throws {
        let directory = try makeModelDirectory()
        defer {
            do {
                try FileManager.default.removeItem(at: directory)
            } catch {
                Issue.record("Failed to remove temporary model directory: \(error)")
            }
        }

        let loader = ModelLoader()
        let fileURLString = directory.absoluteString

        let resolved = try await loader.downloadModel(fileURLString)
        #expect(resolved.standardizedFileURL == directory.standardizedFileURL)

        let isCached = try await loader.isCached(fileURLString)
        #expect(isCached)

        let cachedModels = await loader.cachedModels()
        #expect(cachedModels.contains(directory.standardizedFileURL.path))
    }

    @Test("Clearing a model cache removes resolved download artifacts")
    func clearCacheForModelRemovesResolvedArtifact() async throws {
        let directory = try makeModelDirectory()
        defer {
            do {
                try FileManager.default.removeItem(at: directory)
            } catch {
                Issue.record("Failed to remove temporary model directory: \(error)")
            }
        }

        let loader = ModelLoader()
        let modelID = directory.path

        _ = try await loader.downloadModel(modelID)
        #expect(try await loader.isCached(modelID))

        try await loader.clearCache(for: modelID)

        let isCached = try await loader.isCached(modelID)
        #expect(isCached)

        let cachedModels = await loader.cachedModels()
        #expect(!cachedModels.contains(directory.standardizedFileURL.path))
    }

    private func makeModelDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let config = """
        {
          "architectures": ["TestModel"],
          "model_type": "test"
        }
        """
        try config.write(
            to: directory.appendingPathComponent("config.json"),
            atomically: true,
            encoding: .utf8
        )

        try "{}".write(
            to: directory.appendingPathComponent("tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )

        FileManager.default.createFile(
            atPath: directory.appendingPathComponent("weights.safetensors").path,
            contents: Data()
        )

        return directory.standardizedFileURL
    }
}
#endif
