#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM
import Testing
@testable import MetalFoundationModels

@Suite("Metal Foundation Models API")
struct MetalAdapterAPITests {

    @Test("Metal adapter compiles against current swift-lm surface")
    func metalAdapterTracksCurrentSwiftLMPublicAPI() {
        let loader = MetalModelLoader()
        let loadRepo: @Sendable (String, Progress?) async throws -> LanguageModelContainer =
            loader.load(repo:progress:)
        let loadDirectory: @Sendable (URL) async throws -> LanguageModelContainer =
            loader.load(directory:)
        let loadModel: @Sendable (String, Progress?) async throws -> LanguageModelContainer =
            loader.loadModel(_:progress:)
        let loadLocalModel: @Sendable (URL) async throws -> LanguageModelContainer =
            loader.loadLocalModel(from:)
        let isRepoCached: @Sendable (String) async -> Bool =
            loader.isCached(_:)
        let isDirectoryCached: @Sendable (URL) async -> Bool =
            loader.isCached(directory:)
        let constructModel: @Sendable (LanguageModelContainer, Bool) -> MetalLanguageModel =
            MetalLanguageModel.init(container:showsThinking:)
        let legacyConstructModel: @Sendable (LanguageModelContainer, Bool) -> MetalLanguageModel =
            MetalLanguageModel.init(languageModelContainer:showsThinking:)

        requireLanguageModel(MetalLanguageModel.self)

        _ = loadRepo
        _ = loadDirectory
        _ = loadModel
        _ = loadLocalModel
        _ = isRepoCached
        _ = isDirectoryCached
        _ = constructModel
        _ = legacyConstructModel
    }

    private func requireLanguageModel<T: LanguageModel>(_: T.Type) {}
}
#endif
