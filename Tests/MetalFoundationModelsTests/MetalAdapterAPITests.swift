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
        let constructModel: @Sendable (LanguageModelContainer, Bool, MetalConfiguration) -> MetalLanguageModel =
            MetalLanguageModel.init(container:showsThinking:configuration:)
        let legacyConstructModel: @Sendable (LanguageModelContainer, Bool, MetalConfiguration) -> MetalLanguageModel =
            MetalLanguageModel.init(languageModelContainer:showsThinking:configuration:)

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

    @Test("Existing call sites without configuration still compile thanks to default value")
    func metalAdapterRetainsBackwardCompatibleCallSites() {
        // These two declarations must keep type-checking even after the
        // configuration parameter was added, because the parameter has a
        // default value. They guard against an accidental breaking change
        // (e.g. removing the default) that would force every existing call
        // site to be updated.
        let constructWithoutConfig: @Sendable (LanguageModelContainer) -> MetalLanguageModel = {
            MetalLanguageModel(container: $0)
        }
        let constructWithThinking: @Sendable (LanguageModelContainer, Bool) -> MetalLanguageModel = {
            MetalLanguageModel(container: $0, showsThinking: $1)
        }
        _ = constructWithoutConfig
        _ = constructWithThinking
    }

    private func requireLanguageModel<T: LanguageModel>(_: T.Type) {}
}
#endif
