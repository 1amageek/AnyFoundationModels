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
        let loadModel: @Sendable (String, Progress?) async throws -> InferenceSession =
            loader.loadModel(_:progress:)
        let constructModel: @Sendable (InferenceSession, Bool) -> MetalLanguageModel =
            MetalLanguageModel.init(inferenceSession:showsThinking:)

        requireLanguageModel(MetalLanguageModel.self)

        _ = loadModel
        _ = constructModel
    }

    private func requireLanguageModel<T: LanguageModel>(_: T.Type) {}
}
#endif
