#if MLX_ENABLED
import Foundation
@preconcurrency import MLXLMCommon

enum MLXRuntimeFamily: String, Sendable {
    case llm
    case vlm
    case unknown
}

enum MLXModality: String, Sendable, Hashable {
    case text
    case image
}

struct MLXModelProfile: Sendable {
    enum ThinkingPreference: Sendable {
        case enabled
        case disabled
        case unspecified
    }

    let modelID: String
    let runtimeFamily: MLXRuntimeFamily
    let modalities: Set<MLXModality>
    let thinkingPreference: ThinkingPreference

    var supportsImages: Bool {
        modalities.contains(.image)
    }

    static func fallback(modelID: String) -> Self {
        Self(
            modelID: modelID,
            runtimeFamily: .unknown,
            modalities: [.text],
            thinkingPreference: thinkingPreference(for: modelID)
        )
    }

    static func make(
        modelID: String,
        runtimeFamily: MLXRuntimeFamily,
        modalities: Set<MLXModality>
    ) -> Self {
        Self(
            modelID: modelID,
            runtimeFamily: runtimeFamily,
            modalities: modalities,
            thinkingPreference: thinkingPreference(for: modelID)
        )
    }

    private static func thinkingPreference(for modelID: String) -> ThinkingPreference {
        if modelID.lowercased().contains("qwen") {
            return .disabled
        }
        return .unspecified
    }
}

public struct MLXLoadedModel: Sendable {
    public let container: ModelContainer

    let profile: MLXModelProfile

    init(
        container: ModelContainer,
        profile: MLXModelProfile
    ) {
        self.container = container
        self.profile = profile
    }
}
#endif
