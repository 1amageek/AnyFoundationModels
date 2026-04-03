#if MLX_ENABLED
import Foundation

enum MLXRuntimeFamily: String, Sendable {
    case llm
    case vlm
    case unknown
}

struct MLXModelMetadata: Sendable {
    enum ThinkingPreference: Sendable {
        case enabled
        case disabled
        case unspecified
    }

    let modelID: String
    let runtimeFamily: MLXRuntimeFamily

    var thinkingPreference: ThinkingPreference {
        if modelID.lowercased().contains("qwen") {
            return .disabled
        }
        return .unspecified
    }

    static func fallback(modelID: String) -> Self {
        Self(modelID: modelID, runtimeFamily: .unknown)
    }
}

actor MLXModelMetadataRegistry {
    static let shared = MLXModelMetadataRegistry()

    private var metadataByModelID: [String: MLXModelMetadata] = [:]

    func register(_ metadata: MLXModelMetadata, aliases: [String] = []) {
        metadataByModelID[metadata.modelID] = metadata
        for alias in aliases where !alias.isEmpty {
            metadataByModelID[alias] = metadata
        }
    }

    func metadata(for modelID: String) -> MLXModelMetadata? {
        metadataByModelID[modelID]
    }
}
#endif
