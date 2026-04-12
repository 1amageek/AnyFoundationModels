#if METAL_ENABLED
import Foundation

/// Public errors surfaced by the Metal language model adapter.
public enum MetalLanguageModelError: LocalizedError {
    case unsupportedImageInput(modelName: String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedImageInput(let modelName):
            return "Model '\(modelName)' does not support image input."
        }
    }
}

#endif
