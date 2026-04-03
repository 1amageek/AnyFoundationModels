#if METAL_ENABLED
import Foundation
import SwiftLM

/// Errors that occur during Metal model loading.
public enum MetalModelLoadingError: LocalizedError {
    case loadFailed(modelID: String, underlyingError: Error)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let modelID, let underlyingError):
            return "Failed to load model '\(modelID)': \(underlyingError.localizedDescription)"
        }
    }
}

/// Loads HuggingFace models for Metal inference.
///
/// Downloads safetensors, converts to STAF, compiles to Metal dispatch plan.
/// Caches loaded models for reuse.
public final class MetalModelLoader: Sendable {

    private let cache = ModelCache()

    public init() {}

    /// Load a model from a HuggingFace repository.
    ///
    /// - Parameters:
    ///   - modelID: HuggingFace repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
    ///   - progress: Optional progress tracking.
    /// - Returns: A fully initialized ModelContainer.
    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> ModelContainer {
        let key = ModelCacheKey(modelID: modelID)
        if let cached = await cache.get(key) {
            return cached
        }

        let loader = ModelBundleLoader()
        let container: ModelContainer
        do {
            container = try await loader.load(repo: modelID, progress: progress)
        } catch {
            throw MetalModelLoadingError.loadFailed(modelID: modelID, underlyingError: error)
        }

        await cache.set(container, for: key)
        return container
    }

    /// Check if a model is cached.
    public func isCached(_ modelID: String) async -> Bool {
        let key = ModelCacheKey(modelID: modelID)
        return await cache.get(key) != nil
    }

    /// Clear all cached models.
    public func clearCache() async {
        await cache.clear()
    }
}

// MARK: - Thread-safe Cache

private actor ModelCache {
    private var storage: [ModelCacheKey: ModelContainer] = [:]

    func get(_ key: ModelCacheKey) -> ModelContainer? {
        storage[key]
    }

    func set(_ value: ModelContainer, for key: ModelCacheKey) {
        storage[key] = value
    }

    func clear() {
        storage.removeAll()
    }
}

private struct ModelCacheKey: Hashable {
    let modelID: String
}

#endif
