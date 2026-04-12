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
    private let loadRepository: @Sendable (String, Progress?) async throws -> LanguageModelContainer
    private let loadDirectoryModel: @Sendable (URL) async throws -> LanguageModelContainer

    public init() {
        self.loadRepository = { repo, progress in
            let loader = ModelBundleLoader()
            return try await loader.load(repo: repo, progress: progress)
        }
        self.loadDirectoryModel = { directory in
            let loader = ModelBundleLoader()
            return try await loader.load(directory: directory)
        }
    }

    init(
        loadRepository: @escaping @Sendable (String, Progress?) async throws -> LanguageModelContainer,
        loadDirectoryModel: @escaping @Sendable (URL) async throws -> LanguageModelContainer
    ) {
        self.loadRepository = loadRepository
        self.loadDirectoryModel = loadDirectoryModel
    }

    /// Load a model from a HuggingFace repository.
    ///
    /// Mirrors `SwiftLM.ModelBundleLoader.load(repo:)`.
    public func load(
        repo: String,
        progress: Progress? = nil
    ) async throws -> LanguageModelContainer {
        let key = ModelCacheKey(identifier: "repo:\(repo)")
        if let cached = await cache.get(key) {
            return cached
        }

        let container: LanguageModelContainer
        do {
            container = try await loadRepository(repo, progress)
        } catch {
            throw MetalModelLoadingError.loadFailed(modelID: repo, underlyingError: error)
        }

        await cache.set(container, for: key)
        return container
    }

    /// Load a model from a local snapshot directory.
    ///
    /// Mirrors `SwiftLM.ModelBundleLoader.load(directory:)`.
    public func load(
        directory: URL
    ) async throws -> LanguageModelContainer {
        let standardizedURL = directory.standardizedFileURL
        let key = ModelCacheKey(identifier: "dir:\(standardizedURL.path)")
        if let cached = await cache.get(key) {
            return cached
        }

        let container: LanguageModelContainer
        do {
            container = try await loadDirectoryModel(standardizedURL)
        } catch {
            throw MetalModelLoadingError.loadFailed(
                modelID: standardizedURL.path,
                underlyingError: error
            )
        }

        await cache.set(container, for: key)
        return container
    }

    /// Load a model from a HuggingFace repository.
    ///
    /// - Parameters:
    ///   - modelID: HuggingFace repository ID (e.g., "Qwen/Qwen2.5-0.5B-Instruct").
    ///   - progress: Optional progress tracking.
    /// - Returns: A fully initialized LanguageModelContainer.
    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> LanguageModelContainer {
        try await load(repo: modelID, progress: progress)
    }

    /// Backward-compatible local snapshot loader.
    public func loadLocalModel(from directory: URL) async throws -> LanguageModelContainer {
        try await load(directory: directory)
    }

    /// Check if a model is cached.
    public func isCached(_ modelID: String) async -> Bool {
        let key = ModelCacheKey(identifier: "repo:\(modelID)")
        return await cache.get(key) != nil
    }

    /// Check if a local model directory is cached.
    public func isCached(directory: URL) async -> Bool {
        let key = ModelCacheKey(identifier: "dir:\(directory.standardizedFileURL.path)")
        return await cache.get(key) != nil
    }

    /// Clear all cached models.
    public func clearCache() async {
        await cache.clear()
    }
}

// MARK: - Thread-safe Cache

private actor ModelCache {
    private var storage: [ModelCacheKey: LanguageModelContainer] = [:]

    func get(_ key: ModelCacheKey) -> LanguageModelContainer? {
        storage[key]
    }

    func set(_ value: LanguageModelContainer, for key: ModelCacheKey) {
        storage[key] = value
    }

    func clear() {
        storage.removeAll()
    }
}

private struct ModelCacheKey: Hashable {
    let identifier: String
}

#endif
