#if MLX_ENABLED
import Foundation
import Hub
import MLXLMCommon
import MLXLLM
import MLXVLM

/// Errors that occur during MLX model loading and validation.
public enum MLXModelLoadingError: LocalizedError {
    /// The model could not be loaded.
    case loadFailed(modelID: String, underlyingError: Error)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let modelID, let underlyingError):
            return "Failed to load model '\(modelID)': \(underlyingError.localizedDescription)"
        }
    }
}

/// Unified model loader for HuggingFace models with progress reporting.
///
/// Uses `mlx-swift-lm` to resolve either LLM or VLM models from the Hub
/// and load them into a `ModelContainer`.
public final class ModelLoader {

    private let hubApi: HubApi
    private var modelCache: [String: ModelContainer] = [:]
    private let cacheQueue = DispatchQueue(label: "com.openFoundationModels.modelLoader.cache")

    public init(hubApi: HubApi = HubApi()) {
        self.hubApi = hubApi
    }

    /// Load a model with optional progress reporting.
    ///
    /// Downloads a model from Hugging Face and loads it.
    /// The `modelID` should be a HuggingFace repository ID
    /// (e.g., "mlx-community/Qwen2.5-0.5B-Instruct-4bit").
    ///
    /// - Parameters:
    ///   - modelID: Hugging Face repository ID.
    ///   - progress: Optional `Progress` for download progress reporting.
    /// - Returns: A fully initialized `ModelContainer` ready for generation.
    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> ModelContainer {

        if let localDir = resolveLocalDirectory(from: modelID) {
            return try await loadLocalModel(from: localDir, modelID: modelID)
        }

        if let cached = getCachedModel(modelID) {
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Model loaded from cache", comment: "")
            return cached
        }

        let loadingProgress = progress ?? Progress(totalUnitCount: 0)
        let configuration = ModelConfiguration(id: modelID)
        let container: ModelContainer
        do {
            container = try await loadModelContainer(
                hub: hubApi,
                configuration: configuration
            ) { hubProgress in
                loadingProgress.totalUnitCount = hubProgress.totalUnitCount
                loadingProgress.completedUnitCount = hubProgress.completedUnitCount
                loadingProgress.localizedDescription = hubProgress.localizedDescription
                loadingProgress.localizedAdditionalDescription = hubProgress.localizedAdditionalDescription
            }
        } catch {
            throw MLXModelLoadingError.loadFailed(modelID: modelID, underlyingError: error)
        }

        let containerConfiguration = await container.configuration
        let metadata = inferMetadata(modelID: modelID, modelName: containerConfiguration.name)
        await MLXModelMetadataRegistry.shared.register(
            metadata,
            aliases: [containerConfiguration.name]
        )

        setCachedModel(container, for: modelID)

        loadingProgress.completedUnitCount = loadingProgress.totalUnitCount
        loadingProgress.localizedDescription = NSLocalizedString("Model ready", comment: "")

        return container
    }

    /// Download a model without loading it into memory.
    public func downloadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws {
        _ = try await loadModel(modelID, progress: progress)
    }

    /// Load a model from a local HF directory.
    ///
    /// The directory must contain config.json, safetensors weight files,
    /// and tokenizer files.
    ///
    /// - Parameters:
    ///   - directory: URL to the HF model directory.
    ///   - modelID: Optional model identifier for metadata registration.
    public func loadLocalModel(from directory: URL, modelID: String? = nil) async throws -> ModelContainer {
        let resolvedID = modelID ?? directory.lastPathComponent
        let container: ModelContainer
        do {
            container = try await loadModelContainer(
                hub: hubApi,
                directory: directory
            ) { _ in }
        } catch {
            throw MLXModelLoadingError.loadFailed(modelID: resolvedID, underlyingError: error)
        }

        let containerConfiguration = await container.configuration
        let metadata = inferMetadata(modelID: resolvedID, modelName: containerConfiguration.name)
        await MLXModelMetadataRegistry.shared.register(
            metadata,
            aliases: [containerConfiguration.name]
        )

        setCachedModel(container, for: resolvedID)
        return container
    }

    public func cachedModels() -> [String] {
        cacheQueue.sync {
            Array(modelCache.keys)
        }
    }

    public func clearCache() {
        cacheQueue.sync {
            modelCache.removeAll()
        }
    }

    public func clearCache(for modelID: String) {
        _ = cacheQueue.sync {
            modelCache.removeValue(forKey: modelID)
        }
    }

    public func isCached(_ modelID: String) -> Bool {
        cacheQueue.sync {
            modelCache[modelID] != nil
        }
    }

    // MARK: - Metadata Inference

    /// Infers model metadata from the model ID and name.
    ///
    /// Architecture detection happens during loading via config.json.
    /// Metadata for the planner (runtime family, modality) is inferred
    /// from naming conventions in the model ID.
    private func inferMetadata(modelID: String, modelName: String) -> MLXModelMetadata {
        let lowercased = modelID.lowercased()
        let nameLowercased = modelName.lowercased()
        let searchable = lowercased + " " + nameLowercased

        let isVLM = searchable.contains("vl") || searchable.contains("vision")
        let runtimeFamily: MLXRuntimeFamily = isVLM ? .vlm : .llm

        return MLXModelMetadata(
            modelID: modelID,
            runtimeFamily: runtimeFamily
        )
    }

    // MARK: - Cache

    private func getCachedModel(_ modelID: String) -> ModelContainer? {
        cacheQueue.sync {
            modelCache[modelID]
        }
    }

    private func setCachedModel(_ container: ModelContainer, for modelID: String) {
        cacheQueue.sync {
            modelCache[modelID] = container
        }
    }

    /// Resolve a model ID to a local HF directory if it points to a local path.
    private func resolveLocalDirectory(from modelID: String) -> URL? {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let url: URL
        if let parsed = URL(string: trimmed), parsed.isFileURL {
            url = parsed.standardizedFileURL
        } else {
            let localURL = URL(fileURLWithPath: trimmed).standardizedFileURL
            url = localURL
        }

        // Check if it's a directory containing config.json
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
              isDir.boolValue else {
            return nil
        }
        let configPath = url.appendingPathComponent("config.json").path
        guard FileManager.default.fileExists(atPath: configPath) else {
            return nil
        }
        return url
    }
}
#endif
