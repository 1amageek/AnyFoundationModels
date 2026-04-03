#if MLX_ENABLED
import Foundation
import Hub
import MLXLMCommon

public enum MLXModelLoadingError: LocalizedError {
    case loadFailed(modelID: String, underlyingError: Error)
    case invalidLocalModelDirectory(path: String)
    case missingConfig(modelID: String, directory: URL)
    case invalidConfig(modelID: String, directory: URL, underlyingError: Error)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let modelID, let underlyingError):
            return "Failed to load model '\(modelID)': \(underlyingError.localizedDescription)"
        case .invalidLocalModelDirectory(let path):
            return "The local model directory is invalid: \(path)"
        case .missingConfig(let modelID, let directory):
            return "Missing config.json for model '\(modelID)' at \(directory.path)."
        case .invalidConfig(let modelID, let directory, let underlyingError):
            return "Invalid config.json for model '\(modelID)' at \(directory.path): \(underlyingError.localizedDescription)"
        }
    }
}

public actor ModelLoader {
    private let hubApi: HubApi
    private var loadedModels: [String: MLXLoadedModel] = [:]
    private var resolvedArtifacts: [String: URL] = [:]
    private var inflightArtifacts: [String: Task<URL, Error>] = [:]
    private var inflightLoads: [String: Task<MLXLoadedModel, Error>] = [:]

    public init(hubApi: HubApi = HubApi()) {
        self.hubApi = hubApi
    }

    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> MLXLoadedModel {
        let artifact = try resolveArtifactReference(from: modelID)
        let cacheKey = artifact.cacheKey

        if let cached = loadedModels[cacheKey] {
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Model loaded from cache", comment: "")
            return cached
        }

        if let inflight = inflightLoads[cacheKey] {
            return try await inflight.value
        }

        let task = Task { [hubApi] () async throws -> MLXLoadedModel in
            let directory = try await self.artifactDirectory(
                for: artifact,
                progress: progress,
                hubApi: hubApi
            )
            let profile = try Self.resolveProfile(
                for: artifact.modelID,
                directory: directory
            )

            let container: ModelContainer
            do {
                container = try await loadModelContainer(
                    hub: hubApi,
                    directory: directory
                ) { _ in }
            } catch {
                throw MLXModelLoadingError.loadFailed(
                    modelID: artifact.modelID,
                    underlyingError: error
                )
            }

            progress?.completedUnitCount = progress?.totalUnitCount
                ?? progress?.completedUnitCount
                ?? 1
            progress?.localizedDescription = NSLocalizedString("Model ready", comment: "")

            return MLXLoadedModel(
                container: container,
                profile: profile
            )
        }

        inflightLoads[cacheKey] = task
        defer { inflightLoads[cacheKey] = nil }

        let loadedModel = try await task.value
        loadedModels[cacheKey] = loadedModel
        return loadedModel
    }

    @discardableResult
    public func downloadModel(
        _ modelID: String,
        progress: Progress? = nil
    ) async throws -> URL {
        let artifact = try resolveArtifactReference(from: modelID)
        return try await artifactDirectory(
            for: artifact,
            progress: progress,
            hubApi: hubApi
        )
    }

    public func cachedModels() -> [String] {
        Array(loadedModels.keys)
    }

    public func clearCache() {
        loadedModels.removeAll()
        resolvedArtifacts.removeAll()
    }

    public func clearCache(for modelID: String) throws {
        let artifact = try resolveArtifactReference(from: modelID)
        loadedModels.removeValue(forKey: artifact.cacheKey)
        resolvedArtifacts.removeValue(forKey: artifact.cacheKey)
    }

    public func isCached(_ modelID: String) throws -> Bool {
        let artifact = try resolveArtifactReference(from: modelID)
        if loadedModels[artifact.cacheKey] != nil {
            return true
        }
        if let resolvedArtifact = resolvedArtifacts[artifact.cacheKey] {
            return Self.isUsableModelDirectory(resolvedArtifact)
        }

        switch artifact.source {
        case .local(let directory):
            return Self.isUsableModelDirectory(directory)
        case .remote(let configuration):
            return Self.isUsableModelDirectory(configuration.modelDirectory(hub: hubApi))
        }
    }

    private func resolveArtifactReference(from modelID: String) throws -> ArtifactReference {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.isEmpty == false else {
            throw MLXModelLoadingError.invalidLocalModelDirectory(path: modelID)
        }

        if let localDirectory = resolveLocalDirectory(from: trimmed) {
            return ArtifactReference(
                cacheKey: localDirectory.path,
                modelID: trimmed,
                source: .local(directory: localDirectory)
            )
        }

        return ArtifactReference(
            cacheKey: trimmed,
            modelID: trimmed,
            source: .remote(
                configuration: ModelConfiguration(id: trimmed)
            )
        )
    }

    private func resolveLocalDirectory(from modelID: String) -> URL? {
        let parsedURL: URL
        if let url = URL(string: modelID), url.isFileURL {
            parsedURL = url.standardizedFileURL
        } else {
            parsedURL = URL(fileURLWithPath: modelID).standardizedFileURL
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: parsedURL.path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            return nil
        }
        return parsedURL
    }

    private struct ArtifactReference: Sendable {
        enum Source: Sendable {
            case local(directory: URL)
            case remote(configuration: ModelConfiguration)
        }

        let cacheKey: String
        let modelID: String
        let source: Source
    }

    private func artifactDirectory(
        for artifact: ArtifactReference,
        progress: Progress?,
        hubApi: HubApi
    ) async throws -> URL {
        if let resolved = resolvedArtifacts[artifact.cacheKey],
           Self.isUsableModelDirectory(resolved) {
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Using cached model artifact", comment: "")
            return resolved
        }

        if let inflight = inflightArtifacts[artifact.cacheKey] {
            return try await inflight.value
        }

        let task = Task { () async throws -> URL in
            try await Self.resolveArtifactDirectory(
                artifact,
                hubApi: hubApi,
                progress: progress
            )
        }

        inflightArtifacts[artifact.cacheKey] = task
        defer { inflightArtifacts[artifact.cacheKey] = nil }

        let resolved = try await task.value
        resolvedArtifacts[artifact.cacheKey] = resolved
        return resolved
    }

    private static func resolveArtifactDirectory(
        _ artifact: ArtifactReference,
        hubApi: HubApi,
        progress: Progress?
    ) async throws -> URL {
        switch artifact.source {
        case .local(let directory):
            guard isUsableModelDirectory(directory) else {
                throw MLXModelLoadingError.invalidLocalModelDirectory(path: directory.path)
            }
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Using local model directory", comment: "")
            return directory
        case .remote(let configuration):
            let directory = configuration.modelDirectory(hub: hubApi)
            if isUsableModelDirectory(directory) {
                progress?.completedUnitCount = progress?.totalUnitCount ?? 1
                progress?.localizedDescription = NSLocalizedString("Using downloaded model snapshot", comment: "")
                return directory
            }

            progress?.localizedDescription = NSLocalizedString("Downloading model", comment: "")
            do {
                return try await MLXLMCommon.downloadModel(
                    hub: hubApi,
                    configuration: configuration
                ) { hubProgress in
                    progress?.totalUnitCount = hubProgress.totalUnitCount
                    progress?.completedUnitCount = hubProgress.completedUnitCount
                    progress?.localizedDescription = hubProgress.localizedDescription
                    progress?.localizedAdditionalDescription = hubProgress.localizedAdditionalDescription
                }
            } catch {
                throw MLXModelLoadingError.loadFailed(
                    modelID: artifact.modelID,
                    underlyingError: error
                )
            }
        }
    }

    private static func resolveProfile(
        for modelID: String,
        directory: URL
    ) throws -> MLXModelProfile {
        let configURL = directory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw MLXModelLoadingError.missingConfig(modelID: modelID, directory: directory)
        }

        let data: Data
        do {
            data = try Data(contentsOf: configURL)
        } catch {
            throw MLXModelLoadingError.invalidConfig(
                modelID: modelID,
                directory: directory,
                underlyingError: error
            )
        }

        let jsonObject: Any
        do {
            jsonObject = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw MLXModelLoadingError.invalidConfig(
                modelID: modelID,
                directory: directory,
                underlyingError: error
            )
        }

        guard let dictionary = jsonObject as? [String: Any] else {
            throw MLXModelLoadingError.invalidConfig(
                modelID: modelID,
                directory: directory,
                underlyingError: CocoaError(.coderInvalidValue)
            )
        }

        if containsVisualConfiguration(in: dictionary) {
            return MLXModelProfile.make(
                modelID: modelID,
                runtimeFamily: .vlm,
                modalities: [.text, .image]
            )
        }

        if containsArchitectureMetadata(in: dictionary) {
            return MLXModelProfile.make(
                modelID: modelID,
                runtimeFamily: .llm,
                modalities: [.text]
            )
        }

        return .fallback(modelID: modelID)
    }

    private static func containsArchitectureMetadata(in dictionary: [String: Any]) -> Bool {
        if let architectures = dictionary["architectures"] as? [String], architectures.isEmpty == false {
            return true
        }
        if let modelType = dictionary["model_type"] as? String, modelType.isEmpty == false {
            return true
        }

        for value in dictionary.values {
            if let nestedDictionary = value as? [String: Any],
               containsArchitectureMetadata(in: nestedDictionary) {
                return true
            }
            if let nestedArray = value as? [Any],
               containsArchitectureMetadata(in: nestedArray) {
                return true
            }
        }

        return false
    }

    private static func containsArchitectureMetadata(in array: [Any]) -> Bool {
        for element in array {
            if let nestedDictionary = element as? [String: Any],
               containsArchitectureMetadata(in: nestedDictionary) {
                return true
            }
            if let nestedArray = element as? [Any],
               containsArchitectureMetadata(in: nestedArray) {
                return true
            }
        }

        return false
    }

    private static func resolveProfileLegacy(
        for modelID: String
    ) -> MLXModelProfile {
        return MLXModelProfile.make(
            modelID: modelID,
            runtimeFamily: .llm,
            modalities: [.text]
        )
    }

    private static func isUsableModelDirectory(_ directory: URL) -> Bool {
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: directory.appendingPathComponent("config.json").path) else {
            return false
        }

        guard directoryContains(directory, extension: "safetensors") else {
            return false
        }

        let tokenizerCandidates = [
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
            "spiece.model",
        ]
        return tokenizerCandidates.contains { filename in
            fileManager.fileExists(atPath: directory.appendingPathComponent(filename).path)
        }
    }

    private static func directoryContains(
        _ directory: URL,
        extension pathExtension: String
    ) -> Bool {
        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }

        for case let url as URL in enumerator where url.pathExtension == pathExtension {
            return true
        }
        return false
    }

    private static func containsVisualConfiguration(in dictionary: [String: Any]) -> Bool {
        let multimodalKeys: Set<String> = [
            "vision_config",
            "vision_tower",
            "image_token",
            "image_token_id",
            "image_token_index",
            "multi_modal_projector",
            "mm_projector",
            "mm_projector_type",
        ]

        for (key, value) in dictionary {
            if multimodalKeys.contains(key) {
                return true
            }

            if key == "architectures",
               let architectures = value as? [String],
               architectures.contains(where: { architecture in
                   let normalized = architecture.lowercased()
                   return normalized.contains("vision") || normalized.contains("vl")
               }) {
                return true
            }

            if let nestedDictionary = value as? [String: Any],
               containsVisualConfiguration(in: nestedDictionary) {
                return true
            }

            if let nestedArray = value as? [Any],
               containsVisualConfiguration(in: nestedArray) {
                return true
            }
        }

        return false
    }

    private static func containsVisualConfiguration(in array: [Any]) -> Bool {
        for element in array {
            if let dictionary = element as? [String: Any],
               containsVisualConfiguration(in: dictionary) {
                return true
            }

            if let nestedArray = element as? [Any],
               containsVisualConfiguration(in: nestedArray) {
                return true
            }
        }

        return false
    }
}
#endif
