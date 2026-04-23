#if MLX_ENABLED
import Foundation
import HuggingFace
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
import Tokenizers

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

public enum MLXModelLoadingStatus: Sendable, Equatable {
    case preparing(modelID: String)
    case checkingCache(modelID: String)
    case awaitingInFlightLoad(modelID: String)
    case usingMemoryCache(modelID: String)
    case usingDiskCache(modelID: String)
    case downloading(
        modelID: String,
        phase: String,
        completedUnitCount: Int64?,
        totalUnitCount: Int64?
    )
    case loadingWeights(modelID: String)
    case ready(modelID: String)
}

public typealias MLXModelLoadingStatusHandler = @Sendable (MLXModelLoadingStatus) async -> Void

public actor ModelLoader {
    private static let registeredFactories: Void = {
        ModelFactoryRegistry.shared.addTrampoline {
            VLMModelFactory.shared
        }
        ModelFactoryRegistry.shared.addTrampoline {
            LLMModelFactory.shared
        }
        }()

    private final class DownloadHeartbeatState: @unchecked Sendable {
        private let lock = NSLock()
        private var hasReceivedProgress = false
        private var isFinished = false

        func markProgressReceived() {
            lock.lock()
            hasReceivedProgress = true
            lock.unlock()
        }

        func finish() {
            lock.lock()
            isFinished = true
            lock.unlock()
        }

        func shouldEmitHeartbeat() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            return isFinished == false && hasReceivedProgress == false
        }
    }

    private struct InflightArtifact: Sendable {
        let id: UUID
        let task: Task<URL, Error>
    }

    private struct InflightLoad: Sendable {
        let id: UUID
        let task: Task<MLXLoadedModel, Error>
    }

    private let hubClient: HubClient
    private let downloader: any Downloader
    private let tokenizerLoader: any TokenizerLoader
    private var loadedModels: [String: MLXLoadedModel] = [:]
    private var resolvedArtifacts: [String: URL] = [:]
    private var inflightArtifacts: [String: InflightArtifact] = [:]
    private var inflightLoads: [String: InflightLoad] = [:]

    public init(hubClient: HubClient = HubClient()) {
        self.hubClient = hubClient
        self.downloader = #hubDownloader(hubClient)
        self.tokenizerLoader = #huggingFaceTokenizerLoader()
    }

    public func loadModel(
        _ modelID: String,
        progress: Progress? = nil,
        statusHandler: MLXModelLoadingStatusHandler? = nil
    ) async throws -> MLXLoadedModel {
        Self.ensureModelFactoriesRegistered()
        await emit(.preparing(modelID: modelID), to: statusHandler)
        let artifact = try resolveArtifactReference(from: modelID)
        let cacheKey = artifact.cacheKey

        if let cached = loadedModels[cacheKey] {
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Model loaded from cache", comment: "")
            await emit(.usingMemoryCache(modelID: modelID), to: statusHandler)
            await emit(.ready(modelID: modelID), to: statusHandler)
            return cached
        }

        if let inflight = inflightLoads[cacheKey] {
            await emit(.awaitingInFlightLoad(modelID: modelID), to: statusHandler)
            return try await inflight.task.value
        }

        let downloader = self.downloader
        let tokenizerLoader = self.tokenizerLoader
        let inflightID = UUID()
        let task = Task { () async throws -> MLXLoadedModel in
            let directory = try await self.artifactDirectory(
                for: artifact,
                progress: progress,
                statusHandler: statusHandler
            )
            let profile = try Self.resolveProfile(
                for: artifact.modelID,
                directory: directory
            )

            let container: ModelContainer
            do {
                await self.emit(.loadingWeights(modelID: artifact.modelID), to: statusHandler)
                container = try await Self.loadModelContainer(
                    for: profile,
                    downloader: downloader,
                    tokenizerLoader: tokenizerLoader,
                    directory: directory
                )
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
            await self.emit(.ready(modelID: artifact.modelID), to: statusHandler)

            return MLXLoadedModel(
                container: container,
                profile: profile
            )
        }

        inflightLoads[cacheKey] = InflightLoad(id: inflightID, task: task)
        defer { removeInflightLoad(id: inflightID, for: cacheKey) }

        let loadedModel = try await task.value
        if inflightLoads[cacheKey]?.id == inflightID {
            loadedModels[cacheKey] = loadedModel
        }
        return loadedModel
    }

    @discardableResult
    public func downloadModel(
        _ modelID: String,
        progress: Progress? = nil,
        statusHandler: MLXModelLoadingStatusHandler? = nil
    ) async throws -> URL {
        await emit(.preparing(modelID: modelID), to: statusHandler)
        let artifact = try resolveArtifactReference(from: modelID)
        return try await artifactDirectory(
            for: artifact,
            progress: progress,
            statusHandler: statusHandler
        )
    }

    public func cachedModels() -> [String] {
        Array(Set(loadedModels.keys).union(resolvedArtifacts.keys)).sorted()
    }

    public func clearCache() {
        for inflightArtifact in inflightArtifacts.values {
            inflightArtifact.task.cancel()
        }
        for inflightLoad in inflightLoads.values {
            inflightLoad.task.cancel()
        }
        inflightArtifacts.removeAll()
        inflightLoads.removeAll()
        loadedModels.removeAll()
        resolvedArtifacts.removeAll()
    }

    public func clearCache(for modelID: String) throws {
        let artifact = try resolveArtifactReference(from: modelID)
        inflightArtifacts.removeValue(forKey: artifact.cacheKey)?.task.cancel()
        inflightLoads.removeValue(forKey: artifact.cacheKey)?.task.cancel()
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
        case .remote:
            return false
        }
    }

    private static func ensureModelFactoriesRegistered() {
        _ = registeredFactories
    }

    private static func loadModelContainer(
        for profile: MLXModelProfile,
        downloader: any Downloader,
        tokenizerLoader: any TokenizerLoader,
        directory: URL
    ) async throws -> ModelContainer {
        let configuration = ModelConfiguration(directory: directory)
        switch profile.runtimeFamily {
        case .vlm:
            return try await VLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration
            ) { _ in }
        case .llm:
            return try await LLMModelFactory.shared.loadContainer(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration
            ) { _ in }
        case .unknown:
            return try await MLXLMCommon.loadModelContainer(
                from: downloader,
                using: tokenizerLoader,
                configuration: configuration
            )
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
        statusHandler: MLXModelLoadingStatusHandler?
    ) async throws -> URL {
        await emit(.checkingCache(modelID: artifact.modelID), to: statusHandler)
        if let resolved = resolvedArtifacts[artifact.cacheKey],
           Self.isUsableModelDirectory(resolved) {
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Using cached model artifact", comment: "")
            await emit(.usingDiskCache(modelID: artifact.modelID), to: statusHandler)
            return resolved
        }

        if let inflight = inflightArtifacts[artifact.cacheKey] {
            await emit(.awaitingInFlightLoad(modelID: artifact.modelID), to: statusHandler)
            return try await inflight.task.value
        }

        let downloader = self.downloader
        let inflightID = UUID()
        let task = Task { () async throws -> URL in
            try await Self.resolveArtifactDirectory(
                artifact,
                downloader: downloader,
                progress: progress,
                statusHandler: statusHandler
            )
        }

        inflightArtifacts[artifact.cacheKey] = InflightArtifact(id: inflightID, task: task)
        defer { removeInflightArtifact(id: inflightID, for: artifact.cacheKey) }

        let resolved = try await task.value
        if inflightArtifacts[artifact.cacheKey]?.id == inflightID {
            resolvedArtifacts[artifact.cacheKey] = resolved
        }
        return resolved
    }

    private func removeInflightArtifact(id: UUID, for cacheKey: String) {
        guard inflightArtifacts[cacheKey]?.id == id else {
            return
        }
        inflightArtifacts.removeValue(forKey: cacheKey)
    }

    private func removeInflightLoad(id: UUID, for cacheKey: String) {
        guard inflightLoads[cacheKey]?.id == id else {
            return
        }
        inflightLoads.removeValue(forKey: cacheKey)
    }

    private static func resolveArtifactDirectory(
        _ artifact: ArtifactReference,
        downloader: any Downloader,
        progress: Progress?,
        statusHandler: MLXModelLoadingStatusHandler?
    ) async throws -> URL {
        switch artifact.source {
        case .local(let directory):
            guard isUsableModelDirectory(directory) else {
                throw MLXModelLoadingError.invalidLocalModelDirectory(path: directory.path)
            }
            progress?.completedUnitCount = progress?.totalUnitCount ?? 1
            progress?.localizedDescription = NSLocalizedString("Using local model directory", comment: "")
            await emit(.usingDiskCache(modelID: artifact.modelID), to: statusHandler)
            return directory
        case .remote(let configuration):
            progress?.localizedDescription = NSLocalizedString("Downloading model", comment: "")
            await emit(
                .downloading(
                    modelID: artifact.modelID,
                    phase: "Resolving remote snapshot metadata",
                    completedUnitCount: nil,
                    totalUnitCount: nil
                ),
                to: statusHandler
            )
            let heartbeatState = DownloadHeartbeatState()
            let heartbeatTask = Task {
                while heartbeatState.shouldEmitHeartbeat() {
                    try? await Task.sleep(for: .seconds(2))
                    guard heartbeatState.shouldEmitHeartbeat() else {
                        return
                    }
                    await emit(
                        .downloading(
                            modelID: artifact.modelID,
                            phase: "Resolving remote snapshot metadata",
                            completedUnitCount: nil,
                            totalUnitCount: nil
                        ),
                        to: statusHandler
                    )
                }
            }
            do {
                let resolved = try await MLXLMCommon.resolve(
                    configuration: configuration,
                    from: downloader,
                    useLatest: false
                ) { hubProgress in
                    heartbeatState.markProgressReceived()
                    progress?.totalUnitCount = hubProgress.totalUnitCount
                    progress?.completedUnitCount = hubProgress.completedUnitCount
                    progress?.localizedDescription = hubProgress.localizedDescription
                    progress?.localizedAdditionalDescription = hubProgress.localizedAdditionalDescription
                    let description = hubProgress.localizedDescription?
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let additional = hubProgress.localizedAdditionalDescription?
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    let phase = additional?.isEmpty == false
                        ? additional!
                        : (description?.isEmpty == false ? description! : "Downloading model")
                    let completedUnitCount = hubProgress.completedUnitCount > 0 ? hubProgress.completedUnitCount : nil
                    let totalUnitCount = hubProgress.totalUnitCount > 0 ? hubProgress.totalUnitCount : nil
                    Task {
                        await emit(
                            .downloading(
                                modelID: artifact.modelID,
                                phase: phase,
                                completedUnitCount: completedUnitCount,
                                totalUnitCount: totalUnitCount
                            ),
                            to: statusHandler
                        )
                    }
                }
                heartbeatState.finish()
                heartbeatTask.cancel()
                return resolved.modelDirectory
            } catch {
                heartbeatState.finish()
                heartbeatTask.cancel()
                throw MLXModelLoadingError.loadFailed(
                    modelID: artifact.modelID,
                    underlyingError: error
                )
            }
        }
    }

    private func emit(
        _ status: MLXModelLoadingStatus,
        to statusHandler: MLXModelLoadingStatusHandler?
    ) async {
        guard let statusHandler else {
            return
        }
        await statusHandler(status)
    }

    private static func emit(
        _ status: MLXModelLoadingStatus,
        to statusHandler: MLXModelLoadingStatusHandler?
    ) async {
        guard let statusHandler else {
            return
        }
        await statusHandler(status)
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
