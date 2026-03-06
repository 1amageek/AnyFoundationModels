#if MLX_ENABLED
import Foundation
@preconcurrency import MLXLMCommon

struct MLXPrefixCacheEntry {
    let cacheKey: MLXPrefixCacheKey
    let snapshot: PromptCacheSnapshot
    let createdAt: Date
    let runtimeFamily: MLXRuntimeFamily
    let modality: MLXModalityFamily
}

struct MLXCacheReuseDecision {
    let cache: [KVCache]?
    let prefixTokenCount: Int?
    let outcome: String

    static func noReuse(reason: MLXCacheInvalidationReason) -> Self {
        Self(cache: nil, prefixTokenCount: nil, outcome: reason.rawValue)
    }
}

struct MLXPrefixCacheStore {
    private let capacity: Int
    private var entries: [MLXPrefixCacheKey: MLXPrefixCacheEntry]
    private var accessOrder: [MLXPrefixCacheKey]

    init(capacity: Int = 4) {
        self.capacity = capacity
        self.entries = [:]
        self.accessOrder = []
    }

    mutating func lookup(
        plan: MLXExecutionPlan,
        metadata: MLXModelMetadata
    ) -> MLXCacheReuseDecision {
        guard plan.cachePlan.reuseScope == .prefixReusable,
              let cacheKey = plan.cachePlan.cacheKey
        else {
            return .noReuse(reason: .noReusablePrefix)
        }

        guard let entry = entries[cacheKey] else {
            return MLXCacheReuseDecision(
                cache: nil,
                prefixTokenCount: nil,
                outcome: entries.isEmpty
                    ? MLXCacheInvalidationReason.noStoredCache.rawValue
                    : MLXCacheInvalidationReason.cacheKeyChanged.rawValue
            )
        }

        guard entry.cacheKey == cacheKey,
              entry.runtimeFamily == metadata.runtimeFamily,
              entry.modality == metadata.modalityFamily
        else {
            remove(cacheKey: cacheKey)
            return MLXCacheReuseDecision(
                cache: nil,
                prefixTokenCount: nil,
                outcome: MLXCacheInvalidationReason.cacheKeyChanged.rawValue
            )
        }

        do {
            let cache = try materializePromptCache(from: entry.snapshot)
            touch(cacheKey)
            return MLXCacheReuseDecision(
                cache: cache,
                prefixTokenCount: entry.snapshot.prefixTokenCount,
                outcome: "hit"
            )
        } catch {
            remove(cacheKey: cacheKey)
            Logger.warning("[MLXPrefixCacheStore] Failed to materialize prompt cache: \(error)")
            return MLXCacheReuseDecision(
                cache: nil,
                prefixTokenCount: nil,
                outcome: MLXCacheInvalidationReason.cacheKeyChanged.rawValue
            )
        }
    }

    mutating func store(
        cacheKey: MLXPrefixCacheKey,
        snapshot: PromptCacheSnapshot,
        metadata: MLXModelMetadata
    ) {
        entries[cacheKey] = MLXPrefixCacheEntry(
            cacheKey: cacheKey,
            snapshot: snapshot,
            createdAt: Date(),
            runtimeFamily: metadata.runtimeFamily,
            modality: metadata.modalityFamily
        )
        touch(cacheKey)
        evictIfNeeded()
    }

    mutating func invalidate(cacheKey: MLXPrefixCacheKey) {
        remove(cacheKey: cacheKey)
    }

    private mutating func touch(_ cacheKey: MLXPrefixCacheKey) {
        accessOrder.removeAll { $0 == cacheKey }
        accessOrder.append(cacheKey)
    }

    private mutating func remove(cacheKey: MLXPrefixCacheKey) {
        entries.removeValue(forKey: cacheKey)
        accessOrder.removeAll { $0 == cacheKey }
    }

    private mutating func evictIfNeeded() {
        while entries.count > capacity, let evictedKey = accessOrder.first {
            accessOrder.removeFirst()
            entries.removeValue(forKey: evictedKey)
        }
    }
}
#endif
