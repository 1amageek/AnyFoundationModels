#if MLX_ENABLED
import MLXLMCommon
import Testing
@testable import MLXFoundationModels

@Suite("MLXPrefixCacheStore Tests")
struct MLXPrefixCacheStoreTests {
    @Test("Lookup returns materialized cache and prefix token count")
    func lookupReturnsMaterializedCache() throws {
        var store = MLXPrefixCacheStore(capacity: 4)
        let metadata = denseMetadata()
        let cacheKey = MLXPrefixCacheKey(rawValue: "cache-1")
        let snapshot = sampleSnapshot(prefixTokenCount: 2, seed: 0)

        store.store(cacheKey: cacheKey, snapshot: snapshot, metadata: metadata)

        let decision = store.lookup(plan: makePlan(cacheKey: cacheKey), metadata: metadata)
        #expect(decision.outcome == "hit")
        #expect(decision.prefixTokenCount == 2)

        guard let cache = decision.cache, let arraysCache = cache.first as? ArraysCache else {
            Issue.record("Expected materialized ArraysCache")
            return
        }

        #expect(arraysCache.metaState == [""])
    }

    @Test("Lookup misses when runtime family changes")
    func lookupMissesOnRuntimeMismatch() throws {
        var store = MLXPrefixCacheStore(capacity: 4)
        let cacheKey = MLXPrefixCacheKey(rawValue: "cache-1")
        let snapshot = sampleSnapshot(prefixTokenCount: 2, seed: 0)

        store.store(cacheKey: cacheKey, snapshot: snapshot, metadata: denseMetadata())

        let decision = store.lookup(plan: makePlan(cacheKey: cacheKey), metadata: vlmMetadata())
        #expect(decision.cache == nil)
        #expect(decision.prefixTokenCount == nil)
        #expect(decision.outcome == MLXCacheInvalidationReason.cacheKeyChanged.rawValue)
    }

    @Test("LRU eviction removes least recently used entry")
    func lruEvictionRemovesLeastRecentlyUsedEntry() throws {
        var store = MLXPrefixCacheStore(capacity: 4)
        let metadata = denseMetadata()
        let keys = (1 ... 5).map { MLXPrefixCacheKey(rawValue: "cache-\($0)") }

        for (index, key) in keys.enumerated() {
            let snapshot = sampleSnapshot(prefixTokenCount: index + 1, seed: index * 10)
            store.store(cacheKey: key, snapshot: snapshot, metadata: metadata)
        }

        let oldestDecision = store.lookup(plan: makePlan(cacheKey: keys[0]), metadata: metadata)
        #expect(oldestDecision.cache == nil)
        #expect(oldestDecision.outcome == MLXCacheInvalidationReason.cacheKeyChanged.rawValue)

        let newestDecision = store.lookup(plan: makePlan(cacheKey: keys[4]), metadata: metadata)
        #expect(newestDecision.cache != nil)
        #expect(newestDecision.prefixTokenCount == 5)
    }

    @Test("Each lookup returns an independent materialized cache")
    func eachLookupReturnsIndependentCache() throws {
        var store = MLXPrefixCacheStore(capacity: 4)
        let metadata = denseMetadata()
        let cacheKey = MLXPrefixCacheKey(rawValue: "cache-1")
        let snapshot = sampleSnapshot(prefixTokenCount: 2, seed: 0)
        store.store(cacheKey: cacheKey, snapshot: snapshot, metadata: metadata)

        let firstDecision = store.lookup(plan: makePlan(cacheKey: cacheKey), metadata: metadata)
        let secondDecision = store.lookup(plan: makePlan(cacheKey: cacheKey), metadata: metadata)

        guard let firstCache = firstDecision.cache?.first,
              let secondCache = secondDecision.cache?.first
        else {
            Issue.record("Expected materialized cache")
            return
        }

        #expect(ObjectIdentifier(firstCache as AnyObject) != ObjectIdentifier(secondCache as AnyObject))

        let thirdDecision = store.lookup(plan: makePlan(cacheKey: cacheKey), metadata: metadata)
        let thirdCache = try #require(thirdDecision.cache?.first)
        #expect(ObjectIdentifier(firstCache as AnyObject) != ObjectIdentifier(thirdCache as AnyObject))
    }

    @Test("Invalidate removes a stored cache entry")
    func invalidateRemovesStoredEntry() throws {
        var store = MLXPrefixCacheStore(capacity: 4)
        let metadata = denseMetadata()
        let cacheKey = MLXPrefixCacheKey(rawValue: "cache-1")
        let snapshot = sampleSnapshot(prefixTokenCount: 2, seed: 0)

        store.store(cacheKey: cacheKey, snapshot: snapshot, metadata: metadata)
        store.invalidate(cacheKey: cacheKey)

        let decision = store.lookup(plan: makePlan(cacheKey: cacheKey), metadata: metadata)
        #expect(decision.cache == nil)
        #expect(decision.prefixTokenCount == nil)
        #expect(decision.outcome == MLXCacheInvalidationReason.noStoredCache.rawValue)
    }
}

private func sampleSnapshot(prefixTokenCount: Int, seed: Int) -> PromptCacheSnapshot {
    PromptCacheSnapshot(
        prefixTokenCount: prefixTokenCount,
        cacheClasses: ["ArraysCache"],
        cacheMetaState: [[""]],
        cacheState: [[]],
        metadata: ["seed": String(seed)]
    )
}

private func makePlan(cacheKey: MLXPrefixCacheKey) -> MLXExecutionPlan {
    MLXExecutionPlan(
        input: UserInput(chat: [.system("You are helpful."), .user("Hello")], tools: nil, additionalContext: [:]),
        responseMode: .text,
        toolPolicy: .disabled,
        cachePlan: MLXCachePlan(
            reuseScope: .prefixReusable,
            cacheKey: cacheKey,
            prefixMessages: [.system("You are helpful.")],
            suffixMessages: [.user("Hello")],
            prefixInput: nil
        ),
        promptTokenEstimate: nil,
        schemaFingerprint: nil,
        additionalContext: [:],
        plannerDiagnostics: .init(
            systemMessageCount: 1,
            userMessageCount: 1,
            assistantMessageCount: 0,
            toolMessageCount: 0,
            imageCount: 0,
            toolDefinitionCount: 0,
            latestUserPreview: "hello"
        )
    )
}

private func denseMetadata() -> MLXModelMetadata {
    MLXModelMetadata(
        modelID: "mlx-community/Qwen3.5-9B",
        runtimeFamily: .llm,
        modalityFamily: .text,
        qwen35Variant: nil
    )
}

private func vlmMetadata() -> MLXModelMetadata {
    MLXModelMetadata(
        modelID: "mlx-community/Qwen3.5-4B-MLX-4bit",
        runtimeFamily: .vlm,
        modalityFamily: .conditionalGeneration,
        qwen35Variant: nil
    )
}
#endif
