#if MLX_ENABLED
import Foundation

struct MLXPrefixReuseValidationResult: Sendable, Equatable {
    let acceptedPrefixTokenCount: Int?
    let outcome: String
    let shouldInvalidate: Bool
}

struct MLXPrefixReuseValidator {
    static func validate(
        fullTokens: [Int],
        prefixTokens: [Int],
        requestedPrefixTokenCount: Int?,
        successOutcome: String
    ) -> MLXPrefixReuseValidationResult {
        guard requestedPrefixTokenCount != nil else {
            return MLXPrefixReuseValidationResult(
                acceptedPrefixTokenCount: nil,
                outcome: MLXCacheInvalidationReason.noReusablePrefix.rawValue,
                shouldInvalidate: false
            )
        }

        guard !prefixTokens.isEmpty,
              prefixTokens.count < fullTokens.count,
              fullTokens.count >= prefixTokens.count,
              Array(fullTokens.prefix(prefixTokens.count)) == prefixTokens
        else {
            return MLXPrefixReuseValidationResult(
                acceptedPrefixTokenCount: nil,
                outcome: MLXCacheInvalidationReason.prefixSnapshotMismatch.rawValue,
                shouldInvalidate: true
            )
        }

        return MLXPrefixReuseValidationResult(
            acceptedPrefixTokenCount: prefixTokens.count,
            outcome: successOutcome,
            shouldInvalidate: false
        )
    }
}
#endif
