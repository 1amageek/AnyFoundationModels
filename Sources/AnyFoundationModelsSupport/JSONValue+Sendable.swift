import OpenFoundationModelsExtra

/// Bridge `JSONValue` (type-safe, `Sendable`) to `[String: any Sendable]` for tokenizer
/// tool-spec dictionaries used by swift-transformers and swift-lm.
extension JSONValue {
    public var sendableValue: any Sendable {
        switch self {
        case .null:
            return Optional<String>.none as any Sendable
        case .bool(let b):
            return b
        case .int(let i):
            return i
        case .double(let d):
            return d
        case .string(let s):
            return s
        case .array(let a):
            return a.map(\.sendableValue) as [any Sendable]
        case .object(let o):
            return o.mapValues(\.sendableValue) as [String: any Sendable]
        }
    }
}
