#if METAL_ENABLED
import Foundation

/// Construction-time policy for ``MetalLanguageModel``.
///
/// Mirrors the role of `OllamaConfiguration` in the Ollama backend: a value
/// set once at model construction that forces specific generation parameters
/// for every request, sidestepping the limited per-request surface of
/// `OpenFoundationModels.GenerationOptions`.
///
/// ## Override semantics
///
/// For each field below:
/// - `nil` (or the documented sentinel) means "no forced value": the request
///   uses `GenerationOptions` or the underlying `GenerationParameters`
///   default.
/// - A non-`nil` value means "forced": the per-request `GenerationOptions`
///   value is ignored and the configuration value is applied to every
///   request.
///
/// ## When to use
///
/// Use this for model-specific quirks that should hold across the entire
/// session — for example, a model family that always needs a repetition
/// penalty to avoid loops, or a deployment that must run at a fixed
/// temperature.
public struct MetalConfiguration: Sendable {

    /// Forced sampling temperature. `nil` falls through to
    /// `GenerationOptions.temperature`; if both are `nil`, the
    /// `GenerationParameters` default applies.
    public let temperature: Float?

    /// Forced top-p (nucleus) sampling threshold. `nil` falls through to
    /// the `GenerationParameters` default.
    public let topP: Float?

    /// Forced top-k cutoff. `nil` disables the cutoff.
    public let topK: Int?

    /// Forced minimum probability threshold relative to the best candidate.
    /// `nil` falls through to the `GenerationParameters` default.
    public let minP: Float?

    /// Forced repetition penalty factor. `nil` disables the penalty.
    public let repetitionPenalty: Float?

    /// Forced presence penalty. `nil` disables the penalty.
    public let presencePenalty: Float?

    /// Recent-token window applied to repetition and presence penalties.
    /// Always forced (no fall-through), defaulting to the value the runtime
    /// has historically used.
    public let repetitionContextSize: Int

    /// Forced maximum response tokens. `nil` falls through to
    /// `GenerationOptions.maximumResponseTokens`; if both are `nil`, the
    /// runtime default cap applies.
    public let maxTokens: Int?

    /// Forced hard cap on tokens generated inside the reasoning channel
    /// before the model must close `</think>` (or equivalent). `nil` falls
    /// through to swift-lm's historical generous multiplier
    /// (`max(maxTokens * 16, maxTokens + 256)`), which lets a runaway
    /// reasoning loop run far past the caller's intended `maxTokens`. Set
    /// this to make such loops fail fast — the total raw cap then becomes
    /// `maxTokens + maxReasoningTokens`.
    public let maxReasoningTokens: Int?

    /// Maximum tokens to coalesce into one streamed text chunk. Always
    /// forced; defaults to the value the runtime has historically used.
    public let streamChunkTokenCount: Int

    public init(
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        repetitionContextSize: Int = 64,
        maxTokens: Int? = nil,
        maxReasoningTokens: Int? = nil,
        streamChunkTokenCount: Int = 1
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
        self.repetitionContextSize = repetitionContextSize
        self.maxTokens = maxTokens
        self.maxReasoningTokens = maxReasoningTokens
        self.streamChunkTokenCount = streamChunkTokenCount
    }

    /// Default configuration that preserves the runtime's historical
    /// behavior: no forced sampling, `repetitionContextSize` of 64, and
    /// `streamChunkTokenCount` of 1.
    public static var `default`: MetalConfiguration { MetalConfiguration() }
}

#endif
