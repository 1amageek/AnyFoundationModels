#if METAL_ENABLED
import Foundation
import OpenFoundationModels
import SwiftLM
import Testing
@testable import MetalFoundationModels

/// Pins the construction-time policy contract for ``MetalConfiguration``.
///
/// MetalConfiguration sets values once at model construction that are then
/// merged into the per-request `GenerationParameters`. These tests lock in:
/// (1) defaults preserve the runtime's historical hardcoded values so adding
/// the type is non-breaking, (2) configuration fields force values over
/// `GenerationOptions`, (3) `nil` fields fall through to `GenerationOptions`,
/// (4) every backend-extensible knob actually reaches `GenerationParameters`.
@Suite("MetalConfiguration")
struct MetalConfigurationTests {

    // MARK: - Defaults preserve historical behavior

    @Test("Default configuration matches the runtime's previously hardcoded values")
    func defaultConfigurationKeepsHistoricalDefaults() {
        let config = MetalConfiguration.default
        #expect(config.temperature == nil)
        #expect(config.topP == nil)
        #expect(config.topK == nil)
        #expect(config.minP == nil)
        #expect(config.repetitionPenalty == nil)
        #expect(config.presencePenalty == nil)
        #expect(config.repetitionContextSize == 64)
        #expect(config.maxTokens == nil)
        #expect(config.streamChunkTokenCount == 1)
    }

    @Test("makeParameters with default config and no options reproduces historical request")
    func defaultConfigReproducesHistoricalRequest() {
        let params = MetalLanguageModelRuntime.makeParameters(
            options: nil,
            showsThinking: false,
            configuration: .default
        )
        // Values previously hardcoded in MetalLanguageModelRuntime.makeParameters.
        #expect(params.streamChunkTokenCount == 1)
        #expect(params.repetitionContextSize == 64)
        #expect(params.reasoning == .hidden)
        // GenerationParameters defaults that were left untouched before.
        #expect(params.temperature == 0.6)
        #expect(params.topP == 1.0)
        #expect(params.topK == nil)
        #expect(params.minP == 0)
        #expect(params.repetitionPenalty == nil)
        #expect(params.presencePenalty == nil)
        #expect(params.maxTokens == nil)
    }

    @Test("showsThinking still maps to .separate reasoning regardless of configuration")
    func showsThinkingMapsToSeparateReasoning() {
        let params = MetalLanguageModelRuntime.makeParameters(
            options: nil,
            showsThinking: true,
            configuration: .default
        )
        #expect(params.reasoning == .separate)
    }

    // MARK: - GenerationOptions still flows through when configuration is silent

    @Test("With nil configuration fields, GenerationOptions.temperature wins")
    func optionsTemperatureFlowsThroughWhenConfigSilent() {
        let params = MetalLanguageModelRuntime.makeParameters(
            options: GenerationOptions(temperature: 0.2),
            showsThinking: false,
            configuration: .default
        )
        #expect(params.temperature == 0.2)
    }

    @Test("With nil configuration fields, GenerationOptions.maximumResponseTokens wins")
    func optionsMaxTokensFlowsThroughWhenConfigSilent() {
        let params = MetalLanguageModelRuntime.makeParameters(
            options: GenerationOptions(maximumResponseTokens: 512),
            showsThinking: false,
            configuration: .default
        )
        #expect(params.maxTokens == 512)
    }

    // MARK: - Configuration forces values over GenerationOptions

    @Test("Configured temperature overrides GenerationOptions.temperature")
    func configuredTemperatureOverridesOptions() {
        let config = MetalConfiguration(temperature: 0)
        let params = MetalLanguageModelRuntime.makeParameters(
            options: GenerationOptions(temperature: 0.9),
            showsThinking: false,
            configuration: config
        )
        #expect(params.temperature == 0)
    }

    @Test("Configured maxTokens overrides GenerationOptions.maximumResponseTokens")
    func configuredMaxTokensOverridesOptions() {
        let config = MetalConfiguration(maxTokens: 128)
        let params = MetalLanguageModelRuntime.makeParameters(
            options: GenerationOptions(maximumResponseTokens: 1024),
            showsThinking: false,
            configuration: config
        )
        #expect(params.maxTokens == 128)
    }

    // MARK: - Every extension field reaches GenerationParameters

    @Test("All sampling fields propagate from configuration to GenerationParameters")
    func allSamplingFieldsPropagate() {
        let config = MetalConfiguration(
            temperature: 0.3,
            topP: 0.85,
            topK: 50,
            minP: 0.02,
            repetitionPenalty: 1.15,
            presencePenalty: 0.4,
            repetitionContextSize: 96,
            maxTokens: 256,
            streamChunkTokenCount: 4
        )
        let params = MetalLanguageModelRuntime.makeParameters(
            options: nil,
            showsThinking: false,
            configuration: config
        )
        #expect(params.temperature == 0.3)
        #expect(params.topP == 0.85)
        #expect(params.topK == 50)
        #expect(params.minP == 0.02)
        #expect(params.repetitionPenalty == 1.15)
        #expect(params.presencePenalty == 0.4)
        #expect(params.repetitionContextSize == 96)
        #expect(params.maxTokens == 256)
        #expect(params.streamChunkTokenCount == 4)
    }

    @Test("Sparse configuration only overrides the fields it sets")
    func sparseConfigurationOnlyOverridesSetFields() {
        let config = MetalConfiguration(repetitionPenalty: 1.1)
        let params = MetalLanguageModelRuntime.makeParameters(
            options: GenerationOptions(temperature: 0.7, maximumResponseTokens: 200),
            showsThinking: false,
            configuration: config
        )
        // Forced from configuration.
        #expect(params.repetitionPenalty == 1.1)
        // Falls through to GenerationOptions because configuration is silent on these.
        #expect(params.temperature == 0.7)
        #expect(params.maxTokens == 200)
        // Defaults still apply for everything else.
        #expect(params.topP == 1.0)
        #expect(params.topK == nil)
        #expect(params.minP == 0)
        #expect(params.repetitionContextSize == 64)
    }
}
#endif
