#if OLLAMA_ENABLED
import Foundation
import OpenFoundationModels

// MARK: - GenerationOptions Extension
internal extension GenerationOptions {
    func toOllamaOptions() -> OllamaOptions {
        return OllamaOptions(
            numPredict: maximumResponseTokens,
            temperature: temperature,
            topP: nil  // SamplingMode probabilityThreshold cannot be extracted
        )
    }
}

/// Ollama Language Model Provider for OpenFoundationModels
public final class OllamaLanguageModel: LanguageModel, Sendable {

    // MARK: - Properties
    internal let httpClient: OllamaHTTPClient
    internal let modelName: String
    internal let configuration: OllamaConfiguration

    /// Controls thinking mode for this model instance.
    /// When `nil`, the `think` parameter is omitted from API requests (Ollama uses model defaults).
    /// When `.enabled`, sends `think: true` to explicitly enable thinking output separation.
    /// For structured output requests, this is automatically overridden to `.disabled`.
    public let thinkingMode: ThinkingMode?

    /// Response processor for unified response handling
    private let responseProcessor = ResponseProcessor()

    /// Request builder for creating ChatRequests from Transcripts
    private var requestBuilder: OllamaRequestBuilder {
        OllamaRequestBuilder(configuration: configuration, modelName: modelName, thinkingMode: thinkingMode)
    }

    // MARK: - LanguageModel Protocol Compliance
    public var isAvailable: Bool { true }

    // MARK: - Initialization

    /// Initialize with configuration and model name
    /// - Parameters:
    ///   - configuration: Ollama configuration
    ///   - modelName: Name of the model (e.g., "llama3.2", "mistral")
    public init(
        configuration: OllamaConfiguration,
        modelName: String,
        thinkingMode: ThinkingMode? = .enabled
    ) {
        self.configuration = configuration
        self.modelName = modelName
        self.thinkingMode = thinkingMode
        self.httpClient = OllamaHTTPClient(configuration: configuration)
    }

    /// Convenience initializer with just model name
    /// - Parameter modelName: Name of the model
    public convenience init(modelName: String, thinkingMode: ThinkingMode? = nil) {
        self.init(configuration: OllamaConfiguration(), modelName: modelName, thinkingMode: thinkingMode)
    }

    // MARK: - LanguageModel Protocol Implementation

    public func generate(transcript: Transcript, options: GenerationOptions?) async throws -> Transcript.Entry {
        // Build request using shared builder
        let buildResult = requestBuilder.build(
            transcript: transcript,
            options: options,
            stream: false
        )

        #if DEBUG
        // Log request details
        if let tools = buildResult.request.tools {
            print("[Ollama] Request tools: \(tools.map { $0.function.name })")
        } else {
            print("[Ollama] Request tools: none")
        }
        #endif

        // Send request
        let response: ChatResponse = try await httpClient.send(buildResult.request, to: "/api/chat")

        guard let message = response.message else {
            return createResponseEntry(content: "", reasoning: "")
        }

        #if DEBUG
        // Log response details
        if let thinking = message.thinking, !thinking.isEmpty {
            print("[Ollama] Thinking: \(thinking.prefix(500))...")
        }
        if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            print("[Ollama] Native tool_calls: \(toolCalls.map { $0.function.name })")
        }
        print("[Ollama] Content: \(message.content.prefix(200))...")
        #endif

        // Use ResponseProcessor for unified handling
        switch responseProcessor.process(message) {
        case .toolCalls(let toolCalls):
            #if DEBUG
            print("[Ollama] Processed as: toolCalls (\(toolCalls.count))")
            #endif
            return createToolCallsEntry(from: toolCalls)
        case .response(let response):
            #if DEBUG
            print("[Ollama] Processed as: response")
            #endif
            return createResponseEntry(content: response.content, reasoning: response.reasoning)
        case .empty:
            #if DEBUG
            print("[Ollama] Processed as: empty")
            #endif
            return createResponseEntry(content: "", reasoning: "")
        }
    }

    public func stream(transcript: Transcript, options: GenerationOptions?) -> AsyncThrowingStream<Transcript.Entry, Error> {
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    // Build request using shared builder
                    let buildResult = self.requestBuilder.build(
                        transcript: transcript,
                        options: options,
                        stream: true
                    )

                    // Stream raw responses
                    let rawStream: AsyncThrowingStream<ChatResponse, Error> = await self.httpClient.stream(
                        buildResult.request,
                        to: "/api/chat"
                    )

                    var accumulatedContent = ""
                    var accumulatedThinking = ""
                    var emittedContent = ""
                    var emittedReasoning = ""
                    var nativeToolCalls: [ToolCall] = []
                    var hasYieldedResponseEntry = false

                    for try await chunk in rawStream {
                        // Accumulate content
                        if let content = chunk.message?.content, !content.isEmpty {
                            accumulatedContent += content
                        }

                        // Accumulate thinking content
                        if let thinking = chunk.message?.thinking, !thinking.isEmpty {
                            accumulatedThinking += thinking
                        }

                        // Accumulate native tool calls
                        if let toolCalls = chunk.message?.toolCalls {
                            nativeToolCalls.append(contentsOf: toolCalls)
                        }

                        let containsTextualToolCalls =
                            TextToolCallParser.containsToolCallPatterns(accumulatedContent) ||
                            TextToolCallParser.containsToolCallPatterns(accumulatedThinking)

                        if !chunk.done && nativeToolCalls.isEmpty && !containsTextualToolCalls {
                            let normalized = self.responseProcessor.normalize(
                                content: accumulatedContent,
                                thinking: accumulatedThinking.isEmpty ? nil : accumulatedThinking
                            )
                            if let entry = self.makeDeltaResponseEntry(
                                normalized: normalized,
                                emittedContent: &emittedContent,
                                emittedReasoning: &emittedReasoning
                            ) {
                                continuation.yield(entry)
                                hasYieldedResponseEntry = true
                            }
                        }

                        // On stream completion
                        if chunk.done {
                            // Create a virtual Message from accumulated data
                            let finalMessage = Message(
                                role: .assistant,
                                content: accumulatedContent,
                                toolCalls: nativeToolCalls.isEmpty ? nil : nativeToolCalls,
                                thinking: accumulatedThinking.isEmpty ? nil : accumulatedThinking
                            )

                            // Use ResponseProcessor for unified handling
                            switch self.responseProcessor.process(finalMessage) {
                            case .toolCalls(let toolCalls):
                                continuation.yield(self.createToolCallsEntry(from: toolCalls))

                            case .response(let response):
                                if let entry = self.makeDeltaResponseEntry(
                                    normalized: response,
                                    emittedContent: &emittedContent,
                                    emittedReasoning: &emittedReasoning
                                ) {
                                    continuation.yield(entry)
                                } else if !hasYieldedResponseEntry {
                                    continuation.yield(
                                        self.createResponseEntry(
                                            content: response.content,
                                            reasoning: response.reasoning
                                        )
                                    )
                                }

                            case .empty:
                                if !hasYieldedResponseEntry {
                                    continuation.yield(self.createResponseEntry(content: "", reasoning: ""))
                                }
                            }
                        }
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    public func supports(locale: Locale) -> Bool {
        // Ollama models generally support multiple languages
        return true
    }

    // MARK: - Private Helper Methods

    /// Create tool calls entry from Ollama tool calls
    internal func createToolCallsEntry(from toolCalls: [ToolCall]) -> Transcript.Entry {
        let transcriptToolCalls = toolCalls.map { toolCall in
            // Convert Ollama tool call to Transcript tool call
            // arguments is JSONValue (Codable), encode to JSON string
            let argumentsContent: GeneratedContent

            do {
                let jsonData = try JSONEncoder().encode(toolCall.function.arguments)
                let jsonString = String(data: jsonData, encoding: .utf8) ?? "{}"
                argumentsContent = try GeneratedContent(json: jsonString)
            } catch {
                // Fallback to empty content - use a safer approach
                #if DEBUG
                print("[OllamaLanguageModel] Failed to create GeneratedContent from tool arguments: \(error)")
                #endif
                // Create empty GeneratedContent without force unwrap
                if let emptyContent = try? GeneratedContent(json: "{}") {
                    argumentsContent = emptyContent
                } else {
                    let emptyKeyValuePairs: KeyValuePairs<String, any ConvertibleToGeneratedContent> = [:]
                    argumentsContent = GeneratedContent(properties: emptyKeyValuePairs)
                }
            }

            let toolCall = Transcript.ToolCall(
                id: UUID().uuidString,
                toolName: toolCall.function.name,
                arguments: argumentsContent
            )

            return toolCall
        }

        return .toolCalls(
            Transcript.ToolCalls(
                id: UUID().uuidString,
                transcriptToolCalls
            )
        )
    }

    /// Create response entry from normalized content and reasoning strings.
    private func createResponseEntry(content: String, reasoning: String) -> Transcript.Entry {
        let segments = makeResponseSegments(content: content, reasoning: reasoning)
        return .response(
            Transcript.Response(
                id: UUID().uuidString,
                assetIDs: [],
                segments: segments
            )
        )
    }

    private func makeDeltaResponseEntry(
        normalized: ResponseProcessor.ResponseChannels,
        emittedContent: inout String,
        emittedReasoning: inout String
    ) -> Transcript.Entry? {
        let contentDelta = delta(current: normalized.content, emitted: &emittedContent)
        let reasoningDelta = delta(current: normalized.reasoning, emitted: &emittedReasoning)
        guard !contentDelta.isEmpty || !reasoningDelta.isEmpty else {
            return nil
        }
        return createResponseEntry(content: contentDelta, reasoning: reasoningDelta)
    }

    private func delta(current: String, emitted: inout String) -> String {
        let delta: String
        if current.hasPrefix(emitted) {
            let startIndex = current.index(current.startIndex, offsetBy: emitted.count)
            delta = String(current[startIndex...])
        } else {
            delta = current
        }
        emitted = current
        return delta
    }

    private func makeResponseSegments(content: String, reasoning: String) -> [Transcript.Segment] {
        var segments: [Transcript.Segment] = []
        if !reasoning.isEmpty {
            segments.append(.reasoning(.init(id: UUID().uuidString, content: reasoning)))
        }
        if !content.isEmpty || segments.isEmpty {
            segments.append(.text(.init(id: UUID().uuidString, content: content)))
        }
        return segments
    }

}

#endif
