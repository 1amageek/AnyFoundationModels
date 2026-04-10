# AnyFoundationModels

A unified Swift framework for interacting with multiple language model providers through a single, protocol-based API.

Write against the `LanguageModel` protocol and use **Ollama**, **Claude**, **OpenAI Responses API**, and **on-device MLX** — side by side in the same application.

## Features

- **Unified `LanguageModel` protocol** — one interface for all providers
- **Structured output** via `@Generable` macro — get typed Swift structs from LLM responses
- **Tool calling** with automatic execution loop
- **Streaming** responses with partial content parsing
- **Swift 6 concurrency** — fully `Sendable`, `async`/`await` native
- **Trait-gated compilation** — keep package size small by compiling only the backends you need

## Requirements

- Swift 6.2+
- macOS 15+ / iOS 18+ / visionOS 2+

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/1amageek/AnyFoundationModels.git", from: "0.2.0"),
]
```

Enable the backends you need via traits. Each backend compiles only when its trait is active, keeping your binary small:

```swift
targets: [
    .executableTarget(
        name: "MyApp",
        dependencies: [
            .product(name: "OpenFoundationModels", package: "AnyFoundationModels"),
            .product(name: "ClaudeFoundationModels", package: "AnyFoundationModels",
                     condition: .when(traits: ["Claude"])),
            .product(name: "ResponseFoundationModels", package: "AnyFoundationModels",
                     condition: .when(traits: ["Response"])),
            .product(name: "OllamaFoundationModels", package: "AnyFoundationModels",
                     condition: .when(traits: ["Ollama"])),
            .product(name: "MLXFoundationModels", package: "AnyFoundationModels",
                     condition: .when(traits: ["MLX"])),
        ]
    )
]
```

Build with the traits you need:

```bash
swift build --traits Claude,Response,Ollama
```

## Quick Start

### Text Generation

```swift
import OpenFoundationModels
import ClaudeFoundationModels

let model = ClaudeLanguageModel(
    configuration: ClaudeConfiguration(apiKey: "sk-..."),
    modelName: "claude-sonnet-4-5-20250929"
)
let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "Explain quantum computing in one sentence.")
print(response.content)
```

### Structured Output

Use the `@Generable` macro to get typed Swift structs directly from LLM responses:

```swift
import OpenFoundationModels
import OpenFoundationModelsMacros

@Generable
struct Recipe {
    @Guide(description: "Name of the recipe")
    var name: String
    @Guide(description: "List of ingredients")
    var ingredients: [String]
    @Guide(description: "Step-by-step instructions")
    var steps: [String]
    @Guide(description: "Estimated cooking time in minutes")
    var cookingTimeMinutes: Int
}

let response = try await session.respond(
    to: "Give me a recipe for pancakes",
    generating: Recipe.self
)
print(response.content.name)           // "Classic Pancakes"
print(response.content.ingredients)    // ["flour", "eggs", ...]
```

### Tool Calling

Define tools that the model can invoke automatically:

```swift
import OpenFoundationModels
import OpenFoundationModelsMacros

@Generable
struct WeatherArguments {
    @Guide(description: "City name")
    var city: String
}

struct WeatherTool: Tool {
    var description: String { "Get current weather for a city" }

    func call(arguments: WeatherArguments) async throws -> String {
        return "Sunny, 24°C in \(arguments.city)"
    }
}

let session = LanguageModelSession(
    model: model,
    tools: [WeatherTool()],
    instructions: "You are a helpful assistant."
)

let response = try await session.respond(to: "What's the weather in Tokyo?")
// The session automatically executes the tool and returns the final answer
```

### Streaming

```swift
let stream = session.streamResponse(to: "Write a short story about a cat.")

for try await snapshot in stream {
    print(snapshot.content, terminator: "")
}
```

Streaming also works with structured output:

```swift
let stream = session.streamResponse(
    to: "Give me a recipe for miso soup",
    generating: Recipe.self
)

for try await snapshot in stream {
    // snapshot.content is Recipe.PartiallyGenerated
    print(snapshot.content)
}

let final = try await stream.collect()
print(final.content.name) // fully parsed Recipe
```

## Backends

### Claude (Anthropic API)

```swift
import ClaudeFoundationModels

let config = ClaudeConfiguration(apiKey: "sk-...")
// Or from environment: ClaudeConfiguration.fromEnvironment()

let model = ClaudeLanguageModel(
    configuration: config,
    modelName: "claude-sonnet-4-5-20250929",
    thinkingBudgetTokens: 10000  // optional: enable extended thinking
)
```

### OpenAI Responses API

```swift
import ResponseFoundationModels

let config = ResponseConfiguration(apiKey: "sk-...")
let model = ResponseLanguageModel(configuration: config, model: "gpt-4.1")
```

### Ollama (Local)

```swift
import OllamaFoundationModels

let config = OllamaConfiguration()  // defaults to http://127.0.0.1:11434, temperature: 0
let model = OllamaLanguageModel(configuration: config, modelName: "llama3.2")
```

### MLX (On-Device)

```swift
import MLXFoundationModels

let factory = MLXLanguageModelFactory()
let model = try await factory.makeLanguageModel(
    descriptor: .nanbeige41_3B_8bit
)
```

Custom Hugging Face models:

```swift
let descriptor = MLXModelDescriptor(
    id: "mlx-community/your-model-4bit",
    promptStyle: .llama3
)
let model = try await factory.makeLanguageModel(descriptor: descriptor)
```

## Using Multiple Backends

All backends conform to `LanguageModel`, so you can use them side by side in the same application:

```swift
import ClaudeFoundationModels
import OllamaFoundationModels

let claude = ClaudeLanguageModel(
    configuration: ClaudeConfiguration(apiKey: "sk-..."),
    modelName: "claude-sonnet-4-5-20250929"
)
let ollama = OllamaLanguageModel(
    configuration: OllamaConfiguration(),
    modelName: "llama3.2"
)

// Use different models for different tasks
let session1 = LanguageModelSession(model: claude, instructions: "You are a translator.")
let session2 = LanguageModelSession(model: ollama, instructions: "You are a code reviewer.")

async let translation = session1.respond(to: "Translate to Japanese: Hello")
async let review = session2.respond(to: "Review this function: ...")
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your Application                    │
├─────────────────────────────────────────────────────┤
│              LanguageModelSession                     │
│         (tool loop, transcript, streaming)            │
├─────────────────────────────────────────────────────┤
│              LanguageModel Protocol                   │
│        generate() / stream() / isAvailable            │
├──────────┬──────────┬──────────┬────────────────────┤
│  Claude  │ Response │  Ollama  │        MLX          │
│  (API)   │  (API)   │ (Local)  │    (On-Device)      │
└──────────┴──────────┴──────────┴────────────────────┘
```

### Module Structure

| Module | Description |
|--------|-------------|
| `OpenFoundationModelsCore` | Core protocols: `Generable`, `GeneratedContent`, `GenerationSchema` |
| `OpenFoundationModels` | `LanguageModel`, `LanguageModelSession`, `Transcript`, `Tool` |
| `OpenFoundationModelsMacros` | `@Generable` and `@Guide` macros |
| `OpenFoundationModelsExtra` | Internal accessors for backend implementations |
| `ClaudeFoundationModels` | Anthropic Claude API backend |
| `ResponseFoundationModels` | OpenAI Responses API backend |
| `OllamaFoundationModels` | Ollama local server backend |
| `MLXFoundationModels` | Apple MLX on-device inference backend |

## License

MIT
