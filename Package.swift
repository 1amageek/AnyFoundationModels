// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "AnyFoundationModels",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .macCatalyst(.v26),
        .visionOS(.v26)
    ],
    products: [
        // Backends
        .library(name: "OllamaFoundationModels", targets: ["OllamaFoundationModels"]),
        .library(name: "ClaudeFoundationModels", targets: ["ClaudeFoundationModels"]),
        .library(name: "ResponseFoundationModels", targets: ["ResponseFoundationModels"]),
        .library(name: "MLXFoundationModels", targets: ["MLXFoundationModels"]),
        .library(name: "MetalFoundationModels", targets: ["MetalFoundationModels"]),
    ],
    traits: [
        .trait(name: "Ollama"),
        .trait(name: "Claude"),
        .trait(name: "Response"),
        .trait(name: "MLX"),
        .trait(name: "Metal"),
        .default(enabledTraits: []),
    ],
    dependencies: [
        // Core API
        .package(path: "../OpenFoundationModels"),
        // Claude
        .package(url: "https://github.com/apple/swift-configuration.git", from: "1.2.0"),
        // Metal
        .package(path: "../swift-lm"),
        // MLX
        .package(path: "../mlx-swift-lm"),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.3.0")
        ),
    ],
    targets: [
        // ===== Backend Targets =====
        .target(
            name: "OllamaFoundationModels",
            dependencies: [
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Ollama"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Ollama"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("OLLAMA_ENABLED", .when(traits: ["Ollama"])),
            ]
        ),
        .target(
            name: "ClaudeFoundationModels",
            dependencies: [
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Claude"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Claude"])),
                .product(name: "Configuration", package: "swift-configuration",
                         condition: .when(traits: ["Claude"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("CLAUDE_ENABLED", .when(traits: ["Claude"])),
            ]
        ),
        .target(
            name: "ResponseFoundationModels",
            dependencies: [
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Response"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Response"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("RESPONSE_ENABLED", .when(traits: ["Response"])),
            ]
        ),
        .target(
            name: "MLXFoundationModels",
            dependencies: [
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         moduleAliases: ["Generation": "FoundationGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         moduleAliases: ["Generation": "FoundationGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "MLXLLM", package: "mlx-swift-lm",
                         moduleAliases: ["Generation": "TransformersGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "MLXVLM", package: "mlx-swift-lm",
                         moduleAliases: ["Generation": "TransformersGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm",
                         moduleAliases: ["Generation": "TransformersGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "Hub", package: "swift-transformers",
                         moduleAliases: ["Generation": "TransformersGeneration"],
                         condition: .when(traits: ["MLX"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("MLX_ENABLED", .when(traits: ["MLX"])),
            ]
        ),

        .target(
            name: "MetalFoundationModels",
            dependencies: [
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Metal"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Metal"])),
                .product(name: "SwiftLM", package: "swift-lm",
                         condition: .when(traits: ["Metal"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("METAL_ENABLED", .when(traits: ["Metal"])),
            ]
        ),

        // ===== Tests =====
        .testTarget(
            name: "OllamaFoundationModelsTests",
            dependencies: [
                .target(name: "OllamaFoundationModels", condition: .when(traits: ["Ollama"])),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Ollama"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Ollama"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("OLLAMA_ENABLED", .when(traits: ["Ollama"])),
            ]
        ),
        .testTarget(
            name: "ClaudeFoundationModelsTests",
            dependencies: [
                .target(name: "ClaudeFoundationModels", condition: .when(traits: ["Claude"])),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Claude"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Claude"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("CLAUDE_ENABLED", .when(traits: ["Claude"])),
            ]
        ),
        .testTarget(
            name: "ResponseFoundationModelsTests",
            dependencies: [
                .target(name: "ResponseFoundationModels", condition: .when(traits: ["Response"])),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         condition: .when(traits: ["Response"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         condition: .when(traits: ["Response"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("RESPONSE_ENABLED", .when(traits: ["Response"])),
            ]
        ),
        .testTarget(
            name: "MLXFoundationModelsTests",
            dependencies: [
                .target(name: "MLXFoundationModels", condition: .when(traits: ["MLX"])),
                .product(name: "OpenFoundationModels", package: "OpenFoundationModels",
                         moduleAliases: ["Generation": "FoundationGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "OpenFoundationModelsExtra", package: "OpenFoundationModels",
                         moduleAliases: ["Generation": "FoundationGeneration"],
                         condition: .when(traits: ["MLX"])),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm",
                         moduleAliases: ["Generation": "TransformersGeneration"],
                         condition: .when(traits: ["MLX"])),
            ],
            swiftSettings: [
                .swiftLanguageMode(.v6),
                .define("MLX_ENABLED", .when(traits: ["MLX"])),
            ]
        ),
    ]
)
