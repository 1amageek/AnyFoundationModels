# CLAUDE.md

## Build Commands

```bash
# Build per-trait
swift build --traits Ollama
swift build --traits Claude
swift build --traits Response
swift build --traits MLX
swift build --traits Metal

# Test per-trait
swift test --traits Ollama
swift test --traits Claude
swift test --traits Response
swift test --traits MLX
swift test --traits Metal
```

## Release Verification Policy

swift-lm / Metal backend に影響するリリースは、build と通常テストだけでは完了扱いにしない。

- `Package.swift` の `swift-lm` 更新、`MetalFoundationModels` の変更、Metal 推論に関係する dependency 更新では、必ず実モデル出力を確認してから tag / GitHub Release を作成する。
- 実モデル確認は `MetalModelLoader` と `MetalLanguageModel` を通し、AnyFoundationModels の adapter 経由で短い deterministic prompt を実行する。
- `swift-lm` upstream の release note や upstream test 結果は補助情報であり、AnyFoundationModels 側の実モデル確認の代替にしない。
- 実モデル確認では、使用した model ID または local snapshot path、実行コマンド、出力、成功/失敗を release note または作業ログに残す。
- モデルの download、local cache、Metal runtime、出力品質のいずれかを確認できない場合は release を止め、原因を報告する。
- 既定の heavy test は通常テストからは外し、明示的な環境変数で実行する。現在の release smoke は次の形式で実行する:

```bash
ANYFM_REAL_MODEL_SMOKE=1 \
ANYFM_REAL_MODEL_PATH="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/<snapshot>" \
swift test --traits Metal --filter MetalRealModelSmokeTests
```

## Architecture

5 backend modules that bridge OpenFoundationModels API to external LLM providers:

| Module | Provider | Trait |
|--------|----------|-------|
| OllamaFoundationModels | Ollama REST API | `Ollama` |
| ClaudeFoundationModels | Anthropic Messages API | `Claude` |
| ResponseFoundationModels | OpenAI Responses API | `Response` |
| MLXFoundationModels | MLX local inference | `MLX` |
| MetalFoundationModels | swift-lm Metal inference | `Metal` |

All backends depend on `OpenFoundationModels` and `OpenFoundationModelsExtra`.

## Type Conversion Policy

### Canonical JSON Type: `JSONValue`

`JSONValue`（mattt/JSONSchema package）が JSON データの唯一の型。`[String: Any]` を中間表現として使用することを禁止する。

- `OpenFoundationModelsExtra` が `@_exported import JSONSchema` しているため、各バックエンドは `import OpenFoundationModelsExtra` するだけで `JSONValue` と `JSONSchema` の両方が使える

### Type Relationships

```
JSONSchema package (mattt/JSONSchema)
├── JSONSchema  ... schema definition (object, array, string, etc.)
│                   object の properties は OrderedDictionary<String, JSONSchema>
│                   default/examples/enum/const は全て JSONValue 型
└── JSONValue   ... typed JSON data
                    .null, .bool(Bool), .int(Int), .double(Double),
                    .string(String), .array([JSONValue]), .object([String: JSONValue])
                    Codable, Hashable, Sendable

GeneratedContent (OpenFoundationModelsCore)
├── .jsonString: String      ... JSON string を返す computed property
└── init(json: String)       ... JSON string から生成（partial JSON 対応）
    ※ 現在 fileprivate enum JSONValue を内部に持っているが、
      mattt/JSONSchema の JSONValue に移行予定（下記参照）

OpenFoundationModelsExtra
├── @_exported import JSONSchema  ... JSONValue と JSONSchema を re-export
└── GenerationSchema._jsonSchema  ... schema dictionary → JSONSchema
```

### GeneratedContent の JSONValue 統合方針

GeneratedContent は現在独自の `fileprivate enum JSONValue` を内部に持っている。これを mattt/JSONSchema の `JSONValue` に統合する:

**現状の差分:**

| | GeneratedContent (内部) | mattt/JSONSchema |
|---|---|---|
| number | `.number(Double)` 統一 | `.int(Int)` / `.double(Double)` 分離 |
| object | `.object([String: JSONValue], orderedKeys: [String])` | `.object([String: JSONValue])` |

**移行方針:**
- OpenFoundationModelsCore が JSONSchema package に依存する
- `fileprivate enum JSONValue` を削除し、mattt/JSONSchema の `JSONValue` を使う
- `orderedKeys` は別の仕組みで管理する（JSONSchema の `JSONSchema.object` は `OrderedDictionary` を使用済み）
- int/double の区別は mattt/JSONSchema に合わせる（精度向上）

### Conversion Paths

`GeneratedContent` は `jsonString` を持っている。`JSONValue` は `Codable`。この2つの性質だけで相互変換できる:

```swift
// GeneratedContent → JSONValue
//   jsonString で JSON 文字列を取得し、JSONValue としてデコード
let data = content.jsonString.data(using: .utf8)!
let value = try JSONDecoder().decode(JSONValue.self, from: data)

// JSONValue → GeneratedContent
//   JSONValue をエンコードし、その文字列から GeneratedContent を生成
let data = try JSONEncoder().encode(jsonValue)
let content = try GeneratedContent(json: String(data: data, encoding: .utf8)!)
```

### Prohibited Patterns

- **`[String: Any]` as intermediate representation** — type safety が失われるため禁止。`JSONValue` を使う
- **`JSONSerialization.jsonObject`** — `Any` を返すため禁止。`JSONDecoder().decode(JSONValue.self, ...)` を使う
- **Backend-local JSON value type** — mattt/JSONSchema の `JSONValue` のみを使う。独自定義を作らない

### Shared Helpers in OpenFoundationModelsExtra

Backend 間で重複するロジックは OpenFoundationModelsExtra に配置する:

1. **Transcript extraction** — `extractText`, `extractToolDefinitions`, `extractOptions` etc.
2. **GeneratedContent ↔ JSONValue conversion** — `jsonString` / `init(json:)` 経由
3. **Entry builder helpers** — tool call entry construction

## Naming Conventions

- Use `ID` suffix (not `Id`): `sessionID`, `toolCallID`
- Backend module names: `{Provider}FoundationModels`
