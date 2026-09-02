# Magpie

An on-device, multilingual text-to-speech app for iOS built with SwiftUI. Magpie runs CoreML acoustic models together with an MLX-Swift local transformer to synthesize speech entirely on-device, with no network required at runtime.

## Features

- Multilingual synthesis: English, German, Japanese, Mandarin, Hindi, and Spanish
- 5 built-in speakers (John, Sofia, Aria, Jason, Leo)
- CoreML-based text encoder, prefill, step decoder, and Nanocodec audio decoder
- MLX-Swift local transformer for autoregressive token generation
- Classifier-free guidance (CFG) with optional early-cutoff for faster RTF
- Japanese G2P via OpenJTalk; text normalization via NemoTextProcessing
- SwiftUI UI with progress reporting and audio sharing

## Requirements

- Xcode 16.0+
- iOS 17.0+ deployment target
- Apple Silicon Mac for building
- [xcodegen](https://github.com/yonoz/XcodeGen) to generate the Xcode project from `project.yml`

## Project Layout

```
Magpie/
├── project.yml                      # xcodegen spec
├── MagpieTTSApp/                    # SwiftUI app target (ContentView, ViewModel)
├── Sources/MagpieTTS/               # Core library (tokenizers, transformer, bridges)
├── COpenJTalk/                      # C bridging headers for OpenJTalk
├── OpenJTalk.xcframework/           # Prebuilt OpenJTalk binary (bundled)
├── NemoTextProcessing.xcframework/  # Prebuilt NemoTextProcessing binary (bundled)
```

## Prerequisites

### 1. NemoTextProcessing.xcframework

`NemoTextProcessing.xcframework` (English text normalization, compiled from
[text-processing-rs](https://github.com/FluidInference/text-processing-rs)) is checked into
this repo so the package resolves from a plain `git clone` / Swift Package Manager URL. To
rebuild it, follow that repo's build instructions and replace the directory at the project
root, keeping this layout:

```
Magpie/
└── NemoTextProcessing.xcframework/
    ├── Info.plist
    ├── ios-arm64/
    ├── ios-arm64-simulator/
    └── macos-arm64_x86_64/
```

`Sources/MagpieTTS/NemoTextProcessing.swift` is the Swift wrapper over its C API and is
maintained here alongside it.

### 2. Using MagpieTTS as a package

```swift
.package(url: "https://github.com/smdesai/Magpie", from: "1.0.0")
// target dependency:
.product(name: "MagpieTTS", package: "Magpie")
```

### 3. CoreML Models, Constants, and OpenJTalk.xcframework

The models and constants are downloaded from:

> https://huggingface.co/smdesai/magpie-tts-multilingual-357m-coreml

## Build & Run

```bash
# 1. Ensure NemoTextProcessing.xcframework is in the project root
# 2. Generate the Xcode project
xcodegen generate

# 4. Open and run
open MagpieTTS.xcodeproj
```

Select the `MagpieTTS` scheme and run on an iOS 17+ device or simulator.

## Usage

1. Launch the app.
2. Select a speaker, language, and generation options (temperature, top-k, CFG scale).
3. Enter text and tap **Generate**.
4. Play back the synthesized audio or share the resulting WAV.

## Custom Pronunciation (IPA Override)

Wrap a word in vertical bars to bypass the phoneme dictionary and feed IPA symbols straight to the tokenizer, matching NeMo's `IPATokenizer` convention:

```swift
let result = try await tts.generate(
    text: "I say |təˈmɑtoʊ|, you say |təˈmeɪtoʊ|."
)
```

Write the IPA as a single contiguous string with **no internal spaces** — each Swift character becomes one
token, mirroring how a dictionary entry is stored (e.g. `TOMATO → ['t','ə','ˈ','m','e','ɪ','ˌ','t','o','ʊ']`). Spaces inside the bars get emitted as word-separator tokens and confuse the model.

Supported in all IPA-based tokenizers: English, Spanish, German, French, Italian, Vietnamese, Hindi. Not
supported for Mandarin or Japanese (pinyin/katakana pipelines). Valid symbols are those listed as keys in the corresponding `*_token2id.json` (individual IPA characters plus stress markers `ˈ` and `ˌ`).

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) (≥ 0.30.6) — resolved via Swift Package Manager
- OpenJTalk (vendored as `OpenJTalk.xcframework`)
- NemoTextProcessing (build from [text-processing-rs](https://github.com/FluidInference/text-processing-rs))
