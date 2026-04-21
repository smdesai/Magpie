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
├── NemoTextProcessing.xcframework/  # Text-processing framework (you compile this)
├── models/                          # CoreML models (*.mlmodelc, you provide)
└── constants/                       # Tokenizers, phoneme dicts, speaker embeddings
```

## Prerequisites

### 1. NemoTextProcessing.xcframework

`NemoTextProcessing.xcframework` is **not** checked into this repo. You must build it yourself from:

> https://github.com/FluidInference/text-processing-rs

Follow that repo's build instructions to produce `NemoTextProcessing.xcframework`, then place it at the root of this project so the final layout is:

```
Magpie/
└── NemoTextProcessing.xcframework/
    ├── Info.plist
    ├── ios-arm64/
    ├── ios-arm64-simulator/
    └── macos-arm64_x86_64/
```

The Xcode project links this framework by path (see `project.yml`), so it must live in the current directory alongside `project.yml`.

### 2. CoreML Models, Constants, and OpenJTalk.xcframework

The `models/` directory, the `constants/` directory, and `OpenJTalk.xcframework` can all be obtained by following the build instructions at:

> https://github.com/smdesai/MagpieTTS

After building, place them at the project root so the layout is:

```
Magpie/
├── models/
│   ├── TextEncoder.mlmodelc/
│   ├── DecoderPrefill.mlmodelc/
│   ├── DecoderStep.mlmodelc/
│   └── NanocodecDecoder.mlmodelc/
├── constants/
│   ├── *.json                  # tokenizers, phoneme dicts, speaker info
│   ├── *.npy                   # speaker & audio embeddings
│   ├── local_transformer/      # MLX weights
│   └── open_jtalk_dic/         # OpenJTalk dictionary
└── OpenJTalk.xcframework/
    ├── Info.plist
    ├── ios-arm64/
    ├── ios-arm64-simulator/
    └── macos-arm64/
```

The CoreML models and the `constants/` folder are bundled as app resources via `project.yml`. The app will fail to launch synthesis if any of them are missing.

## Build & Run

```bash
# 1. Ensure NemoTextProcessing.xcframework is in the project root
# 2. Ensure the four *.mlmodelc bundles are in models/
# 3. Generate the Xcode project
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
