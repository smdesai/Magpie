//
//  MagpieTTS.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import Accelerate
import CoreML
import Foundation
import MLX
import os

// MARK: - Public Types

public enum MagpieTTSError: LocalizedError {
    case modelNotFound(String)
    case constantsNotFound(String)
    case invalidConfiguration(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let n): return "CoreML model not found: \(n)"
        case .constantsNotFound(let p): return "Constants not found: \(p)"
        case .invalidConfiguration(let m): return "Invalid configuration: \(m)"
        case .generationFailed(let m): return "Generation failed: \(m)"
        }
    }
}

public struct GenerationOptions: Sendable {
    public var speaker: Int
    public var temperature: Float
    public var topK: Int
    public var maxSteps: Int
    public var useCFG: Bool
    public var cfgScale: Float
    /// When set, CFG is only used for the first N generation steps, then switches to
    /// CFG-free (single decoder call per step) for the remainder. Reduces RTF by ~2x
    /// after the cutoff while preserving quality from early CFG guidance.
    /// `nil` means CFG runs for all steps (default behavior).
    public var cfgSteps: Int?
    public var seed: UInt64

    public init(
        speaker: Int = 0, temperature: Float = 0.6, topK: Int = 80,
        maxSteps: Int = 500, useCFG: Bool = true, cfgScale: Float = 2.5,
        cfgSteps: Int? = nil, seed: UInt64 = 42
    ) {
        self.speaker = speaker
        self.temperature = temperature
        self.topK = topK
        self.maxSteps = maxSteps
        self.useCFG = useCFG
        self.cfgScale = cfgScale
        self.cfgSteps = cfgSteps
        self.seed = seed
    }
}

public struct GenerationProgress: Sendable {
    public enum Phase: Sendable {
        case loadingModels
        case encodingText
        case prefillingContext(step: Int, total: Int)
        case generating(step: Int, maxSteps: Int)
        case decodingAudio
    }
    public let phase: Phase
}

public struct GenerationResult: Sendable {
    public let audioSamples: [Float]
    public let sampleRate: Int
    public let framesGenerated: Int
    public let generationTimeSeconds: Double
    /// Breakdown: time spent in decoder_step calls (seconds)
    public let decoderTimeSeconds: Double
    /// Breakdown: time spent in nanocodec decode calls (seconds)
    public let codecTimeSeconds: Double
    /// Number of decoder_step predictions made (conditional + unconditional)
    public let decoderCallCount: Int
    /// Breakdown: time spent in local transformer sampling (seconds)
    public let localTransformerTimeSeconds: Double
    /// Which local transformer backend was used
    public let localTransformerBackend: String
    /// Time to first audio chunk (streaming only, 0 for non-streaming)
    public let timeToFirstAudioSeconds: Double

    public var durationSeconds: Double { Double(audioSamples.count) / Double(sampleRate) }

    /// Encode as 16-bit PCM WAV.
    public var wavData: Data { encodeWAV(samples: audioSamples, sampleRate: sampleRate) }

    /// Log generation performance to console
    public func logPerformance(label: String = "generate") {
        let dur = String(format: "%.2f", durationSeconds)
        let gen = String(format: "%.2f", generationTimeSeconds)
        let rtfx = String(format: "%.2f", durationSeconds / generationTimeSeconds)
        let dec = String(format: "%.2f", decoderTimeSeconds)
        let lt = String(format: "%.2f", localTransformerTimeSeconds)
        let codec = String(format: "%.2f", codecTimeSeconds)
        let ltPct = String(
            format: "%.1f", localTransformerTimeSeconds / generationTimeSeconds * 100)
        let ttfa =
            timeToFirstAudioSeconds > 0 ? String(format: "%.2f", timeToFirstAudioSeconds) : "-"
        print(
            """
            [MagpieTTS:\(label)] \(dur)s audio in \(gen)s (\(rtfx)x RTFx) | TTFA: \(ttfa)s
              Decoder: \(dec)s (\(decoderCallCount) calls) | LT: \(lt)s (\(ltPct)%) [\(localTransformerBackend)] | Codec: \(codec)s
              Frames: \(framesGenerated) | Per-frame LT: \(String(format: "%.2f", localTransformerTimeSeconds / Double(max(framesGenerated, 1)) * 1000))ms
            """)
    }
}

// MARK: - Internal Config

struct ModelConfig {
    let numCodebooks: Int
    let vocabSize: Int
    let sampleRate: Int
    let codecSamplesPerFrame: Int
    let audioBosId: Int
    let audioEosId: Int
    let dModel: Int
    let nLayers: Int
    let nHeads: Int
    let dHead: Int
    let minGeneratedFrames: Int

    init(json: [String: Any]) throws {
        guard let dec = json["decoder"] as? [String: Any],
            let st = json["special_tokens"] as? [String: Any],
            let inf = json["inference"] as? [String: Any],
            let numCodebooks = json["num_audio_codebooks"] as? Int,
            let vocabSize = json["num_all_tokens_per_codebook"] as? Int,
            let sampleRate = json["output_sample_rate"] as? Int,
            let codecSamplesPerFrame = json["codec_samples_per_frame"] as? Int,
            let audioBosId = st["audio_bos_id"] as? Int,
            let audioEosId = st["audio_eos_id"] as? Int,
            let dModel = dec["d_model"] as? Int,
            let nLayers = dec["n_layers"] as? Int,
            let nHeads = dec["sa_n_heads"] as? Int
        else {
            throw MagpieTTSError.invalidConfiguration("constants.json: missing or malformed fields")
        }
        self.numCodebooks = numCodebooks
        self.vocabSize = vocabSize
        self.sampleRate = sampleRate
        self.codecSamplesPerFrame = codecSamplesPerFrame
        self.audioBosId = audioBosId
        self.audioEosId = audioEosId
        self.dModel = dModel
        self.nLayers = nLayers
        self.nHeads = nHeads
        self.dHead = dModel / nHeads
        self.minGeneratedFrames = (inf["min_generated_frames"] as? Int) ?? 4
    }
}

// MARK: - MagpieTTS

/// On-device speech synthesis using CoreML.
///
/// Usage:
/// ```swift
/// let tts = MagpieTTS(modelDirectory: modelsURL)
/// try await tts.prepare()
/// let result = try await tts.generate(text: "Hello, world!")
/// try result.wavData.write(to: outputURL)
/// ```
public final class MagpieTTS {

    private let modelDirectory: URL
    private let constantsDirectory: URL
    private let buildDirectory: URL
    private let computeUnits: MLComputeUnits
    private var multiTokenizer: MultiLanguageTokenizer?

    // Loaded state
    private var textEncoder: MLModel?
    private var decoderStep: MLModel?
    private var decoderPrefill: MLModel?  // Optional batched prefill (110x faster than step loop)
    private var nanocodec: MLModel?
    private var config: ModelConfig?
    private var speakerEmbeddings: [[Float]]?  // [speaker] → flat (T_ctx * dModel)
    private var speakerContextLength = 0
    private var audioEmbeddings: [[Float]]?  // [codebook] → flat (vocabSize * dModel)
    private var audioEmbeddingsMLX: [MLXArray]?  // [codebook] → (vocabSize, dModel) on GPU
    private var ltWeights: LocalTransformerWeights?
    private var ltWeightsMLX: LocalTransformerWeightsMLX?
    /// Toggle between MLX (true) and Accelerate (false) for local transformer.
    /// Accelerate is ~2.5x faster for this workload (small 256×256 tensors).
    public var useMLXLocalTransformer = false

    // Cached unconditional CFG prefill (text/speaker-independent, computed once)
    private var cachedUncondCaches: [String: MLMultiArray]?
    private var cachedUncondPositions: [String: MLMultiArray]?

    /// Serializes the lazy-load paths so concurrent generate() callers don't
    /// double-load constants/models or observe partially-mutated state.
    /// Once a loader has run, the loaded properties become write-once.
    private let loadLock = OSAllocatedUnfairLock<Void>(initialState: ())

    // Decoder step I/O key mappings set by convert/convert_decoder_step.py.
    private static let cacheOutKeys: [String] = (0 ..< 12).map { "new_cache\($0)" }
    private static let posOutKeys: [String] = (0 ..< 12).map { "new_position\($0)" }

    /// Initialize with a model directory containing `build/` and `constants/` subdirectories.
    public init(modelDirectory: URL, computeUnits: MLComputeUnits = .cpuAndGPU) {
        self.modelDirectory = modelDirectory
        self.constantsDirectory = modelDirectory.appendingPathComponent("constants")
        self.buildDirectory = modelDirectory.appendingPathComponent("build")
        self.computeUnits = computeUnits
    }

    /// Initialize with explicit paths to constants and compiled model directories.
    /// Use this when loading from an app bundle where resources are in separate locations.
    public init(
        constantsDirectory: URL, buildDirectory: URL, computeUnits: MLComputeUnits = .cpuAndGPU
    ) {
        self.modelDirectory = constantsDirectory.deletingLastPathComponent()
        self.constantsDirectory = constantsDirectory
        self.buildDirectory = buildDirectory
        self.computeUnits = computeUnits
    }

    /// Pre-load all models, constants, and tokenizer. Optional — `generate` will load lazily.
    /// Also pre-computes the unconditional CFG prefill cache (speaker/text-independent).
    public func prepare() async throws {
        try loadLock.withLock { _ in
            if config == nil { try loadConstants() }
            if textEncoder == nil { try loadCoreMLModels() }
            if multiTokenizer == nil { try loadTokenizer() }
        }
        try precomputeUncondPrefill()
    }

    /// Load speaker names from speaker_info.json. Returns array of (index, name) pairs.
    public static func loadSpeakerInfo(constantsDirectory: URL) -> [(index: Int, name: String)] {
        let siURL = constantsDirectory.appendingPathComponent("speaker_info.json")
        guard FileManager.default.fileExists(atPath: siURL.path),
            let data = try? Data(contentsOf: siURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let numSpeakers = json["num_speakers"] as? Int
        else {
            return [(0, "Default")]
        }
        if let names = json["names"] as? [String: String] {
            return (0 ..< numSpeakers).map { i in
                (i, names["\(i)"] ?? "Speaker \(i)")
            }
        }
        return (0 ..< numSpeakers).map { ($0, "Speaker \($0)") }
    }

    /// Generate speech from text in any supported language.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - language: Target language (default: `.english`).
    ///   - options: Generation parameters (speaker, temperature, CFG, etc.).
    ///   - progress: Optional progress callback. **Invoked on a background task
    ///     context** — dispatch to `MainActor` if updating SwiftUI state.
    /// - Returns: Audio samples, sample rate, and WAV data.
    public func generate(
        text: String,
        language: Language = .english,
        options: GenerationOptions = .init(),
        progress: ((GenerationProgress) -> Void)? = nil
    ) async throws -> GenerationResult {
        let tokenIDs = try requireTokenizer().tokenize(text, language: language)
        return try await generate(tokenIDs: tokenIDs, options: options, progress: progress)
    }

    /// Generate speech from text token IDs.
    ///
    /// - Parameters:
    ///   - tokenIDs: Text token IDs from the NeMo tokenizer.
    ///   - options: Generation parameters (speaker, temperature, CFG, etc.).
    ///   - progress: Optional callback for UI updates.
    /// - Returns: Audio samples, sample rate, and WAV data.
    public func generate(
        tokenIDs: [Int32],
        options: GenerationOptions = .init(),
        progress: ((GenerationProgress) -> Void)? = nil
    ) async throws -> GenerationResult {
        try await generateInternal(
            tokenIDs: tokenIDs, options: options, progress: progress)
    }

    /// Generate speech with streaming audio chunks.
    ///
    /// Calls `onAudioChunk` periodically with new audio samples during generation,
    /// enabling playback to start before generation completes.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - language: Target language.
    ///   - options: Generation parameters.
    ///   - chunkFrames: Number of codec frames between audio chunk emissions (default 15 ≈ 0.7s).
    ///   - progress: Optional progress callback. **Invoked on a background task
    ///     context** — dispatch to `MainActor` if updating SwiftUI state.
    ///   - onAudioChunk: Called with `(newSamples, sampleRate)` for each decoded
    ///     chunk. **Invoked on a background task context.** Audio playback engines
    ///     like AVAudioEngine accept buffers from any thread; SwiftUI consumers
    ///     must dispatch to `MainActor` themselves.
    /// - Returns: Complete generation result.
    public func generateStreaming(
        text: String,
        language: Language = .english,
        options: GenerationOptions = .init(),
        chunkFrames: Int = 15,
        progress: ((GenerationProgress) -> Void)? = nil,
        onAudioChunk: @escaping ([Float], Int) -> Void
    ) async throws -> GenerationResult {
        let tokenIDs = try requireTokenizer().tokenize(text, language: language)
        return try await generateInternal(
            tokenIDs: tokenIDs, options: options, chunkFrames: chunkFrames,
            progress: progress, onAudioChunk: onAudioChunk)
    }

    // MARK: - Lazy-Load Helpers

    private func requireTokenizer() throws -> MultiLanguageTokenizer {
        try loadLock.withLock { _ in
            if multiTokenizer == nil { try loadTokenizer() }
            guard let tok = multiTokenizer else {
                throw MagpieTTSError.generationFailed("Tokenizer failed to load")
            }
            return tok
        }
    }

    private func requireConfig() throws -> ModelConfig {
        try loadLock.withLock { _ in
            if config == nil { try loadConstants() }
            guard let cfg = config else {
                throw MagpieTTSError.generationFailed("Config failed to load")
            }
            return cfg
        }
    }

    /// Ensure constants and CoreML models are loaded. Safe to call concurrently —
    /// only one loader runs at a time, and subsequent callers observe the loaded
    /// state.
    private func ensureLoaded() throws {
        try loadLock.withLock { _ in
            if config == nil { try loadConstants() }
            if textEncoder == nil { try loadCoreMLModels() }
        }
    }

    // MARK: - Internal Generation Pipeline

    /// Unified generation pipeline used by both `generate` and `generateStreaming`.
    ///
    /// When `chunkFrames` and `onAudioChunk` are provided, audio is decoded and
    /// emitted incrementally during generation (streaming mode). Otherwise, audio
    /// is decoded once at the end.
    private func generateInternal(
        tokenIDs: [Int32],
        options: GenerationOptions,
        chunkFrames: Int? = nil,
        progress: ((GenerationProgress) -> Void)? = nil,
        onAudioChunk: (([Float], Int) -> Void)? = nil
    ) async throws -> GenerationResult {
        try ensureLoaded()

        guard let cfg = config,
            let encoder = textEncoder,
            let speakers = speakerEmbeddings,
            let audioEmbs = audioEmbeddings,
            let ltW = ltWeights
        else {
            throw MagpieTTSError.generationFailed("Models or constants not loaded")
        }
        let maxTextLen = 256
        let maxSeqLen = 512
        let maxCodecFrames = 256
        let numCb = cfg.numCodebooks
        var rng = SplitMix64(seed: options.seed)

        // ---- 1. Encode text ----
        progress?(GenerationProgress(phase: .encodingText))

        var tokensBuf = [Int32](repeating: 0, count: maxTextLen)
        let tLen = min(tokenIDs.count, maxTextLen)
        for i in 0 ..< tLen { tokensBuf[i] = tokenIDs[i] }

        var maskBuf = [Float](repeating: 0, count: maxTextLen)
        for i in 0 ..< tLen { maskBuf[i] = 1 }

        let tokensArr = try int32Array(tokensBuf, shape: [1, maxTextLen])
        let maskArr = try floatArray(maskBuf, shape: [1, maxTextLen])

        let encResult = try await encoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "text_tokens": MLFeatureValue(multiArray: tokensArr),
                "text_mask": MLFeatureValue(multiArray: maskArr),
            ]))
        guard let encoderOutput = encResult.featureValue(for: "encoder_output")?.multiArrayValue
        else {
            throw MagpieTTSError.generationFailed("TextEncoder missing encoder_output")
        }

        let uncondEncOut: MLMultiArray?
        let uncondMask: MLMultiArray?
        if options.useCFG {
            uncondEncOut = try zeroArray(shape: encoderOutput.shape.map { $0.intValue })
            var um = [Float](repeating: 0, count: maxTextLen)
            um[0] = 1
            uncondMask = try floatArray(um, shape: [1, maxTextLen])
        } else {
            uncondEncOut = nil
            uncondMask = nil
        }

        // ---- 2. Init KV caches ----
        let nL = cfg.nLayers
        let nH = cfg.nHeads
        let dH = cfg.dHead
        let dM = cfg.dModel

        func makeCaches() throws -> ([String: MLMultiArray], [String: MLMultiArray]) {
            var c = [String: MLMultiArray]()
            var p = [String: MLMultiArray]()
            for i in 0 ..< nL {
                c["cache\(i)"] = try zeroArray(shape: [2, 1, maxSeqLen, nH, dH])
                p["position\(i)"] = try floatArray([0], shape: [1])
            }
            return (c, p)
        }

        var (caches, positions) = try makeCaches()

        // ---- 3. Prefill speaker context ----
        guard options.speaker < speakers.count else {
            throw MagpieTTSError.invalidConfiguration(
                "speaker \(options.speaker) out of range (have \(speakers.count))")
        }
        let spkEmb = speakers[options.speaker]
        let tCtx = speakerContextLength
        var uCaches = [String: MLMultiArray]()
        var uPositions = [String: MLMultiArray]()

        if let prefillModel = decoderPrefill {
            progress?(GenerationProgress(phase: .prefillingContext(step: 0, total: tCtx)))

            let spkArr = try floatArray(spkEmb, shape: [1, tCtx, dM])
            let condResult = try await prefillModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "audio_embed": MLFeatureValue(multiArray: spkArr),
                    "encoder_output": MLFeatureValue(multiArray: encoderOutput),
                    "encoder_mask": MLFeatureValue(multiArray: maskArr),
                ]))
            try parsePrefillCaches(
                condResult, into: &caches, positions: &positions, tCtx: tCtx, nLayers: nL)

            if options.useCFG {
                if let cached = cachedUncondCaches, let cachedPos = cachedUncondPositions {
                    uCaches = try copyCaches(cached)
                    uPositions = try copyCaches(cachedPos)
                } else {
                    let zeroEmb = try zeroArray(shape: [1, tCtx, dM])
                    let uncondResult = try await prefillModel.prediction(
                        from: MLDictionaryFeatureProvider(dictionary: [
                            "audio_embed": MLFeatureValue(multiArray: zeroEmb),
                            "encoder_output": MLFeatureValue(multiArray: uncondEncOut!),
                            "encoder_mask": MLFeatureValue(multiArray: uncondMask!),
                        ]))
                    try parsePrefillCaches(
                        uncondResult, into: &uCaches, positions: &uPositions, tCtx: tCtx,
                        nLayers: nL)
                    cachedUncondCaches = try copyCaches(uCaches)
                    cachedUncondPositions = try copyCaches(uPositions)
                }
            }
            progress?(GenerationProgress(phase: .prefillingContext(step: tCtx, total: tCtx)))
        } else {
            if options.useCFG, let cached = cachedUncondCaches,
                let cachedPos = cachedUncondPositions
            {
                uCaches = try copyCaches(cached)
                uPositions = try copyCaches(cachedPos)
            } else if options.useCFG {
                (uCaches, uPositions) = try makeCaches()
            }
            let hasUncondCache = cachedUncondCaches != nil

            for t in 0 ..< tCtx {
                try Task.checkCancellation()
                let slice = Array(spkEmb[t * dM ..< (t + 1) * dM])
                let ctxArr = try floatArray(slice, shape: [1, 1, dM])
                _ = try runDecoder(
                    audio: ctxArr, enc: encoderOutput, mask: maskArr,
                    caches: &caches, positions: &positions)
                if options.useCFG && !hasUncondCache {
                    let zeroCtx = try floatArray(
                        [Float](repeating: 0, count: dM), shape: [1, 1, dM])
                    _ = try runDecoder(
                        audio: zeroCtx, enc: uncondEncOut!, mask: uncondMask!,
                        caches: &uCaches, positions: &uPositions)
                }
                if (t + 1) % 20 == 0 {
                    progress?(
                        GenerationProgress(phase: .prefillingContext(step: t + 1, total: tCtx)))
                }
            }
        }

        // ---- 4. Autoregressive generation ----
        let startTime = CFAbsoluteTimeGetCurrent()
        var currentCodes = [Int32](repeating: Int32(cfg.audioBosId), count: numCb)
        var allPredictions = [[Int32]]()
        var totalDecoderTime: Double = 0
        var totalCodecTime: Double = 0
        var totalLTTime: Double = 0
        var decoderCalls = 0
        var ttfa: Double = 0
        var lastDecodedSampleCount = 0
        var lastDecodedFrameCount = 0

        for step in 0 ..< options.maxSteps {
            try Task.checkCancellation()

            let audioEmbed = try embedAudioCodes(currentCodes)

            let t0 = CFAbsoluteTimeGetCurrent()
            let condHidden = try runDecoder(
                audio: audioEmbed, enc: encoderOutput, mask: maskArr,
                caches: &caches, positions: &positions)
            totalDecoderTime += CFAbsoluteTimeGetCurrent() - t0
            decoderCalls += 1
            let condFloats = readFloats(condHidden)

            let useCFGThisStep =
                options.useCFG && (options.cfgSteps == nil || step < options.cfgSteps!)
            var uncondFloats: [Float]? = nil
            if useCFGThisStep {
                let t1 = CFAbsoluteTimeGetCurrent()
                let uHidden = try runDecoder(
                    audio: audioEmbed, enc: uncondEncOut!, mask: uncondMask!,
                    caches: &uCaches, positions: &uPositions)
                totalDecoderTime += CFAbsoluteTimeGetCurrent() - t1
                decoderCalls += 1
                uncondFloats = readFloats(uHidden)
            }

            let forbidEOS = step < cfg.minGeneratedFrames
            let ltStart = CFAbsoluteTimeGetCurrent()
            let nextCodes: [Int32]
            if useMLXLocalTransformer {
                guard let ltWMLX = ltWeightsMLX else {
                    throw MagpieTTSError.generationFailed(
                        "MLX local transformer weights not loaded")
                }
                nextCodes = localTransformerSampleMLX(
                    decoderHidden: condFloats, weights: ltWMLX,
                    audioEmbeddings: audioEmbs, audioEmbeddingsMLX: audioEmbeddingsMLX,
                    numCodebooks: numCb,
                    temperature: options.temperature, topK: options.topK,
                    forbidEOS: forbidEOS, uncondDecoderHidden: uncondFloats,
                    cfgScale: options.cfgScale, rng: &rng
                )
            } else {
                nextCodes = localTransformerSample(
                    decoderHidden: condFloats, weights: ltW,
                    audioEmbeddings: audioEmbs, numCodebooks: numCb,
                    temperature: options.temperature, topK: options.topK,
                    forbidEOS: forbidEOS, uncondDecoderHidden: uncondFloats,
                    cfgScale: options.cfgScale, rng: &rng
                )
            }
            totalLTTime += CFAbsoluteTimeGetCurrent() - ltStart

            if nextCodes.contains(Int32(cfg.audioEosId)), step >= cfg.minGeneratedFrames { break }

            allPredictions.append(nextCodes)
            currentCodes = nextCodes

            if step % 20 == 0 {
                progress?(
                    GenerationProgress(phase: .generating(step: step, maxSteps: options.maxSteps)))
            }

            // Streaming: periodic decode + emit chunk
            if let chunkFrames = chunkFrames, let onAudioChunk = onAudioChunk {
                let framesSinceLastDecode = allPredictions.count - lastDecodedFrameCount
                if framesSinceLastDecode >= chunkFrames {
                    let ct0 = CFAbsoluteTimeGetCurrent()
                    let chunkAudio = try decodeFrames(
                        allPredictions, numCb: numCb, maxCodecFrames: maxCodecFrames, cfg: cfg)
                    totalCodecTime += CFAbsoluteTimeGetCurrent() - ct0
                    let newSamples = Array(chunkAudio.dropFirst(lastDecodedSampleCount))
                    lastDecodedSampleCount = chunkAudio.count
                    lastDecodedFrameCount = allPredictions.count
                    if !newSamples.isEmpty {
                        if ttfa == 0 { ttfa = CFAbsoluteTimeGetCurrent() - startTime }
                        onAudioChunk(newSamples, cfg.sampleRate)
                    }
                }
            }
        }

        let genTime = CFAbsoluteTimeGetCurrent() - startTime

        guard !allPredictions.isEmpty else {
            throw MagpieTTSError.generationFailed("No audio frames generated")
        }

        // ---- 5. Final decode ----
        progress?(GenerationProgress(phase: .decodingAudio))

        let ct0 = CFAbsoluteTimeGetCurrent()
        let finalAudio = try decodeFrames(
            allPredictions, numCb: numCb, maxCodecFrames: maxCodecFrames, cfg: cfg)
        totalCodecTime += CFAbsoluteTimeGetCurrent() - ct0

        // Emit remaining samples for streaming
        if let onAudioChunk = onAudioChunk {
            let remainingSamples = Array(finalAudio.dropFirst(lastDecodedSampleCount))
            if !remainingSamples.isEmpty {
                onAudioChunk(remainingSamples, cfg.sampleRate)
            }
        }

        // Peak normalize
        var audio = finalAudio
        var peak: Float = 0
        vDSP_maxmgv(audio, 1, &peak, vDSP_Length(audio.count))
        if peak > 0 {
            var scale = Float(0.9) / peak
            vDSP_vsmul(audio, 1, &scale, &audio, 1, vDSP_Length(audio.count))
        }

        let label = chunkFrames != nil ? "streaming" : "generate"
        let result = GenerationResult(
            audioSamples: audio,
            sampleRate: cfg.sampleRate,
            framesGenerated: allPredictions.count,
            generationTimeSeconds: genTime,
            decoderTimeSeconds: totalDecoderTime,
            codecTimeSeconds: totalCodecTime,
            decoderCallCount: decoderCalls,
            localTransformerTimeSeconds: totalLTTime,
            localTransformerBackend: useMLXLocalTransformer ? "MLX" : "Accelerate",
            timeToFirstAudioSeconds: ttfa
        )
        result.logPerformance(label: label)
        return result
    }

    // MARK: - Text Normalization

    /// Normalize text for TTS: convert written forms (numbers, currency, dates, etc.)
    /// to spoken form so the phoneme tokenizer produces natural-sounding output.
    /// Only applies to English for now; other languages pass through unchanged.
    private static func normalizeForTTS(_ text: String, language: Language) -> String {
        guard language == .english else { return text }
        return NemoTextProcessing.tnNormalizeSentence(foldDashesToAscii(text))
    }

    /// Replace Unicode dash variants with ASCII hyphen-minus.
    /// macOS `TextEditor` "smart dashes" rewrite ASCII `-` to U+2013 in date-shaped
    /// inputs (e.g. "2026-05-06" → "2026–05–06"), which then bypasses NeMo's
    /// date taggers since they match `\-` only.
    private static func foldDashesToAscii(_ text: String) -> String {
        var result = text
        for dash in ["\u{2010}", "\u{2011}", "\u{2012}", "\u{2013}", "\u{2014}", "\u{2015}", "\u{2212}"] {
            if result.contains(dash) {
                result = result.replacingOccurrences(of: dash, with: "-")
            }
        }
        return result
    }

    // MARK: - Phoneme Span Masking

    /// Replace `|...|` IPA override spans with unique alphabetic placeholders.
    ///
    /// Placeholders are all-letter words that carry no meaning to the NeMo TN
    /// grammars (so they survive normalization unchanged) and contain no
    /// whitespace or sentence punctuation (so the chunker can't cut them).
    /// The tokenizer sees the restored `|...|` after `unmaskPhonemeSpans`.
    /// Unterminated spans (a stray `|`) are left as literal text.
    private static func maskPhonemeSpans(_ text: String) -> (masked: String, spans: [String]) {
        var spans = [String]()
        var result = ""
        var cursor = text.startIndex
        while cursor < text.endIndex {
            guard let open = text[cursor...].firstIndex(of: "|") else {
                result.append(contentsOf: text[cursor...])
                break
            }
            let afterOpen = text.index(after: open)
            guard let close = text[afterOpen...].firstIndex(of: "|") else {
                result.append(contentsOf: text[cursor...])
                break
            }
            result.append(contentsOf: text[cursor ..< open])
            spans.append(String(text[afterOpen ..< close]))
            result.append(phonemeSpanPlaceholder(index: spans.count - 1))
            cursor = text.index(after: close)
        }
        return (result, spans)
    }

    /// Restore phoneme spans that were hidden by `maskPhonemeSpans`.
    /// Case-insensitive so TN case changes (if any) don't lose the mapping.
    private static func unmaskPhonemeSpans(_ text: String, spans: [String]) -> String {
        guard !spans.isEmpty else { return text }
        var result = text
        for (i, span) in spans.enumerated() {
            result = result.replacingOccurrences(
                of: phonemeSpanPlaceholder(index: i),
                with: "|\(span)|",
                options: .caseInsensitive)
        }
        return result
    }

    /// Build a unique all-letter placeholder for span index `i`.
    private static func phonemeSpanPlaceholder(index: Int) -> String {
        var n = index
        var suffix = ""
        repeat {
            suffix.append(Character(UnicodeScalar(UInt8(97 + n % 26))))
            n /= 26
        } while n > 0
        return "zmgpphon\(suffix)zmgpphon"
    }

    // MARK: - Chunked Generation (Long Text)

    /// Sentence boundary regex.
    /// - Latin: split after `.!?` followed by whitespace and an uppercase letter.
    /// - CJK: split after `。！？`.
    /// Uses NSRegularExpression because Swift Regex literals don't support
    /// lookbehind on this toolchain.
    private static let sentenceBoundary = try! NSRegularExpression(
        pattern: #"(?<=[.!?])\s+(?=[A-Z])|(?<=[。！？])"#)

    /// Split text into individual sentences for chunked generation.
    /// Each sentence is generated independently  to avoid the model hitting EOS
    /// before vocalizing all text in a chunk.
    /// Sentences that exceed the model's text token limit (256) are split further
    /// on clause boundaries (commas) or word boundaries.
    private func chunkText(_ text: String, language: Language) throws -> [String] {
        let maxTokens = 240  // Leave margin below maxTextLen=256

        let sentences = text.components(separatedBy: .newlines)
            .flatMap { line in
                Self.splitOn(line, regex: Self.sentenceBoundary)
            }

        // Second pass: split sentences that exceed the token limit
        var result = [String]()
        for sentence in (sentences.isEmpty ? [text] : sentences) {
            let tokens = try requireTokenizer().tokenize(sentence, language: language)
            if tokens.count <= maxTokens {
                result.append(sentence)
            } else {
                // Try splitting on commas first
                result.append(
                    contentsOf: try splitLongChunk(
                        sentence, language: language, maxTokens: maxTokens))
            }
        }

        return result.isEmpty ? [text] : result
    }

    /// Clause boundary regex (Latin + CJK comma/semicolon).
    /// NSRegularExpression because Swift Regex literals don't support lookbehind here.
    private static let clauseBoundary = try! NSRegularExpression(
        pattern: #"(?<=[,;,;、])\s*"#)

    /// Split `text` at every match of `regex`, trim, and drop empties.
    private static func splitOn(_ text: String, regex: NSRegularExpression) -> [String] {
        let nsText = text as NSString
        let matches = regex.matches(
            in: text, range: NSRange(location: 0, length: nsText.length))
        var parts = [String]()
        var lastEnd = 0
        for match in matches {
            let r = NSRange(location: lastEnd, length: match.range.location - lastEnd)
            let s = nsText.substring(with: r).trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { parts.append(s) }
            lastEnd = match.range.location + match.range.length
        }
        let tail = nsText.substring(from: lastEnd).trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { parts.append(tail) }
        return parts
    }

    /// Split a chunk that exceeds the token limit on clause boundaries, then word boundaries.
    private func splitLongChunk(_ text: String, language: Language, maxTokens: Int) throws
        -> [String]
    {
        let clauses = Self.splitOn(text, regex: Self.clauseBoundary)
        if clauses.count > 1 {
            // Group clauses to fit within token limit
            var grouped = [String]()
            var current = ""
            for clause in clauses {
                let candidate = current.isEmpty ? clause : "\(current) \(clause)"
                let tokens = try requireTokenizer().tokenize(candidate, language: language)
                if tokens.count <= maxTokens {
                    current = candidate
                } else {
                    if !current.isEmpty { grouped.append(current) }
                    current = clause
                }
            }
            if !current.isEmpty { grouped.append(current) }

            // If grouping helped, return; otherwise fall through to word splitting
            let firstGroupTokenCount = try requireTokenizer().tokenize(
                grouped[0], language: language
            ).count
            if grouped.count > 1 || firstGroupTokenCount <= maxTokens {
                // Recursively check each group
                var final = [String]()
                for g in grouped {
                    let tokens = try requireTokenizer().tokenize(g, language: language)
                    if tokens.count <= maxTokens {
                        final.append(g)
                    } else {
                        final.append(
                            contentsOf: try splitOnWords(
                                g, language: language, maxTokens: maxTokens))
                    }
                }
                return final
            }
        }

        return try splitOnWords(text, language: language, maxTokens: maxTokens)
    }

    /// Last-resort split: divide text at word boundaries near the midpoint.
    private func splitOnWords(_ text: String, language: Language, maxTokens: Int) throws -> [String]
    {
        let words = text.split(separator: " ").map(String.init)
        guard words.count > 1 else { return [text] }

        // Binary search for the longest prefix that fits
        var lo = 1
        var hi = words.count
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            let candidate = words[0 ..< mid].joined(separator: " ")
            let tokens = try requireTokenizer().tokenize(candidate, language: language)
            if tokens.count <= maxTokens {
                lo = mid
            } else {
                hi = mid - 1
            }
        }

        let first = words[0 ..< lo].joined(separator: " ")
        let rest = words[lo...].joined(separator: " ")

        if rest.isEmpty { return [first] }

        // Recursively split the remainder
        var result = [first]
        let restTokens = try requireTokenizer().tokenize(rest, language: language)
        if restTokens.count <= maxTokens {
            result.append(rest)
        } else {
            result.append(
                contentsOf: try splitOnWords(rest, language: language, maxTokens: maxTokens))
        }
        return result
    }

    /// Crossfade two audio buffers. Returns combined audio with overlap region blended.
    private static func crossfade(_ a: [Float], _ b: [Float], overlapSamples: Int) -> [Float] {
        guard overlapSamples > 0, a.count >= overlapSamples else {
            return a + b
        }
        let fadeLen = min(overlapSamples, b.count)
        var result = Array(a.dropLast(fadeLen))
        for i in 0 ..< fadeLen {
            let t = Float(i) / Float(fadeLen)
            result.append(a[a.count - fadeLen + i] * (1 - t) + b[i] * t)
        }
        result.append(contentsOf: b.dropFirst(fadeLen))
        return result
    }

    /// Generate speech from long text by splitting into chunks.
    ///
    /// Automatically splits text at sentence boundaries and generates each chunk
    /// independently, crossfading the audio at boundaries.
    public func generateLong(
        text: String,
        language: Language = .english,
        options: GenerationOptions = .init(),
        progress: ((GenerationProgress) -> Void)? = nil
    ) async throws -> GenerationResult {
        _ = try requireTokenizer()  // ensure tokenizer is loaded for chunkText

        // Protect |...| phoneme spans from TN and the chunker.
        let (masked, spans) = Self.maskPhonemeSpans(text)
        // Normalize before chunking so token counts reflect expanded text
        let normalized = Self.normalizeForTTS(masked, language: language)
        let chunks = try chunkText(normalized, language: language)

        // Single chunk — use normal generate
        if chunks.count <= 1 {
            return try await generate(
                text: Self.unmaskPhonemeSpans(normalized, spans: spans),
                language: language, options: options, progress: progress)
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        var allAudio = [Float]()
        var totalFrames = 0
        var totalDecoderTime: Double = 0
        var totalCodecTime: Double = 0
        var totalLTTime: Double = 0
        var totalDecoderCalls = 0
        guard let cfg = config else {
            throw MagpieTTSError.generationFailed("Config not loaded")
        }
        let overlapSamples = cfg.sampleRate / 10  // 100ms crossfade

        for (i, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            progress?(GenerationProgress(phase: .generating(step: i, maxSteps: chunks.count)))

            let result = try await generate(
                text: Self.unmaskPhonemeSpans(chunk, spans: spans),
                language: language, options: options, progress: nil)

            if allAudio.isEmpty {
                allAudio = result.audioSamples
            } else {
                allAudio = Self.crossfade(
                    allAudio, result.audioSamples, overlapSamples: overlapSamples)
            }
            totalFrames += result.framesGenerated
            totalDecoderTime += result.decoderTimeSeconds
            totalCodecTime += result.codecTimeSeconds
            totalDecoderCalls += result.decoderCallCount
            totalLTTime += result.localTransformerTimeSeconds
        }

        // Peak normalize combined audio
        var peak: Float = 0
        vDSP_maxmgv(allAudio, 1, &peak, vDSP_Length(allAudio.count))
        if peak > 0 {
            var scale = Float(0.9) / peak
            vDSP_vsmul(allAudio, 1, &scale, &allAudio, 1, vDSP_Length(allAudio.count))
        }

        return GenerationResult(
            audioSamples: allAudio,
            sampleRate: cfg.sampleRate,
            framesGenerated: totalFrames,
            generationTimeSeconds: CFAbsoluteTimeGetCurrent() - startTime,
            decoderTimeSeconds: totalDecoderTime,
            codecTimeSeconds: totalCodecTime,
            decoderCallCount: totalDecoderCalls,
            localTransformerTimeSeconds: totalLTTime,
            localTransformerBackend: useMLXLocalTransformer ? "MLX" : "Accelerate",
            timeToFirstAudioSeconds: 0
        )
    }

    /// Generate long text with streaming audio chunks.
    ///
    /// Splits text at sentence boundaries and streams audio from each chunk.
    public func generateLongStreaming(
        text: String,
        language: Language = .english,
        options: GenerationOptions = .init(),
        chunkFrames: Int = 75,
        progress: ((GenerationProgress) -> Void)? = nil,
        onAudioChunk: @escaping ([Float], Int) -> Void
    ) async throws -> GenerationResult {
        _ = try requireTokenizer()  // ensure tokenizer is loaded for chunkText

        // Protect |...| phoneme spans from TN and the chunker.
        let (masked, spans) = Self.maskPhonemeSpans(text)
        // Normalize before chunking so token counts reflect expanded text
        let normalized = Self.normalizeForTTS(masked, language: language)
        let chunks = try chunkText(normalized, language: language)

        // Single chunk — use normal streaming
        if chunks.count <= 1 {
            return try await generateStreaming(
                text: Self.unmaskPhonemeSpans(normalized, spans: spans),
                language: language, options: options,
                chunkFrames: chunkFrames, progress: progress, onAudioChunk: onAudioChunk)
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        var allAudio = [Float]()
        var totalFrames = 0
        var totalDecoderTime: Double = 0
        var totalCodecTime: Double = 0
        var totalLTTime: Double = 0
        var totalDecoderCalls = 0

        for (i, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            progress?(GenerationProgress(phase: .generating(step: i, maxSteps: chunks.count)))

            let result = try await generateStreaming(
                text: Self.unmaskPhonemeSpans(chunk, spans: spans),
                language: language, options: options,
                chunkFrames: chunkFrames, progress: nil, onAudioChunk: onAudioChunk)

            allAudio.append(contentsOf: result.audioSamples)
            totalFrames += result.framesGenerated
            totalDecoderTime += result.decoderTimeSeconds
            totalCodecTime += result.codecTimeSeconds
            totalDecoderCalls += result.decoderCallCount
            totalLTTime += result.localTransformerTimeSeconds
        }

        return GenerationResult(
            audioSamples: allAudio,
            sampleRate: try requireConfig().sampleRate,
            framesGenerated: totalFrames,
            generationTimeSeconds: CFAbsoluteTimeGetCurrent() - startTime,
            decoderTimeSeconds: totalDecoderTime,
            codecTimeSeconds: totalCodecTime,
            decoderCallCount: totalDecoderCalls,
            localTransformerTimeSeconds: totalLTTime,
            localTransformerBackend: useMLXLocalTransformer ? "MLX" : "Accelerate",
            timeToFirstAudioSeconds: 0
        )
    }

    /// Decode accumulated codec frames into audio samples.
    private func decodeFrames(
        _ predictions: [[Int32]], numCb: Int, maxCodecFrames: Int, cfg: ModelConfig
    ) throws -> [Float] {
        guard let codec = nanocodec else {
            throw MagpieTTSError.generationFailed("Nanocodec not loaded")
        }
        let numFrames = min(predictions.count, maxCodecFrames)
        var codecBuf = [Int32](repeating: 0, count: numCb * maxCodecFrames)
        for t in 0 ..< numFrames {
            for cb in 0 ..< numCb {
                codecBuf[cb * maxCodecFrames + t] = predictions[t][cb]
            }
        }

        let codecResult = try codec.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "tokens": MLFeatureValue(
                    multiArray: try int32Array(codecBuf, shape: [1, numCb, maxCodecFrames]))
            ]),
            options: MLPredictionOptions()
        )

        guard let audioArr = codecResult.featureValue(for: "audio")?.multiArrayValue else {
            throw MagpieTTSError.generationFailed("Nanocodec missing audio output")
        }
        var audio = readFloats(audioArr)
        let expected = numFrames * cfg.codecSamplesPerFrame
        if audio.count > expected { audio = Array(audio.prefix(expected)) }
        return audio
    }

    // MARK: - Private: Loading

    private func loadConstants() throws {
        let cDir = constantsDirectory
        let jsonURL = cDir.appendingPathComponent("constants.json")
        guard FileManager.default.fileExists(atPath: jsonURL.path) else {
            throw MagpieTTSError.constantsNotFound(jsonURL.path)
        }
        guard
            let json = try JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL))
                as? [String: Any]
        else {
            throw MagpieTTSError.invalidConfiguration("constants.json: malformed root")
        }
        let cfg = try ModelConfig(json: json)
        config = cfg

        // Speaker embeddings
        let siURL = cDir.appendingPathComponent("speaker_info.json")
        guard
            let si = try JSONSerialization.jsonObject(with: Data(contentsOf: siURL))
                as? [String: Any],
            let numSpeakers = si["num_speakers"] as? Int,
            let tCtx = si["T"] as? Int
        else {
            throw MagpieTTSError.invalidConfiguration("speaker_info.json: malformed")
        }
        speakerContextLength = tCtx
        speakerEmbeddings = try (0 ..< numSpeakers).map {
            try NpyReader.load(url: cDir.appendingPathComponent("speaker_\($0).npy")).data
        }

        // Audio code embeddings
        let embs = try (0 ..< cfg.numCodebooks).map {
            try NpyReader.load(url: cDir.appendingPathComponent("audio_embedding_\($0).npy")).data
        }
        audioEmbeddings = embs

        // Local transformer weights
        ltWeights = try LocalTransformerWeights.load(
            from: cDir.appendingPathComponent("local_transformer"))
        ltWeightsMLX = try LocalTransformerWeightsMLX.load(
            from: cDir.appendingPathComponent("local_transformer"))

        // Pre-convert audio embeddings to MLXArray for GPU-side lookup
        let vocabSize = 2024  // audio codebook vocab size
        audioEmbeddingsMLX = convertAudioEmbeddingsToMLX(
            embs, vocabSize: vocabSize, dModel: cfg.dModel)
    }

    private func loadTokenizer() throws {
        multiTokenizer = try MultiLanguageTokenizer(constantsDirectory: constantsDirectory)
    }

    /// Pre-compute the unconditional CFG prefill KV caches.
    /// The unconditional path uses zero audio + zero encoder output, so it's identical
    /// across all text inputs and speakers. Computing it once saves tCtx decoder calls per generation.
    private func precomputeUncondPrefill() throws {
        guard let cfg = config else {
            throw MagpieTTSError.generationFailed("Config not loaded")
        }
        let maxTextLen = 256
        let maxSeqLen = 512
        let nL = cfg.nLayers
        let nH = cfg.nHeads
        let dH = cfg.dHead
        let dM = cfg.dModel
        let tCtx = speakerContextLength

        var uCaches = [String: MLMultiArray]()
        var uPositions = [String: MLMultiArray]()

        if let prefillModel = decoderPrefill {
            // Batched: single forward pass
            let zeroEmb = try zeroArray(shape: [1, tCtx, dM])
            let uncondEncOut = try zeroArray(shape: [1, maxTextLen, dM])
            var um = [Float](repeating: 0, count: maxTextLen)
            um[0] = 1
            let uncondMask = try floatArray(um, shape: [1, maxTextLen])

            let result = try prefillModel.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "audio_embed": MLFeatureValue(multiArray: zeroEmb),
                    "encoder_output": MLFeatureValue(multiArray: uncondEncOut),
                    "encoder_mask": MLFeatureValue(multiArray: uncondMask),
                ]), options: MLPredictionOptions())
            try parsePrefillCaches(
                result, into: &uCaches, positions: &uPositions, tCtx: tCtx, nLayers: nL)
        } else {
            // Sequential fallback
            for i in 0 ..< nL {
                uCaches["cache\(i)"] = try zeroArray(shape: [2, 1, maxSeqLen, nH, dH])
                uPositions["position\(i)"] = try floatArray([0], shape: [1])
            }

            let uncondEncOut = try zeroArray(shape: [1, maxTextLen, dM])
            var um = [Float](repeating: 0, count: maxTextLen)
            um[0] = 1
            let uncondMask = try floatArray(um, shape: [1, maxTextLen])
            let zeroCtx = try floatArray([Float](repeating: 0, count: dM), shape: [1, 1, dM])

            for _ in 0 ..< tCtx {
                _ = try runDecoder(
                    audio: zeroCtx, enc: uncondEncOut, mask: uncondMask,
                    caches: &uCaches, positions: &uPositions)
            }
        }

        cachedUncondCaches = uCaches
        cachedUncondPositions = uPositions
    }

    private func copyMLMultiArray(_ source: MLMultiArray) throws -> MLMultiArray {
        let copy = try MLMultiArray(shape: source.shape, dataType: source.dataType)
        let bytesPerElement: Int
        switch source.dataType {
        case .float16: bytesPerElement = 2
        case .float32: bytesPerElement = 4
        case .double: bytesPerElement = 8
        case .int32: bytesPerElement = 4
        @unknown default: bytesPerElement = 4
        }
        memcpy(copy.dataPointer, source.dataPointer, source.count * bytesPerElement)
        return copy
    }

    private func copyCaches(_ caches: [String: MLMultiArray]) throws -> [String: MLMultiArray] {
        var result = [String: MLMultiArray]()
        for (key, value) in caches {
            result[key] = try copyMLMultiArray(value)
        }
        return result
    }

    private func loadCoreMLModels() throws {
        let mlCfg = MLModelConfiguration()
        mlCfg.computeUnits = computeUnits

        func load(_ name: String, computeUnit: MLComputeUnits) throws -> MLModel {
            let cu = MLModelConfiguration()
            cu.computeUnits = computeUnit

            // Prefer compiled .mlmodelc, fall back to .mlpackage
            let compiled = buildDirectory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiled.path) {
                return try MLModel(contentsOf: compiled, configuration: cu)
            }
            let pkg = buildDirectory.appendingPathComponent("\(name).mlpackage")
            guard FileManager.default.fileExists(atPath: pkg.path) else {
                throw MagpieTTSError.modelNotFound(name)
            }
            let compiledURL = try MLModel.compileModel(at: pkg)
            return try MLModel(contentsOf: compiledURL, configuration: cu)
        }

        textEncoder = try load("TextEncoder", computeUnit: .cpuAndNeuralEngine)
        decoderStep = try load("DecoderStep", computeUnit: .cpuOnly)
        nanocodec = try load("NanocodecDecoder", computeUnit: .cpuOnly)

        // Optional: batched prefill model (falls back to step loop if absent)
        func tryLoad(_ name: String, computeUnit: MLComputeUnits) -> MLModel? {
            let cu = MLModelConfiguration()
            cu.computeUnits = computeUnit

            let compiled = buildDirectory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiled.path) {
                return try? MLModel(contentsOf: compiled, configuration: cu)
            }
            let pkg = buildDirectory.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: pkg.path),
                let url = try? MLModel.compileModel(at: pkg)
            {
                return try? MLModel(contentsOf: url, configuration: cu)
            }
            return nil
        }
        decoderPrefill = tryLoad("DecoderPrefill", computeUnit: .cpuAndNeuralEngine)
    }

    // MARK: - Private: Prefill Cache Parsing

    /// Parse prefill model outputs into the cache/position format used by decoder_step.
    private func parsePrefillCaches(
        _ result: MLFeatureProvider,
        into caches: inout [String: MLMultiArray],
        positions: inout [String: MLMultiArray],
        tCtx: Int,
        nLayers: Int
    ) throws {
        for i in 0 ..< nLayers {
            guard let cache = result.featureValue(for: "prefill_cache\(i)")?.multiArrayValue else {
                throw MagpieTTSError.generationFailed("Prefill model missing cache for layer \(i)")
            }
            caches["cache\(i)"] = cache
            positions["position\(i)"] = try floatArray([Float(tCtx)], shape: [1])
        }
    }

    // MARK: - Private: Decoder Step

    private func runDecoder(
        audio: MLMultiArray, enc: MLMultiArray, mask: MLMultiArray,
        caches: inout [String: MLMultiArray], positions: inout [String: MLMultiArray]
    ) throws -> MLMultiArray {
        guard let cfg = config, let decoder = decoderStep else {
            throw MagpieTTSError.generationFailed("Decoder not loaded")
        }
        var dict: [String: MLFeatureValue] = [
            "audio_embed": MLFeatureValue(multiArray: audio),
            "encoder_output": MLFeatureValue(multiArray: enc),
            "encoder_mask": MLFeatureValue(multiArray: mask),
        ]
        for i in 0 ..< cfg.nLayers {
            guard let cache = caches["cache\(i)"], let position = positions["position\(i)"] else {
                throw MagpieTTSError.generationFailed("Missing KV cache for layer \(i)")
            }
            dict["cache\(i)"] = MLFeatureValue(multiArray: cache)
            dict["position\(i)"] = MLFeatureValue(multiArray: position)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        let out = try decoder.prediction(from: provider, options: MLPredictionOptions())
        for i in 0 ..< cfg.nLayers {
            guard let newCache = out.featureValue(for: Self.cacheOutKeys[i])?.multiArrayValue,
                let newPos = out.featureValue(for: Self.posOutKeys[i])?.multiArrayValue
            else {
                throw MagpieTTSError.generationFailed("Decoder missing output for layer \(i)")
            }
            caches["cache\(i)"] = newCache
            positions["position\(i)"] = newPos
        }
        guard let hidden = out.featureValue(for: "decoder_hidden")?.multiArrayValue else {
            throw MagpieTTSError.generationFailed("Decoder missing hidden-state output")
        }
        return hidden  // (1, 1, dModel)
    }

    // MARK: - Private: Audio Embedding

    private func embedAudioCodes(_ codes: [Int32]) throws -> MLMultiArray {
        guard let cfg = config, let embs = audioEmbeddings else {
            throw MagpieTTSError.generationFailed("Config or audio embeddings not loaded")
        }
        let dM = cfg.dModel
        let nCb = cfg.numCodebooks
        let vDM = vDSP_Length(dM)
        var emb = [Float](repeating: 0, count: dM)
        for cb in 0 ..< nCb {
            let off = Int(codes[cb]) * dM
            embs[cb].withUnsafeBufferPointer { table in
                vDSP_vadd(emb, 1, table.baseAddress! + off, 1, &emb, 1, vDM)
            }
        }
        var inv = 1.0 / Float(nCb)
        vDSP_vsmul(emb, 1, &inv, &emb, 1, vDM)
        return try floatArray(emb, shape: [1, 1, dM])
    }

    // MARK: - Private: MLMultiArray Helpers

    private func floatArray(_ data: [Float], shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        data.withUnsafeBufferPointer { src in
            a.dataPointer.assumingMemoryBound(to: Float.self)
                .update(from: src.baseAddress!, count: src.count)
        }
        return a
    }

    private func int32Array(_ data: [Int32], shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        data.withUnsafeBufferPointer { src in
            a.dataPointer.assumingMemoryBound(to: Int32.self)
                .update(from: src.baseAddress!, count: src.count)
        }
        return a
    }

    private func zeroArray(shape: [Int]) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float32)
        memset(a.dataPointer, 0, a.count * MemoryLayout<Float>.stride)
        return a
    }

    private func readFloats(_ array: MLMultiArray) -> [Float] {
        let n = array.count
        switch array.dataType {
        case .float32:
            return Array(
                UnsafeBufferPointer(
                    start: array.dataPointer.assumingMemoryBound(to: Float.self), count: n))
        case .float16:
            let src = array.dataPointer.assumingMemoryBound(to: Float16.self)
            return [Float](unsafeUninitializedCapacity: n) { buffer, count in
                for i in 0 ..< n { buffer[i] = Float(src[i]) }
                count = n
            }
        default:
            fatalError("Unsupported MLMultiArray dataType: \(array.dataType.rawValue)")
        }
    }
}

// MARK: - WAV Encoder

private func encodeWAV(samples: [Float], sampleRate: Int) -> Data {
    let n = samples.count
    let bytesPerSample = 2
    let dataSize = n * bytesPerSample

    var d = Data(capacity: 44 + dataSize)
    func u16(_ v: UInt16) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
    func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }

    // RIFF header
    d.append(contentsOf: [0x52, 0x49, 0x46, 0x46])  // "RIFF"
    u32(UInt32(36 + dataSize))
    d.append(contentsOf: [0x57, 0x41, 0x56, 0x45])  // "WAVE"

    // fmt chunk
    d.append(contentsOf: [0x66, 0x6D, 0x74, 0x20])  // "fmt "
    u32(16)  // chunk size
    u16(1)  // PCM
    u16(1)  // mono
    u32(UInt32(sampleRate))
    u32(UInt32(sampleRate * bytesPerSample))
    u16(UInt16(bytesPerSample))
    u16(16)  // bits per sample

    // data chunk
    d.append(contentsOf: [0x64, 0x61, 0x74, 0x61])  // "data"
    u32(UInt32(dataSize))
    for s in samples {
        let clamped = max(-1, min(1, s))
        let i16 = Int16(clamped * Float(Int16.max))
        withUnsafeBytes(of: i16.littleEndian) { d.append(contentsOf: $0) }
    }
    return d
}
