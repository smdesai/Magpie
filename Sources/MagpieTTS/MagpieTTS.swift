import Accelerate
import CoreML
import Foundation
import MLX

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
            let inf = json["inference"] as? [String: Any]
        else {
            throw MagpieTTSError.invalidConfiguration("Missing decoder/special_tokens/inference")
        }
        numCodebooks = json["num_audio_codebooks"] as! Int
        vocabSize = json["num_all_tokens_per_codebook"] as! Int
        sampleRate = json["output_sample_rate"] as! Int
        codecSamplesPerFrame = json["codec_samples_per_frame"] as! Int
        audioBosId = st["audio_bos_id"] as! Int
        audioEosId = st["audio_eos_id"] as! Int
        dModel = dec["d_model"] as! Int
        nLayers = dec["n_layers"] as! Int
        nHeads = dec["sa_n_heads"] as! Int
        dHead = dModel / nHeads
        minGeneratedFrames = (inf["min_generated_frames"] as? Int) ?? 4
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

    // Decoder step I/O key mappings (from traced CoreML model variable names)
    private static let cacheOutKeys: [String] = {
        var keys = (0 ..< 11).map { "new_cache_\($0 * 2 + 1)" }
        keys.append("new_cache")  // layer 11
        return keys
    }()
    private static let posOutKeys: [String] = (0 ..< 12).map { "var_\(169 + $0 * 177)" }

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
        try loadConstants()
        try loadCoreMLModels()
        try loadTokenizer()
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
    ///   - progress: Optional callback for UI updates.
    /// - Returns: Audio samples, sample rate, and WAV data.
    public func generate(
        text: String,
        language: Language = .english,
        options: GenerationOptions = .init(),
        progress: ((GenerationProgress) -> Void)? = nil
    ) async throws -> GenerationResult {
        if multiTokenizer == nil { try loadTokenizer() }
        let tokenIDs = try multiTokenizer!.tokenize(text, language: language)
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
    ///   - progress: Optional callback for UI updates.
    ///   - onAudioChunk: Called with (newSamples, sampleRate) for each decoded chunk.
    /// - Returns: Complete generation result.
    public func generateStreaming(
        text: String,
        language: Language = .english,
        options: GenerationOptions = .init(),
        chunkFrames: Int = 15,
        progress: ((GenerationProgress) -> Void)? = nil,
        onAudioChunk: @escaping ([Float], Int) -> Void
    ) async throws -> GenerationResult {
        if multiTokenizer == nil { try loadTokenizer() }
        let tokenIDs = try multiTokenizer!.tokenize(text, language: language)
        return try await generateInternal(
            tokenIDs: tokenIDs, options: options, chunkFrames: chunkFrames,
            progress: progress, onAudioChunk: onAudioChunk)
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
        if config == nil { try loadConstants() }
        if textEncoder == nil { try loadCoreMLModels() }

        let cfg = config!
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

        let encResult = try await textEncoder!.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "text_tokens": MLFeatureValue(multiArray: tokensArr),
                "text_mask": MLFeatureValue(multiArray: maskArr),
            ]))
        let encoderOutput = encResult.featureValue(for: "encoder_output")!.multiArrayValue!

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
        let spkEmb = speakerEmbeddings![options.speaker]
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
                nextCodes = localTransformerSampleMLX(
                    decoderHidden: condFloats, weights: ltWeightsMLX!,
                    audioEmbeddings: audioEmbeddings!, audioEmbeddingsMLX: audioEmbeddingsMLX,
                    numCodebooks: numCb,
                    temperature: options.temperature, topK: options.topK,
                    forbidEOS: forbidEOS, uncondDecoderHidden: uncondFloats,
                    cfgScale: options.cfgScale, rng: &rng
                )
            } else {
                nextCodes = localTransformerSample(
                    decoderHidden: condFloats, weights: ltWeights!,
                    audioEmbeddings: audioEmbeddings!, numCodebooks: numCb,
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
        return NemoTextProcessing.tnNormalizeSentence(text)
    }

    // MARK: - Chunked Generation (Long Text)

    /// Split text into individual sentences for chunked generation.
    /// Each sentence is generated independently to avoid the model hitting EOS
    /// before vocalizing all text in a chunk.
    /// Sentences that exceed the model's text token limit (256) are split further
    /// on clause boundaries (commas) or word boundaries.
    private func chunkText(_ text: String, language: Language) throws -> [String] {
        let maxTokens = 240  // Leave margin below maxTextLen=256

        // Split on sentence-ending punctuation.
        // For Latin: split after .!? followed by whitespace and uppercase letter.
        // For CJK: split after 。！？ (Chinese/Japanese sentence-ending punctuation).
        let pattern = #"(?<=[.!?])\s+(?=[A-Z])|(?<=[。！？])"#
        let sentences = text.components(separatedBy: .newlines)
            .flatMap { line -> [String] in
                guard let regex = try? NSRegularExpression(pattern: pattern) else {
                    return [line]
                }
                let nsLine = line as NSString
                var parts = [String]()
                var lastEnd = 0
                let matches = regex.matches(
                    in: line, range: NSRange(location: 0, length: nsLine.length))
                for match in matches {
                    let range = NSRange(location: lastEnd, length: match.range.location - lastEnd)
                    let s = nsLine.substring(with: range).trimmingCharacters(
                        in: .whitespacesAndNewlines)
                    if !s.isEmpty { parts.append(s) }
                    lastEnd = match.range.location + match.range.length
                }
                let remainder = nsLine.substring(from: lastEnd).trimmingCharacters(
                    in: .whitespacesAndNewlines)
                if !remainder.isEmpty { parts.append(remainder) }
                return parts
            }
            .filter { !$0.isEmpty }

        // Second pass: split sentences that exceed the token limit
        var result = [String]()
        for sentence in (sentences.isEmpty ? [text] : sentences) {
            let tokens = try multiTokenizer!.tokenize(sentence, language: language)
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

    /// Split a chunk that exceeds the token limit on clause boundaries, then word boundaries.
    private func splitLongChunk(_ text: String, language: Language, maxTokens: Int) throws
        -> [String]
    {
        // Try splitting on commas/semicolons (Latin and CJK)
        let clausePattern = #"(?<=[,;，；、])\s*"#
        if let regex = try? NSRegularExpression(pattern: clausePattern) {
            let nsText = text as NSString
            let matches = regex.matches(
                in: text, range: NSRange(location: 0, length: nsText.length))
            if !matches.isEmpty {
                var clauses = [String]()
                var lastEnd = 0
                for match in matches {
                    let range = NSRange(location: lastEnd, length: match.range.location - lastEnd)
                    let s = nsText.substring(with: range).trimmingCharacters(
                        in: .whitespacesAndNewlines)
                    if !s.isEmpty { clauses.append(s) }
                    lastEnd = match.range.location + match.range.length
                }
                let remainder = nsText.substring(from: lastEnd).trimmingCharacters(
                    in: .whitespacesAndNewlines)
                if !remainder.isEmpty { clauses.append(remainder) }

                // Group clauses to fit within token limit
                var grouped = [String]()
                var current = ""
                for clause in clauses {
                    let candidate = current.isEmpty ? clause : "\(current) \(clause)"
                    let tokens = try multiTokenizer!.tokenize(candidate, language: language)
                    if tokens.count <= maxTokens {
                        current = candidate
                    } else {
                        if !current.isEmpty { grouped.append(current) }
                        current = clause
                    }
                }
                if !current.isEmpty { grouped.append(current) }

                // If grouping helped, return; otherwise fall through to word splitting
                let firstGroupTokenCount = try multiTokenizer!.tokenize(
                    grouped[0], language: language
                ).count
                if grouped.count > 1 || firstGroupTokenCount <= maxTokens {
                    // Recursively check each group
                    var final = [String]()
                    for g in grouped {
                        let tokens = try multiTokenizer!.tokenize(g, language: language)
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
            let tokens = try multiTokenizer!.tokenize(candidate, language: language)
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
        let restTokens = try multiTokenizer!.tokenize(rest, language: language)
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
        if multiTokenizer == nil { try loadTokenizer() }

        // Normalize before chunking so token counts reflect expanded text
        let normalized = Self.normalizeForTTS(text, language: language)
        let chunks = try chunkText(normalized, language: language)

        // Single chunk — use normal generate
        if chunks.count <= 1 {
            return try await generate(
                text: normalized, language: language, options: options, progress: progress)
        }

        let startTime = CFAbsoluteTimeGetCurrent()
        var allAudio = [Float]()
        var totalFrames = 0
        var totalDecoderTime: Double = 0
        var totalCodecTime: Double = 0
        var totalLTTime: Double = 0
        var totalDecoderCalls = 0
        let overlapSamples = config!.sampleRate / 10  // 100ms crossfade

        for (i, chunk) in chunks.enumerated() {
            try Task.checkCancellation()
            progress?(GenerationProgress(phase: .generating(step: i, maxSteps: chunks.count)))

            let result = try await generate(
                text: chunk, language: language, options: options, progress: nil)

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
            sampleRate: config!.sampleRate,
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
        if multiTokenizer == nil { try loadTokenizer() }

        // Normalize before chunking so token counts reflect expanded text
        let normalized = Self.normalizeForTTS(text, language: language)
        let chunks = try chunkText(normalized, language: language)

        // Single chunk — use normal streaming
        if chunks.count <= 1 {
            return try await generateStreaming(
                text: normalized, language: language, options: options,
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
                text: chunk, language: language, options: options,
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
            sampleRate: config!.sampleRate,
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
        let numFrames = min(predictions.count, maxCodecFrames)
        var codecBuf = [Int32](repeating: 0, count: numCb * maxCodecFrames)
        for t in 0 ..< numFrames {
            for cb in 0 ..< numCb {
                codecBuf[cb * maxCodecFrames + t] = predictions[t][cb]
            }
        }

        let codecResult = try nanocodec!.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "tokens": MLFeatureValue(
                    multiArray: try int32Array(codecBuf, shape: [1, numCb, maxCodecFrames]))
            ]),
            options: MLPredictionOptions()
        )

        var audio = readFloats(codecResult.featureValue(for: "audio")!.multiArrayValue!)
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
        let json =
            try JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL)) as! [String: Any]
        config = try ModelConfig(json: json)

        // Speaker embeddings
        let siURL = cDir.appendingPathComponent("speaker_info.json")
        let si = try JSONSerialization.jsonObject(with: Data(contentsOf: siURL)) as! [String: Any]
        let numSpeakers = si["num_speakers"] as! Int
        speakerContextLength = si["T"] as! Int
        speakerEmbeddings = try (0 ..< numSpeakers).map {
            try NpyReader.load(url: cDir.appendingPathComponent("speaker_\($0).npy")).data
        }

        // Audio code embeddings
        audioEmbeddings = try (0 ..< config!.numCodebooks).map {
            try NpyReader.load(url: cDir.appendingPathComponent("audio_embedding_\($0).npy")).data
        }

        // Local transformer weights
        ltWeights = try LocalTransformerWeights.load(
            from: cDir.appendingPathComponent("local_transformer"))
        ltWeightsMLX = try LocalTransformerWeightsMLX.load(
            from: cDir.appendingPathComponent("local_transformer"))

        // Pre-convert audio embeddings to MLXArray for GPU-side lookup
        let vocabSize = 2024  // audio codebook vocab size
        let dModel = config!.dModel
        audioEmbeddingsMLX = convertAudioEmbeddingsToMLX(
            audioEmbeddings!, vocabSize: vocabSize, dModel: dModel)
    }

    private func loadTokenizer() throws {
        multiTokenizer = try MultiLanguageTokenizer(constantsDirectory: constantsDirectory)
    }

    /// Pre-compute the unconditional CFG prefill KV caches.
    /// The unconditional path uses zero audio + zero encoder output, so it's identical
    /// across all text inputs and speakers. Computing it once saves tCtx decoder calls per generation.
    private func precomputeUncondPrefill() throws {
        let cfg = config!
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

        func load(_ name: String) throws -> MLModel {
            // Prefer compiled .mlmodelc, fall back to .mlpackage
            let compiled = buildDirectory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiled.path) {
                return try MLModel(contentsOf: compiled, configuration: mlCfg)
            }
            let pkg = buildDirectory.appendingPathComponent("\(name).mlpackage")
            guard FileManager.default.fileExists(atPath: pkg.path) else {
                throw MagpieTTSError.modelNotFound(name)
            }
            let compiledURL = try MLModel.compileModel(at: pkg)
            return try MLModel(contentsOf: compiledURL, configuration: mlCfg)
        }

        textEncoder = try load("TextEncoder")
        decoderStep = try load("DecoderStep")
        nanocodec = try load("NanocodecDecoder")

        // Optional: batched prefill model (falls back to step loop if absent)
        func tryLoad(_ name: String) -> MLModel? {
            let compiled = buildDirectory.appendingPathComponent("\(name).mlmodelc")
            if FileManager.default.fileExists(atPath: compiled.path) {
                return try? MLModel(contentsOf: compiled, configuration: mlCfg)
            }
            let pkg = buildDirectory.appendingPathComponent("\(name).mlpackage")
            if FileManager.default.fileExists(atPath: pkg.path),
                let url = try? MLModel.compileModel(at: pkg)
            {
                return try? MLModel(contentsOf: url, configuration: mlCfg)
            }
            return nil
        }
        decoderPrefill = tryLoad("DecoderPrefill")
    }

    // MARK: - Private: Prefill Cache Parsing

    /// Output key names from decoder_prefill model (discovered on first use)
    private var prefillOutputKeys: [String]?

    /// Parse prefill model outputs into the cache/position format used by decoder_step.
    private func parsePrefillCaches(
        _ result: MLFeatureProvider,
        into caches: inout [String: MLMultiArray],
        positions: inout [String: MLMultiArray],
        tCtx: Int,
        nLayers: Int
    ) throws {
        // Discover output key names on first call.
        // CoreML names are like var_214, var_383, ... — must sort by numeric suffix
        // to preserve layer order (alphabetical sort would put var_1059 before var_214).
        if prefillOutputKeys == nil {
            var keys = [String]()
            for name in result.featureNames {
                if let val = result.featureValue(for: name)?.multiArrayValue,
                    val.shape.count == 5
                {  // (2, B, max_seq, H, D)
                    keys.append(name)
                }
            }
            keys.sort { a, b in
                let numA = Int(a.split(separator: "_").last ?? "") ?? 0
                let numB = Int(b.split(separator: "_").last ?? "") ?? 0
                return numA < numB
            }
            prefillOutputKeys = keys
        }

        let keys = prefillOutputKeys!
        guard keys.count == nLayers else {
            throw MagpieTTSError.generationFailed(
                "Prefill model returned \(keys.count) caches, expected \(nLayers)")
        }

        for i in 0 ..< nLayers {
            caches["cache\(i)"] = result.featureValue(for: keys[i])!.multiArrayValue!
            positions["position\(i)"] = try floatArray([Float(tCtx)], shape: [1])
        }
    }

    // MARK: - Private: Decoder Step

    @discardableResult
    private func runDecoder(
        audio: MLMultiArray, enc: MLMultiArray, mask: MLMultiArray,
        caches: inout [String: MLMultiArray], positions: inout [String: MLMultiArray]
    ) throws -> MLMultiArray {
        var dict: [String: MLFeatureValue] = [
            "audio_embed": MLFeatureValue(multiArray: audio),
            "encoder_output": MLFeatureValue(multiArray: enc),
            "encoder_mask": MLFeatureValue(multiArray: mask),
        ]
        for i in 0 ..< config!.nLayers {
            dict["cache\(i)"] = MLFeatureValue(multiArray: caches["cache\(i)"]!)
            dict["position\(i)"] = MLFeatureValue(multiArray: positions["position\(i)"]!)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)
        let out = try decoderStep!.prediction(from: provider, options: MLPredictionOptions())
        for i in 0 ..< config!.nLayers {
            caches["cache\(i)"] = out.featureValue(for: Self.cacheOutKeys[i])!.multiArrayValue!
            positions["position\(i)"] = out.featureValue(for: Self.posOutKeys[i])!.multiArrayValue!
        }
        return out.featureValue(for: "input")!.multiArrayValue!  // hidden state (1, 1, dModel)
    }

    // MARK: - Private: Audio Embedding

    private func embedAudioCodes(_ codes: [Int32]) throws -> MLMultiArray {
        let dM = config!.dModel
        let nCb = config!.numCodebooks
        let vDM = vDSP_Length(dM)
        var emb = [Float](repeating: 0, count: dM)
        for cb in 0 ..< nCb {
            let off = Int(codes[cb]) * dM
            audioEmbeddings![cb].withUnsafeBufferPointer { table in
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
