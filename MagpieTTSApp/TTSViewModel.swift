//
//  TTSViewModel.swift
//  MagpieTTSApp
//
//  Created by Sachin Desai on 3/8/26.
//

import AVFoundation
import Combine
import SwiftUI

struct Voice: Identifiable, Hashable {
    let id: Int
    let name: String
}

@MainActor
final class TTSViewModel: ObservableObject {

    // MARK: - Inputs

    @Published var text: String = "Hello, this is Magpie, a text to speech model running on device."
    @Published var selectedLanguage: Language = .english
    @Published var selectedVoice: Voice
    @Published var useMLX: Bool = false {
        didSet { tts?.useMLXLocalTransformer = useMLX }
    }

    // MARK: - State

    @Published var isGenerating = false
    @Published var isPreparing = false
    @Published var isPlaying = false
    @Published var isStreaming = false
    @Published var progress: String = ""
    @Published var statusMessage: String = "Ready"
    @Published var errorMessage: String?
    @Published var generationStats: String?
    @Published var hasAudio = false

    // MARK: - Data

    let voices: [Voice]
    let languages: [Language] = Language.allCases.filter(\.supportsTextInput)

    // MARK: - Private

    private var tts: MagpieTTS?
    private var audioData: Data?
    private var audioPlayer: AVAudioPlayer?
    private var audioPlayerDelegate: AudioPlayerDelegate?
    private var isPrepared = false

    // Streaming playback
    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var streamingFormat: AVAudioFormat?
    private var pendingBufferCount = 0

    init() {
        let bundle = Bundle.main

        // constants/ is added as a folder reference → appears as "constants" in bundle
        let constantsURL =
            bundle.url(forResource: "constants", withExtension: nil)
            ?? bundle.resourceURL!.appendingPathComponent("constants")

        let speakers = MagpieTTS.loadSpeakerInfo(constantsDirectory: constantsURL)
        self.voices = speakers.map { Voice(id: $0.index, name: $0.name) }
        self.selectedVoice = voices.first ?? Voice(id: 0, name: "Default")

        // .mlmodelc directories are copied to bundle root
        let buildURL = bundle.resourceURL!

        self.tts = MagpieTTS(
            constantsDirectory: constantsURL,
            buildDirectory: buildURL,
            computeUnits: .cpuAndGPU
        )

        // Eagerly load models in background so they're ready when user taps Generate
        isPreparing = true
        statusMessage = "Loading models..."
        Task {
            do {
                try await tts?.prepare()
                isPrepared = true
                statusMessage = "Ready"
            } catch {
                statusMessage = "Model load failed"
                errorMessage = error.localizedDescription
            }
            isPreparing = false
        }
    }

    // MARK: - Actions

    func generate() {
        guard !isGenerating else { return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "Please enter some text."
            return
        }

        isGenerating = true
        hasAudio = false
        audioData = nil
        errorMessage = nil
        generationStats = nil
        progress = ""

        let expandedText = PronunciationExpander.expand(
            text, language: selectedLanguage,
            dictionaryText: UserDefaults.standard.string(forKey: "pronunciationDictionary") ?? ""
        )

        Task {
            do {
                let result = try await tts!.generateLong(
                    text: expandedText,
                    language: selectedLanguage,
                    options: GenerationOptions(
                        speaker: selectedVoice.id,
                        seed: UInt64.random(in: 0 ... UInt64.max)
                    )
                ) { [weak self] prog in
                    Task { @MainActor in
                        guard let self else { return }
                        switch prog.phase {
                        case .loadingModels:
                            self.progress = "Loading models..."
                        case .encodingText:
                            self.progress = "Encoding text..."
                        case .prefillingContext(let step, let total):
                            self.progress = "Preparing voice... \(step)/\(total)"
                        case .generating(let step, let maxSteps):
                            self.progress = "Generating... \(step)/\(maxSteps)"
                        case .decodingAudio:
                            self.progress = "Decoding audio..."
                        }
                    }
                }

                audioData = result.wavData
                hasAudio = true
                let dur = String(format: "%.1f", result.durationSeconds)
                let gen = String(format: "%.1f", result.generationTimeSeconds)
                let rtfx = String(
                    format: "%.1f", result.durationSeconds / result.generationTimeSeconds)
                let lt = String(format: "%.2f", result.localTransformerTimeSeconds)
                generationStats =
                    "\(dur)s audio in \(gen)s (\(rtfx)x RTFx) | LT: \(lt)s [\(result.localTransformerBackend)]"
                statusMessage = "Done"
                progress = ""
            } catch {
                errorMessage = error.localizedDescription
                progress = ""
                statusMessage = "Error"
            }
            isGenerating = false
        }
    }

    func stream() {
        guard !isGenerating else { return }
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            errorMessage = "Please enter some text."
            return
        }

        isGenerating = true
        isStreaming = true
        hasAudio = false
        audioData = nil
        errorMessage = nil
        generationStats = nil
        progress = ""

        let expandedText = PronunciationExpander.expand(
            text, language: selectedLanguage,
            dictionaryText: UserDefaults.standard.string(forKey: "pronunciationDictionary") ?? ""
        )

        Task {
            do {
                try startStreamingEngine(sampleRate: 22050)

                let result = try await tts!.generateLongStreaming(
                    text: expandedText,
                    language: selectedLanguage,
                    options: GenerationOptions(
                        speaker: selectedVoice.id,
                        cfgSteps: 20,
                        seed: UInt64.random(in: 0 ... UInt64.max)
                    ),
                    chunkFrames: 75,
                    progress: { [weak self] prog in
                        Task { @MainActor in
                            guard let self else { return }
                            switch prog.phase {
                            case .loadingModels:
                                self.progress = "Loading models..."
                            case .encodingText:
                                self.progress = "Encoding text..."
                            case .prefillingContext(let step, let total):
                                self.progress = "Preparing voice... \(step)/\(total)"
                            case .generating(let step, let maxSteps):
                                self.progress = "Generating... \(step)/\(maxSteps)"
                            case .decodingAudio:
                                self.progress = "Decoding audio..."
                            }
                        }
                    },
                    onAudioChunk: { [weak self] samples, _ in
                        Task { @MainActor in
                            self?.scheduleAudioChunk(samples)
                        }
                    }
                )

                audioData = result.wavData
                hasAudio = true
                let dur = String(format: "%.1f", result.durationSeconds)
                let gen = String(format: "%.1f", result.generationTimeSeconds)
                let rtfx = String(
                    format: "%.1f", result.durationSeconds / result.generationTimeSeconds)
                let ttfa =
                    result.timeToFirstAudioSeconds > 0
                    ? String(format: "%.2f", result.timeToFirstAudioSeconds) : "-"
                generationStats = "\(dur)s in \(gen)s (\(rtfx)x) | TTFA: \(ttfa)s"
                statusMessage = "Done"
                progress = ""

                // Wait for all buffers to finish playing
                await waitForStreamingPlaybackEnd()
            } catch {
                errorMessage = error.localizedDescription
                progress = ""
                statusMessage = "Error"
            }

            stopStreamingEngine()
            isGenerating = false
            isStreaming = false
        }
    }

    func play() {
        guard let data = audioData else { return }

        if isPlaying {
            audioPlayer?.stop()
            isPlaying = false
            return
        }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback)
            try AVAudioSession.sharedInstance().setActive(true)
            audioPlayer = try AVAudioPlayer(data: data)
            audioPlayerDelegate = AudioPlayerDelegate { [weak self] in
                Task { @MainActor in self?.isPlaying = false }
            }
            audioPlayer?.delegate = audioPlayerDelegate
            audioPlayer?.play()
            isPlaying = true
        } catch {
            errorMessage = "Playback error: \(error.localizedDescription)"
        }
    }

    func stopStreaming() {
        // Cancel by stopping the engine; Task.checkCancellation will catch it
        stopStreamingEngine()
        isStreaming = false
        isGenerating = false
    }

    func shareItem() -> AudioShareItem? {
        guard let data = audioData else { return nil }
        return AudioShareItem(data: data, filename: "magpie_tts.wav")
    }

    var defaultTextForLanguage: String {
        switch selectedLanguage {
        case .english: return "Hello, this is Magpie, a text to speech model running on device."
        case .spanish: return "Hola, esta es una prueba de síntesis de voz."
        case .german: return "Hallo, das ist ein Test der Sprachsynthese."
        case .mandarin: return "你好，这是语音合成的测试。"
        case .french: return "Bonjour, ceci est un test de synthèse vocale."
        case .hindi: return "नमस्ते, यह वाक् संश्लेषण का परीक्षण है।"
        case .italian: return "Ciao, questo è un test di sintesi vocale."
        case .vietnamese: return "Xin chào, đây là bài kiểm tra tổng hợp giọng nói."
        case .japanese: return "こんにちは、これはデバイス上で動作する音声合成のテストです。"
        }
    }

    // MARK: - Streaming Engine

    private func startStreamingEngine(sampleRate: Int) throws {
        try AVAudioSession.sharedInstance().setCategory(.playback)
        try AVAudioSession.sharedInstance().setActive(true)

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        player.play()

        self.audioEngine = engine
        self.playerNode = player
        self.streamingFormat = format
        self.pendingBufferCount = 0
        self.isPlaying = true
    }

    private func scheduleAudioChunk(_ samples: [Float]) {
        guard let player = playerNode, let format = streamingFormat else { return }

        let buffer = AVAudioPCMBuffer(
            pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        let channelData = buffer.floatChannelData![0]
        samples.withUnsafeBufferPointer { src in
            channelData.update(from: src.baseAddress!, count: samples.count)
        }

        pendingBufferCount += 1
        player.scheduleBuffer(buffer) { [weak self] in
            Task { @MainActor in
                self?.pendingBufferCount -= 1
            }
        }
    }

    private func waitForStreamingPlaybackEnd() async {
        // Poll until all scheduled buffers have been played
        while pendingBufferCount > 0 {
            try? await Task.sleep(nanoseconds: 100_000_000)  // 100ms
        }
    }

    private func stopStreamingEngine() {
        playerNode?.stop()
        audioEngine?.stop()
        playerNode = nil
        audioEngine = nil
        streamingFormat = nil
        pendingBufferCount = 0
        isPlaying = false
    }
}

// MARK: - Share Item

struct AudioShareItem: Transferable {
    let data: Data
    let filename: String

    static var transferRepresentation: some TransferRepresentation {
        DataRepresentation(exportedContentType: .wav) { item in
            item.data
        }
    }

    var temporaryURL: URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        try? data.write(to: url)
        return url
    }
}

// MARK: - Audio Player Delegate

private final class AudioPlayerDelegate: NSObject, AVAudioPlayerDelegate {
    let onFinish: () -> Void
    init(onFinish: @escaping () -> Void) { self.onFinish = onFinish }
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        onFinish()
    }
}
