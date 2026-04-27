//
//  TTSViewModel.swift
//  MagpieTTSApp
//
//  Created by Sachin Desai on 3/8/26.
//

import AVFoundation
import ActivityKit
import Combine
import SwiftUI
import UIKit

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
    let languages: [Language] = Language.allCases

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
    private var isInterrupted = false
    private var interruptionObserver: NSObjectProtocol?
    private var routeChangeObserver: NSObjectProtocol?
    private var engineConfigObserver: NSObjectProtocol?
    private var becameActiveObserver: NSObjectProtocol?

    // Live Activity
    private var currentActivity: Activity<TTSActivityAttributes>?
    private var activityStartedAt = Date()
    private var playedAudioSamples = 0
    private var streamingSampleRate = 22050
    private var lastActivityUpdate = Date.distantPast
    private var lastActivityState: TTSActivityAttributes.ContentState?

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
                        self?.progress = Self.progressString(for: prog.phase)
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

        startLiveActivity(textPreview: expandedText)

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
                            self.progress = Self.progressString(for: prog.phase)
                            switch prog.phase {
                            case .loadingModels:
                                self.updateLiveActivity(phase: .preparingVoice)
                            case .encodingText:
                                self.updateLiveActivity(phase: .encoding)
                            case .prefillingContext(let step, let total):
                                self.updateLiveActivity(
                                    phase: .preparingVoice, step: step, maxSteps: total)
                            case .generating(let step, let maxSteps):
                                self.updateLiveActivity(
                                    phase: .generating, step: step, maxSteps: maxSteps)
                            case .decodingAudio:
                                self.updateLiveActivity(phase: .finishing)
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

            endLiveActivity()
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
        endLiveActivity()
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

    // MARK: - Helpers

    private static func progressString(for phase: GenerationProgress.Phase) -> String {
        switch phase {
        case .loadingModels: return "Loading models..."
        case .encodingText: return "Encoding text..."
        case .prefillingContext(let s, let t): return "Preparing voice... \(s)/\(t)"
        case .generating(let s, let m): return "Generating... \(s)/\(m)"
        case .decodingAudio: return "Decoding audio..."
        }
    }

    // MARK: - Live Activity

    private func startLiveActivity(textPreview: String) {
        guard ActivityAuthorizationInfo().areActivitiesEnabled else {
            print("[TTSViewModel] Live Activities disabled by user")
            return
        }
        let attributes = TTSActivityAttributes(
            voiceName: selectedVoice.name,
            languageName: selectedLanguage.displayName)
        activityStartedAt = Date()
        playedAudioSamples = 0
        let state = TTSActivityAttributes.ContentState(
            textPreview: String(textPreview.prefix(80)),
            phase: .encoding, step: 0, maxSteps: 0,
            audioElapsedSeconds: 0, startedAt: activityStartedAt)
        do {
            currentActivity = try Activity.request(
                attributes: attributes,
                content: .init(state: state, staleDate: nil))
            lastActivityState = state
            lastActivityUpdate = Date()
        } catch {
            print("[TTSViewModel] Live Activity start failed: \(error)")
        }
    }

    /// Rate-limited update. ActivityKit throttles rapid updates (~1/sec),
    /// so coalesce and only push when the phase changes or enough time has
    /// passed. Always pushes on phase transitions.
    private func updateLiveActivity(
        phase: TTSActivityAttributes.ContentState.Phase,
        step: Int = 0, maxSteps: Int = 0,
        textPreview: String? = nil,
        force: Bool = false
    ) {
        guard let activity = currentActivity else { return }
        let audioElapsed = Double(playedAudioSamples) / Double(streamingSampleRate)
        let preview = textPreview ?? lastActivityState?.textPreview ?? ""
        let newState = TTSActivityAttributes.ContentState(
            textPreview: preview, phase: phase,
            step: step, maxSteps: maxSteps,
            audioElapsedSeconds: audioElapsed,
            startedAt: activityStartedAt)

        let phaseChanged = lastActivityState?.phase != phase
        let elapsed = Date().timeIntervalSince(lastActivityUpdate)
        guard force || phaseChanged || elapsed > 1.0 else { return }

        Task { await activity.update(.init(state: newState, staleDate: nil)) }
        lastActivityState = newState
        lastActivityUpdate = Date()
    }

    private func endLiveActivity() {
        guard let activity = currentActivity else { return }
        let finalState = TTSActivityAttributes.ContentState(
            textPreview: lastActivityState?.textPreview ?? "",
            phase: .done, step: lastActivityState?.step ?? 0,
            maxSteps: lastActivityState?.maxSteps ?? 0,
            audioElapsedSeconds: Double(playedAudioSamples) / Double(streamingSampleRate),
            startedAt: activityStartedAt)
        Task {
            await activity.end(
                .init(state: finalState, staleDate: nil),
                dismissalPolicy: .after(Date().addingTimeInterval(3)))
        }
        currentActivity = nil
        lastActivityState = nil
    }

    // MARK: - Streaming Engine

    private func startStreamingEngine(sampleRate: Int) throws {
        // `.playback` + `.spokenAudio` marks this as long-form speech.
        // `.interruptSpokenAudioAndMixWithOthers` makes the session mixable,
        // which is required for background reactivation after a cancelled
        // phone call (nonmixable sessions hit
        // AVAudioSessionErrorCode.cannotInterruptOthers). TTS still takes
        // priority over other spoken audio (Siri, navigation, podcasts).
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playback, mode: .spokenAudio,
            options: [.interruptSpokenAudioAndMixWithOthers])
        try session.setActive(true)

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        try engine.start()
        player.play()

        // Schedule a short silence buffer so there's continuous audio output
        // from session-activate through first-generated-chunk. Without this,
        // a slow TTFA can let the system throttle/suspend the audio route.
        scheduleSilence(durationSeconds: 0.25, format: format, on: player)

        self.audioEngine = engine
        self.playerNode = player
        self.streamingFormat = format
        self.streamingSampleRate = sampleRate
        self.pendingBufferCount = 0
        self.isPlaying = true

        registerAudioSessionObservers()
    }

    private func scheduleSilence(
        durationSeconds: Double, format: AVAudioFormat, on player: AVAudioPlayerNode
    ) {
        let frames = AVAudioFrameCount(Double(format.sampleRate) * durationSeconds)
        guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return }
        buf.frameLength = frames
        // floatChannelData is already zeroed; nothing more to do
        player.scheduleBuffer(buf, completionHandler: nil)
    }

    private func registerAudioSessionObservers() {
        let nc = NotificationCenter.default

        interruptionObserver = nc.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil, queue: .main
        ) { [weak self] note in
            Task { @MainActor in self?.handleInterruption(note) }
        }

        routeChangeObserver = nc.addObserver(
            forName: AVAudioSession.routeChangeNotification,
            object: nil, queue: .main
        ) { [weak self] note in
            Task { @MainActor in self?.handleRouteChange(note) }
        }

        // Fires when hardware format changes out from under the engine (e.g.,
        // a phone call reassigns the audio route). Deferred to `.ended` if
        // we're currently in an interruption — the hardware isn't usable yet.
        engineConfigObserver = nc.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: audioEngine, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.handleEngineConfigChange() }
        }

        // Fallback when `.ended` is never delivered (cancelled / missed calls
        // sometimes don't post it). When the app returns to foreground and we
        // still intend to be streaming, try to recover the audio graph.
        becameActiveObserver = nc.addObserver(
            forName: UIApplication.didBecomeActiveNotification,
            object: nil, queue: .main
        ) { [weak self] _ in
            Task { @MainActor in self?.handleBecameActive() }
        }
    }

    private func unregisterAudioSessionObservers() {
        let nc = NotificationCenter.default
        if let obs = interruptionObserver { nc.removeObserver(obs) }
        if let obs = routeChangeObserver { nc.removeObserver(obs) }
        if let obs = engineConfigObserver { nc.removeObserver(obs) }
        if let obs = becameActiveObserver { nc.removeObserver(obs) }
        interruptionObserver = nil
        routeChangeObserver = nil
        engineConfigObserver = nil
        becameActiveObserver = nil
    }

    /// Apple's documented resume pattern: reactivate session, ensure engine
    /// is running, resume the player. No teardown, no retry ladder — Apple
    /// explicitly recommends against rebuilding the engine during an
    /// interruption, and `shouldResume` is only a hint. If reactivation
    /// fails here, `didBecomeActive` is the documented fallback.
    @discardableResult
    private func tryResumePlayback() -> Bool {
        guard isStreaming else { return true }
        if let engine = audioEngine, engine.isRunning {
            playerNode?.play()
            return true
        }

        let session = AVAudioSession.sharedInstance()
        do {
            try session.setActive(true)
        } catch {
            print("[TTSViewModel] session activate failed: \(error)")
            return false
        }
        if let engine = audioEngine, !engine.isRunning {
            do { try engine.start() } catch {
                print("[TTSViewModel] engine start failed: \(error)")
                return false
            }
        }
        playerNode?.play()
        print("[TTSViewModel] playback resumed")
        return true
    }

    /// Handles real hardware/route changes (sample rate, format). Apple
    /// explicitly says this is NOT a resume trigger — it's for rewiring
    /// graph connections if the mainMixer's format changed. Deferred if an
    /// interruption is active; re-syncs on `.ended`.
    private func handleEngineConfigChange() {
        if isInterrupted {
            print("[TTSViewModel] config change during interruption — deferring")
            return
        }
        guard let engine = audioEngine, let player = playerNode,
            let format = streamingFormat
        else { return }
        print("[TTSViewModel] AVAudioEngineConfigurationChange — rewiring graph")
        engine.disconnectNodeOutput(player)
        engine.connect(player, to: engine.mainMixerNode, format: format)
        if !engine.isRunning {
            try? engine.start()
            if isStreaming { player.play() }
        }
    }

    private func handleBecameActive() {
        guard isStreaming else { return }
        if let engine = audioEngine, engine.isRunning { return }
        print("[TTSViewModel] didBecomeActive with engine down — resuming")
        isInterrupted = false
        _ = tryResumePlayback()
    }

    private func handleInterruption(_ note: Notification) {
        guard let info = note.userInfo,
            let raw = info[AVAudioSessionInterruptionTypeKey] as? UInt,
            let type = AVAudioSession.InterruptionType(rawValue: raw)
        else { return }

        switch type {
        case .began:
            print("[TTSViewModel] interruption began (isStreaming=\(isStreaming))")
            isInterrupted = true
            playerNode?.pause()
        case .ended:
            let opts =
                (info[AVAudioSessionInterruptionOptionKey] as? UInt)
                .map(AVAudioSession.InterruptionOptions.init(rawValue:)) ?? []
            let systemSaysResume = opts.contains(.shouldResume)
            print(
                "[TTSViewModel] interruption ended (shouldResume=\(systemSaysResume), "
                    + "isStreaming=\(isStreaming))")
            isInterrupted = false
            // `isStreaming` is the source of truth for "should we still be
            // playing?" — it's only cleared when the user taps Stop or the
            // generation completes via `stopStreamingEngine`. `shouldResume`
            // is informational (Apple's docs say it's only a hint).
            guard isStreaming else { return }
            _ = tryResumePlayback()
        @unknown default:
            break
        }
    }

    private func handleRouteChange(_ note: Notification) {
        guard let info = note.userInfo,
            let raw = info[AVAudioSessionRouteChangeReasonKey] as? UInt,
            let reason = AVAudioSession.RouteChangeReason(rawValue: raw)
        else { return }
        // Pause on headphone unplug so audio doesn't blast through the speaker.
        if reason == .oldDeviceUnavailable {
            playerNode?.pause()
        }
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
        let sampleCount = samples.count
        player.scheduleBuffer(buffer) { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.pendingBufferCount -= 1
                self.playedAudioSamples += sampleCount
                // Push periodic audioElapsed refreshes while the Live
                // Activity is live (rate-limited inside).
                if self.currentActivity != nil,
                    let last = self.lastActivityState
                {
                    self.updateLiveActivity(
                        phase: last.phase,
                        step: last.step, maxSteps: last.maxSteps)
                }
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

        unregisterAudioSessionObservers()
        // Release the audio session so music apps resume cleanly.
        try? AVAudioSession.sharedInstance().setActive(
            false, options: [.notifyOthersOnDeactivation])
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
