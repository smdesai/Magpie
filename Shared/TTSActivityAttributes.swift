//
//  TTSActivityAttributes.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 4/21/26.
//

import ActivityKit
import Foundation

/// Live Activity attributes for an in-progress MagpieTTS streaming session.
/// Shared between the main app (which creates/updates the activity) and the
/// widget extension (which renders lock-screen + Dynamic Island views).
public struct TTSActivityAttributes: ActivityAttributes {

    public struct ContentState: Codable, Hashable {
        /// Short preview of the text being spoken (first ~40 chars).
        public var textPreview: String
        /// Current pipeline phase.
        public var phase: Phase
        /// Autoregressive generation step (0-based). 0 until generation starts.
        public var step: Int
        /// Upper bound on generation steps (maxSteps from GenerationOptions).
        public var maxSteps: Int
        /// Seconds of audio already played back on device.
        public var audioElapsedSeconds: Double
        /// Start time in absolute seconds-since-reference-date; used for
        /// elapsed-wallclock rendering on the lock screen.
        public var startedAt: Date

        public enum Phase: String, Codable, Hashable {
            case encoding  // encoding text
            case preparingVoice  // decoder prefill
            case generating  // autoregressive generation + streaming
            case finishing  // post-generation, waiting for last buffers
            case done  // everything played; activity will end shortly
        }

        public init(
            textPreview: String, phase: Phase, step: Int, maxSteps: Int,
            audioElapsedSeconds: Double, startedAt: Date
        ) {
            self.textPreview = textPreview
            self.phase = phase
            self.step = step
            self.maxSteps = maxSteps
            self.audioElapsedSeconds = audioElapsedSeconds
            self.startedAt = startedAt
        }
    }

    /// Voice / speaker label shown at the top of the activity.
    public var voiceName: String
    /// Language label (e.g., "English").
    public var languageName: String

    public init(voiceName: String, languageName: String) {
        self.voiceName = voiceName
        self.languageName = languageName
    }
}
