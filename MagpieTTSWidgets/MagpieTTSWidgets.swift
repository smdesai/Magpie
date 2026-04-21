//
//  MagpieTTSWidgets.swift
//  MagpieTTSWidgets
//
//  Created by Sachin Desai on 4/21/26.
//

import ActivityKit
import SwiftUI
import WidgetKit

@main
struct MagpieTTSWidgetsBundle: WidgetBundle {
    var body: some Widget {
        TTSLiveActivityWidget()
    }
}

struct TTSLiveActivityWidget: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: TTSActivityAttributes.self) { context in
            LockScreenView(attributes: context.attributes, state: context.state)
                .activityBackgroundTint(Color.black.opacity(0.85))
                .activitySystemActionForegroundColor(.white)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    HStack(spacing: 6) {
                        Image(systemName: "waveform")
                            .foregroundStyle(.white)
                        Text(context.attributes.voiceName)
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.white)
                            .lineLimit(1)
                    }
                }
                DynamicIslandExpandedRegion(.trailing) {
                    Text(audioElapsedString(context.state.audioElapsedSeconds))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.white)
                        .monospacedDigit()
                }
                DynamicIslandExpandedRegion(.center) {
                    ProgressBar(state: context.state)
                        .frame(height: 3)
                        .padding(.horizontal, 2)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(phaseLabel(context.state.phase))
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.secondary)
                        Text(context.state.textPreview)
                            .font(.caption)
                            .foregroundStyle(.white)
                            .lineLimit(2)
                    }
                }
            } compactLeading: {
                Image(systemName: "waveform")
                    .foregroundStyle(.white)
            } compactTrailing: {
                Text(audioElapsedString(context.state.audioElapsedSeconds))
                    .font(.caption2.weight(.semibold))
                    .monospacedDigit()
                    .foregroundStyle(.white)
            } minimal: {
                Image(systemName: "waveform")
                    .foregroundStyle(.white)
            }
            .widgetURL(URL(string: "magpietts://activity"))
            .keylineTint(.white)
        }
    }
}

// MARK: - Lock Screen

private struct LockScreenView: View {
    let attributes: TTSActivityAttributes
    let state: TTSActivityAttributes.ContentState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Image(systemName: "waveform")
                    .foregroundStyle(.white)
                Text(attributes.voiceName)
                    .font(.headline)
                    .foregroundStyle(.white)
                Text("· \(attributes.languageName)")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                Text(audioElapsedString(state.audioElapsedSeconds))
                    .font(.headline)
                    .monospacedDigit()
                    .foregroundStyle(.white)
            }

            Text(state.textPreview)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.9))
                .lineLimit(2)

            ProgressBar(state: state)
                .frame(height: 4)

            HStack {
                Text(phaseLabel(state.phase))
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                Spacer()
                if state.phase == .generating, state.maxSteps > 0 {
                    Text("step \(state.step)/\(state.maxSteps)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}

// MARK: - Progress Bar

private struct ProgressBar: View {
    let state: TTSActivityAttributes.ContentState

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(.white.opacity(0.2))
                Capsule().fill(.white).frame(width: geo.size.width * fraction)
            }
        }
    }

    private var fraction: Double {
        switch state.phase {
        case .encoding: return 0.05
        case .preparingVoice: return 0.15
        case .generating:
            guard state.maxSteps > 0 else { return 0.3 }
            // generation occupies the 15%..95% band of the bar
            let gen = min(1.0, Double(state.step) / Double(state.maxSteps))
            return 0.15 + gen * 0.80
        case .finishing: return 0.98
        case .done: return 1.0
        }
    }
}

// MARK: - Helpers

private func phaseLabel(_ phase: TTSActivityAttributes.ContentState.Phase) -> String {
    switch phase {
    case .encoding: return "Encoding text"
    case .preparingVoice: return "Preparing voice"
    case .generating: return "Speaking"
    case .finishing: return "Finishing"
    case .done: return "Done"
    }
}

private func audioElapsedString(_ seconds: Double) -> String {
    let total = Int(seconds.rounded())
    let m = total / 60
    let s = total % 60
    return String(format: "%d:%02d", m, s)
}
