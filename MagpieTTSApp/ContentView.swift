import SwiftUI

struct ContentView: View {
    @StateObject private var vm = TTSViewModel()
    @State private var showShareSheet = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // MARK: - Configuration
                    configSection

                    // MARK: - Text Input
                    textSection

                    // MARK: - Model Loading
                    if vm.isPreparing {
                        HStack(spacing: 8) {
                            ProgressView()
                                .controlSize(.small)
                            Text(vm.statusMessage)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(10)
                        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                    }

                    // MARK: - Actions
                    actionBar

                    // MARK: - Progress
                    if vm.isGenerating {
                        progressSection
                    }

                    // MARK: - Results
                    if vm.hasAudio {
                        resultsSection
                    }

                    // MARK: - Error
                    if let error = vm.errorMessage {
                        errorBanner(error)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
            }
            .background(Color(.systemGroupedBackground))
            .onTapGesture {
                UIApplication.shared.sendAction(
                    #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
            .toolbar {
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 6) {
                        Image(systemName: "bird")
                            .font(.headline)
                        Text("Magpie TTS")
                            .font(.headline)
                    }
                }
            }
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    // MARK: - Config Section

    private var configSection: some View {
        VStack(spacing: 0) {
            // Language
            HStack {
                Label("Language", systemImage: "globe")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Language", selection: $vm.selectedLanguage) {
                    ForEach(vm.languages, id: \.self) { lang in
                        Text(displayName(for: lang)).tag(lang)
                    }
                }
                .pickerStyle(.menu)
                .tint(.primary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .onChange(of: vm.selectedLanguage) { _, _ in
                vm.text = vm.defaultTextForLanguage
            }

            Divider().padding(.horizontal, 12)

            // Voice
            HStack {
                Label("Voice", systemImage: "person.wave.2")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Voice", selection: $vm.selectedVoice) {
                    ForEach(vm.voices) { voice in
                        Text(voice.name).tag(voice)
                    }
                }
                .pickerStyle(.menu)
                .tint(.primary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
        }
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Text Input

    private var textSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label("Text", systemImage: "text.alignleft")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)
                Spacer()
                if !vm.text.isEmpty {
                    Button {
                        vm.text = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    .buttonStyle(.plain)
                }
            }

            TextEditor(text: $vm.text)
                .frame(minHeight: 160, maxHeight: 600)
                .padding(10)
                .font(.body)
                .scrollContentBackground(.hidden)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                .overlay(
                    RoundedRectangle(cornerRadius: 10)
                        .strokeBorder(.quaternary, lineWidth: 1)
                )
        }
    }

    // MARK: - Action Bar

    private var actionBar: some View {
        let inputDisabled =
            vm.isPreparing || vm.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty

        return HStack(spacing: 10) {
            // Generate
            Button {
                UIApplication.shared.sendAction(
                    #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                vm.generate()
            } label: {
                Label("Generate", systemImage: "waveform")
                    .font(.subheadline.weight(.medium))
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.capsule)
            .tint(.accentColor)
            .disabled(vm.isGenerating || inputDisabled)

            // Stream
            Button {
                UIApplication.shared.sendAction(
                    #selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                if vm.isStreaming { vm.stopStreaming() } else { vm.stream() }
            } label: {
                Label(
                    vm.isStreaming ? "Stop" : "Stream",
                    systemImage: vm.isStreaming ? "stop.fill" : "play.fill"
                )
                .font(.subheadline.weight(.medium))
            }
            .buttonStyle(.bordered)
            .buttonBorderShape(.capsule)
            .tint(vm.isStreaming ? .red : .orange)
            .disabled((vm.isGenerating && !vm.isStreaming) || inputDisabled)

            Spacer()

            // Play (only when audio available)
            if vm.hasAudio {
                Button {
                    vm.play()
                } label: {
                    Image(systemName: vm.isPlaying ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title2)
                }
                .tint(.green)

                // Share
                if let item = vm.shareItem() {
                    ShareLink(
                        item: item.temporaryURL,
                        preview: SharePreview(
                            "Magpie TTS Audio", image: Image(systemName: "waveform"))
                    ) {
                        Image(systemName: "square.and.arrow.up.circle.fill")
                            .font(.title2)
                    }
                    .tint(.blue)
                }
            }
        }
    }

    // MARK: - Progress

    private var progressSection: some View {
        HStack(spacing: 8) {
            ProgressView()
                .controlSize(.small)
            Text(vm.progress)
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(12)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Results

    private var resultsSection: some View {
        Group {
            if let stats = vm.generationStats {
                HStack(spacing: 6) {
                    Image(systemName: "clock")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Text(stats)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                }
                .padding(10)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
            }
        }
        .transition(.opacity.combined(with: .move(edge: .bottom)))
        .animation(.spring(duration: 0.3), value: vm.hasAudio)
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(3)
            Spacer()
        }
        .padding(10)
        .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
    }

    // MARK: - Helpers

    private func displayName(for lang: Language) -> String {
        switch lang {
        case .english: return "English"
        case .spanish: return "Spanish"
        case .german: return "German"
        case .mandarin: return "Mandarin"
        case .japanese: return "Japanese"
        case .french: return "French"
        case .hindi: return "Hindi"
        case .italian: return "Italian"
        case .vietnamese: return "Vietnamese"
        }
    }
}

#Preview {
    ContentView()
}
