//
//  SettingsView.swift
//  MagpieTTSApp
//
//  Created by Sachin Desai on 4/18/26.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("pronunciationDictionary") private var dictionary: String = ""
    @Environment(\.dismiss) private var dismiss
    @State private var confirmClear = false

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 12) {
                helpBlurb

                TextEditor(text: $dictionary)
                    .font(.body.monospaced())
                    .scrollContentBackground(.hidden)
                    .padding(10)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
                    .overlay(
                        RoundedRectangle(cornerRadius: 10)
                            .strokeBorder(.quaternary, lineWidth: 1)
                    )
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .keyboardType(.asciiCapable)
            }
            .padding(16)
            .navigationTitle("Pronunciation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Clear", role: .destructive) {
                        confirmClear = true
                    }
                    .disabled(dictionary.isEmpty)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                        .fontWeight(.semibold)
                }
            }
            .confirmationDialog(
                "Clear all entries?",
                isPresented: $confirmClear,
                titleVisibility: .visible
            ) {
                Button("Clear", role: .destructive) { dictionary = "" }
                Button("Cancel", role: .cancel) {}
            }
        }
    }

    private var helpBlurb: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("One `key=value` per line. Applied before generation.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Text(
                """
                • English / Spanish / German — value is IPA (auto-wrapped in |…|)
                  e.g. `magpie=mæɡpaɪ`
                • French / Italian / Vietnamese / Hindi — value is a respelling
                  e.g. `Worcestershire=Wouss-ter-cheur`
                • Mandarin / Japanese — not supported, entries ignored
                • Lines starting with `#` are comments
                • Inline |…| in the input text always wins
                """
            )
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
    }
}

#Preview {
    SettingsView()
}
