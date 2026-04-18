//
//  PronunciationExpander.swift
//  MagpieTTSApp
//
//  Created by Sachin Desai on 4/18/26.
//

import Foundation

/// Applies a user-defined `key=value` pronunciation dictionary to input text
/// before it reaches the tokenizer.
///
/// - For IPA-phoneme languages (English / Spanish / German) the value is
///   treated as IPA and wrapped in `|…|` so the tokenizer passes it through
///   unchanged.
/// - For byte-level / char-level languages (French / Italian / Vietnamese /
///   Hindi) the value is substituted verbatim — a grapheme "respelling".
/// - Mandarin and Japanese are skipped (structured tokenizers).
/// - Any text already inside an inline `|…|` span is preserved, so inline
///   overrides in the source text always win.
enum PronunciationExpander {

    static func expand(_ text: String, language: Language, dictionaryText: String) -> String {
        let entries = parse(dictionaryText)
        guard !entries.isEmpty else { return text }

        let style: Style
        switch language {
        case .english, .spanish, .german:
            style = .wrapInPipes
        case .french, .italian, .vietnamese, .hindi:
            style = .literal
        case .mandarin, .japanese:
            return text
        }

        return applyReplacements(to: text, entries: entries, style: style)
    }

    // MARK: - Private

    private enum Style {
        case wrapInPipes
        case literal
    }

    private struct Entry {
        let key: String
        let value: String
    }

    private static func parse(_ raw: String) -> [Entry] {
        var entries: [Entry] = []
        for rawLine in raw.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty, !line.hasPrefix("#") else { continue }
            guard let eq = line.firstIndex(of: "=") else { continue }
            let key = line[..<eq].trimmingCharacters(in: .whitespaces)
            let value = line[line.index(after: eq)...].trimmingCharacters(in: .whitespaces)
            guard !key.isEmpty, !value.isEmpty else { continue }
            entries.append(Entry(key: String(key), value: String(value)))
        }
        return entries
    }

    /// Split text into alternating (unprotected, pipe-protected) segments,
    /// apply replacements only to unprotected ones, and rejoin.
    private static func applyReplacements(
        to text: String, entries: [Entry], style: Style
    ) -> String {
        let pipeRegex = try! NSRegularExpression(pattern: #"\|[^\|]*\|"#)
        let nsText = text as NSString
        let fullRange = NSRange(location: 0, length: nsText.length)
        let pipeMatches = pipeRegex.matches(in: text, range: fullRange)

        var pieces: [(String, protected: Bool)] = []
        var cursor = 0
        for match in pipeMatches {
            if match.range.location > cursor {
                let r = NSRange(location: cursor, length: match.range.location - cursor)
                pieces.append((nsText.substring(with: r), false))
            }
            pieces.append((nsText.substring(with: match.range), true))
            cursor = match.range.location + match.range.length
        }
        if cursor < nsText.length {
            let r = NSRange(location: cursor, length: nsText.length - cursor)
            pieces.append((nsText.substring(with: r), false))
        }

        return pieces.map { piece in
            piece.protected ? piece.0 : replaceAll(in: piece.0, entries: entries, style: style)
        }.joined()
    }

    private static func replaceAll(in text: String, entries: [Entry], style: Style) -> String {
        var s = text
        for entry in entries {
            let replacement: String
            switch style {
            case .wrapInPipes: replacement = "|\(entry.value)|"
            case .literal: replacement = entry.value
            }
            let escapedKey = NSRegularExpression.escapedPattern(for: entry.key)
            let pattern = "\\b\(escapedKey)\\b"
            guard
                let regex = try? NSRegularExpression(
                    pattern: pattern,
                    options: [.caseInsensitive, .useUnicodeWordBoundaries]
                )
            else { continue }
            let nsS = s as NSString
            let template = NSRegularExpression.escapedTemplate(for: replacement)
            s = regex.stringByReplacingMatches(
                in: s, range: NSRange(location: 0, length: nsS.length), withTemplate: template
            )
        }
        return s
    }
}
