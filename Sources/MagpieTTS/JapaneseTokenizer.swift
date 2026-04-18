//
//  JapaneseTokenizer.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import Foundation

/// Japanese katakana accent G2P tokenizer for Magpie TTS.
///
/// Replicates NeMo's `JapaneseKatakanaAccentG2p` + `IPATokenizer` pipeline:
/// 1. Run OpenJTalk frontend → NJD word features
/// 2. Group words into accent chains (chain_flag)
/// 3. Calculate pitch accent pattern (0=low, 1=high) per mora
/// 4. Emit pitch marker + katakana mora tokens
/// 5. Insert punctuation inline
/// 6. Map through token2id
final class JapaneseTokenizer {

    private let token2id: [String: Int]
    private let tokens: Set<String>
    private let punctSet: Set<String>
    private let padWithSpace: Bool

    // Mora splitting: matches katakana mora units
    // Pattern: full kana + optional small kana | standalone small kana | chōonpu
    private static let moraRegex = try! NSRegularExpression(
        pattern: "[ア-ンヴ][ャュョァィゥェォヮ]?|[ァィゥェォヵヶッャュョヮ]|ー"
    )

    // Part-of-speech tags for punctuation in OpenJTalk
    private static let punctPOS: Set<String> = ["記号", "補助記号"]

    init(constantsDirectory: URL) throws {
        // Load token2id
        let t2iURL = constantsDirectory.appendingPathComponent("japanese_phoneme_token2id.json")
        guard FileManager.default.fileExists(atPath: t2iURL.path) else {
            throw MagpieTTSError.constantsNotFound("japanese_phoneme_token2id.json")
        }
        let t2iData = try Data(contentsOf: t2iURL)
        self.token2id = try JSONSerialization.jsonObject(with: t2iData) as! [String: Int]
        self.tokens = Set(token2id.keys)

        // Load punctuation list
        let punctURL = constantsDirectory.appendingPathComponent(
            "japanese_phoneme_punctuation.json")
        guard FileManager.default.fileExists(atPath: punctURL.path) else {
            throw MagpieTTSError.constantsNotFound("japanese_phoneme_punctuation.json")
        }
        let punctData = try Data(contentsOf: punctURL)
        let punctList = try JSONSerialization.jsonObject(with: punctData) as! [String]
        self.punctSet = Set(punctList)

        // Load pad_with_space from metadata
        let metaURL = constantsDirectory.appendingPathComponent("tokenizer_metadata.json")
        if FileManager.default.fileExists(atPath: metaURL.path) {
            let metaData = try Data(contentsOf: metaURL)
            let metaJson = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
            if let padDict = metaJson["pad_with_space"] as? [String: Bool] {
                self.padWithSpace = padDict["japanese_phoneme"] ?? true
            } else {
                self.padWithSpace = true
            }
        } else {
            self.padWithSpace = true
        }

        // Initialize OpenJTalk with bundled dictionary
        let dictPath = constantsDirectory.appendingPathComponent("open_jtalk_dic").path
        guard OpenJTalkBridge.initialize(dictionaryPath: dictPath) else {
            throw MagpieTTSError.invalidConfiguration(
                "Failed to initialize OpenJTalk with dictionary at \(dictPath)"
            )
        }
    }

    /// Encode Japanese text to local token IDs (before language offset).
    func encode(_ text: String) -> [Int] {
        let g2pResult = g2p(text)
        return encodeFromG2P(g2pResult)
    }

    // MARK: - G2P Pipeline

    /// Convert text to a sequence of tokens (pitch markers, katakana, punctuation).
    private func g2p(_ text: String) -> [String] {
        guard let words = OpenJTalkBridge.runFrontend(text), !words.isEmpty else {
            return []
        }

        var result = [String]()
        var currentChain = [OpenJTalkBridge.NJDWord]()
        var chainAcc = 0

        for word in words {
            // Check if this word is punctuation
            if isPunctuation(word) {
                // Flush current chain
                if !currentChain.isEmpty {
                    result.append(contentsOf: processChain(currentChain, acc: chainAcc))
                    currentChain.removeAll()
                }
                // Insert punctuation
                let normalized = word.string.precomposedStringWithCompatibilityMapping  // NFKC
                if punctSet.contains(normalized) {
                    result.append(normalized)
                } else if punctSet.contains(word.string) {
                    result.append(word.string)
                }
                continue
            }

            // Check if all ASCII letters
            if word.string.allSatisfy({ $0.isASCII && $0.isLetter }) {
                // Flush current chain
                if !currentChain.isEmpty {
                    result.append(contentsOf: processChain(currentChain, acc: chainAcc))
                    currentChain.removeAll()
                }
                // Insert ASCII letters as uppercase
                for ch in word.string.uppercased() {
                    result.append(String(ch))
                }
                continue
            }

            // Regular word — add to chain
            if word.chainFlag == 1 && !currentChain.isEmpty {
                // Continue current chain
                currentChain.append(word)
            } else {
                // Flush previous chain and start new one
                if !currentChain.isEmpty {
                    result.append(contentsOf: processChain(currentChain, acc: chainAcc))
                }
                currentChain = [word]
                chainAcc = word.acc
            }
        }

        // Flush remaining chain
        if !currentChain.isEmpty {
            result.append(contentsOf: processChain(currentChain, acc: chainAcc))
        }

        return result
    }

    /// Process a chain of words: split into mora, calculate pitch pattern,
    /// and return interleaved pitch markers + katakana tokens.
    private func processChain(_ chain: [OpenJTalkBridge.NJDWord], acc: Int) -> [String] {
        // Collect all mora from the chain
        var allMora = [String]()
        for word in chain {
            let pron = word.pron
            // Skip words with no pronunciation (e.g., "*")
            if pron == "*" || pron.isEmpty { continue }
            let mora = splitMora(pron)
            allMora.append(contentsOf: mora)
        }

        guard !allMora.isEmpty else { return [] }

        // Calculate pitch pattern
        let pitchPattern = calculatePitchPattern(acc: acc, totalMora: allMora.count)

        // Interleave pitch markers with mora
        var result = [String]()
        for (i, mora) in allMora.enumerated() {
            // Pitch marker: "0" for low, "1" for high
            if i < pitchPattern.count {
                result.append(pitchPattern[i] == 0 ? "0" : "1")
            }
            result.append(mora)
        }

        return result
    }

    /// Split a katakana pronunciation string into individual mora.
    private func splitMora(_ pron: String) -> [String] {
        let nsString = pron as NSString
        let matches = Self.moraRegex.matches(
            in: pron,
            range: NSRange(location: 0, length: nsString.length)
        )
        return matches.map { nsString.substring(with: $0.range) }
    }

    /// Calculate Japanese pitch accent pattern.
    ///
    /// - acc=0 (Heiban/flat): L-H-H-H...
    /// - acc=1 (Atamadaka/head-high): H-L-L-L...
    /// - acc=N, 1<N<mora_count (Nakadaka/mid-fall): L-H...H-L...L
    /// - acc>=mora_count (Odaka/tail-fall): L-H-H...H
    private func calculatePitchPattern(acc: Int, totalMora: Int) -> [Int] {
        guard totalMora > 0 else { return [] }

        if acc == 0 {
            // Heiban: low then all high
            return [0] + Array(repeating: 1, count: totalMora - 1)
        } else if acc == 1 {
            // Atamadaka: high then all low
            return [1] + Array(repeating: 0, count: totalMora - 1)
        } else if acc >= totalMora {
            // Odaka: low then all high
            return [0] + Array(repeating: 1, count: totalMora - 1)
        } else {
            // Nakadaka: low, then high up to acc, then low
            return [0]
                + Array(repeating: 1, count: acc - 1)
                + Array(repeating: 0, count: totalMora - acc)
        }
    }

    /// Check if an NJD word represents punctuation.
    private func isPunctuation(_ word: OpenJTalkBridge.NJDWord) -> Bool {
        // Check by POS
        if Self.punctPOS.contains(word.pos) { return true }
        // Also check if the string itself is in the punctuation set
        if punctSet.contains(word.string) { return true }
        let normalized = word.string.precomposedStringWithCompatibilityMapping
        if punctSet.contains(normalized) { return true }
        return false
    }

    // MARK: - Token Encoding

    /// Encode G2P output through token2id (matches IPATokenizer pattern).
    private func encodeFromG2P(_ phonemes: [String]) -> [Int] {
        var ps = [String]()
        let space = " "

        for p in phonemes {
            if p == space {
                if !ps.isEmpty && ps.last != space { ps.append(p) }
            } else if tokens.contains(p) {
                ps.append(p)
            } else if punctSet.contains(p) {
                ps.append(p)
            }
            // Unknown symbols silently skipped
        }

        while let last = ps.last, last == space { ps.removeLast() }

        if padWithSpace {
            ps = [space] + ps + [space]
        }

        return ps.compactMap { token2id[$0] }
    }
}
