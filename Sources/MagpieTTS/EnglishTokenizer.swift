//
//  EnglishTokenizer.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import Foundation

/// English IPA tokenizer for Magpie TTS.
///
/// Replicates NeMo's `IPATokenizer` + `IpaG2p` pipeline:
///   text → NFC normalize → word split → phoneme dict lookup → token IDs
///
/// Requires `english_phoneme_dict.json` and `english_token2id.json` in the constants directory
/// (exported by the Python `export_constants.py` script).
public final class EnglishTokenizer {

    private let phonemeDict: [String: [String]]  // UPPERCASE word → phoneme list
    private let token2id: [String: Int]
    private let eosTokenId: Int
    private let spaceId: Int
    private let tokens: Set<String>
    private let punctList: Set<Character>

    // Word | |unchanged| | non-word
    // Matches NeMo's _WORDS_RE_EN = r"([A-Za-z]+(?:[A-Za-z\-']*[A-Za-z]+)*)|(\|[^|]*\|)|([^A-Za-z|]+)"
    private static let wordRegex =
        #/([A-Za-z]+(?:[A-Za-z\-']*[A-Za-z]+)*)|(\|[^\|]*\|)|([^A-Za-z\|]+)/#

    public init(constantsDirectory: URL, eosTokenId: Int = 2361) throws {
        let tokDir = constantsDirectory.appendingPathComponent("tokenizer")
        let dictURL = tokDir.appendingPathComponent("english_phoneme_dict.json")
        let t2iURL = tokDir.appendingPathComponent("english_token2id.json")

        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw MagpieTTSError.constantsNotFound("english_phoneme_dict.json")
        }
        guard FileManager.default.fileExists(atPath: t2iURL.path) else {
            throw MagpieTTSError.constantsNotFound("english_token2id.json")
        }

        let dictData = try Data(contentsOf: dictURL)
        guard
            let rawDict = try JSONSerialization.jsonObject(with: dictData)
                as? [String: [String]]
        else {
            throw MagpieTTSError.invalidConfiguration("english_phoneme_dict.json: malformed")
        }
        self.phonemeDict = rawDict

        let t2iData = try Data(contentsOf: t2iURL)
        guard let rawT2i = try JSONSerialization.jsonObject(with: t2iData) as? [String: Int]
        else {
            throw MagpieTTSError.invalidConfiguration("english_token2id.json: malformed")
        }
        self.token2id = rawT2i

        self.eosTokenId = eosTokenId
        guard let spaceId = rawT2i[" "] else {
            throw MagpieTTSError.invalidConfiguration("english_token2id.json: missing space token")
        }
        self.spaceId = spaceId
        self.tokens = Set(rawT2i.keys)
        self.punctList = Set("!\"(),-.:;?[]{}/")
    }

    /// Tokenize English text to token IDs ready for the TTS model.
    public func tokenize(_ text: String) -> [Int32] {
        let normalized = text.precomposedStringWithCanonicalMapping  // NFC
        let words = wordTokenize(normalized)
        let phonemes = g2p(words)
        var ids = encode(phonemes)
        ids.append(Int32(eosTokenId))
        return ids
    }

    // MARK: - Word Tokenization

    /// Split text into (tokens, isUnchanged) pairs.
    /// Matches NeMo's `english_word_tokenize`.
    private func wordTokenize(_ text: String) -> [(tokens: [String], unchanged: Bool)] {
        var result = [(tokens: [String], unchanged: Bool)]()
        for match in text.matches(of: Self.wordRegex) {
            let (_, word, unchanged, punct) = match.output
            if let w = word, !w.isEmpty {
                result.append(([w.lowercased()], false))
            } else if let u = unchanged, !u.isEmpty {
                // Strip | delimiters, split by space
                let inner = u.dropFirst().dropLast()
                result.append((inner.split(separator: " ").map(String.init), true))
            } else if let p = punct, !p.isEmpty {
                result.append(([String(p)], false))
            }
        }
        return result
    }

    // MARK: - Grapheme to Phoneme

    /// Convert word tokens to phoneme/grapheme sequence.
    /// Matches NeMo's `IpaG2p.__call__` + `parse_one_word`.
    private func g2p(_ words: [(tokens: [String], unchanged: Bool)]) -> [String] {
        var result = [String]()
        for (tokens, unchanged) in words {
            if unchanged {
                // Pipe-delimited unchanged text — keep as-is
                for (i, token) in tokens.enumerated() {
                    if i > 0 { result.append(" ") }
                    result.append(contentsOf: token.map(String.init))
                }
                continue
            }

            for token in tokens {
                let phonemes = parseOneWord(token)
                result.append(contentsOf: phonemes)
            }
        }
        return result
    }

    /// Convert a single word to phonemes or graphemes.
    /// Matches NeMo's `IpaG2p.parse_one_word` for en-US.
    private func parseOneWord(_ word: String) -> [String] {
        let upper = word.uppercased()

        // Pure punctuation / non-alpha → keep as characters
        if upper.allSatisfy({ !$0.isLetter && !$0.isNumber }) {
            return upper.map(String.init)
        }

        // 's suffix handling (with apostrophe)
        if upper.count > 2, upper.hasSuffix("'S") {
            let stem = String(upper.dropLast(2))
            if phonemeDict[upper] == nil, let stemPhonemes = phonemeDict[stem] {
                return appendPossessiveSuffix(stemPhonemes, lastChar: stem.last!)
            }
        }

        // s suffix handling (without apostrophe)
        if upper.count > 1, upper.hasSuffix("S"), !upper.hasSuffix("'S") {
            let stem = String(upper.dropLast(1))
            if phonemeDict[upper] == nil, let stemPhonemes = phonemeDict[stem] {
                if stem.last == "T" || stem.last == "t" {
                    return stemPhonemes + ["s"]
                } else {
                    return stemPhonemes + ["z"]
                }
            }
        }

        // Dictionary lookup — use first pronunciation
        if let phonemes = phonemeDict[upper] {
            return phonemes
        }

        // OOV → uppercase graphemes
        return upper.map(String.init)
    }

    /// Append possessive suffix phonemes based on final character.
    private func appendPossessiveSuffix(_ phonemes: [String], lastChar: Character) -> [String] {
        switch lastChar {
        case "T", "t": return phonemes + ["s"]
        case "S", "s": return phonemes + ["ɪ", "z"]
        default: return phonemes + ["z"]
        }
    }

    // MARK: - Encode

    /// Map phoneme/grapheme sequence to token IDs.
    /// Matches NeMo's `IPATokenizer.encode_from_g2p`.
    private func encode(_ phonemes: [String]) -> [Int32] {
        var ps = [String]()
        let space = " "

        for p in phonemes {
            if p == space {
                if !ps.isEmpty, ps.last != space { ps.append(p) }
            } else if tokens.contains(p) {
                ps.append(p)
            } else if punctList.contains(Character(p)) {
                ps.append(p)
            }
            // Unknown symbols are silently skipped (matching NeMo)
        }

        // Remove trailing spaces
        while ps.last == space { ps.removeLast() }

        return ps.compactMap { token2id[$0] }.map(Int32.init)
    }
}
