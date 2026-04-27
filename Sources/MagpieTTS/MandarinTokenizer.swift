//
//  MandarinTokenizer.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import Foundation

// MARK: - Jieba Word Segmenter

/// Minimal jieba-compatible word segmenter for Mandarin Chinese.
/// Uses a DAG (directed acyclic graph) built from a frequency dictionary
/// with Viterbi dynamic programming to find the optimal segmentation.
final class JiebaSegmenter {

    /// Word → log(frequency / total). Unknown words get log(1 / total).
    private let logFreq: [String: Double]
    private let logTotal: Double
    private let defaultLogFreq: Double

    /// Prefix set for building the DAG — all prefixes of all dictionary words.
    private let prefixes: Set<String>

    init(constantsDirectory: URL) throws {
        let url = constantsDirectory.appendingPathComponent("tokenizer")
            .appendingPathComponent("mandarin_jieba_dict.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_jieba_dict.json")
        }
        let data = try Data(contentsOf: url)
        guard let freqDict = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw MagpieTTSError.invalidConfiguration("mandarin_jieba_dict.json: malformed")
        }

        let total = Double(freqDict.values.reduce(0, +))
        self.logTotal = log(total)

        var logF = [String: Double]()
        logF.reserveCapacity(freqDict.count)
        var pfx = Set<String>()

        for (word, freq) in freqDict {
            logF[word] = log(Double(freq)) - logTotal
            // Add all prefixes of this word
            let chars = Array(word)
            for i in 1 ... chars.count {
                pfx.insert(String(chars[0 ..< i]))
            }
        }

        self.logFreq = logF
        self.defaultLogFreq = log(1.0) - logTotal
        self.prefixes = pfx
    }

    /// Segment text into words using DAG + Viterbi DP.
    func cut(_ text: String) -> [String] {
        let chars = Array(text)
        guard !chars.isEmpty else { return [] }

        let n = chars.count
        let dag = buildDAG(chars)
        let route = calcRoute(chars, dag: dag, n: n)

        var result = [String]()
        var i = 0
        while i < n {
            let end = route[i].end + 1
            let word = String(chars[i ..< end])
            result.append(word)
            i = end
        }
        return result
    }

    /// Build DAG: for each position i, find all j where text[i..j] is a word.
    private func buildDAG(_ chars: [Character]) -> [[Int]] {
        let n = chars.count
        var dag = [[Int]](repeating: [], count: n)

        for i in 0 ..< n {
            var j = i
            var frag = ""
            while j < n {
                frag.append(chars[j])
                if logFreq[frag] != nil {
                    dag[i].append(j)
                }
                if !prefixes.contains(frag) {
                    break
                }
                j += 1
            }
            if dag[i].isEmpty {
                dag[i] = [i]  // single character
            }
        }
        return dag
    }

    /// Viterbi DP: find the path that maximizes sum of log frequencies.
    /// Works backward from end to start.
    private func calcRoute(_ chars: [Character], dag: [[Int]], n: Int) -> [(
        logProb: Double, end: Int
    )] {
        var route = [(logProb: Double, end: Int)](repeating: (0.0, 0), count: n + 1)
        // route[n] = (0, 0) — sentinel

        for i in stride(from: n - 1, through: 0, by: -1) {
            var bestProb = -Double.infinity
            var bestEnd = i
            for j in dag[i] {
                let word = String(chars[i ... j])
                let freq = logFreq[word] ?? defaultLogFreq
                let prob = freq + route[j + 1].logProb
                if prob > bestProb {
                    bestProb = prob
                    bestEnd = j
                }
            }
            route[i] = (bestProb, bestEnd)
        }
        return route
    }
}

// MARK: - Pinyin Converter

/// Converts Chinese characters to pinyin (TONE3 format, e.g., "ni3", "hao3").
/// Uses a character dictionary (41K entries) with phrase dictionary override (47K entries)
/// for polyphone disambiguation. Replicates pypinyin's lazy_pinyin behavior.
final class PinyinConverter {

    /// Single character → list of pinyin readings (TONE3 format).
    /// First entry is the default reading.
    private let charDict: [Character: [String]]

    /// Multi-character phrase → list of pinyin readings (one per character).
    private let phraseDict: [String: [String]]

    init(constantsDirectory: URL) throws {
        let tokDir = constantsDirectory.appendingPathComponent("tokenizer")

        // Load character → pinyin dict
        let charURL = tokDir.appendingPathComponent("mandarin_pypinyin_char_dict.json")
        guard FileManager.default.fileExists(atPath: charURL.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_pypinyin_char_dict.json")
        }
        let charData = try Data(contentsOf: charURL)
        guard
            let charJson = try JSONSerialization.jsonObject(with: charData)
                as? [String: [String]]
        else {
            throw MagpieTTSError.invalidConfiguration("mandarin_pypinyin_char_dict.json: malformed")
        }
        var cd = [Character: [String]]()
        cd.reserveCapacity(charJson.count)
        for (key, value) in charJson {
            if let ch = key.first, key.count == 1 {
                cd[ch] = value
            }
        }
        self.charDict = cd

        // Load phrase → pinyin dict
        let phraseURL = tokDir.appendingPathComponent("mandarin_pypinyin_phrase_dict.json")
        guard FileManager.default.fileExists(atPath: phraseURL.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_pypinyin_phrase_dict.json")
        }
        let phraseData = try Data(contentsOf: phraseURL)
        guard
            let phraseDict = try JSONSerialization.jsonObject(with: phraseData)
                as? [String: [String]]
        else {
            throw MagpieTTSError.invalidConfiguration(
                "mandarin_pypinyin_phrase_dict.json: malformed")
        }
        self.phraseDict = phraseDict
    }

    /// Convert a word (from jieba segmentation) to pinyin sequence.
    /// Chinese characters → pinyin; non-Chinese characters → individual characters.
    /// Replicates: lazy_pinyin(word, style=TONE3, neutral_tone_with_five=True,
    ///                         errors=lambda en: [c for c in en])
    func wordToPinyin(_ word: String) -> [String] {
        // Check phrase dict first (for multi-char polyphone disambiguation)
        if word.count > 1, let pinyins = phraseDict[word] {
            return pinyins
        }

        var result = [String]()
        for char in word {
            if let pinyins = charDict[char] {
                // Use first (default) pronunciation
                result.append(pinyins[0])
            } else {
                // Non-Chinese character: pass through as-is
                result.append(String(char))
            }
        }
        return result
    }
}

// MARK: - Mandarin Tokenizer

/// Complete Mandarin Chinese tokenizer for Magpie TTS.
/// Pipeline: text → jieba word segmentation → pypinyin → phoneme dict → token IDs.
/// Replicates NeMo's ChineseG2p + IPATokenizer pipeline.
final class MandarinTokenizer {

    private let jieba: JiebaSegmenter
    private let pinyin: PinyinConverter
    private let phonemeDict: [String: [String]]  // pinyin syllable → IPA phonemes
    private let toneDict: [Character: String]  // tone digit → prefixed tone
    private let asciiLetterDict: [Character: String]  // ASCII letter → output token
    private let token2id: [String: Int]
    private let tokens: Set<String>
    private let padWithSpace: Bool

    init(constantsDirectory: URL) throws {
        self.jieba = try JiebaSegmenter(constantsDirectory: constantsDirectory)
        self.pinyin = try PinyinConverter(constantsDirectory: constantsDirectory)

        // Load pinyin → phoneme dict (413 entries)
        let pdURL = constantsDirectory.appendingPathComponent("tokenizer")
            .appendingPathComponent("mandarin_phoneme_pinyin_dict.json")
        guard FileManager.default.fileExists(atPath: pdURL.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_phoneme_pinyin_dict.json")
        }
        let pdData = try Data(contentsOf: pdURL)
        guard
            let phonemeDict = try JSONSerialization.jsonObject(with: pdData)
                as? [String: [String]]
        else {
            throw MagpieTTSError.invalidConfiguration(
                "mandarin_phoneme_pinyin_dict.json: malformed")
        }
        self.phonemeDict = phonemeDict

        // Load tone dict (5 entries: "1"→"#1", etc.)
        let toneURL = constantsDirectory.appendingPathComponent("tokenizer")
            .appendingPathComponent("mandarin_phoneme_tone_dict.json")
        guard FileManager.default.fileExists(atPath: toneURL.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_phoneme_tone_dict.json")
        }
        let toneData = try Data(contentsOf: toneURL)
        guard let toneJson = try JSONSerialization.jsonObject(with: toneData) as? [String: String]
        else {
            throw MagpieTTSError.invalidConfiguration("mandarin_phoneme_tone_dict.json: malformed")
        }
        var td = [Character: String]()
        for (k, v) in toneJson {
            if let ch = k.first { td[ch] = v }
        }
        self.toneDict = td

        // Load ASCII letter dict (26 entries: "A"→"A", etc.)
        let asciiURL = constantsDirectory.appendingPathComponent("tokenizer")
            .appendingPathComponent("mandarin_phoneme_ascii_letter_dict.json")
        guard FileManager.default.fileExists(atPath: asciiURL.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_phoneme_ascii_letter_dict.json")
        }
        let asciiData = try Data(contentsOf: asciiURL)
        guard let asciiJson = try JSONSerialization.jsonObject(with: asciiData) as? [String: String]
        else {
            throw MagpieTTSError.invalidConfiguration(
                "mandarin_phoneme_ascii_letter_dict.json: malformed")
        }
        var ad = [Character: String]()
        for (k, v) in asciiJson {
            if let ch = k.first { ad[ch] = v }
        }
        self.asciiLetterDict = ad

        // Load token2id
        let t2iURL = constantsDirectory.appendingPathComponent("tokenizer")
            .appendingPathComponent("mandarin_phoneme_token2id.json")
        guard FileManager.default.fileExists(atPath: t2iURL.path) else {
            throw MagpieTTSError.constantsNotFound("mandarin_phoneme_token2id.json")
        }
        let t2iData = try Data(contentsOf: t2iURL)
        guard let token2id = try JSONSerialization.jsonObject(with: t2iData) as? [String: Int]
        else {
            throw MagpieTTSError.invalidConfiguration("mandarin_phoneme_token2id.json: malformed")
        }
        self.token2id = token2id
        self.tokens = Set(token2id.keys)

        // Load pad_with_space from metadata
        let metaURL = constantsDirectory.appendingPathComponent("tokenizer")
            .appendingPathComponent("tokenizer_metadata.json")
        if FileManager.default.fileExists(atPath: metaURL.path),
            let metaData = try? Data(contentsOf: metaURL),
            let metaJson = try? JSONSerialization.jsonObject(with: metaData) as? [String: Any]
        {
            if let padDict = metaJson["pad_with_space"] as? [String: Bool] {
                self.padWithSpace = padDict["mandarin_phoneme"] ?? true
            } else {
                self.padWithSpace = true
            }
        } else {
            self.padWithSpace = true
        }
    }

    /// Encode text to local token IDs (before language offset).
    func encode(_ text: String) -> [Int] {
        // Step 1: Lowercase ASCII (NeMo: set_grapheme_case with ascii_letter_case="lower")
        let lowered = text.lowercased()

        // Step 2: jieba word segmentation
        let words = jieba.cut(lowered)

        // Step 3: Convert each word to pinyin, then to phonemes
        var phonemeSeq = [String]()
        for word in words {
            let pinyinSeq = pinyin.wordToPinyin(word)
            for py in pinyinSeq {
                guard !py.isEmpty else { continue }
                let lastChar = py.last!
                if let tone = toneDict[lastChar] {
                    // This is a pinyin syllable with tone number
                    let syllable = String(py.dropLast())
                    if let phonemes = phonemeDict[syllable] {
                        phonemeSeq.append(contentsOf: phonemes)
                        phonemeSeq.append(tone)
                    }
                    // Skip unknown syllables (matches NeMo behavior)
                } else if let letter = asciiLetterDict[Character(py.uppercased())] {
                    // ASCII letter
                    phonemeSeq.append(letter)
                } else {
                    // Punctuation or other symbol — pass through
                    phonemeSeq.append(py)
                }
            }
        }

        // Step 4: Encode through IPATokenizer logic
        return encodeFromG2P(phonemeSeq)
    }

    private func encodeFromG2P(_ phonemes: [String]) -> [Int] {
        var ps = [String]()
        let space = " "

        for p in phonemes {
            if p == space {
                if !ps.isEmpty && ps.last != space { ps.append(p) }
            } else if tokens.contains(p) {
                ps.append(p)
            }
            // Unknown symbols silently skipped (same as NeMo)
        }

        while let last = ps.last, last == space { ps.removeLast() }

        if padWithSpace {
            ps = [space] + ps + [space]
        }

        return ps.compactMap { token2id[$0] }
    }
}
