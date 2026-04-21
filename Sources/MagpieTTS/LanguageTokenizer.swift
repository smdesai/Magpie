//
//  LanguageTokenizer.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import Foundation

/// Supported languages for Magpie TTS.
public enum Language: String, CaseIterable, Sendable {
    case english = "english_phoneme"
    case spanish = "spanish_phoneme"
    case german = "german_phoneme"
    case mandarin = "mandarin_phoneme"
    case japanese = "japanese_phoneme"
    case french = "french_chartokenizer"
    case hindi = "hindi_chartokenizer"
    case italian = "italian_phoneme"
    case vietnamese = "vietnamese_phoneme"

    /// All languages support text-to-tokens natively in Swift.
    public var supportsTextInput: Bool {
        return true
    }

    /// Human-readable name for UI display.
    public var displayName: String {
        switch self {
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

// MARK: - Tokenizer Metadata

/// Loaded from `tokenizer_metadata.json`. Contains offsets and config for all languages.
struct TokenizerMetadata {
    let offsets: [String: Int]
    let eosTokenId: Int

    init(constantsDirectory: URL) throws {
        let url = constantsDirectory.appendingPathComponent("tokenizer_metadata.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MagpieTTSError.constantsNotFound("tokenizer_metadata.json")
        }
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        self.offsets = json["offsets"] as! [String: Int]
        self.eosTokenId = json["eos_token_id"] as! Int
    }
}

// MARK: - Multi-Language Tokenizer

/// Dispatches text tokenization to the appropriate language-specific tokenizer.
/// All tokenizers return global token IDs (with language offset applied) plus model EOS.
public final class MultiLanguageTokenizer {

    private let constantsDirectory: URL
    private let metadata: TokenizerMetadata
    private var tokenizers: [Language: Any] = [:]

    public init(constantsDirectory: URL) throws {
        self.constantsDirectory = constantsDirectory
        self.metadata = try TokenizerMetadata(constantsDirectory: constantsDirectory)
    }

    /// Tokenize text for the given language. Returns global token IDs including model EOS.
    public func tokenize(_ text: String, language: Language) throws -> [Int32] {
        guard language.supportsTextInput else {
            throw MagpieTTSError.invalidConfiguration(
                "\(language) requires pre-tokenized input. Use generate(tokenIDs:) with Python-generated tokens."
            )
        }

        let offset = metadata.offsets[language.rawValue]!
        let eos = metadata.eosTokenId

        switch language {
        case .english:
            let tok = try getEnglishTokenizer()
            return tok.tokenize(text)

        case .spanish:
            let tok = try getIPATokenizer(language: .spanish)
            var ids = tok.encode(text).map { Int32($0 + offset) }
            ids.append(Int32(eos))
            return ids

        case .german:
            let tok = try getIPATokenizer(language: .german)
            var ids = tok.encode(text).map { Int32($0 + offset) }
            ids.append(Int32(eos))
            return ids

        case .french, .italian, .vietnamese:
            var ids = ByT5Encoder.encode(text, offset: offset)
            ids.append(Int32(eos))
            return ids

        case .hindi:
            let tok = try getHindiTokenizer()
            var ids = tok.encode(text).map { Int32($0 + offset) }
            ids.append(Int32(eos))
            return ids

        case .mandarin:
            let tok = try getMandarinTokenizer()
            var ids = tok.encode(text).map { Int32($0 + offset) }
            ids.append(Int32(eos))
            return ids

        case .japanese:
            let tok = try getJapaneseTokenizer()
            var ids = tok.encode(text).map { Int32($0 + offset) }
            ids.append(Int32(eos))
            return ids
        }
    }

    /// Apply language offset + model EOS to pre-computed local token IDs.
    /// Use this for Mandarin/Japanese where tokenization was done in Python.
    public func applyOffset(_ localIDs: [Int32], language: Language) -> [Int32] {
        let offset = Int32(metadata.offsets[language.rawValue]!)
        var ids = localIDs.map { $0 + offset }
        ids.append(Int32(metadata.eosTokenId))
        return ids
    }

    // MARK: - Lazy Loading

    private func getEnglishTokenizer() throws -> EnglishTokenizer {
        if let tok = tokenizers[.english] as? EnglishTokenizer { return tok }
        let tok = try EnglishTokenizer(constantsDirectory: constantsDirectory)
        tokenizers[.english] = tok
        return tok
    }

    private func getIPATokenizer(language: Language) throws -> IPALanguageTokenizer {
        if let tok = tokenizers[language] as? IPALanguageTokenizer { return tok }
        let tok = try IPALanguageTokenizer(
            language: language, constantsDirectory: constantsDirectory)
        tokenizers[language] = tok
        return tok
    }

    private func getHindiTokenizer() throws -> HindiCharTokenizer {
        if let tok = tokenizers[.hindi] as? HindiCharTokenizer { return tok }
        let tok = try HindiCharTokenizer(constantsDirectory: constantsDirectory)
        tokenizers[.hindi] = tok
        return tok
    }

    private func getMandarinTokenizer() throws -> MandarinTokenizer {
        if let tok = tokenizers[.mandarin] as? MandarinTokenizer { return tok }
        let tok = try MandarinTokenizer(constantsDirectory: constantsDirectory)
        tokenizers[.mandarin] = tok
        return tok
    }

    private func getJapaneseTokenizer() throws -> JapaneseTokenizer {
        if let tok = tokenizers[.japanese] as? JapaneseTokenizer { return tok }
        let tok = try JapaneseTokenizer(constantsDirectory: constantsDirectory)
        tokenizers[.japanese] = tok
        return tok
    }
}

// MARK: - IPA Language Tokenizer (Spanish, German)

/// IPA G2P tokenizer for languages that use phoneme dictionaries.
/// Replicates NeMo's IpaG2p + IPATokenizer for non-English locales.
final class IPALanguageTokenizer {

    private let phonemeDict: [String: [String]]
    private let token2id: [String: Int]
    private let tokens: Set<String>
    private let heteronyms: Set<String>
    private let punctList: Set<String>
    private let graphemeCase: GraphemeCase
    private let graphemePrefix: String
    private let padWithSpace: Bool

    enum GraphemeCase { case upper, mixed }

    // Regex for Latin + accented characters (any_locale_word_tokenize)
    private static let anyLocaleWordRegex = try! NSRegularExpression(
        pattern:
            #"([A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+(?:[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF\-']*[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]+)*)|(\|[^\|]*\|)|([^A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF\|]+)"#
    )

    init(language: Language, constantsDirectory: URL) throws {
        let name = language.rawValue

        // Load phoneme dict
        let dictURL = constantsDirectory.appendingPathComponent("\(name)_phoneme_dict.json")
        guard FileManager.default.fileExists(atPath: dictURL.path) else {
            throw MagpieTTSError.constantsNotFound("\(name)_phoneme_dict.json")
        }
        let dictData = try Data(contentsOf: dictURL)
        self.phonemeDict = try JSONSerialization.jsonObject(with: dictData) as! [String: [String]]

        // Load token2id
        let t2iURL = constantsDirectory.appendingPathComponent("\(name)_token2id.json")
        guard FileManager.default.fileExists(atPath: t2iURL.path) else {
            throw MagpieTTSError.constantsNotFound("\(name)_token2id.json")
        }
        let t2iData = try Data(contentsOf: t2iURL)
        self.token2id = try JSONSerialization.jsonObject(with: t2iData) as! [String: Int]
        self.tokens = Set(token2id.keys)

        // Load heteronyms (optional)
        let hetURL = constantsDirectory.appendingPathComponent("\(name)_heteronyms.json")
        if FileManager.default.fileExists(atPath: hetURL.path) {
            let hetData = try Data(contentsOf: hetURL)
            let hetList = try JSONSerialization.jsonObject(with: hetData) as! [String]
            self.heteronyms = Set(hetList)
        } else {
            self.heteronyms = Set()
        }

        // Language-specific config from tokenizer_metadata.json
        switch language {
        case .german:
            self.graphemeCase = .mixed
            self.graphemePrefix = "#"
            self.padWithSpace = true
        case .spanish:
            self.graphemeCase = .upper
            self.graphemePrefix = ""
            self.padWithSpace = true
        default:
            self.graphemeCase = .upper
            self.graphemePrefix = ""
            self.padWithSpace = false
        }

        // Load punct list from metadata
        let metaURL = constantsDirectory.appendingPathComponent("tokenizer_metadata.json")
        if FileManager.default.fileExists(atPath: metaURL.path) {
            let metaData = try Data(contentsOf: metaURL)
            let metaJson = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
            if let punctLists = metaJson["punct_lists"] as? [String: [String]],
                let punct = punctLists[name]
            {
                self.punctList = Set(punct)
            } else {
                self.punctList = Set(#"!"(),-.:;?[]{}/""#.map(String.init))
            }
        } else {
            self.punctList = Set(#"!"(),-.:;?[]{}/""#.map(String.init))
        }
    }

    /// Encode text to local token IDs (before offset).
    func encode(_ text: String) -> [Int] {
        // Preprocessing: NFC normalize, replace right quote
        let preprocessed = anyLocalePreprocess(text)
        let words = wordTokenize(preprocessed)
        let phonemes = g2p(words)
        return encodeFromG2P(phonemes)
    }

    private func anyLocalePreprocess(_ text: String) -> String {
        var result = text.precomposedStringWithCanonicalMapping  // NFC
        result = result.replacingOccurrences(of: "\u{2019}", with: "'")  // right single quote → apostrophe
        return result
    }

    private func wordTokenize(_ text: String) -> [(tokens: [String], unchanged: Bool)] {
        let nsText = text as NSString
        let regex = Self.anyLocaleWordRegex
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))

        var result = [(tokens: [String], unchanged: Bool)]()
        for match in matches {
            let word = rangeStr(nsText, match, 1)
            let unchanged = rangeStr(nsText, match, 2)
            let punct = rangeStr(nsText, match, 3)

            if let w = word {
                // any_locale: no lowercasing (unlike English)
                result.append(([w], false))
            } else if let u = unchanged {
                let inner = String(u.dropFirst().dropLast())
                result.append((inner.split(separator: " ").map(String.init), true))
            } else if let p = punct {
                result.append(([p], false))
            }
        }
        return result
    }

    private func rangeStr(_ s: NSString, _ match: NSTextCheckingResult, _ group: Int) -> String? {
        let r = match.range(at: group)
        guard r.location != NSNotFound else { return nil }
        let str = s.substring(with: r)
        return str.isEmpty ? nil : str
    }

    private func g2p(_ words: [(tokens: [String], unchanged: Bool)]) -> [String] {
        var result = [String]()
        for (tokens, unchanged) in words {
            if unchanged {
                for (i, token) in tokens.enumerated() {
                    if i > 0 { result.append(" ") }
                    result.append(contentsOf: prependPrefix(token))
                }
                continue
            }
            for token in tokens {
                let pron = parseOneWord(token)
                result.append(contentsOf: pron)
            }
        }
        return result
    }

    private func parseOneWord(_ word: String) -> [String] {
        let cased = applyGraphemeCase(word)

        // Pure punctuation
        if cased.unicodeScalars.allSatisfy({ !CharacterSet.alphanumerics.contains($0) }) {
            return Array(cased).map(String.init)
        }

        // Heteronyms → graphemize
        if !heteronyms.isEmpty && heteronyms.contains(cased) {
            return prependPrefix(cased)
        }

        // For German mixed case: also check uppercase version
        if graphemeCase == .mixed {
            if let prons = phonemeDict[cased], !prons.isEmpty {
                return prons
            }
            let upper = cased.uppercased()
            if upper != cased, let prons = phonemeDict[upper], !prons.isEmpty {
                return prons
            }
        } else {
            // Upper case (Spanish, etc.)
            if let prons = phonemeDict[cased], !prons.isEmpty {
                return prons
            }
        }

        // OOV: try splitting by hyphens
        let subwords = cased.split(separator: "-")
        if subwords.count > 1 {
            var result = [String]()
            for (i, sub) in subwords.enumerated() {
                let subStr = applyGraphemeCase(String(sub))
                if let prons = phonemeDict[subStr], !prons.isEmpty {
                    result.append(contentsOf: prons)
                } else if graphemeCase == .mixed, let prons = phonemeDict[subStr.uppercased()],
                    !prons.isEmpty
                {
                    result.append(contentsOf: prons)
                } else {
                    result.append(contentsOf: prependPrefix(subStr))
                }
                if i < subwords.count - 1 { result.append("-") }
            }
            return result
        }

        // Final fallback: graphemize
        return prependPrefix(cased)
    }

    private func applyGraphemeCase(_ word: String) -> String {
        switch graphemeCase {
        case .upper: return word.uppercased()
        case .mixed: return word
        }
    }

    private func prependPrefix(_ word: String) -> [String] {
        return word.map { "\(graphemePrefix)\($0)" }
    }

    private func encodeFromG2P(_ phonemes: [String]) -> [Int] {
        var ps = [String]()
        let space = " "

        for p in phonemes {
            if p == space {
                if !ps.isEmpty && ps.last != space { ps.append(p) }
            } else if tokens.contains(p) {
                ps.append(p)
            } else if punctList.contains(p) {
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

// MARK: - ByT5 Encoder (French, Italian, Vietnamese)

/// ByT5 byte-level tokenizer. Maps UTF-8 bytes to token IDs (byte + 3).
/// Used for French, Italian, and Vietnamese.
enum ByT5Encoder {

    /// Encode text to global token IDs (with offset). Includes ByT5 EOS token.
    static func encode(_ text: String, offset: Int) -> [Int32] {
        let bytes = Array(text.utf8)
        var ids = bytes.map { Int32($0) + 3 + Int32(offset) }
        ids.append(Int32(1 + offset))  // ByT5 EOS (local ID 1)
        return ids
    }
}

// MARK: - Hindi Char Tokenizer

/// Character-level tokenizer for Hindi (Devanagari + ASCII).
/// Replicates NeMo's HindiCharsTokenizer.
final class HindiCharTokenizer {

    private let token2id: [String: Int]
    private let tokens: Set<String>
    private let punctList: Set<String>

    init(constantsDirectory: URL) throws {
        let t2iURL = constantsDirectory.appendingPathComponent("hindi_chartokenizer_token2id.json")
        guard FileManager.default.fileExists(atPath: t2iURL.path) else {
            throw MagpieTTSError.constantsNotFound("hindi_chartokenizer_token2id.json")
        }
        let t2iData = try Data(contentsOf: t2iURL)
        self.token2id = try JSONSerialization.jsonObject(with: t2iData) as! [String: Int]
        self.tokens = Set(token2id.keys)
        self.punctList = Set(#"!"(),-.:;?[]{}/""#.map(String.init))
    }

    /// Encode text to local token IDs (before offset).
    func encode(_ text: String) -> [Int] {
        // Preprocessing: NFC normalize, replace right quote
        var preprocessed = text.precomposedStringWithCanonicalMapping
        preprocessed = preprocessed.replacingOccurrences(of: "\u{2019}", with: "'")

        var cs = [String]()
        let space = " "

        // Iterate over Unicode scalars, not Characters, because Swift's Character
        // combines Devanagari combining marks with base characters (grapheme clusters).
        // NeMo processes text code-point by code-point.
        for scalar in preprocessed.unicodeScalars {
            let s = String(scalar)
            if s == space {
                if !cs.isEmpty && cs.last != space { cs.append(s) }
            } else if tokens.contains(s) {
                cs.append(s)
            } else if punctList.contains(s) {
                cs.append(s)
            }
        }

        while let last = cs.last, last == space { cs.removeLast() }

        // HindiCharsTokenizer uses pad_with_space=True
        cs = [space] + cs + [space]

        return cs.compactMap { token2id[$0] }
    }
}
