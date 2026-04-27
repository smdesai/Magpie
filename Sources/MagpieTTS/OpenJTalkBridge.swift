//
//  OpenJTalkBridge.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import COpenJTalk
import Foundation
import os

/// Swift bridge for OpenJTalk Japanese G2P frontend.
/// Follows the NemoTextProcessing.swift pattern.
public enum OpenJTalkBridge {

    private static let initLock = OSAllocatedUnfairLock<Bool>(initialState: false)

    /// NJD word feature from OpenJTalk frontend.
    struct NJDWord {
        let string: String  // surface form
        let pos: String  // part of speech (品詞)
        let pron: String  // pronunciation (katakana)
        let acc: Int  // accent type
        let moraSize: Int  // number of mora
        let chainFlag: Int  // chain flag (0/-1 = new chain, 1 = continuation)
    }

    /// Initialize OpenJTalk with the MeCab dictionary at the given path.
    /// Idempotent — subsequent calls are no-ops. Thread-safe.
    @discardableResult
    static func initialize(dictionaryPath: String) -> Bool {
        initLock.withLock { initialized in
            guard !initialized else { return true }
            guard let cPath = dictionaryPath.cString(using: .utf8) else { return false }
            initialized = (openjtalk_init(cPath) == 1)
            return initialized
        }
    }

    /// Run OpenJTalk G2P frontend, returning NJD word features.
    static func runFrontend(_ text: String) -> [NJDWord]? {
        guard initLock.withLock({ $0 }) else { return nil }
        guard let cString = text.cString(using: .utf8) else { return nil }
        guard let resultPtr = openjtalk_g2p(cString) else { return nil }
        defer { openjtalk_free_string(resultPtr) }

        let jsonString = String(cString: resultPtr)
        guard let data = jsonString.data(using: .utf8),
            let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]]
        else { return nil }

        return array.compactMap { dict in
            guard let string = dict["string"] as? String,
                let pos = dict["pos"] as? String,
                let pron = dict["pron"] as? String,
                let acc = dict["acc"] as? Int,
                let moraSize = dict["mora_size"] as? Int,
                let chainFlag = dict["chain_flag"] as? Int
            else { return nil }
            return NJDWord(
                string: string, pos: pos, pron: pron,
                acc: acc, moraSize: moraSize, chainFlag: chainFlag)
        }
    }

    /// Release all OpenJTalk resources.
    static func cleanup() {
        initLock.withLock { initialized in
            if initialized {
                openjtalk_destroy()
                initialized = false
            }
        }
    }
}
