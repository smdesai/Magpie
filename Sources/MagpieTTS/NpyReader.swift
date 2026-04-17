import Foundation

enum NpyError: Error {
    case invalidMagic
    case unsupportedVersion(UInt8, UInt8)
    case parseError(String)
    case unsupportedDtype(String)
}

struct NpyArray {
    let data: [Float]
    let shape: [Int]
}

/// Minimal reader for NumPy .npy files (v1 and v2). Supports float32 and float64.
enum NpyReader {

    static func load(url: URL) throws -> NpyArray {
        let raw = try Data(contentsOf: url)
        guard raw.count >= 10,
            raw[0] == 0x93,
            raw[1] == 0x4E, raw[2] == 0x55, raw[3] == 0x4D,
            raw[4] == 0x50, raw[5] == 0x59
        else { throw NpyError.invalidMagic }

        let major = raw[6]
        let headerLen: Int
        let headerStart: Int

        switch major {
        case 1:
            headerLen = Int(raw[8]) | (Int(raw[9]) << 8)
            headerStart = 10
        case 2:
            guard raw.count >= 12 else { throw NpyError.parseError("File too short for v2 header") }
            headerLen =
                Int(raw[8]) | (Int(raw[9]) << 8) | (Int(raw[10]) << 16) | (Int(raw[11]) << 24)
            headerStart = 12
        default:
            throw NpyError.unsupportedVersion(major, raw[7])
        }

        guard
            let header = String(
                data: raw[headerStart ..< headerStart + headerLen], encoding: .ascii)
        else {
            throw NpyError.parseError("Cannot decode header")
        }

        if header.contains("'fortran_order': True") {
            throw NpyError.parseError("Fortran order not supported")
        }

        let dtype = try parseDtype(header)
        let shape = try parseShape(header)
        let total = shape.reduce(1, *)
        let dataStart = headerStart + headerLen
        let slice = raw[dataStart...]

        switch dtype {
        case "<f4", "=f4", "f4":
            let floats = slice.withUnsafeBytes {
                Array($0.bindMemory(to: Float.self).prefix(total))
            }
            return NpyArray(data: floats, shape: shape)
        case "<f8", "=f8", "f8":
            let doubles = slice.withUnsafeBytes {
                Array($0.bindMemory(to: Double.self).prefix(total))
            }
            return NpyArray(data: doubles.map { Float($0) }, shape: shape)
        default:
            throw NpyError.unsupportedDtype(dtype)
        }
    }

    // MARK: - Header parsing

    private static func parseDtype(_ h: String) throws -> String {
        guard let r = h.range(of: "'descr':") else { throw NpyError.parseError("No descr") }
        let after = h[r.upperBound...]
        guard let q1 = after.firstIndex(of: "'") else {
            throw NpyError.parseError("No descr quote")
        }
        let inner = after[after.index(after: q1)...]
        guard let q2 = inner.firstIndex(of: "'") else {
            throw NpyError.parseError("No descr end quote")
        }
        return String(inner[..<q2])
    }

    private static func parseShape(_ h: String) throws -> [Int] {
        guard let open = h.firstIndex(of: "(") else { throw NpyError.parseError("No shape tuple") }
        let after = h[h.index(after: open)...]
        guard let close = after.firstIndex(of: ")") else {
            throw NpyError.parseError("No closing paren")
        }
        return after[..<close]
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
            .compactMap { Int($0) }
    }
}
