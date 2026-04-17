import Accelerate
import Foundation

// MARK: - Seedable PRNG (SplitMix64)

struct SplitMix64: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e37_79b9_7f4a_7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
        z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
        return z ^ (z >> 31)
    }
}

// MARK: - Local Transformer Weights

struct LocalTransformerWeights {
    static let dim = 256
    static let ffnDim = 1024
    static let maxPositions = 10

    let inProjWeight: [Float]  // (256, 768)
    let inProjBias: [Float]  // (256,)
    let posEmb: [Float]  // (10, 256)
    let norm1Weight: [Float]  // (256,)
    let saQkvWeight: [Float]  // (768, 256)
    let saOWeight: [Float]  // (256, 256)
    let norm2Weight: [Float]  // (256,)
    let ffnW1: [Float]  // (1024, 256) squeezed from (1024, 256, 1)
    let ffnW2: [Float]  // (256, 1024) squeezed from (256, 1024, 1)
    let outProjWeights: [[Float]]  // 8 × (2024, 256)
    let outProjBiases: [[Float]]  // 8 × (2024,)

    static func load(from directory: URL) throws -> LocalTransformerWeights {
        func npy(_ name: String) throws -> [Float] {
            try NpyReader.load(url: directory.appendingPathComponent(name)).data
        }
        var outW = [[Float]]()
        var outB = [[Float]]()
        for i in 0 ..< 8 {
            outW.append(try npy("out_proj_\(i)_weight.npy"))
            outB.append(try npy("out_proj_\(i)_bias.npy"))
        }
        return LocalTransformerWeights(
            inProjWeight: try npy("in_proj_weight.npy"),
            inProjBias: try npy("in_proj_bias.npy"),
            posEmb: try npy("pos_emb.npy"),
            norm1Weight: try npy("norm1_weight.npy"),
            saQkvWeight: try npy("sa_qkv_weight.npy"),
            saOWeight: try npy("sa_o_weight.npy"),
            norm2Weight: try npy("norm2_weight.npy"),
            ffnW1: try npy("ffn_conv1_weight.npy"),  // (1024,256,1) read flat = (1024,256)
            ffnW2: try npy("ffn_conv2_weight.npy"),
            outProjWeights: outW,
            outProjBiases: outB
        )
    }
}

// MARK: - Matrix multiply helpers (Accelerate BLAS)

/// C = A @ B^T   where A is (M,K), B is (N,K), C is (M,N)
private func matmulT(
    _ a: [Float], _ b: [Float], M: Int, N: Int, K: Int,
    alpha: Float = 1, into c: inout [Float]
) {
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            c.withUnsafeMutableBufferPointer { cp in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    Int32(M), Int32(N), Int32(K),
                    alpha, ap.baseAddress!, Int32(K),
                    bp.baseAddress!, Int32(K),
                    0, cp.baseAddress!, Int32(N))
            }
        }
    }
}

/// C = A @ B   where A is (M,K), B is (K,N), C is (M,N)
private func matmul(
    _ a: [Float], _ b: [Float], M: Int, N: Int, K: Int,
    into c: inout [Float]
) {
    a.withUnsafeBufferPointer { ap in
        b.withUnsafeBufferPointer { bp in
            c.withUnsafeMutableBufferPointer { cp in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    Int32(M), Int32(N), Int32(K),
                    1, ap.baseAddress!, Int32(K),
                    bp.baseAddress!, Int32(N),
                    0, cp.baseAddress!, Int32(N))
            }
        }
    }
}

// MARK: - Element-wise helpers

private func layerNorm(_ x: [Float], weight: [Float], T: Int, D: Int, eps: Float = 1e-5) -> [Float]
{
    var out = [Float](repeating: 0, count: T * D)
    x.withUnsafeBufferPointer { xBuf in
        out.withUnsafeMutableBufferPointer { outBuf in
            weight.withUnsafeBufferPointer { wBuf in
                let xp = xBuf.baseAddress!
                let op = outBuf.baseAddress!
                let wp = wBuf.baseAddress!
                let vD = vDSP_Length(D)
                for t in 0 ..< T {
                    let off = t * D
                    var mean: Float = 0
                    vDSP_meanv(xp + off, 1, &mean, vD)
                    var negMean = -mean
                    vDSP_vsadd(xp + off, 1, &negMean, op + off, 1, vD)
                    var sumSq: Float = 0
                    vDSP_dotpr(op + off, 1, op + off, 1, &sumSq, vD)
                    var invStd = 1.0 / (sumSq / Float(D) + eps).squareRoot()
                    vDSP_vsmul(op + off, 1, &invStd, op + off, 1, vD)
                    vDSP_vmul(op + off, 1, wp, 1, op + off, 1, vD)
                }
            }
        }
    }
    return out
}

private func gelu(_ x: [Float]) -> [Float] {
    let n = x.count
    var result = [Float](repeating: 0, count: n)
    x.withUnsafeBufferPointer { xBuf in
        result.withUnsafeMutableBufferPointer { rBuf in
            let xp = xBuf.baseAddress!
            let rp = rBuf.baseAddress!
            let vN = vDSP_Length(n)
            var k = Float(0.044715)
            var c = Float(0.7978845608028654)
            var one: Float = 1.0
            var half: Float = 0.5
            // r = x^2
            vDSP_vsq(xp, 1, rp, 1, vN)
            // r = k * x^2
            vDSP_vsmul(rp, 1, &k, rp, 1, vN)
            // r = 1 + k * x^2
            vDSP_vsadd(rp, 1, &one, rp, 1, vN)
            // r = x * (1 + k * x^2)
            vDSP_vmul(xp, 1, rp, 1, rp, 1, vN)
            // r = c * x * (1 + k * x^2)
            vDSP_vsmul(rp, 1, &c, rp, 1, vN)
            // r = tanh(r)
            var nn = Int32(n)
            vvtanhf(rp, rp, &nn)
            // r = 1 + tanh(...)
            vDSP_vsadd(rp, 1, &one, rp, 1, vN)
            // r = x * (1 + tanh(...))
            vDSP_vmul(xp, 1, rp, 1, rp, 1, vN)
            // r = 0.5 * r
            vDSP_vsmul(rp, 1, &half, rp, 1, vN)
        }
    }
    return result
}

// MARK: - Local Transformer Forward

/// 1-layer causal transformer.  Input / output: flat (T, 256).
func localTransformerForward(_ seq: [Float], T: Int, _ w: LocalTransformerWeights) -> [Float] {
    let D = LocalTransformerWeights.dim  // 256
    let F = LocalTransformerWeights.ffnDim  // 1024
    let vTD = vDSP_Length(T * D)

    // x = seq + posEmb[:T]
    var x = [Float](repeating: 0, count: T * D)
    seq.withUnsafeBufferPointer { sp in
        w.posEmb.withUnsafeBufferPointer { pp in
            vDSP_vadd(sp.baseAddress!, 1, pp.baseAddress!, 1, &x, 1, vTD)
        }
    }

    // --- Self-attention block ---
    let residual1 = x
    let xn1 = layerNorm(x, weight: w.norm1Weight, T: T, D: D)

    // QKV: (T,256) @ (768,256)^T → (T,768)
    var qkv = [Float](repeating: 0, count: T * 3 * D)
    matmulT(xn1, w.saQkvWeight, M: T, N: 3 * D, K: D, into: &qkv)

    // Split Q, K, V
    var q = [Float](repeating: 0, count: T * D)
    var k = [Float](repeating: 0, count: T * D)
    var v = [Float](repeating: 0, count: T * D)
    for t in 0 ..< T {
        for d in 0 ..< D {
            q[t * D + d] = qkv[t * 3 * D + d]
            k[t * D + d] = qkv[t * 3 * D + D + d]
            v[t * D + d] = qkv[t * 3 * D + 2 * D + d]
        }
    }

    // Attention scores: (T,D) @ (T,D)^T * scale → (T,T)
    let scale = 1.0 / Float(D).squareRoot()
    var attn = [Float](repeating: 0, count: T * T)
    matmulT(q, k, M: T, N: T, K: D, alpha: scale, into: &attn)

    // Causal mask + row softmax (pointer-based, avoids ArraySlice COW)
    attn.withUnsafeMutableBufferPointer { buf in
        let p = buf.baseAddress!
        for i in 0 ..< T {
            let rowStart = i * T
            for j in (i + 1) ..< T { p[rowStart + j] = -1e9 }
            var mx = p[rowStart]
            for j in 1 ..< T { if p[rowStart + j] > mx { mx = p[rowStart + j] } }
            var sum: Float = 0
            for j in 0 ..< T {
                p[rowStart + j] = exp(p[rowStart + j] - mx)
                sum += p[rowStart + j]
            }
            let invSum = 1.0 / sum
            for j in 0 ..< T { p[rowStart + j] *= invSum }
        }
    }

    // Context: attn @ V → (T,D)
    var saOut = [Float](repeating: 0, count: T * D)
    matmul(attn, v, M: T, N: D, K: T, into: &saOut)

    // Output projection
    var saProj = [Float](repeating: 0, count: T * D)
    matmulT(saOut, w.saOWeight, M: T, N: D, K: D, into: &saProj)

    // Residual
    vDSP_vadd(residual1, 1, saProj, 1, &x, 1, vTD)

    // --- FFN block ---
    let residual2 = x
    let xn2 = layerNorm(x, weight: w.norm2Weight, T: T, D: D)

    // FFN up: (T,256) @ (1024,256)^T → (T,1024)
    var h = [Float](repeating: 0, count: T * F)
    matmulT(xn2, w.ffnW1, M: T, N: F, K: D, into: &h)
    h = gelu(h)

    // FFN down: (T,1024) @ (256,1024)^T → (T,256)
    var ffnOut = [Float](repeating: 0, count: T * D)
    matmulT(h, w.ffnW2, M: T, N: D, K: F, into: &ffnOut)

    // Residual
    vDSP_vadd(residual2, 1, ffnOut, 1, &x, 1, vTD)
    return x
}

// MARK: - KV Cache for Local Transformer

struct LocalTransformerCache {
    private(set) var kCache: [Float]  // (maxT, D)
    private(set) var vCache: [Float]  // (maxT, D)
    private(set) var length: Int

    init(maxT: Int) {
        let D = LocalTransformerWeights.dim
        kCache = [Float](repeating: 0, count: maxT * D)
        vCache = [Float](repeating: 0, count: maxT * D)
        length = 0
    }

    mutating func append(k: UnsafePointer<Float>, v: UnsafePointer<Float>) {
        let D = LocalTransformerWeights.dim
        let off = length * D
        kCache.withUnsafeMutableBufferPointer { buf in
            memcpy(buf.baseAddress! + off, k, D * MemoryLayout<Float>.stride)
        }
        vCache.withUnsafeMutableBufferPointer { buf in
            memcpy(buf.baseAddress! + off, v, D * MemoryLayout<Float>.stride)
        }
        length += 1
    }
}

// MARK: - Cached Single-Token Forward Step

/// Single-token forward step with KV cache. Returns output for the new token (D elements).
/// Avoids recomputing Q/K/V for all previous positions.
func localTransformerStepCached(
    _ input: [Float],
    position: Int,
    cache: inout LocalTransformerCache,
    _ w: LocalTransformerWeights
) -> [Float] {
    let D = LocalTransformerWeights.dim
    let F = LocalTransformerWeights.ffnDim
    let vD = vDSP_Length(D)

    // x = input + posEmb[position]
    var x = [Float](repeating: 0, count: D)
    w.posEmb.withUnsafeBufferPointer { pp in
        vDSP_vadd(input, 1, pp.baseAddress! + position * D, 1, &x, 1, vD)
    }

    let residual1 = x
    let xn1 = layerNorm(x, weight: w.norm1Weight, T: 1, D: D)

    // QKV for single token: (1, D) @ (3D, D)^T → (1, 3D)
    var qkv = [Float](repeating: 0, count: 3 * D)
    matmulT(xn1, w.saQkvWeight, M: 1, N: 3 * D, K: D, into: &qkv)

    // Append new K, V to cache (avoid Array allocation — use pointers)
    qkv.withUnsafeBufferPointer { buf in
        cache.append(k: buf.baseAddress! + D, v: buf.baseAddress! + 2 * D)
    }
    let T_cache = cache.length

    // Attention: q @ K_cache^T * scale → (1, T_cache)
    let scale = 1.0 / Float(D).squareRoot()
    var attn = [Float](repeating: 0, count: T_cache)
    // q is qkv[0..<D], K_cache is (T_cache, D) — BLAS reads first T_cache rows
    qkv.withUnsafeBufferPointer { qkvBuf in
        cache.kCache.withUnsafeBufferPointer { kBuf in
            attn.withUnsafeMutableBufferPointer { aBuf in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    1, Int32(T_cache), Int32(D),
                    scale, qkvBuf.baseAddress!, Int32(D),
                    kBuf.baseAddress!, Int32(D),
                    0, aBuf.baseAddress!, Int32(T_cache))
            }
        }
    }

    // Softmax
    attn.withUnsafeMutableBufferPointer { buf in
        let p = buf.baseAddress!
        let vT = vDSP_Length(T_cache)
        var mx: Float = 0
        vDSP_maxv(p, 1, &mx, vT)
        var negMx = -mx
        vDSP_vsadd(p, 1, &negMx, p, 1, vT)
        var nn = Int32(T_cache)
        vvexpf(p, p, &nn)
        var sum: Float = 0
        vDSP_sve(p, 1, &sum, vT)
        vDSP_vsdiv(p, 1, &sum, p, 1, vT)
    }

    // Context: attn @ V_cache → (1, D)
    var context = [Float](repeating: 0, count: D)
    attn.withUnsafeBufferPointer { aBuf in
        cache.vCache.withUnsafeBufferPointer { vBuf in
            context.withUnsafeMutableBufferPointer { cBuf in
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    1, Int32(D), Int32(T_cache),
                    1, aBuf.baseAddress!, Int32(T_cache),
                    vBuf.baseAddress!, Int32(D),
                    0, cBuf.baseAddress!, Int32(D))
            }
        }
    }

    // Output projection
    var saProj = [Float](repeating: 0, count: D)
    matmulT(context, w.saOWeight, M: 1, N: D, K: D, into: &saProj)

    // Residual
    vDSP_vadd(residual1, 1, saProj, 1, &x, 1, vD)

    // FFN
    let residual2 = x
    let xn2 = layerNorm(x, weight: w.norm2Weight, T: 1, D: D)

    var h = [Float](repeating: 0, count: F)
    matmulT(xn2, w.ffnW1, M: 1, N: F, K: D, into: &h)
    h = gelu(h)

    var ffnOut = [Float](repeating: 0, count: D)
    matmulT(h, w.ffnW2, M: 1, N: D, K: F, into: &ffnOut)

    vDSP_vadd(residual2, 1, ffnOut, 1, &x, 1, vD)
    return x
}

// MARK: - Top-K Sampling

func sampleTopK(
    logits: inout [Float], temperature: Float, topK: Int,
    rng: inout SplitMix64
) -> Int {
    let n = logits.count
    let vN = vDSP_Length(n)

    // Top-k filter
    if topK > 0, topK < n {
        var sorted = logits
        vDSP_vsort(&sorted, vN, 1)  // ascending
        let threshold = sorted[n - topK]
        let negInf = -Float.infinity
        logits.withUnsafeMutableBufferPointer { buf in
            let p = buf.baseAddress!
            for i in 0 ..< n where p[i] < threshold { p[i] = negInf }
        }
    }

    logits.withUnsafeMutableBufferPointer { buf in
        let p = buf.baseAddress!
        // Temperature
        var invT = 1.0 / Swift.max(temperature, 1e-8)
        vDSP_vsmul(p, 1, &invT, p, 1, vN)

        // Softmax: subtract max, exp, normalize
        var mx: Float = 0
        vDSP_maxv(p, 1, &mx, vN)
        var negMx = -mx
        vDSP_vsadd(p, 1, &negMx, p, 1, vN)
        var nn = Int32(n)
        vvexpf(p, p, &nn)
        var sum: Float = 0
        vDSP_sve(p, 1, &sum, vN)
        vDSP_vsdiv(p, 1, &sum, p, 1, vN)
    }

    // Sample
    let r = Float.random(in: 0 ..< 1, using: &rng)
    var cumulative: Float = 0
    for i in 0 ..< n {
        cumulative += logits[i]
        if cumulative > r { return i }
    }
    return n - 1
}

// MARK: - Local Transformer Codebook Sampling

/// Forbidden special token IDs
let forbiddenAllowEOS: Set<Int> = [2016, 2018, 2019, 2020, 2021, 2022, 2023]
let forbiddenForbidEOS: Set<Int> = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

/// Sample 8 codebook tokens autoregressively via local transformer with KV caching.
/// CFG is applied at the codebook logit level (matching NeMo).
func localTransformerSample(
    decoderHidden: [Float],
    weights: LocalTransformerWeights,
    audioEmbeddings: [[Float]],
    numCodebooks: Int,
    temperature: Float,
    topK: Int,
    forbidEOS: Bool,
    uncondDecoderHidden: [Float]?,
    cfgScale: Float,
    rng: inout SplitMix64
) -> [Int32] {
    let forbidden = forbidEOS ? forbiddenForbidEOS : forbiddenAllowEOS
    let useCFG = uncondDecoderHidden != nil && cfgScale != 1.0
    let ltD = LocalTransformerWeights.dim  // 256
    let dModel = 768
    let vocab = 2024
    let vVocab = vDSP_Length(vocab)

    // Project decoder hidden (768) → LT dim (256)
    func project(_ hidden: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: ltD)
        matmulT(hidden, weights.inProjWeight, M: 1, N: ltD, K: dModel, into: &out)
        vDSP_vadd(out, 1, weights.inProjBias, 1, &out, 1, vDSP_Length(ltD))
        return out
    }

    // Initialize KV caches (max entries = numCodebooks + 1 for initial token)
    let maxT = numCodebooks + 1
    var condCache = LocalTransformerCache(maxT: maxT)
    var uncondCache = LocalTransformerCache(maxT: maxT)

    // Process initial token through cached step
    var condLast = localTransformerStepCached(
        project(decoderHidden), position: 0, cache: &condCache, weights)
    var uncondLast = [Float]()
    if useCFG {
        uncondLast = localTransformerStepCached(
            project(uncondDecoderHidden!), position: 0, cache: &uncondCache, weights)
    }

    var codes = [Int32](repeating: 0, count: numCodebooks)

    for cb in 0 ..< numCodebooks {
        // Logits from last output: (1, 256) @ (2024, 256)^T + bias
        var condLogits = [Float](repeating: 0, count: vocab)
        matmulT(condLast, weights.outProjWeights[cb], M: 1, N: vocab, K: ltD, into: &condLogits)
        vDSP_vadd(condLogits, 1, weights.outProjBiases[cb], 1, &condLogits, 1, vVocab)

        var finalLogits: [Float]
        if useCFG {
            var uncondLogits = [Float](repeating: 0, count: vocab)
            matmulT(
                uncondLast, weights.outProjWeights[cb], M: 1, N: vocab, K: ltD,
                into: &uncondLogits)
            vDSP_vadd(uncondLogits, 1, weights.outProjBiases[cb], 1, &uncondLogits, 1, vVocab)

            // CFG: scale * cond + (1 - scale) * uncond
            finalLogits = [Float](repeating: 0, count: vocab)
            var ics = 1.0 - cfgScale
            vDSP_vsmul(uncondLogits, 1, &ics, &finalLogits, 1, vVocab)
            var cs = cfgScale
            vDSP_vsma(condLogits, 1, &cs, finalLogits, 1, &finalLogits, 1, vVocab)
        } else {
            finalLogits = condLogits
        }

        // Mask forbidden tokens
        for tok in forbidden where tok < vocab { finalLogits[tok] = -.infinity }

        // Sample
        codes[cb] = Int32(
            sampleTopK(
                logits: &finalLogits, temperature: temperature,
                topK: topK, rng: &rng))

        // Embed sampled token → project → cached step for next codebook
        if cb < numCodebooks - 1 {
            let idx = Int(codes[cb])
            let emb = Array(audioEmbeddings[cb][idx * dModel ..< (idx + 1) * dModel])
            let nextInput = project(emb)
            let nextPos = cb + 1
            condLast = localTransformerStepCached(
                nextInput, position: nextPos, cache: &condCache, weights)
            if useCFG {
                uncondLast = localTransformerStepCached(
                    nextInput, position: nextPos, cache: &uncondCache, weights)
            }
        }
    }

    return codes
}
