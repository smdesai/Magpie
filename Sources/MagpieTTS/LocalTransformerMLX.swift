//
//  LocalTransformerMLX.swift
//  MagpieTTS
//
//  Created by Sachin Desai on 3/8/26.
//

import Foundation
import MLX
import MLXNN

// MARK: - MLX-Based Local Transformer Weights

struct LocalTransformerWeightsMLX {
    static let dim = 256
    static let ffnDim = 1024
    static let maxPositions = 10

    // All weight matrices are pre-transposed at load time so we can use
    // matmul(x, w) instead of matmul(x, w.T) at inference time.
    let inProjWeight: MLXArray  // (768, 256) — pre-transposed from (256, 768)
    let inProjBias: MLXArray  // (256,)
    let posEmb: MLXArray  // (10, 256)
    let norm1Weight: MLXArray  // (256,)
    let saQkvWeight: MLXArray  // (256, 768) — pre-transposed from (768, 256)
    let saOWeight: MLXArray  // (256, 256) — pre-transposed from (256, 256)
    let norm2Weight: MLXArray  // (256,)
    let ffnW1: MLXArray  // (256, 1024) — pre-transposed from (1024, 256)
    let ffnW2: MLXArray  // (1024, 256) — pre-transposed from (256, 1024)
    let outProjWeights: [MLXArray]  // 8 × (256, 2024) — pre-transposed from (2024, 256)
    let outProjBiases: [MLXArray]  // 8 × (2024,)

    static func load(from directory: URL) throws -> LocalTransformerWeightsMLX {
        func npy(_ name: String) throws -> MLXArray {
            try MLX.loadArray(url: directory.appendingPathComponent(name))
        }

        var outW = [MLXArray]()
        var outB = [MLXArray]()
        for i in 0 ..< 8 {
            outW.append(try npy("out_proj_\(i)_weight.npy").T)
            outB.append(try npy("out_proj_\(i)_bias.npy"))
        }

        let weights = LocalTransformerWeightsMLX(
            inProjWeight: try npy("in_proj_weight.npy").T,
            inProjBias: try npy("in_proj_bias.npy"),
            posEmb: try npy("pos_emb.npy"),
            norm1Weight: try npy("norm1_weight.npy"),
            saQkvWeight: try npy("sa_qkv_weight.npy").T,
            saOWeight: try npy("sa_o_weight.npy").T,
            norm2Weight: try npy("norm2_weight.npy"),
            ffnW1: try npy("ffn_conv1_weight.npy").squeezed(axis: -1).T,
            ffnW2: try npy("ffn_conv2_weight.npy").squeezed(axis: -1).T,
            outProjWeights: outW,
            outProjBiases: outB
        )

        // Force evaluation so all weights are materialized on the GPU
        eval(
            weights.inProjWeight, weights.inProjBias, weights.posEmb,
            weights.norm1Weight, weights.saQkvWeight, weights.saOWeight,
            weights.norm2Weight, weights.ffnW1, weights.ffnW2)
        eval(outW)
        eval(outB)

        return weights
    }
}

// MARK: - Compiled Forward Pass

/// 1-layer causal transformer forward pass. Input: (T, 256), output: (T, 256).
/// Weights are captured by the closure; only the sequence tensor is passed as input.
private func makeForwardFn(_ w: LocalTransformerWeightsMLX) -> @Sendable (MLXArray) -> MLXArray {
    let D = LocalTransformerWeightsMLX.dim  // 256
    let scale = MLXArray(1.0 / Float(D).squareRoot())

    return { (seq: MLXArray) -> MLXArray in
        let T = seq.dim(0)

        // x = seq + posEmb[:T]
        var x = seq + w.posEmb[0 ..< T]

        // --- Self-attention block ---
        let residual1 = x
        let xn1 = MLXFast.layerNorm(x, weight: w.norm1Weight, eps: 1e-5)

        // QKV projection: (T, 256) @ (256, 768) → (T, 768)
        let qkv = matmul(xn1, w.saQkvWeight)

        // Split Q, K, V along last axis
        let q = qkv[0..., 0 ..< D]
        let k = qkv[0..., D ..< 2 * D]
        let v = qkv[0..., 2 * D ..< 3 * D]

        // Attention: Q @ K^T * scale → (T, T)
        var attn = matmul(q, k.T) * scale

        // Causal mask: -1e9 above diagonal
        if T > 1 {
            let mask = triu(MLXArray.ones([T, T]) * Float(-1e9), k: 1)
            attn = attn + mask
        }
        attn = softmax(attn, axis: -1)

        // Context: attn @ V → (T, D)
        let saOut = matmul(attn, v)

        // Output projection: (T, D) @ (D, D) → (T, D)
        let saProj = matmul(saOut, w.saOWeight)

        // Residual
        x = residual1 + saProj

        // --- FFN block ---
        let residual2 = x
        let xn2 = MLXFast.layerNorm(x, weight: w.norm2Weight, eps: 1e-5)

        // FFN up: (T, 256) @ (256, 1024) → (T, 1024)
        var h = matmul(xn2, w.ffnW1)
        h = geluApproximate(h)

        // FFN down: (T, 1024) @ (1024, 256) → (T, 256)
        let ffnOut = matmul(h, w.ffnW2)

        // Residual
        x = residual2 + ffnOut
        return x
    }
}

// MARK: - Pre-converted Audio Embeddings for GPU Lookup

/// Convert flat [Float] audio embeddings to MLXArray for GPU-side take() lookup.
/// Returns [numCodebooks] × MLXArray of shape (vocabSize, dModel).
func convertAudioEmbeddingsToMLX(_ embeddings: [[Float]], vocabSize: Int, dModel: Int) -> [MLXArray]
{
    embeddings.map { flat in
        let arr = MLXArray(flat, [vocabSize, dModel])
        arr.eval()
        return arr
    }
}

// MARK: - MLX Local Transformer Codebook Sampling (Optimized)

/// Sample 8 codebook tokens autoregressively via compiled MLX local transformer.
/// CFG is applied at the codebook logit level (matching NeMo).
func localTransformerSampleMLX(
    decoderHidden: [Float],
    weights: LocalTransformerWeightsMLX,
    audioEmbeddings: [[Float]],
    audioEmbeddingsMLX: [MLXArray]?,
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
    let ltD = LocalTransformerWeightsMLX.dim  // 256
    let dModel = 768
    let vocab = 2024
    let maxT = numCodebooks + 1  // max sequence length: 1 initial + 8 codebooks
    var topKScratch = [Float](repeating: 0, count: vocab)

    let forward = makeForwardFn(weights)

    // Project decoder hidden (768) → LT dim (256)
    func project(_ hidden: MLXArray) -> MLXArray {
        matmul(hidden, weights.inProjWeight) + weights.inProjBias
    }

    // Pre-allocate sequence buffers (maxT, ltD) and fill first row
    let condHiddenMLX = MLXArray(decoderHidden, [1, dModel])
    let firstProj = project(condHiddenMLX)

    // Build sequence by pre-allocating with zeros and updating via scatter
    var condBuf = MLXArray.zeros([maxT, ltD])
    condBuf = condBuf.at[0].add(firstProj.squeezed(axis: 0))
    var condT = 1

    var uncondBuf = MLXArray.zeros([maxT, ltD])
    var uncondT = 0
    if useCFG {
        let uncondHiddenMLX = MLXArray(uncondDecoderHidden!, [1, dModel])
        let uncondProj = project(uncondHiddenMLX)
        uncondBuf = uncondBuf.at[0].add(uncondProj.squeezed(axis: 0))
        uncondT = 1
    }

    var codes = [Int32](repeating: 0, count: numCodebooks)

    for cb in 0 ..< numCodebooks {
        // Conditional LT forward on active slice
        let condSeq = condBuf[0 ..< condT]
        let condOut = forward(condSeq)
        let condLast = condOut[condT - 1].reshaped(1, ltD)

        // Logits: (1, 256) @ (256, 2024) + bias → (1, 2024)
        let condLogitsMLX = matmul(condLast, weights.outProjWeights[cb]) + weights.outProjBiases[cb]

        var finalLogits: [Float]
        if useCFG {
            let uncondSeq = uncondBuf[0 ..< uncondT]
            let uncondOut = forward(uncondSeq)
            let uncondLast = uncondOut[uncondT - 1].reshaped(1, ltD)
            let uncondLogitsMLX =
                matmul(uncondLast, weights.outProjWeights[cb]) + weights.outProjBiases[cb]

            // CFG blend on GPU: scale * cond + (1 - scale) * uncond
            let blended =
                MLXArray(cfgScale) * condLogitsMLX + MLXArray(1.0 - cfgScale) * uncondLogitsMLX
            finalLogits = blended.reshaped(vocab).asArray(Float.self)
        } else {
            finalLogits = condLogitsMLX.reshaped(vocab).asArray(Float.self)
        }

        // Mask forbidden tokens
        for tok in forbidden where tok < vocab { finalLogits[tok] = -.infinity }

        // Sample using existing CPU-based top-k with SplitMix64 for reproducibility
        codes[cb] = Int32(
            sampleTopK(
                logits: &finalLogits, scratch: &topKScratch,
                temperature: temperature, topK: topK, rng: &rng))

        // Embed sampled token via GPU take() or CPU fallback, then project
        let idx = Int(codes[cb])
        let nextInput: MLXArray
        if let embMLX = audioEmbeddingsMLX {
            // GPU-side embedding lookup — no CPU↔GPU copy
            let emb = embMLX[cb].take(MLXArray(Int32(idx)), axis: 0).reshaped(1, dModel)
            nextInput = project(emb)
        } else {
            // Fallback: CPU array slice → MLXArray
            let embOffset = idx * dModel
            let emb = Array(audioEmbeddings[cb][embOffset ..< embOffset + dModel])
            let embMLXArr = MLXArray(emb, [1, dModel])
            nextInput = project(embMLXArr)
        }

        // Scatter into pre-allocated buffer at next position
        condBuf = condBuf.at[condT].add(nextInput.squeezed(axis: 0))
        condT += 1

        if useCFG {
            uncondBuf = uncondBuf.at[uncondT].add(nextInput.squeezed(axis: 0))
            uncondT += 1
        }
    }

    return codes
}
