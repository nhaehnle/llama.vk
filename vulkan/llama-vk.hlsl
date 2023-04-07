// dxc -spirv -T cs_6_6 -E KernelRmsNorm1 -fspv-target-env=vulkan1.3 -enable-16bit-types llama-vk.hlsl

// Descriptor set 0: Pass globals
// Descriptor set 1: Kernel-specific
// Descriptor set 2: Layer globals

#include "llama-vk-shader.h"

#define DEBUG_FIRST_RMS_NORM 0

enum WeightFormat {
    WeightFormatQ4_0gpu = 2,
};

[[vk::constant_id(0)]] const uint specNEmbd = 6656; // LLaMa 30B
[[vk::constant_id(1)]] const uint specNCtx = 2048; // LLaMa
[[vk::constant_id(2)]] const uint specNFF = 17920; // LLaMa 30B
[[vk::constant_id(3)]] const uint specNVocab = 32000; // all LLaMas
[[vk::constant_id(4)]] const float specRotaryTheta = 10000.0;

[[vk::binding(0, 0)]] cbuffer ForwardPassConstants {
    GlobalConstantBuffer g_constants;
};

[[vk::binding(1, 0)]] ByteAddressBuffer bufferHistoryIndex;
[[vk::binding(2, 0)]] ByteAddressBuffer bufferEmbedding;

[[vk::binding(0, 1)]] ByteAddressBuffer bufferInput;
[[vk::binding(1, 1)]] RWByteAddressBuffer bufferOutput;

[[vk::binding(0, 2)]] ByteAddressBuffer bufferAttentionNorm;
[[vk::binding(1, 2)]] ByteAddressBuffer bufferWq;
[[vk::binding(2, 2)]] ByteAddressBuffer bufferWk;
[[vk::binding(3, 2)]] ByteAddressBuffer bufferWv;
[[vk::binding(4, 2)]] ByteAddressBuffer bufferWo;

[[vk::binding(5, 2)]] RWByteAddressBuffer bufferKeys;
[[vk::binding(6, 2)]] RWByteAddressBuffer bufferValues;

#define NUM_WGP_THREADS 256 // multiple of a wave size!

groupshared float gRmsScratch[NUM_WGP_THREADS];

void rmsNormTopHalf(uint tid, uint numPerLane, half activations[32]) {
    // Compute sum of squares in parallel streams to hide ALU latency
    float means[3] = { 0, 0, 0 };

    uint idx;
    [[unroll]] for (idx = 0; idx + 3 <= numPerLane; idx += 3) {
        [[unroll]] for (uint i = 0; i < 3; ++i) {
            float x = activations[idx + i];
            means[i] += x * x;
        }
    }
    [[unroll]] for (uint i = 0; idx + i < numPerLane; ++i) {
        float x = activations[idx + i];
        means[i] += x * x;
    }

    // Reduce across threads
    gRmsScratch[tid] = means[0] + means[1] + means[2];
}

void rmsNormBottomHalf(uint tid, uint numThreads, uint numPerLane, inout half activations[32]) {
    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();
    uint numWaves = (numThreads + waveSize - 1) / waveSize;

    float mean = 0.0;
    [[unroll]] for (uint wave = 0; wave < numWaves; ++wave)
        mean += gRmsScratch[wave * waveSize + lane];
    mean = WaveActiveSum(mean);
    mean *= 1.0 / specNEmbd;

    if (tid >= numThreads)
        return;

    // Write activations out to memory
    half scale = half(1.0 / sqrt(g_constants.rmsEpsilon + mean));

    [[unroll]] for (uint idx = 0; idx < numPerLane; ++idx)
        activations[idx] *= scale;
}

void storeActivations(uint tid, uint numThreads, uint numPerLane, half activations[32],
                      uint base, bool storeSwizzled) {
    uint offset;
    if (storeSwizzled)
        offset = 16 * tid;
    else
        offset = 2 * numPerLane * tid;

    uint idx = 0;
    [[unroll]] for (; idx + 8 <= numPerLane; idx += 8) {
        half4 vec1;
        half4 vec2;
        [[unroll]] for (uint i = 0; i < 4; ++i) {
            vec1[i] = activations[idx + i];
            vec2[i] = activations[idx + i + 4];
        }
        bufferOutput.Store(base + offset, vec1);
        bufferOutput.Store(base + offset + 8, vec2);
        if (storeSwizzled)
            offset += numThreads * 16;
        else
            offset += 16;
    }

    if (storeSwizzled)
        offset = 2 * idx * numThreads + 4 * tid;
    [[unroll]] for (; idx + 2 <= numPerLane; idx += 2) {
        half2 word;
        word.x = activations[idx];
        word.y = activations[idx + 1];
        bufferOutput.Store(base + offset, word);
        if (storeSwizzled)
            offset += numThreads * 4;
        else
            offset += 4;
    }

    if (idx < numPerLane) {
        if (storeSwizzled)
            offset = 2 * idx * numThreads + 2 * tid;
        bufferOutput.Store(base + offset, activations[idx]);
    }
}

// In a single workgroup, compute the entire
//
//   attentionNorm * rmsNorm(embedding)
//
// activations vector for a single token.
//
// Each lane process either 16 or 32 vector elements, as this simplifies the
// dequantization.
//
// Uses full compute subgroups.
[numthreads(NUM_WGP_THREADS, 1, 1)]
void KernelThinFp16FirstRmsNorm(uint3 localTid : SV_GroupThreadID) {
    uint rmsNEmbdPerLane = (specNEmbd + NUM_WGP_THREADS - 1) / NUM_WGP_THREADS;
    if (rmsNEmbdPerLane > 16)
        rmsNEmbdPerLane = 32;
    else
        rmsNEmbdPerLane = 16;

    uint numThreads = specNEmbd / rmsNEmbdPerLane;

    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();
    uint waveId = localTid.x / waveSize; // TODO: Can we get DXC to emit SubgroupId?
    uint tid = waveId * waveSize + lane;
    uint numLiveWaves = (numThreads + waveSize - 1) / waveSize;

    // We keep all activations in registers
    half activations[32];

    if (waveId < numLiveWaves) {
        // Step 1: Fetch and dequantize activations into registers
        uint rowScaleBytes = 4 * specNEmbd / 64;
        uint rowWeightBytes = 32 * specNEmbd / 64;
        uint totalScaleBytes = rowScaleBytes * specNVocab;
        uint baseScales = g_constants.currentToken * rowScaleBytes;
        uint baseWeights = totalScaleBytes + g_constants.currentToken * rowWeightBytes;

        uint doubleBlockIdx = tid * rmsNEmbdPerLane / 64;
        uint subBlockIdx = (tid * rmsNEmbdPerLane / 16) % 4;

        if (tid < numThreads) {
            // Workaround: HLSL forces the load offset to be a multiple of 4 :(
            half2 scales = bufferEmbedding.Load<half2>(baseScales + 4 * doubleBlockIdx);
            half scale = scales[subBlockIdx / 2];
            uint4 weights;
            if (rmsNEmbdPerLane == 32) {
                weights = bufferEmbedding.Load<uint4>(baseWeights + 32 * doubleBlockIdx);
            } else {
                weights.xy = bufferEmbedding.Load<uint2>(baseWeights + 32 * doubleBlockIdx + 8 * subBlockIdx);
            }

#if DEBUG_FIRST_RMS_NORM
            if (tid < 4) {
                printf("tid: %u token: %u, baseScales: %u baseWeights: %u\n", tid, g_constants.currentToken, baseScales, baseWeights);
                printf("tid: %u doubleBlockIdx: %u, scale: %f, weights: %v4x\n", tid, doubleBlockIdx, (float)scale, weights);
            }
#endif

            [[unroll]] for (uint idx = 0; idx < rmsNEmbdPerLane; idx += 8) {
                for (uint nibble = 0; nibble < 8; ++nibble) {
                    uint bits = (weights[idx / 8] >> (4 * nibble)) & 0xf;
                    activations[idx + nibble] = (half)((int)bits - 8) * scale;
                }
            }
        } else {
            [[unroll]] for (uint idx = 0; idx < rmsNEmbdPerLane; ++idx)
                activations[idx] = 0;
        }

#if DEBUG_FIRST_RMS_NORM
        if (tid < 2) {
            printf("tid: %u, numThreads: %u, per lane: %u %f %f %f %f\n", tid, numThreads, rmsNEmbdPerLane,
                (float)activations[0], (float)activations[1], (float)activations[2], (float)activations[3]);
        }
#endif

        rmsNormTopHalf(tid, rmsNEmbdPerLane, activations);
    }

    GroupMemoryBarrierWithGroupSync();

    if (waveId < numLiveWaves) {
        // TODO: attention norm!
        rmsNormBottomHalf(tid, numThreads, rmsNEmbdPerLane, activations);

        storeActivations(tid, numThreads, rmsNEmbdPerLane, activations, 0, false);
    }
}

// TODO: Use specialization constants
// TODO: Target wave64 in WGP mode
[numthreads(NUM_WGP_THREADS, 1, 1)]
void KernelThinFp16RmsNorm(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    // LLaMa 30B: rmsNEmbdPerLane = 26
    // LLaMa 65B: rmsNEmbdPerLane = 32
    const uint rmsNEmbdPerLane = (specNEmbd + NUM_WGP_THREADS - 1) / NUM_WGP_THREADS;

    // Load all activations into registers
    half activations[/* rmsNEmbdPerLane */ 32]; // TODO:

    uint base = 0; // TODO
    uint offset = 16 * localTid.x;
    uint idx = 0;
    [[unroll]] for (; idx + 8 <= rmsNEmbdPerLane; idx += 8) {
        half4 vec1 = bufferInput.Load<half4>(base + offset);
        half4 vec2 = bufferInput.Load<half4>(base + offset + 8);
        [[unroll]] for (uint i = 0; i < 4; ++i) {
            activations[idx + i] = vec1[i];
            activations[idx + i + 4] = vec2[i];
        }
        offset += NUM_WGP_THREADS * 16;
    }

    offset = 2 * idx * NUM_WGP_THREADS + 4 * localTid.x;
    [[unroll]] for (; idx + 2 <= rmsNEmbdPerLane; idx += 2) {
        half2 word = bufferInput.Load<half2>(base + offset);
        activations[idx] = word.x;
        activations[idx + 1] = word.y;
        offset += NUM_WGP_THREADS * 4;
    }

    if (idx < rmsNEmbdPerLane) {
        offset = 2 * idx * NUM_WGP_THREADS + 2 * localTid.x;
        activations[idx] = bufferInput.Load<half>(base + offset);
    }

    rmsNormTopHalf(localTid.x, rmsNEmbdPerLane, activations);

    GroupMemoryBarrierWithGroupSync();

    // TODO: attention norm!
    rmsNormBottomHalf(localTid.x, NUM_WGP_THREADS, rmsNEmbdPerLane, activations);

    storeActivations(localTid.x, NUM_WGP_THREADS, rmsNEmbdPerLane, activations, base, true);
}

#define NUM_THIN_ATTENTION_THREADS 128 // = n_rot

groupshared half gAttentionScratch16[3 * 128]; // 0.75 kB
groupshared float gAttentionScratch32[2 * 128]; // 1 kB
groupshared float gAttentionQK[2048]; // 8 kB; TODO: use specNCtx?!

// Self-attention module. Use one workgroup per attention head.
//
// For LLaMa 30B on Radeon RX 7900 XTX, this means we mostly have one workgroup
// per WGP (52 attention heads for 48 WGPs). But we do want to be able to fit
// 2 workgroups per WGP.
//
// TODO: Use specialization constants
// TODO: Target wave32 in WGP mode
[numthreads(NUM_THIN_ATTENTION_THREADS, 1, 1)]
void KernelThinFp16Attention(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint head = gid.x;
    const uint n_rot = 128; // = NUM_THIN_ATTENTION_THREADS

    uint i, j;

    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();

    uint historyIndex;
    bool haveInitialHistory = localTid.x >= 1 && localTid.x <= g_constants.currentHistoryLength;
    if (haveInitialHistory) {
        uint group = g_constants.currentHistoryBase / specNCtx;
        uint idx = g_constants.currentHistoryBase + g_constants.currentHistoryLength - localTid.x;
        idx = specNCtx * group + (idx % specNCtx);

        historyIndex = bufferHistoryIndex.Load<uint16_t>(2 * idx);
    }

    // Step 1a: Calculate Query, Key, Value, one element per lane
    // Assume Q4_0gpu format.
    uint nDoubleBlocks = specNEmbd / 64;
    uint nScaleColumnBytes = nDoubleBlocks * 4;
    uint matrixScaleBytes = specNEmbd * nScaleColumnBytes;

    uint scaleOffset = 0 + (n_rot * head + localTid.x) * 4;
    uint scaleStride = specNEmbd * 4;
    uint weightsOffset = matrixScaleBytes + (n_rot * head + localTid.x) * 16;
    uint weightsStride = specNEmbd * 32;
    uint activationOffset = 0;

    float qResult = 0;
    float kResult = 0;
    float vResult = 0;

    for (uint blockIdx = 0; blockIdx < nDoubleBlocks; ++blockIdx) {
        uint16_t2 qRawWeights[2][4];
        uint16_t2 kRawWeights[2][4];
        uint16_t2 vRawWeights[2][4];
        half2 activations[2][4][4];
        uint sub;
        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            [[unroll]] for (i = 0; i < 4; ++i) {
                [[unroll]] for (uint nibble = 0; nibble < 4; ++nibble) {
                    activations[sub][i][nibble] = bufferInput.Load<half2>(activationOffset);
                    activationOffset += 4;
                }

                qRawWeights[sub][i] =
                        bufferWq.Load<uint16_t2>(weightsOffset + (4 * sub + i) * 4);
                kRawWeights[sub][i] =
                        bufferWk.Load<uint16_t2>(weightsOffset + (4 * sub + i) * 4);
                vRawWeights[sub][i] =
                        bufferWv.Load<uint16_t2>(weightsOffset + (4 * sub + i) * 4);
            }
        }

        half2 qScales = bufferWq.Load<half2>(scaleOffset);
        half2 kScales = bufferWk.Load<half2>(scaleOffset);
        half2 vScales = bufferWv.Load<half2>(scaleOffset);

        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            float qSum = 0;
            float kSum = 0;
            float vSum = 0;

            [[unroll]] for (i = 0; i < 4; ++i) {
                [[unroll]] for (uint nibble = 0; nibble < 4; ++nibble) {
                    uint16_t2 utmp;
                    half2 prod;

                    utmp = (qRawWeights[sub][i] >> (nibble * 4)) & 0xf - 8;
                    prod = (half2)utmp * activations[sub][i][nibble];
                    qSum += prod.x + prod.y;

                    utmp = (kRawWeights[sub][i] >> (nibble * 4)) & 0xf - 8;
                    prod = (half2)utmp * activations[sub][i][nibble];
                    kSum += prod.x + prod.y;

                    utmp = (vRawWeights[sub][i] >> (nibble * 4)) & 0xf - 8;
                    prod = (half2)utmp * activations[sub][i][nibble];
                    vSum += prod.x + prod.y;
                }
            }

            qResult += qScales[sub] * qSum;
            kResult += kScales[sub] * kSum;
            vResult += vScales[sub] * vSum;
        }

        scaleOffset += scaleStride;
        weightsOffset += weightsStride;
    }

    // Step 1b: Apply rotary position embedding on Q and K
    //
    // ATTN: There's an implicit assumption here that pairs of lanes correspond
    //       to pairs of thread indices! This is a fairly reasonable assumption
    //       in a 1D workgroup. This avoids a barrier.
    //
    // Each lane does one half of a 2D rotation (or, equivalently, one part of
    // a complex multiplication by e^it).

    float qOther = WaveReadLaneAt(qResult, lane ^ 1);
    float kOther = WaveReadLaneAt(kResult, lane ^ 1);

    float invfreq = pow(specRotaryTheta, float(localTid.x / 2) * (-1.0 / (n_rot / 2)));
    float t = float(g_constants.currentRotaryPosition) * invfreq;
    float cost = cos(t);
    float sint = sin(t);

    sint = (lane & 1) == 1 ? sint : -sint;
    qResult = cost * qResult + sint * qOther;
    kResult = cost * kResult + sint * kOther;

    // Step 1c: Broadcast and store to prepare for Q * K calculation
    uint currentStorageOffset = 2 * (g_constants.currentStorageIndex * specNEmbd + head * n_rot + localTid.x);
    bufferKeys.Store<half>(currentStorageOffset, (half)kResult);
    bufferValues.Store<half>(currentStorageOffset, (half)vResult);

    gAttentionScratch16[0 * n_rot + localTid.x] = (half)qResult;
    gAttentionScratch16[1 * n_rot + localTid.x] = (half)kResult;
    gAttentionScratch16[2 * n_rot + localTid.x] = (half)vResult;

    GroupMemoryBarrierWithGroupSync();

    half4 q[n_rot / 4];
    [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
        [[unroll]] for (j = 0; j < 4; ++j)
            q[i][j] = gAttentionScratch16[4 * i + j];
    }

    // Step 2: Q * K; one history entry per lane
    float maxQK = -1.#INF;

    uint history = localTid.x;
    for (history = localTid.x;
         history <= g_constants.currentHistoryLength;
         history += NUM_THIN_ATTENTION_THREADS)
    {
        half4 k[n_rot / 4];

        if (history == 0) {
            [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
                [[unroll]] for (j = 0; j < 4; ++j)
                    k[i][j] = gAttentionScratch16[4 * i + j];
            }
        } else {
            uint historyStorageOffset = 2 * (historyIndex * specNEmbd + head * n_rot);
            [[unroll]] for (i = 0; i < n_rot / 4; ++i)
                k[i] = bufferKeys.Load<half4>(historyStorageOffset + 8 * i);
        }

        uint nextHistoryIndex;
        if (history + NUM_THIN_ATTENTION_THREADS <= g_constants.currentHistoryLength) {
            uint group = g_constants.currentHistoryBase / specNCtx;
            uint idx =
                    g_constants.currentHistoryBase
                    + g_constants.currentHistoryLength
                    - (history + NUM_THIN_ATTENTION_THREADS);
            idx = specNCtx * group + (idx % specNCtx);

            nextHistoryIndex = bufferHistoryIndex.Load<uint16_t>(2 * idx);
        }

        float qk = 0;
        [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
            qk += dot(q[i], k[i]);
        }
        gAttentionQK[history] = qk;
        maxQK = max(maxQK, qk);

        historyIndex = nextHistoryIndex;
    }

    // Step 3: bulk of softmax(QK)
    //
    // Multiplication by 1/sum is applied in the next step.

    gAttentionScratch32[localTid.x] = maxQK;

    GroupMemoryBarrierWithGroupSync();

    maxQK = -1.#INF;
    [[unroll]] for (i = 0; i < NUM_THIN_ATTENTION_THREADS / waveSize; ++i)
        maxQK = max(maxQK, gAttentionScratch32[i * waveSize + lane]);
    maxQK = WaveActiveSum(maxQK);

    float sum = 0.0;

    for (history = localTid.x;
         history <= g_constants.currentHistoryLength;
         history += NUM_THIN_ATTENTION_THREADS)
    {
        float qk = gAttentionQK[history];
        qk = exp(qk - maxQK);
        gAttentionQK[history] = qk;

        sum += qk;
    }

    gAttentionScratch32[NUM_THIN_ATTENTION_THREADS + localTid.x] = sum;

    GroupMemoryBarrierWithGroupSync();

    sum = 0.0;
    [[unroll]] for (i = 0; i < NUM_THIN_ATTENTION_THREADS / waveSize; ++i)
        sum += gAttentionScratch32[i * waveSize + lane];
    sum = WaveActiveSum(sum);

    float recipSum = 1.0 / sum;

    // Step 4: softmax(QK) * V
    //
    // Each thread computes two output elements. The workgroup is split in half,
    // with the low half of threads processing even history entries and the
    // high half processing odd entries.
    //
    // TODO: This likely needs more unrolling to keep VRAM busy.
    float2 oResult = 0;
    bool highHalf = localTid.x >= n_rot / 2;

    history = highHalf ? 1 : 0;
    uint halfTid = localTid.x % (n_rot / 2);

    if (highHalf && history <= g_constants.currentHistoryLength) {
        uint group = g_constants.currentHistoryBase / specNCtx;
        uint idx =
                g_constants.currentHistoryBase
                + g_constants.currentHistoryLength
                - (history + NUM_THIN_ATTENTION_THREADS);
        idx = specNCtx * group + (idx % specNCtx);

        historyIndex = bufferHistoryIndex.Load<uint16_t>(2 * idx);
    }

    for (; history <= g_constants.currentHistoryLength; history += 2) {
        half2 v;
        if (history == 0) {
            [[unroll]] for (i = 0; i < 2; ++i)
                v[i] = gAttentionScratch16[2 * n_rot + 2 * halfTid + i];
        } else {
            uint storageOffset = 2 * (historyIndex * specNEmbd + head * n_rot + 2 * halfTid);
            v = bufferValues.Load<half2>(storageOffset);
        }

        if (history + 2 <= g_constants.currentHistoryLength) {
            uint group = g_constants.currentHistoryBase / specNCtx;
            uint idx =
                    g_constants.currentHistoryBase
                    + g_constants.currentHistoryLength
                    - (history + NUM_THIN_ATTENTION_THREADS);
            idx = specNCtx * group + (idx % specNCtx);

            historyIndex = bufferHistoryIndex.Load<uint16_t>(2 * idx);
        }

        oResult += gAttentionQK[history] * v;
    }

    // The low half is probably done before the high half (because the first
    // iteration fetches from local shared memory), so let the second half do
    // the final sum.
    if (!highHalf) {
        gAttentionScratch32[2 * halfTid + 0] = oResult[0];
        gAttentionScratch32[2 * halfTid + 1] = oResult[1];
    }

    GroupMemoryBarrierWithGroupSync();

    if (highHalf) {
        oResult[0] += gAttentionScratch32[2 * halfTid + 0];
        oResult[1] += gAttentionScratch32[2 * halfTid + 1];

        oResult *= recipSum;

        bufferOutput.Store<half2>(2 * (head * n_rot + 2 * halfTid), half2(oResult));
    }
}
