// dxc -spirv -T cs_6_6 -E KernelRmsNorm1 -fspv-target-env=vulkan1.3 -enable-16bit-types llama-vk.hlsl

// Descriptor set 0: Pass globals
// Descriptor set 1: Kernel-specific
// Descriptor set 2: Layer globals

#include "llama-vk-shader.h"

#define DEBUG_FIRST_RMS_NORM 0
#define DEBUG_ATTENTION 0
#define DEBUG_ASSERTIONS 1


#if DEBUG_ASSERTIONS
#define assert(cond) do { } while (false)
#else
#define assert(cond) \
    do { \
        if (!(cond)) \
            printf("assert failed at %s:%d: %s", __FILE__, __LINE__, #cond); \
    } while (false)
#endif

enum WeightFormat {
    WeightFormatQ4_0gpu = 2,
};

[[vk::constant_id(0)]] const uint specNEmbd = 6656; // LLaMa 30B
[[vk::constant_id(1)]] const uint specNCtx = 2048; // all LLaMas
[[vk::constant_id(2)]] const uint specNFF = 17920; // LLaMa 30B
[[vk::constant_id(3)]] const uint specNVocab = 32000; // all LLaMas
[[vk::constant_id(4)]] const uint specNHead = 52; // LLaMa 30B
[[vk::constant_id(5)]] const float specRotaryTheta = 10000.0;
[[vk::constant_id(10)]] const uint specMode = 0; // meaning depends on the kernel

[[vk::binding(0, 0)]] cbuffer ForwardPassConstants {
    GlobalConstantBuffer g_constants;
};

[[vk::binding(1, 0)]] ByteAddressBuffer bufferHistoryIndex;
[[vk::binding(2, 0)]] ByteAddressBuffer bufferEmbedding;

[[vk::binding(3, 0)]] RWByteAddressBuffer bufferBypass;
[[vk::binding(4, 0)]] RWByteAddressBuffer bufferStage1;
[[vk::binding(5, 0)]] RWByteAddressBuffer bufferStage2;

[[vk::binding(0, 1)]] ByteAddressBuffer bufferAttentionNorm;
[[vk::binding(1, 1)]] ByteAddressBuffer bufferWq;
[[vk::binding(2, 1)]] ByteAddressBuffer bufferWk;
[[vk::binding(3, 1)]] ByteAddressBuffer bufferWv;
[[vk::binding(4, 1)]] ByteAddressBuffer bufferWo;

[[vk::binding(5, 1)]] RWByteAddressBuffer bufferKeys;
[[vk::binding(6, 1)]] RWByteAddressBuffer bufferValues;

[[vk::binding(7, 1)]] ByteAddressBuffer bufferFfnNorm;
[[vk::binding(8, 1)]] ByteAddressBuffer bufferW1;
[[vk::binding(9, 1)]] ByteAddressBuffer bufferW2;
[[vk::binding(10, 1)]] ByteAddressBuffer bufferW3;

#define NUM_WGP_THREADS 256 // multiple of a wave size!

groupshared float gRmsScratch[NUM_WGP_THREADS];

void rmsNormTopHalf(uint tid, uint numPerLane, half2 activations[16]) {
    // Compute sum of squares in parallel streams to hide ALU latency
    float means[3] = { 0, 0, 0 };

    uint idx;
    [[unroll]] for (idx = 0; idx + 3 <= numPerLane / 2; idx += 3) {
        [[unroll]] for (uint i = 0; i < 3; ++i) {
            means[i] += dot(activations[idx + i], activations[idx + i]);
        }
    }
    [[unroll]] for (uint i = 0; idx + i < numPerLane / 2; ++i) {
        means[i] += dot(activations[idx + i], activations[idx + i]);
    }

    // Reduce across threads
    gRmsScratch[tid] = means[0] + means[1] + means[2];
}

void rmsNormBottomHalf(uint tid, uint numThreads, uint numPerLane, inout half2 activations[16]) {
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

    if (tid == 0)
        printf("RMS scale: %f\n", (float)scale);

    [[unroll]] for (uint idx = 0; idx < numPerLane / 2; ++idx)
        activations[idx] *= scale;
}

// WARNING: numPerLane *must* be a multiple of 2 because HLSL currently requires
//          buffer offsets to be a multiple of 4!
void storeNormActivations(uint tid, uint numThreads, uint numPerLane, half2 activations[16],
                          bool swizzled) {
    uint offset;
    uint idx;

    // Load and multiply norm weights.
    if (swizzled)
        offset = 16 * tid;
    else
        offset = 2 * numPerLane * tid;

    [[unroll]] for (idx = 0; idx + 4 <= numPerLane / 2; idx += 4) {
        if (specMode == 0) {
            activations[idx + 0] *= bufferAttentionNorm.Load<half2>(offset + 0);
            activations[idx + 1] *= bufferAttentionNorm.Load<half2>(offset + 4);
            activations[idx + 2] *= bufferAttentionNorm.Load<half2>(offset + 8);
            activations[idx + 3] *= bufferAttentionNorm.Load<half2>(offset + 12);
        } else {
            activations[idx + 0] *= bufferFfnNorm.Load<half2>(offset + 0);
            activations[idx + 1] *= bufferFfnNorm.Load<half2>(offset + 4);
            activations[idx + 2] *= bufferFfnNorm.Load<half2>(offset + 8);
            activations[idx + 3] *= bufferFfnNorm.Load<half2>(offset + 12);
        }
        if (swizzled)
            offset += numThreads * 16;
        else
            offset += 16;
    }

    if (swizzled)
        offset = 2 * idx * numThreads + 4 * tid;
    [[unroll]] for (; idx < numPerLane / 2; ++idx) {
        if (specMode == 0) {
            activations[idx] *= bufferAttentionNorm.Load<half2>(offset);
        } else {
            activations[idx] *= bufferFfnNorm.Load<half2>(offset);
        }
        if (swizzled)
            offset += numThreads * 4;
        else
            offset += 4;
    }

    assert(idx == numPerLane);

    // Store result
    if (swizzled)
        offset = 16 * tid;
    else
        offset = 2 * numPerLane * tid;

    [[unroll]] for (idx = 0; idx + 4 <= numPerLane / 2; idx += 4) {
        bufferStage1.Store<half2>(offset +  0, activations[idx + 0]);
        bufferStage1.Store<half2>(offset +  4, activations[idx + 1]);
        bufferStage1.Store<half2>(offset +  8, activations[idx + 2]);
        bufferStage1.Store<half2>(offset + 12, activations[idx + 3]);
        if (swizzled)
            offset += numThreads * 16;
        else
            offset += 16;
    }

    if (swizzled)
        offset = 2 * idx * numThreads + 4 * tid;
    [[unroll]] for (; idx < numPerLane / 2; ++idx) {
        bufferStage1.Store(offset, activations[idx]);
        if (swizzled)
            offset += numThreads * 4;
        else
            offset += 4;
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
    half2 activations[/* rmsNEmbdPerLane / 2 */ 16];

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

            [[unroll]] for (uint word = 0; word < rmsNEmbdPerLane / 8; ++word) {
                uint lowWeights = weights[word] & 0x0f0f0f0f;
                uint highWeights = (weights[word] >> 4) & 0x0f0f0f0f;

                for (uint byte = 0; byte < 4; ++byte) {
                    int16_t2 itmp;
                    itmp.x = (int16_t)((lowWeights >> (8 * byte)) & 0xff);
                    itmp.y = (int16_t)((highWeights >> (8 * byte)) & 0xff);

                    activations[4 * word + byte] = (half2)(itmp - 8) * scale;
                }
            }

            uint offset = 2 * rmsNEmbdPerLane * tid;
            [[unroll]] for (uint idx = 0; idx < rmsNEmbdPerLane / 2; ++idx)
                bufferBypass.Store<half2>(offset + 4 * idx, activations[idx]);
        } else {
            [[unroll]] for (uint idx = 0; idx < rmsNEmbdPerLane / 2; ++idx)
                activations[idx] = 0;
        }

#if DEBUG_FIRST_RMS_NORM
        if (tid < 2) {
            printf("tid: %u, numThreads: %u, per lane: %u %v2f %v2f %v2f %v2f\n", tid, numThreads, rmsNEmbdPerLane,
                (float2)activations[0], (float2)activations[1], (float2)activations[2], (float2)activations[3]);
        }
#endif

        rmsNormTopHalf(tid, rmsNEmbdPerLane, activations);
    }

    GroupMemoryBarrierWithGroupSync();

    if (waveId < numLiveWaves) {
        rmsNormBottomHalf(tid, numThreads, rmsNEmbdPerLane, activations);

        storeNormActivations(tid, numThreads, rmsNEmbdPerLane, activations, false);
    }
}

// RMS norm followed by element-wise multiplication.
//
// A single workgroup processes the entire activations vector.
//
// TODO: Use specialization constants
// TODO: Target wave64 in WGP mode
[numthreads(NUM_WGP_THREADS, 1, 1)]
void KernelThinFp16RmsNorm(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    // LLaMa 30B: rmsNEmbdPerLane = 26
    // LLaMa 65B: rmsNEmbdPerLane = 32
    const uint rmsNEmbdPerLane = (specNEmbd + NUM_WGP_THREADS - 1) / NUM_WGP_THREADS;

    // Load all activations into registers
    half2 activations[/* rmsNEmbdPerLane / 2 */ 16]; // TODO: Spec constant?

    uint offset = 16 * localTid.x;
    uint idx = 0;
    [[unroll]] for (; idx + 4 <= rmsNEmbdPerLane / 2; idx += 4) {
        activations[idx + 0] = bufferBypass.Load<half2>(offset + 0);
        activations[idx + 1] = bufferBypass.Load<half2>(offset + 4);
        activations[idx + 2] = bufferBypass.Load<half2>(offset + 8);
        activations[idx + 3] = bufferBypass.Load<half2>(offset + 12);
        offset += NUM_WGP_THREADS * 16;
    }

    offset = 2 * idx * NUM_WGP_THREADS + 4 * localTid.x;
    [[unroll]] for (; idx < rmsNEmbdPerLane / 2; ++idx) {
        activations[idx] = bufferBypass.Load<half2>(offset);
        offset += NUM_WGP_THREADS * 4;
    }

    // WARNING: HLSL can't reliably load/store individual 16-bit values because
    //          buffer offset must be a multiple of 4.
    assert(idx == rmsNEmbdPerLane);

    rmsNormTopHalf(localTid.x, rmsNEmbdPerLane, activations);

    GroupMemoryBarrierWithGroupSync();

    rmsNormBottomHalf(localTid.x, NUM_WGP_THREADS, rmsNEmbdPerLane, activations);

    storeNormActivations(localTid.x, NUM_WGP_THREADS, rmsNEmbdPerLane, activations, true);
}

#define NUM_THIN_ATTENTION_THREADS 128 // = n_rot

groupshared half gAttentionScratch16[3 * 128]; // 0.75 kB
groupshared float gAttentionScratch32[2 * 128]; // 1 kB
groupshared float gAttentionQK[2048]; // 8 kB; TODO: use specNCtx?!

// Self-attention module, except for final multiply with Wo. Uses one workgroup
// per attention head.
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

    uint tid = localTid.x;
    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();

    uint historyIndex;
    bool haveInitialHistory = tid >= 1 && tid <= g_constants.currentHistoryLength;
    if (haveInitialHistory) {
        uint group = g_constants.currentHistoryBase / specNCtx;
        uint idx = g_constants.currentHistoryBase + g_constants.currentHistoryLength - tid;
        idx = specNCtx * group + (idx % specNCtx);

        historyIndex = bufferHistoryIndex.Load<uint>(4 * idx);
    }

    // Step 1a: Calculate Query, Key, Value, one element per lane
    // Assume Q4_0_SWZ format.
    uint nDoubleBlocks = specNEmbd / 64;
    uint matrixScaleBytes = 4 * nDoubleBlocks * specNEmbd;

    uint scaleOffset = 0 + (n_rot * head + tid) * 4;
    uint scaleStride = specNEmbd * 4;
    uint weightsOffset = matrixScaleBytes + (n_rot * head + tid) * 32;
    uint weightsStride = specNEmbd * 32;
    uint activationOffset = 0;

    float qResult = 0;
    float kResult = 0;
    float vResult = 0;

    for (uint blockIdx = 0; blockIdx < nDoubleBlocks; ++blockIdx) {
        uint4 qRawWeights[2];
        uint4 kRawWeights[2];
        uint4 vRawWeights[2];
        half2 activations[2][4][4];
        uint sub;
        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            [[unroll]] for (i = 0; i < 4; ++i) {
                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    activations[sub][i][byte] = bufferStage1.Load<half2>(activationOffset);
                    activationOffset += 4;
                }
            }

            qRawWeights[sub] = bufferWq.Load<uint4>(weightsOffset + 16 * sub);
            kRawWeights[sub] = bufferWk.Load<uint4>(weightsOffset + 16 * sub);
            vRawWeights[sub] = bufferWv.Load<uint4>(weightsOffset + 16 * sub);
        }

        half2 qScales = bufferWq.Load<half2>(scaleOffset);
        half2 kScales = bufferWk.Load<half2>(scaleOffset);
        half2 vScales = bufferWv.Load<half2>(scaleOffset);

        bool dbg = false; //head == 0 && tid == 0 && blockIdx < 2;

        if (dbg) {
            printf("block: %u qScales = %v2f\n", blockIdx, (float2)qScales);
            printf("block: %u qWeights = %v4x %v4x\n", blockIdx, qRawWeights[0], qRawWeights[1]);
            printf("block: %u activations = %v2f %v2f %v2f %v2f\n", blockIdx,
                (float2)activations[0][0][0], (float2)activations[0][0][1], (float2)activations[0][0][2], (float2)activations[0][0][3]);
        }

        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            float qSum = 0;
            float kSum = 0;
            float vSum = 0;

            [[unroll]] for (i = 0; i < 4; ++i) {
                uint qLowWeights = qRawWeights[sub][i] & 0x0f0f0f0f;
                uint qHighWeights = (qRawWeights[sub][i] >> 4) & 0x0f0f0f0f;
                uint kLowWeights = kRawWeights[sub][i] & 0x0f0f0f0f;
                uint kHighWeights = (kRawWeights[sub][i] >> 4) & 0x0f0f0f0f;
                uint vLowWeights = vRawWeights[sub][i] & 0x0f0f0f0f;
                uint vHighWeights = (vRawWeights[sub][i] >> 4) & 0x0f0f0f0f;

                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    int16_t2 utmp;
                    half2 prod;

                    utmp.x = (int16_t)((qLowWeights >> (8 * byte)) & 0xff);
                    utmp.y = (int16_t)((qHighWeights >> (8 * byte)) & 0xff);
                    prod = (half2)(utmp - 8) * activations[sub][i][byte];
                    qSum += prod.x + prod.y;

                    if (dbg && sub == 0 && i == 0) {
                        printf("block: %u sub: %u i: %u byte: %u utmp: %v2u activations: %v2f prod: %v2f q: %f\n",
                            blockIdx, sub, i, byte, (uint2)utmp, (float2)activations[sub][i][byte], (float2)prod, qSum);
                    }

                    utmp.x = (int16_t)((kLowWeights >> (8 * byte)) & 0xff);
                    utmp.y = (int16_t)((kHighWeights >> (8 * byte)) & 0xff);
                    prod = (half2)(utmp - 8) * activations[sub][i][byte];
                    kSum += prod.x + prod.y;

                    utmp.x = (int16_t)((vLowWeights >> (8 * byte)) & 0xff);
                    utmp.y = (int16_t)((vHighWeights >> (8 * byte)) & 0xff);
                    prod = (half2)(utmp - 8) * activations[sub][i][byte];
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
#if DEBUG_ATTENTION
    if (head < 4 && tid % 35 == 0) {
        printf("head: %u tid: %3u (i = %3u) init: q = %+8f k = %+8f v = %+8f\n", head, tid,
            128 * head + tid, qResult, kResult, vResult);
    }
#endif
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

    float invfreq = pow(specRotaryTheta, float(tid / 2) * (-1.0 / (n_rot / 2)));
    float t = float(g_constants.currentRotaryPosition) * invfreq;
    float cost = cos(t);
    float sint = sin(t);

    sint = (lane & 1) == 1 ? sint : -sint;
    qResult = cost * qResult + sint * qOther;
    kResult = cost * kResult + sint * kOther;
#if DEBUG_ATTENTION
    if (head == 0 && tid < 8) {
        printf("tid: %u rope: q = %f k = %f qOther = %f kOther = %f cos(t) = %f sgn sin(t) = %f\n", tid,
            qResult, kResult, qOther, kOther, cost, sint);
    }
#endif

    // Step 1c: Broadcast and store to prepare for Q * K calculation
    uint currentStorageOffset = 2 * (g_constants.currentStorageIndex * specNEmbd + head * n_rot + tid);
    bufferKeys.Store<half>(currentStorageOffset, (half)kResult);
    bufferValues.Store<half>(currentStorageOffset, (half)vResult);

    gAttentionScratch16[0 * n_rot + tid] = (half)qResult;
    gAttentionScratch16[1 * n_rot + tid] = (half)kResult;
    gAttentionScratch16[2 * n_rot + tid] = (half)vResult;

    GroupMemoryBarrierWithGroupSync();

    half4 q[n_rot / 4];
    [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
        [[unroll]] for (j = 0; j < 4; ++j)
            q[i][j] = gAttentionScratch16[0 * n_rot + 4 * i + j];
    }

    // Step 2: Q * K; one history entry per lane
    float maxQK = -1.#INF;

    uint history = tid;
    for (history = tid;
         history <= g_constants.currentHistoryLength;
         history += NUM_THIN_ATTENTION_THREADS)
    {
        half4 k[n_rot / 4];

        if (history == 0) {
            [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
                [[unroll]] for (j = 0; j < 4; ++j)
                    k[i][j] = gAttentionScratch16[1 * n_rot + 4 * i + j];
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
                    + specNCtx - history;
            idx = specNCtx * group + (idx % specNCtx);

            nextHistoryIndex = bufferHistoryIndex.Load<uint>(4 * idx);
        }

        float qk = 0;
        [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
            qk += dot(q[i], k[i]);
        }
        gAttentionQK[history] = qk;
        maxQK = max(maxQK, qk);
#if DEBUG_ATTENTION
        if (head < 8) {
            printf("head: %u tid: %u history: %u qk: %f maxQK: %f\n", head, tid, history, qk, maxQK);
        }
#endif
        historyIndex = nextHistoryIndex;
    }

    // Step 3: bulk of softmax(QK)
    //
    // Multiplication by 1/sum is applied in the next step.

    gAttentionScratch32[tid] = maxQK;

    GroupMemoryBarrierWithGroupSync();

    maxQK = -1.#INF;
    [[unroll]] for (i = 0; i < NUM_THIN_ATTENTION_THREADS / waveSize; ++i)
        maxQK = max(maxQK, gAttentionScratch32[i * waveSize + lane]);
    maxQK = WaveActiveMax(maxQK);

    float sum = 0.0;

    for (history = tid;
         history <= g_constants.currentHistoryLength;
         history += NUM_THIN_ATTENTION_THREADS)
    {
        float qk = gAttentionQK[history];
        qk = exp(qk - maxQK);
        gAttentionQK[history] = qk;

        sum += qk;
    }

    gAttentionScratch32[NUM_THIN_ATTENTION_THREADS + tid] = sum;

    GroupMemoryBarrierWithGroupSync();

    sum = 0.0;
    [[unroll]] for (i = 0; i < NUM_THIN_ATTENTION_THREADS / waveSize; ++i)
        sum += gAttentionScratch32[NUM_THIN_ATTENTION_THREADS + i * waveSize + lane];
    sum = WaveActiveSum(sum);

    float recipSum = 1.0 / sum;

    // Step 4: softmax(QK) * V
    //
    // Each thread computes two output elements, if only due to the difficulties
    // of loading individual 16-bit values.
    //
    // The workgroup is partitioned, with the low partition processing
    // even history entries and the high partition processing odd entries.
    //
    // TODO: This likely needs more unrolling to keep VRAM busy.
    float2 oResult = 0;
    uint partition = tid / (n_rot / 2);
    uint partitionTid = tid % (n_rot / 2);

    history = partition;

    if (history != 0 && history <= g_constants.currentHistoryLength) {
        uint group = g_constants.currentHistoryBase / specNCtx;
        uint idx =
                g_constants.currentHistoryBase
                + g_constants.currentHistoryLength
                + specNCtx - history;
        idx = specNCtx * group + (idx % specNCtx);

        historyIndex = bufferHistoryIndex.Load<uint>(4 * idx);
    }

    for (; history <= g_constants.currentHistoryLength; history += 2) {
        half2 v;
        if (history == 0) {
            [[unroll]] for (i = 0; i < 2; ++i)
                v[i] = gAttentionScratch16[2 * n_rot + 2 * partitionTid + i];
        } else {
            uint storageOffset = 2 * (historyIndex * specNEmbd + head * n_rot + 2 * partitionTid);
            v = bufferValues.Load<half2>(storageOffset);
        }

        if (history + 2 <= g_constants.currentHistoryLength) {
            uint group = g_constants.currentHistoryBase / specNCtx;
            uint idx =
                    g_constants.currentHistoryBase
                    + g_constants.currentHistoryLength
                    + specNCtx - history;
            idx = specNCtx * group + (idx % specNCtx);

            historyIndex = bufferHistoryIndex.Load<uint>(4 * idx);
        }

        oResult += gAttentionQK[history] * v;
    }

    // The low half is probably done before the high half (because the first
    // iteration fetches from local shared memory), so let the second half do
    // the final sum.
    if (partition == 0) {
        gAttentionScratch32[2 * partitionTid + 0] = oResult[0];
        gAttentionScratch32[2 * partitionTid + 1] = oResult[1];
    }

    GroupMemoryBarrierWithGroupSync();

    if (partition == 1) {
        oResult[0] += gAttentionScratch32[2 * partitionTid + 0];
        oResult[1] += gAttentionScratch32[2 * partitionTid + 1];

        oResult *= recipSum;

        bufferStage2.Store<half2>(2 * (head * n_rot + 2 * partitionTid), half2(oResult));
    }
}

// Matrix-vector multiply-add with Q4_0_SWZ matrix.
//
// Every lane computes a single output element.
//
// TODO: Use specialization constants or push constants?
// TODO: Target wave32 in WGP mode
[numthreads(NUM_THIN_MATMUL_THREADS, 1, 1)]
void KernelThinFp16MatMulAdd(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint outIdx = gid.x * NUM_THIN_MATMUL_THREADS + localTid.x;

    uint numIn = specMode == 0 ? specNEmbd : specNFF;
    uint numOut = specNEmbd;

    // Step 1: Matrix multiply.
    // Assume Q4_0_SWZ format.
    uint nDoubleBlocks = numIn / 64;
    uint matrixScaleBytes = 4 * nDoubleBlocks * numOut;

    uint scaleOffset = 0 + outIdx * 4;
    uint scaleStride = numOut * 4;
    uint weightsOffset = matrixScaleBytes + outIdx * 32;
    uint weightsStride = numOut * 32;

    // NOTE: Can only load at offsets that are a multiple of 4 :(
    half2 prev = bufferBypass.Load<half2>(4 * (outIdx / 2));
    float result = prev[outIdx % 2];

    uint inOffset = 0;
    for (uint blockIdx = 0; blockIdx < nDoubleBlocks; ++blockIdx) {
        uint4 rawWeights[2];
        half2 activations[2][4][4];
        uint sub, i;
        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            if (specMode == 0) {
                rawWeights[sub] = bufferWo.Load<uint4>(weightsOffset + 16 * sub);
            } else {
                rawWeights[sub] = bufferW2.Load<uint4>(weightsOffset + 16 * sub);
            }

            [[unroll]] for (i = 0; i < 4; ++i) {
                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    activations[sub][i][byte] = bufferStage2.Load<half2>(inOffset);
                    inOffset += 4;
                }
            }
        }

        half2 scales;
        if (specMode == 0) {
            scales = bufferWo.Load<half2>(scaleOffset);
        } else {
            scales = bufferW2.Load<half2>(scaleOffset);
        }

        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            float sum = 0;

            [[unroll]] for (i = 0; i < 4; ++i) {
                uint lowWeights = rawWeights[sub][i] & 0x0f0f0f0f;
                uint highWeights = (rawWeights[sub][i] >> 4) & 0x0f0f0f0f;

                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    int16_t2 utmp;
                    half2 prod;

                    utmp.x = (int16_t)((lowWeights >> (8 * byte)) & 0xff);
                    utmp.y = (int16_t)((highWeights >> (8 * byte)) & 0xff);
                    prod = (half2)(utmp - 8) * activations[sub][i][byte];
                    sum += prod.x + prod.y;
                }
            }

            result += scales[sub] * sum;
        }

        scaleOffset += scaleStride;
        weightsOffset += weightsStride;
    }

    // NOTE: Can only store at offsets that are a multiple of 4 :(
    //
    // ATTN: Assume a linear layout of threads!
    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();
    float otherResult = WaveReadLaneAt(result, lane ^ 1);

    if (outIdx % 2 == 0) {
        half2 data = half2(result, otherResult);
        bufferBypass.Store<half2>(4 * (outIdx / 2), data);
    }
}

// Main part of feed forward network: SILU(x * W1) * (x * W3)
//
// Every lane computes a single output element.
//
// TODO: Use specialization constants or push constants?
// TODO: Target wave32 in WGP mode
[numthreads(NUM_THIN_MATMUL_THREADS, 1, 1)]
void KernelThinFp16Ffn(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint outIdx = gid.x * NUM_THIN_MATMUL_THREADS + localTid.x;

    uint numIn = specNEmbd;
    uint numOut = specNFF;

    // Step 1: Matrix multiplies.
    // Assume Q4_0_SWZ format.
    uint nDoubleBlocks = numIn / 64;
    uint matrixScaleBytes = 4 * nDoubleBlocks * numOut;

    uint scaleOffset = 0 + outIdx * 4;
    uint scaleStride = numOut * 4;
    uint weightsOffset = matrixScaleBytes + outIdx * 32;
    uint weightsStride = numOut * 32;

    float result1 = 0;
    float result3 = 0;

    uint inOffset = 0;
    for (uint blockIdx = 0; blockIdx < nDoubleBlocks; ++blockIdx) {
        uint4 rawWeights1[2];
        uint4 rawWeights3[2];
        half2 activations[2][4][4];
        uint sub, i;
        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            rawWeights1[sub] = bufferW1.Load<uint4>(weightsOffset + 16 * sub);
            rawWeights3[sub] = bufferW3.Load<uint4>(weightsOffset + 16 * sub);

            [[unroll]] for (i = 0; i < 4; ++i) {
                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    activations[sub][i][byte] = bufferStage1.Load<half2>(inOffset);
                    inOffset += 4;
                }
            }
        }

        half2 scales1 = bufferW1.Load<half2>(scaleOffset);
        half2 scales3 = bufferW3.Load<half2>(scaleOffset);

        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            float sum1 = 0;
            float sum3 = 0;

            [[unroll]] for (i = 0; i < 4; ++i) {
                uint lowWeights1 = rawWeights1[sub][i] & 0x0f0f0f0f;
                uint highWeights1 = (rawWeights1[sub][i] >> 4) & 0x0f0f0f0f;
                uint lowWeights3 = rawWeights3[sub][i] & 0x0f0f0f0f;
                uint highWeights3 = (rawWeights3[sub][i] >> 4) & 0x0f0f0f0f;

                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    int16_t2 utmp;
                    half2 prod;

                    utmp.x = (int16_t)((lowWeights1 >> (8 * byte)) & 0xff);
                    utmp.y = (int16_t)((highWeights1 >> (8 * byte)) & 0xff);
                    prod = (half2)(utmp - 8) * activations[sub][i][byte];
                    sum1 += prod.x + prod.y;

                    utmp.x = (int16_t)((lowWeights3 >> (8 * byte)) & 0xff);
                    utmp.y = (int16_t)((highWeights3 >> (8 * byte)) & 0xff);
                    prod = (half2)(utmp - 8) * activations[sub][i][byte];
                    sum3 += prod.x + prod.y;
                }
            }

            result1 += scales1[sub] * sum1;
            result3 += scales3[sub] * sum3;
        }

        scaleOffset += scaleStride;
        weightsOffset += weightsStride;
    }

    // Step 2: SILU and multiply
    float result = result1/(1.0 + exp(-result1)) * result3;

    // NOTE: Can only store at offsets that are a multiple of 4 :(
    //
    // ATTN: Assume a linear layout of threads!
    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();
    float otherResult = WaveReadLaneAt(result, lane ^ 1);

    if (outIdx % 2 == 0) {
        half2 data = half2(result, otherResult);
        bufferStage2.Store<half2>(4 * (outIdx / 2), data);
    }
}
