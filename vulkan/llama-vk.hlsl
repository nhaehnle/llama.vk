// dxc -spirv -T cs_6_6 -E KernelRmsNorm1 -fspv-target-env=vulkan1.3 -enable-16bit-types llama-vk.hlsl

// Descriptor set 0: Pass globals
// Descriptor set 1: Kernel-specific
// Descriptor set 2: Layer globals

#include "llama-vk-shader.h"

#define DEBUG_FIRST_RMS_NORM 0
#define DEBUG_ATTENTION 0
#define DEBUG_OUTPUT 1
#define DEBUG_ASSERTIONS 0


#if DEBUG_ASSERTIONS
#define assert(cond) \
    do { \
        if (!(cond)) \
            printf("assert failed at %s:%d: %s", __FILE__, __LINE__, #cond); \
    } while (false)
#else
#define assert(cond) do { } while (false)
#endif

[[vk::ext_instruction(/* OpBitcast */ 124)]]
uint16_t asuint16(half x);

[[vk::ext_instruction(/* OpBitcast */ 124)]]
half ashalf(uint16_t x);

[[vk::constant_id(0)]] const uint specNEmbd = 6656; // LLaMa 30B
[[vk::constant_id(1)]] const uint specNCtx = 2048; // all LLaMas
[[vk::constant_id(2)]] const uint specNFF = 17920; // LLaMa 30B
[[vk::constant_id(3)]] const uint specNVocab = 32000; // all LLaMas
[[vk::constant_id(4)]] const uint specNHead = 52; // LLaMa 30B
[[vk::constant_id(5)]] const float specRotaryTheta = 10000.0;
[[vk::constant_id(10)]] const uint specMode = 0; // meaning depends on the kernel

[[vk::binding(0, 0)]] ConstantBuffer<GlobalConstantBuffer> g_constants;

[[vk::binding(1, 0)]] ByteAddressBuffer bufferHistoryIndex;
[[vk::binding(2, 0)]] RWByteAddressBuffer bufferHistoryTokens;
[[vk::binding(3, 0)]] ByteAddressBuffer bufferEmbedding;

[[vk::binding(4, 0)]] RWByteAddressBuffer bufferBypass;
[[vk::binding(5, 0)]] RWByteAddressBuffer bufferStage1;
[[vk::binding(6, 0)]] RWByteAddressBuffer bufferStage2;

[[vk::binding(7, 0)]] ByteAddressBuffer bufferModelNorm;
[[vk::binding(8, 0)]] ByteAddressBuffer bufferModelOutput;

[[vk::binding(9, 0)]] RWStructuredBuffer<OutputScratch> bufferOutputScratch;
[[vk::binding(10, 0)]] RWStructuredBuffer<ResultBuffer> bufferResult;

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

[[vk::push_constant]] UploadPushConstants g_upload;

[[vk::binding(0, 0)]] ByteAddressBuffer uploadSrc;
[[vk::binding(1, 0)]] RWByteAddressBuffer uploadDst;

#define NUM_WGP_THREADS 256 // multiple of a wave size!

groupshared float gRmsScratch[NUM_WGP_THREADS];

void rmsNormTopHalf(uint tid, uint numPerLane, half2 activations[16]) {
    // Compute sum of squares in parallel streams to hide ALU latency
    float means[3] = { 0, 0, 0 };

    uint idx;
    [[unroll]] for (idx = 0; idx + 3 <= numPerLane / 2; idx += 3) {
        [[unroll]] for (uint i = 0; i < 3; ++i) {
            float2 x = activations[idx + i];
            means[i] += dot(x, x);
        }
    }
    [[unroll]] for (uint i = 0; idx + i < numPerLane / 2; ++i) {
        float2 x = activations[idx + i];
        means[i] += dot(x, x);
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
        } else if (specMode == 1) {
            activations[idx + 0] *= bufferFfnNorm.Load<half2>(offset + 0);
            activations[idx + 1] *= bufferFfnNorm.Load<half2>(offset + 4);
            activations[idx + 2] *= bufferFfnNorm.Load<half2>(offset + 8);
            activations[idx + 3] *= bufferFfnNorm.Load<half2>(offset + 12);
        } else {
            activations[idx + 0] *= bufferModelNorm.Load<half2>(offset + 0);
            activations[idx + 1] *= bufferModelNorm.Load<half2>(offset + 4);
            activations[idx + 2] *= bufferModelNorm.Load<half2>(offset + 8);
            activations[idx + 3] *= bufferModelNorm.Load<half2>(offset + 12);
        }
        if (swizzled)
            offset += numThreads * 16;
        else
            offset += 16;
    }

    if (swizzled)
        offset = 4 * idx * numThreads + 4 * tid;
    [[unroll]] for (; idx < numPerLane / 2; ++idx) {
        if (specMode == 0) {
            activations[idx] *= bufferAttentionNorm.Load<half2>(offset);
        } else if (specMode == 1) {
            activations[idx] *= bufferFfnNorm.Load<half2>(offset);
        } else {
            activations[idx] *= bufferModelNorm.Load<half2>(offset);
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
        offset = 4 * idx * numThreads + 4 * tid;
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
                weights = bufferEmbedding.Load<uint4>(baseWeights + 32 * doubleBlockIdx + 8 * subBlockIdx);
            } else {
                weights.xy = bufferEmbedding.Load<uint2>(baseWeights + 32 * doubleBlockIdx + 8 * subBlockIdx);
            }

#if DEBUG_FIRST_RMS_NORM
            if (tid < 2) {
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

    offset = 4 * idx * NUM_WGP_THREADS + 4 * localTid.x;
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
    if (head == 0 && tid / 8 < 1) {
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
        printf("tid: %u rope: q = %+8f k = %+8f qOther = %+8f kOther = %+8f cos(t) = %f sgn sin(t) = %f\n", tid,
            qResult, kResult, qOther, kOther, cost, sint);
    }
#endif

    // Step 1c: Broadcast and store to prepare for Q * K calculation
    //
    // NOTE: Work around having to use multiples of 4 as storage offset
    uint currentStorageOffset = 2 * (g_constants.currentStorageIndex * specNEmbd + head * n_rot + tid);
    float vOther = WaveReadLaneAt(vResult, lane ^ 1);
    kOther = WaveReadLaneAt(kResult, lane ^ 1);
    if ((lane & 1) == 0) {
        bufferKeys.Store<half2>(currentStorageOffset, half2(kResult, kOther));
        bufferValues.Store<half2>(currentStorageOffset, half2(vResult, vOther));
    }

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
#if DEBUG_ATTENTION
        if (head == 0 && history < 2) {
            printf("head: %u history: %u historyIndex: %u\n", head, history, historyIndex);
            printf("head: %u history: %u k: %+8v4f, %+8v4f\n", head, history, (float4)k[0], (float4)k[1]);
        }
#endif
        uint nextHistoryIndex;
        if (history + NUM_THIN_ATTENTION_THREADS <= g_constants.currentHistoryLength) {
            uint group = g_constants.currentHistoryBase / specNCtx;
            uint idx =
                    g_constants.currentHistoryBase
                    + g_constants.currentHistoryLength
                    + specNCtx - history - NUM_THIN_ATTENTION_THREADS;
            idx = specNCtx * group + (idx % specNCtx);

            nextHistoryIndex = bufferHistoryIndex.Load<uint>(4 * idx);
        }

        float qk = 0;
        [[unroll]] for (i = 0; i < n_rot / 4; ++i) {
            qk += dot(q[i], k[i]);
        }
        qk *= 1.0 / sqrt(n_rot);
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

#if DEBUG_ATTENTION
        if (head == 0 && partitionTid < 8) {
            printf("head: %u tid: %3u history: %u historyIndex: %u\n", head, tid, history, historyIndex);
        }
#endif

        if (history + 2 <= g_constants.currentHistoryLength) {
            uint group = g_constants.currentHistoryBase / specNCtx;
            uint idx =
                    g_constants.currentHistoryBase
                    + g_constants.currentHistoryLength
                    + specNCtx - history - 2;
            idx = specNCtx * group + (idx % specNCtx);

            historyIndex = bufferHistoryIndex.Load<uint>(4 * idx);
        }

        oResult += gAttentionQK[history] * v;
#if DEBUG_ATTENTION
        if (head == 0 && partitionTid < 8) {
            printf("head: %u tid: %3u history: %u - weight: %+8f v: %+8v2f oResult: %+8v2f\n",
                head, tid, history, gAttentionQK[history], (float2)v, oResult);
        }
#endif
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

// Multiply-add of vector with matrix.
//
// mode selects the matrix:
//  0 -> Wo
//  1 -> W2
//  2 -> output
float thinMatMul(uint mode, uint outIdx, float accum) {
    uint numIn;
    uint numOut;

    if (mode == 0) {
        numIn = specNEmbd;
        numOut = specNEmbd;
    } else if (mode == 1) {
        numIn = specNFF;
        numOut = specNEmbd;
    } else {
        numIn = specNEmbd;
        numOut = specNVocab;
    }

    // Step 1: Matrix multiply.
    // Assume Q4_0_SWZ format.
    uint nDoubleBlocks = numIn / 64;
    uint matrixScaleBytes = 4 * nDoubleBlocks * numOut;

    uint scaleOffset = 0 + outIdx * 4;
    uint scaleStride = numOut * 4;
    uint weightsOffset = matrixScaleBytes + outIdx * 32;
    uint weightsStride = numOut * 32;

    uint inOffset = 0;
    for (uint blockIdx = 0; blockIdx < nDoubleBlocks; ++blockIdx) {
        uint4 rawWeights[2];
        half2 activations[2][4][4];
        uint sub, i;
        [[unroll]] for (sub = 0; sub < 2; ++sub) {
            if (mode == 0) {
                rawWeights[sub] = bufferWo.Load<uint4>(weightsOffset + 16 * sub);
            } else if (mode == 1) {
                rawWeights[sub] = bufferW2.Load<uint4>(weightsOffset + 16 * sub);
            } else {
                rawWeights[sub] = bufferModelOutput.Load<uint4>(weightsOffset + 16 * sub);
            }

            [[unroll]] for (i = 0; i < 4; ++i) {
                [[unroll]] for (uint byte = 0; byte < 4; ++byte) {
                    if (mode <= 1) {
                        activations[sub][i][byte] = bufferStage2.Load<half2>(inOffset);
                    } else {
                        activations[sub][i][byte] = bufferStage1.Load<half2>(inOffset);
                    }
                    inOffset += 4;
                }
            }
        }

        half2 scales;
        if (mode == 0) {
            scales = bufferWo.Load<half2>(scaleOffset);
        } else if (mode == 1) {
            scales = bufferW2.Load<half2>(scaleOffset);
        } else {
            scales = bufferModelOutput.Load<half2>(scaleOffset);
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

            accum += scales[sub] * sum;
        }

        scaleOffset += scaleStride;
        weightsOffset += weightsStride;
    }

    return accum;
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

    // Step 1: Matrix multiply.
    //
    // NOTE: Can only load at offsets that are a multiple of 4 :(
    half2 prev = bufferBypass.Load<half2>(4 * (outIdx / 2));
    float result = prev[outIdx % 2];

    result = thinMatMul(specMode, outIdx, result);

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

#define NUM_OUTPUT_LOCAL_POOL 2047

groupshared struct {
    Histogram histogram[2]; // 2kB
    uint pool1[NUM_OUTPUT_LOCAL_POOL + 1]; // 8kB
    uint pool2[NUM_OUTPUT_LOCAL_POOL + 1]; // 8kB
    uint topKUpperBound;
    uint tmp1;
    uint tmp2;
} g_output;

uint histogramIndex(uint key) {
    // key is at most 255
    return key ^ (key >> 3);
}

uint encodeOutput(uint token, float weight) {
    uint x = (uint)asuint16((half)weight) << 16;
    if ((x & 0x80000000) == 0)
        x ^= 0x7fff0000;
    return x | token;
}

void decodeOutput(uint x, out uint token, out float weight) {
    token = x & 0xffff;
    if ((x & 0x80000000) == 0)
        x ^= 0x7fff0000;
    weight = ashalf((uint16_t)(x >> 16));
}

void processHistogram(uint hist, uint waveId, bool topK, bool accumulate, bool debug) {
    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();
    uint numWaves = NUM_OUTPUT_THREADS / waveSize;
    uint wave;

    if (waveId == 0) {
        uint numWaves = 256 / waveSize;
        uint laneBuckets[16]; // ATTN: Assume waveSize >= 16
        uint laneAccum = 0;

        [[unroll]] for (wave = 0; wave < numWaves; ++wave) {
            laneBuckets[wave] = g_output.histogram[hist].bucket[histogramIndex(lane * numWaves + wave)];
            laneAccum += laneBuckets[wave];
        }

        uint lanePrefix = WavePrefixSum(laneAccum);

        if (debug) {
            printf("histogram lane: %2u lanePrefix: %5u buckets: %4u %4u %4u %4u\n",
                lane, lanePrefix,
                laneBuckets[0], laneBuckets[1], laneBuckets[2], laneBuckets[3]);
        }

        if (accumulate) {
            uint accum = lanePrefix;
            [[unroll]] for (wave = 0; wave < numWaves; ++wave) {
                g_output.histogram[hist].bucket[histogramIndex(lane * numWaves + wave)] = accum;
                accum += laneBuckets[wave];
            }
        }

        if (topK) {
            bool someAccepted = lanePrefix < g_constants.topK;
            uint maxLane = WaveActiveCountBits(someAccepted) - 1;

            lanePrefix = WaveReadLaneAt(lanePrefix, maxLane);

            uint maxSub = 0;
            [[unroll]] for (wave = 0; wave < numWaves - 1; ++wave) {
                lanePrefix += WaveReadLaneAt(laneBuckets[wave], maxLane);
                maxSub = lanePrefix < g_constants.topK ? wave + 1 : maxSub;
            }

            g_output.topKUpperBound = ((maxLane * numWaves + maxSub + 1) << 24) - 1;
        }
    }

    GroupMemoryBarrierWithGroupSync();
}

void loadFromGlobalPool(uint tid, uint numGroups, inout uint base, uint globalPoolSize,
                        uint topKUpperBound, uint numInbounds, uint maxBoundary) {
    uint lane = WaveGetLaneIndex();

    for (base = 0; base < globalPoolSize; base += numGroups * NUM_OUTPUT_THREADS) {
        uint group;
        uint entries[4];
        bool inbounds[4];

        uint curBase = min(base + tid * numGroups, (globalPoolSize / numGroups) * numGroups);
        [[unroll]] for (group = 0; group < numGroups; ++group) {
            entries[group] = bufferStage2.Load<uint>(4 * (curBase + group));
            inbounds[group] = curBase + group < globalPoolSize;
        }

        bool boundary[4];
        uint prefixBoundary[4];
        uint numBoundary = 0;
        [[unroll]] for (group = 0; group < numGroups; ++group) {
            inbounds[group] = inbounds[group] &&
                              entries[group] <= topKUpperBound;
            boundary[group] = inbounds[group] &&
                              entries[group] >= (topKUpperBound & 0xff000000);
            prefixBoundary[group] = numBoundary + WavePrefixCountBits(boundary[group]);
            numBoundary += WaveActiveCountBits(boundary[group]);
        }

        if (numBoundary) {
            uint preBoundary;
            if (lane == 0) {
                InterlockedAdd(g_output.tmp2, numBoundary, preBoundary);
            }
            preBoundary = WaveReadLaneAt(preBoundary, 0);

            [[unroll]] for (group = 0; group < numGroups; ++group) {
                prefixBoundary[group] += preBoundary;
                inbounds[group] = inbounds[group] && (!boundary[group] || prefixBoundary[group] < maxBoundary);
            }
        }

        uint prefixInbounds[4];
        uint numInbounds = 0;
        [[unroll]] for (group = 0; group < numGroups; ++group) {
            prefixInbounds[group] = numInbounds + WavePrefixCountBits(inbounds[group]);
            numInbounds += WaveActiveCountBits(inbounds[group]);
        }

        if (!numInbounds)
            continue;

        uint preInbounds;
        if (lane == 0) {
            InterlockedAdd(g_output.tmp1, numInbounds, preInbounds);
        }
        preInbounds = WaveReadLaneAt(preInbounds, 0);

        [[unroll]] for (group = 0; group < numGroups; ++group) {
            prefixInbounds[group] += preInbounds;
            prefixInbounds[group] = inbounds[group] ? prefixInbounds[group] : NUM_OUTPUT_LOCAL_POOL;
            g_output.pool1[prefixInbounds[group]] = entries[group];
        }

        // TODO: This doesn't work with numGroups != 1. Compiler / driver bug?!?
        [[unroll]] for (group = 0; group < numGroups; ++group) {
            if (inbounds[group]) {
                uint idx = histogramIndex((entries[group] >> 16) & 0xff);
                uint noReturn;
                // printf("tid: %u group %u histogram key: %u\n", tid, group, (entries[group] >> 16) & 0xff);
                InterlockedAdd(g_output.histogram[1].bucket[idx], 1, noReturn);
            }
        }
    }
}

// Output kernel. This does:
//
//  * output multiply (one token per thread)
//  * top-K sampling (at most 256)
//  * softmax
//  * sort + top-p cut-off
//  * sample
//
// The idea is that the result can be fed back directly to the next pass without
// going to the CPU.
//
// This kernel starts out with one thread per token and then narrows to a single
// workgroup that is chosen dynamically.
[numthreads(NUM_OUTPUT_THREADS, 1, 1)]
void KernelThinFp16Output(uint2 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint tid = localTid.x;
    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();
    uint waveId = tid / waveSize;

    g_output.histogram[0].bucket[tid] = 0;
    g_output.histogram[1].bucket[tid] = 0;
    g_output.tmp1 = 0;
    g_output.tmp2 = 0;

    if (g_constants.repeatLastN != 0) {
        g_output.pool1[tid] = 0;

        GroupMemoryBarrierWithGroupSync();

        uint lastN = min(g_constants.repeatLastN, g_constants.currentHistoryLength + 1);
        if (tid < lastN) {
            uint group = g_constants.currentHistoryBase / specNCtx;
            uint idx = (g_constants.currentHistoryBase + g_constants.currentHistoryLength + specNCtx - tid) % specNCtx;
            uint token = bufferHistoryTokens.Load<uint>(4 * (group * specNCtx + idx));
            uint base = NUM_OUTPUT_THREADS * gid.x;

            if (base <= token && token < base + NUM_OUTPUT_THREADS)
                g_output.pool1[token - base] = 1;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Step 1: Multiply with output matrix and build local histogram.
    uint output;
    {
        uint token = NUM_OUTPUT_THREADS * gid.x + tid;
        float weight = thinMatMul(2, token, 0);

        if (g_constants.repeatLastN != 0) {
            uint base = NUM_OUTPUT_THREADS * gid.x;
            if (g_output.pool1[token - base] != 0)
                weight -= g_constants.repeatPenalty;
        }

        output = encodeOutput(token, weight);

        uint noReturn;
        InterlockedAdd(g_output.histogram[0].bucket[histogramIndex(output >> 24)], 1, noReturn);
    }

    GroupMemoryBarrierWithGroupSync();

    // Step 2: Accumulate local histogram to global histogram.
    {
        uint our = g_output.histogram[0].bucket[tid];
        uint pre;
        InterlockedAdd(bufferOutputScratch[0].histogram.bucket[tid], our, pre);
        g_output.histogram[1].bucket[tid] = our + pre;
    }

    GroupMemoryBarrierWithGroupSync();

    // Step 3: Filter entries and write them to the global pool
    {
        processHistogram(1, waveId, /* topK */ true, /* accumulate */ true, /* debug */ false);
        uint topKUpperBound = g_output.topKUpperBound;

        bool isLive = output <= topKUpperBound;
        uint count = WaveActiveCountBits(isLive);
        uint relIdx = WavePrefixCountBits(isLive);

        {
            uint waveRelIdx;
            if (lane == 0)
                InterlockedAdd(g_output.tmp1, count, waveRelIdx);
            relIdx += WaveReadLaneAt(waveRelIdx, 0);
        }

        GroupMemoryBarrierWithGroupSync();

        if (tid == 0) {
            uint numInbounds = g_output.tmp1;
            uint groupIdx;
            InterlockedAdd(bufferOutputScratch[0].poolSize, numInbounds, groupIdx);
            g_output.tmp1 = groupIdx;

            // printf("gid: %u numInbounds: %u groupIdx: %u topKUpperBound: %8x\n", gid.x,
            //        numInbounds, groupIdx, topKUpperBound);
        }

        GroupMemoryBarrierWithGroupSync();

        relIdx += g_output.tmp1;

        // if (gid.x == 0) {
        //     printf("tid: %u relIdx: %u isLive: %u output: %8x\n", tid, relIdx, isLive, output);
        // }

        if (isLive) {
            bufferStage2.Store(4 * relIdx, output);
        }
    }

    // Step 4: Signal that we committed to the pool. Check if we're the last
    // workgroup.
    AllMemoryBarrierWithGroupSync();

    if (tid == 0) {
        uint pre;
        InterlockedAdd(bufferOutputScratch[0].committed, NUM_OUTPUT_THREADS, pre);
        g_output.tmp1 = pre + NUM_OUTPUT_THREADS;
    }

    AllMemoryBarrierWithGroupSync();

    if (g_output.tmp1 < specNVocab)
        return; // not the last workgroup, exit
#if DEBUG_OUTPUT
    if (tid == 0)
        printf("output: gid %u chosen\n", gid.x);
#endif
    // Step 5: Fetch the pool size and histogram.
    uint globalPoolSize = bufferOutputScratch[0].poolSize;
    g_output.histogram[0].bucket[tid] = bufferOutputScratch[0].histogram.bucket[tid];
    g_output.histogram[1].bucket[tid] = 0;
    g_output.tmp1 = 0;
    g_output.tmp2 = 0;
#if DEBUG_OUTPUT
    if (tid == 0)
        printf("globalPoolSize: %u\n", globalPoolSize);
#endif
    if (globalPoolSize > specNVocab) {
        if (tid == 0)
            printf("error: globalPoolSize: %u\n", globalPoolSize);
        globalPoolSize = specNVocab;
    }

    GroupMemoryBarrierWithGroupSync();

    // Step 6: Fetch pool entries and do the first histogram for a radix sort of the top-K
    processHistogram(0, waveId, /* topK */ true, /* accumulate */ true, /* debug */ false);

    uint topKUpperBound = g_output.topKUpperBound;

    uint numInbounds;
    if ((topKUpperBound >> 24) == 255)
        numInbounds = NUM_OUTPUT_THREADS;
    else
        numInbounds = g_output.histogram[0].bucket[histogramIndex((topKUpperBound >> 24) + 1)];

    uint maxBoundary = NUM_OUTPUT_LOCAL_POOL - g_output.histogram[0].bucket[histogramIndex(topKUpperBound >> 24)];

    assert(maxBoundary > 0);

#if DEBUG_OUTPUT
    if (tid == 0)
        printf("numInbounds: %u maxBoundary: %u topKUpperBound: %8x\n", numInbounds, maxBoundary, topKUpperBound);
#endif

    if (numInbounds > NUM_OUTPUT_LOCAL_POOL) {
        // Refine the upper bound?
        if (tid == 0)
            printf("TODO: numInbounds too large: %u\n", numInbounds);
    }

    numInbounds = min(numInbounds, NUM_OUTPUT_LOCAL_POOL);

    uint base = 0;
    loadFromGlobalPool(tid, 1, base, globalPoolSize, topKUpperBound, numInbounds, maxBoundary);

    GroupMemoryBarrierWithGroupSync();

    // Step 7: Sort from pool1 into pool2 according to first stage of radix sort
    processHistogram(1, waveId, /* topK */ false, /* accumulate */ true, /* debug */ false);

    for (uint srcIdx = tid; srcIdx < numInbounds; srcIdx += NUM_OUTPUT_THREADS) {
        uint entry = g_output.pool1[srcIdx];
        uint dstIdx;
        InterlockedAdd(g_output.histogram[1].bucket[histogramIndex((entry >> 16) & 0xff)], 1, dstIdx);
        // printf("radix1 tid: %u dstIdx: %u entry: %8x\n", tid, dstIdx, entry);
        g_output.pool2[dstIdx] = entry;
    }

    GroupMemoryBarrierWithGroupSync();

    bufferOutputScratch[0].histogram.bucket[tid] = 0;

    // Step 8: Sort from pool2 into pool1 according to the second stage of radix sort
    //
    // From this point on, we narrow further to a single wave.
    //
    // The copy needs to be stable, i.e. it needs to preserve the relative order
    // of entries that go into the same bucket. Doing this with a single wave
    // of at most 32 lanes is simpler.

    if (waveId != 0)
        return;

    bufferOutputScratch[0].poolSize = 0;
    bufferOutputScratch[0].committed = 0;

    uint subWaveSize = min(waveSize, 32);
    if (lane < subWaveSize) {
        for (uint srcIdx = lane; srcIdx < numInbounds; srcIdx += subWaveSize) {
            uint entry = g_output.pool2[srcIdx];

            uint dummy;
            g_output.histogram[1].bucket[entry >> 24] = 0;
            InterlockedAdd(g_output.histogram[1].bucket[entry >> 24], 1u << lane, dummy);
            uint mask = g_output.histogram[1].bucket[entry >> 24];
            uint leader = firstbithigh(mask);
            uint dstBase = 0;
            if (lane == leader) {
                InterlockedAdd(g_output.histogram[0].bucket[histogramIndex(entry >> 24)],
                               countbits(mask), dstBase);
            }
            dstBase = WaveReadLaneAt(dstBase, leader);

            // printf("radix2 tid: %2u entry: %8x dstBase: %3u mask: %8x leader: %u\n", tid, entry, dstBase,
            //         mask, leader);

            uint dstIdx = dstBase + countbits(mask & ((1u << lane) - 1));
            g_output.pool1[dstIdx] = entry;
        }
    }

    // Step 9: Compute softmax, topP, and sample
    //

    uint idx;
    float scale = 1.0 / g_constants.temp;
    float maxLogit = -1.#INF;

    for (idx = lane; idx < g_constants.topK; idx += waveSize) {
        uint token;
        float weight;
        decodeOutput(g_output.pool1[idx], token, weight);

        // TODO: Repetition penalty?
        float logit = weight * scale;
        g_output.pool2[idx] = asuint(logit);
        maxLogit = max(maxLogit, logit);
    }

    maxLogit = WaveActiveMax(maxLogit);

    float sum = 0;
    for (idx = lane; idx < g_constants.topK; idx += waveSize) {
        float logit = asfloat(g_output.pool2[idx]) - maxLogit;
        float p = exp(logit);
        g_output.pool2[idx] = asuint(p);
        sum += p;
    }
    sum = WaveActiveSum(sum);
    float recip_sum = 1.0 / sum;

    float accum = 0;
    float topPEnd = 1.0;
    uint row;
    for (row = 0; row * waveSize < g_constants.topK; ++row) {
        idx = row * waveSize + lane;

        float p = 0;
        if (idx < g_constants.topK)
            p = asfloat(g_output.pool2[idx]) * recip_sum;

        float cump = accum + WavePrefixSum(p);
        accum += WaveActiveSum(p);

        bool inTopP = cump < g_constants.topP;
        uint numInTopP = WaveActiveCountBits(inTopP);
        topPEnd = numInTopP > 0 ? WaveReadLaneAt(cump + p, numInTopP - 1) : topPEnd;

#if DEBUG_OUTPUT
        if (idx < g_constants.topK) {
            uint token;
            float dummy;
            decodeOutput(g_output.pool1[idx], token, dummy);
            printf("%3u: token: %5u p: %8f cump + p: %8f\n", idx, token, p, cump + p);
        }
#endif

        g_output.pool2[idx] = asuint(cump + p);
    }

    float rand = g_constants.rand * topPEnd;
    uint numSkip = 0;
    for (row = 0; row * waveSize < g_constants.topK; ++row) {
        idx = row * waveSize + lane;

        float cump = asfloat(g_output.pool2[idx]);
        bool skip = cump < rand && idx + 1 < g_constants.topK;
        uint numLocalSkip = WaveActiveCountBits(skip);

        numSkip += numLocalSkip;
        if (numLocalSkip < waveSize)
            break;
    }

    uint chosen = g_output.pool1[numSkip];
    uint token;
    float weight;
    decodeOutput(chosen, token, weight);

#if DEBUG_OUTPUT
    if (lane == 0) {
        printf("chosen token: %u (numSkip: %u, const.rand: %8f rand: %8f %8x)\n",
               token, numSkip, g_constants.rand, rand, chosen);
    }
#endif

    bufferResult[0].token = token;
}

#define NUM_UPLOAD_THREADS 256

// Upload data linearly while converting from F32 to F16.
//
// Requires 1D numElements, multiple of 4 elements.
//
// 8kiB of loads in flight per workgroup.
[numthreads(NUM_UPLOAD_THREADS, 1, 1)]
void KernelUploadF32toF16(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint fetchStride = NUM_UPLOAD_THREADS * 4;
    uint outerStride = fetchStride * 2 * g_upload.numWorkgroups;
    uint index = 4 * (gid.x * 2 * NUM_UPLOAD_THREADS + localTid.x);

    while (index + fetchStride < g_upload.numElements[0]) {
        float4 load[2];
        load[0] = uploadSrc.Load<float4>(4 * (index + 0));
        load[1] = uploadSrc.Load<float4>(4 * (index + fetchStride));
        uploadDst.Store<half4>(2 * (index +           0), (half4)load[0]);
        uploadDst.Store<half4>(2 * (index + fetchStride), (half4)load[1]);
        index += outerStride;
    }

    if (index < g_upload.numElements[0]) {
        float4 load = uploadSrc.Load<float4>(4 * index);
        uploadDst.Store<half4>(2 * index, (half4)load[0]);
    }
}

// Offsets are relative to the start of the copy region.
void copyQ4DoubleBlock(uint srcOffset, uint dstScaleOffset, uint dstWeightsOffset) {
    uint dwords[10];
    uint i;

    [[unroll]] for (i = 0; i < 10; ++i)
        dwords[i] = uploadSrc.Load<uint>(srcOffset + 4 * i);

    half2 scale;
    uint4 weights[2];
    scale.x = (half)asfloat(dwords[0]);
    scale.y = (half)asfloat(dwords[5]);
    [[unroll]] for (i = 0; i < 4; ++i) {
        weights[0][i] = dwords[1 + i];
        weights[1][i] = dwords[6 + i];
    }

    uploadDst.Store<half2>(dstScaleOffset, scale);
    uploadDst.Store<uint4>(dstWeightsOffset +  0, weights[0]);
    uploadDst.Store<uint4>(dstWeightsOffset + 16, weights[1]);
}

// Upload Q4_0 from file format to a linear layout.
//
// Requires 2D numElements, numElements[0] must be a multiple of 64.
//
// Each thread handles a double block at a time (40 bytes -> 36 bytes).
//
// 10kiB of loads in flight per workgroup.
[numthreads(NUM_UPLOAD_THREADS, 1, 1)]
void KernelUploadQ4_0_linear(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint doubleBlocksPerRow = g_upload.numElements[0] / 64;
    uint numScaleBytes = doubleBlocksPerRow * g_upload.numElements[1] * 4;

    uint blockCount = g_upload.rowCount * doubleBlocksPerRow;
    uint blockStride = NUM_UPLOAD_THREADS * g_upload.numWorkgroups;

    uint dstScaleBase = 0;
    uint dstWeightsBase = numScaleBytes;

    uint blockIdx = NUM_UPLOAD_THREADS * gid.x + localTid.x;

    while (blockIdx < blockCount) {
        uint srcOffset = blockIdx * 40;
        uint dstScaleOffset   = dstScaleBase   + (g_upload.rowBegin * doubleBlocksPerRow + blockIdx) * 4;
        uint dstWeightsOffset = dstWeightsBase + (g_upload.rowBegin * doubleBlocksPerRow + blockIdx) * 32;
        copyQ4DoubleBlock(srcOffset, dstScaleOffset, dstWeightsOffset);

        blockIdx += blockStride;
    }
}

// Upload Q4_0 from file format to a swizzled layout.
//
// numElements[0] must be a multiple of 64.
// numElements[1] must be a multiple of 16.
//
// Each thread handles a double block at a time (40 bytes -> 36 bytes).
// We're loading a 16x16 tile of double blocks at a time (corresponds to
// 16x1024 elements).
//
// 10kiB of loads in flight per workgroup.
#if 0
[numthreads(NUM_UPLOAD_THREADS, 1, 1)]
void KernelUploadQ4_0_swz(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint doubleBlocksPerRow = g_upload.numElements[0] / 64;
    uint srcRowBytes = doubleBlocksPerRow * 40;
    uint numScaleBytes = doubleBlocksPerRow * g_upload.numElements[1] * 4;
    uint rowStride = 16 * g_upload.numWorkgroups;

    uint row = 16 * gid.x + localTid.x / 16;

    while (row < rowCount) {
        uint col = localTid.x % 16;

        while (col < doubleBlocksPerRow) {
            uint srcOffset = srcRowBytes * row + col * 40;
            uint dstScaleOffset   =                  4 * (g_upload.rowBegin + row + col * g_upload.numElements[1]);
            uint dstWeightsOffset = numScaleBytes + 32 * (g_upload.rowBegin + row + col * g_upload.numElements[1]);

            copyQ4DoubleBlock(srcOffset, dstScaleOffset, dstWeightsOffset);

            col += 16;
        }

        row += rowStride;
    }
}
#else
groupshared half2 g_uploadScaleBuffer[16 * 16];

[numthreads(NUM_UPLOAD_THREADS, 1, 1)]
void KernelUploadQ4_0_swz(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    uint doubleBlocksPerRow = g_upload.numElements[0] / 64;
    uint srcRowBytes = doubleBlocksPerRow * 40;
    uint numScaleBytes = doubleBlocksPerRow * g_upload.numElements[1] * 4;
    uint rowStride = 16 * g_upload.numWorkgroups;

    uint groupRow = 16 * gid.x;

    while (groupRow < g_upload.rowCount) {
        uint groupCol = 0;

        while (groupCol < doubleBlocksPerRow) {
            uint localRow = localTid.x / 16;
            uint localCol = localTid.x % 16;
            uint col = groupCol + localCol;
            uint row = groupRow + localRow;

            if (col < doubleBlocksPerRow) {
                uint srcOffset = srcRowBytes * row + col * 40;
                uint dstWeightsOffset = numScaleBytes + 32 * (g_upload.rowBegin + row + col * g_upload.numElements[1]);

                uint dwords[10];
                uint i;

                [[unroll]] for (i = 0; i < 10; ++i)
                    dwords[i] = uploadSrc.Load<uint>(srcOffset + 4 * i);

                half2 scale;
                uint4 weights[2];
                scale.x = (half)asfloat(dwords[0]);
                scale.y = (half)asfloat(dwords[5]);
                [[unroll]] for (i = 0; i < 4; ++i) {
                    weights[0][i] = dwords[1 + i];
                    weights[1][i] = dwords[6 + i];
                }

                uploadDst.Store<uint4>(dstWeightsOffset +  0, weights[0]);
                uploadDst.Store<uint4>(dstWeightsOffset + 16, weights[1]);

                // Swizzle to reduce bank conflicts.
                g_uploadScaleBuffer[17 * localRow ^ localCol] = scale;
            }

            GroupMemoryBarrierWithGroupSync();

            localRow = localTid.x % 16;
            localCol = localTid.x / 16;
            col = groupCol + localCol;
            row = groupRow + localRow;

            half2 scale = g_uploadScaleBuffer[17 * localRow ^ localCol];

            GroupMemoryBarrierWithGroupSync();

            if (col < doubleBlocksPerRow) {
                uint dstScaleOffset = 4 * (g_upload.rowBegin + row + col * g_upload.numElements[1]);
                uploadDst.Store<half2>(dstScaleOffset, scale);
            }

            groupCol += 16;
        }

        groupRow += rowStride;
    }
}
#endif
