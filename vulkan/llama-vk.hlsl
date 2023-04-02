// dxc -spirv -T cs_6_6 -E KernelRmsNorm1 -fspv-target-env=vulkan1.3 -enable-16bit-types llama-vk.hlsl

// Descriptor set 0: Pass globals
// Descriptor set 1: Kernel-specific
// Descriptor set 2: Layer globals

typedef float16_t activation;

enum WeightFormat {
    WeightFormatQ4_0gpu = 2,
};

[[vk::constant_id(0)]] const uint specNEmbd = 6656; // LLaMa 30B
[[vk::constant_id(1)]] const uint specNCtx = 2048; // LLaMa
[[vk::constant_id(2)]] const float specRotaryTheta = 10000.0;

[[vk::binding(0, 0)]] cbuffer ForwardPassConstants {
    struct {
        float rmsEpsilon;
        uint currentRotaryPosition;
        uint currentStorageIndex;
        uint currentHistoryBase;
        uint currentHistoryLength;
        uint numKeyValueEntries;
    } g_constants;
};

[[vk::binding(1, 0)]] ByteAddressBuffer bufferHistoryIndex;

[[vk::binding(0, 1)]] RWByteAddressBuffer bufferInput;
[[vk::binding(1, 1)]] RWByteAddressBuffer bufferOutput;

[[vk::binding(1, 2)]] RWByteAddressBuffer bufferWq;
[[vk::binding(2, 2)]] RWByteAddressBuffer bufferWk;
[[vk::binding(3, 2)]] RWByteAddressBuffer bufferWv;
[[vk::binding(4, 2)]] RWByteAddressBuffer bufferWo;

[[vk::binding(5, 2)]] RWByteAddressBuffer bufferKeys;
[[vk::binding(6, 2)]] RWByteAddressBuffer bufferValues;

#define NUM_WGP_THREADS 256
#define NUM_WAVE_THREADS 64

groupshared float gRmsScratch[NUM_WGP_THREADS];

// TODO: Use specialization constants
// TODO: Target wave64 in WGP mode
[numthreads(NUM_WGP_THREADS, 1, 1)]
void KernelThinFp16RmsNorm(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
    // LLaMa 30B: rmsNEmbdPerLane = 26
    const uint rmsNEmbdPerLane = (specNEmbd + NUM_WGP_THREADS - 1) / NUM_WGP_THREADS;

    // Load all activations into registers
    half activations[/* rmsNEmbdPerLane */ 32]; // TODO

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
        if (NUM_WGP_THREADS * idx + localTid.x < specNEmbd) {
            activations[idx] = bufferInput.Load<half>(base + offset);
        } else {
            activations[idx] = 0;
        }
    }

    // Compute sum of squares in parallel
    float means[3] = { 0, 0, 0 };
    [[unroll]] for (idx = 0; idx + 3 <= rmsNEmbdPerLane; idx += 3) {
        [[unroll]] for (uint i = 0; i < 3; ++i) {
            float x = activations[idx + i];
            means[i] += x * x;
        }
    }
    [[unroll]] for (uint i = 0; idx + i < rmsNEmbdPerLane; ++i) {
        float x = activations[idx + i];
        means[i] += x * x;
    }

    // Reduce across threads
    gRmsScratch[localTid.x] = means[0] + means[1] + means[2];

    GroupMemoryBarrierWithGroupSync();

    uint lane = WaveGetLaneIndex();
    uint waveSize = WaveGetLaneCount();

    float mean = 0.0;
    [[unroll]] for (i = 0; i < NUM_WGP_THREADS / waveSize; ++i)
        mean += gRmsScratch[i * waveSize + lane];
    mean = WaveActiveSum(mean);
    mean *= 1.0 / specNEmbd;

    // Write activations out to memory
    half scale = half(1.0 / sqrt(g_constants.rmsEpsilon + mean));

    offset = 16 * localTid.x;
    idx = 0;
    [[unroll]] for (; idx + 8 <= rmsNEmbdPerLane; idx += 8) {
        half4 vec1;
        half4 vec2;
        [[unroll]] for (uint i = 0; i < 4; ++i) {
            vec1[i] = scale * activations[idx + i];
            vec2[i] = scale * activations[idx + i + 4];
        }
        bufferOutput.Store(base + offset, vec1);
        bufferOutput.Store(base + offset + 8, vec2);
        offset += NUM_WGP_THREADS * 16;
    }

    offset = 2 * idx * NUM_WGP_THREADS + 4 * localTid.x;
    [[unroll]] for (; idx + 2 <= rmsNEmbdPerLane; idx += 2) {
        half2 word;
        word.x = scale * activations[idx];
        word.y = scale * activations[idx + 1];
        bufferOutput.Store(base + offset, word);
        offset += NUM_WGP_THREADS * 4;
    }

    if (idx < rmsNEmbdPerLane) {
        offset = 2 * idx * NUM_WGP_THREADS + 2 * localTid.x;
        if (NUM_WGP_THREADS * idx + localTid.x < specNEmbd) {
            bufferOutput.Store(base + offset, scale * activations[idx]);
        }
    }
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
