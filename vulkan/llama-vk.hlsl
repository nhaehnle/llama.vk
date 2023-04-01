// dxc -spirv -T cs_6_6 -E KernelRmsNorm1 -fspv-target-env=vulkan1.3 -enable-16bit-types llama-vk.hlsl

typedef float16_t activation;

[[vk::constant_id(1)]] const uint specNEmbd = 6656; // LLaMa 30B

[[vk::binding(0, 0)]] cbuffer ForwardPassConstants {
    struct {
        uint numTokens;
        float rmsEpsilon;
    } g_constants;
};

[[vk::binding(0, 1)]] RWByteAddressBuffer bufferInput;
[[vk::binding(1, 1)]] RWByteAddressBuffer bufferOutput;

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

// TODO: Use specialization constants
// TODO: Target wave64 in WGP mode
[numthreads(NUM_WAVE_THREADS, 1, 1)]
void KernelThinFp16MatMul(uint3 gid : SV_GroupID, uint3 localTid : SV_GroupThreadID) {
}
