#pragma once

#ifdef LLAMA_HOST
namespace shader {

using uint = uint32_t;
#endif

#define NUM_THIN_MATMUL_THREADS 128
#define NUM_OUTPUT_THREADS 256

struct GlobalConstantBuffer {
    float rmsEpsilon;
    uint currentRotaryPosition;
    uint currentToken;
    uint currentStorageIndex;
    uint currentHistoryBase;
    uint currentHistoryLength;
    uint numKeyValueEntries;
    uint topK;
    float topP;
    float temp;
    float rand;
    uint repeatLastN;
    float repeatPenalty;
};

struct ResultBuffer {
    uint token;
};

struct Histogram {
    uint bucket[256];
};

struct OutputScratch {
    Histogram histogram;
    uint poolSize;
    uint padding1[63];
    uint committed;
    uint padding2[63];
};

struct UploadPushConstants {
    uint numElements[2];
    uint numWorkgroups;
    uint rowBegin;
    uint rowCount;
};

#ifdef LLAMA_HOST
}
#endif
