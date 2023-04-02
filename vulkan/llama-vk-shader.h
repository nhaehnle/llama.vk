#pragma once

#ifdef LLAMA_HOST
namespace shader {

using uint = uint32_t;
#endif

struct GlobalConstantBuffer {
    float rmsEpsilon;
    uint currentRotaryPosition;
    uint currentStorageIndex;
    uint currentHistoryBase;
    uint currentHistoryLength;
    uint numKeyValueEntries;
};

struct HostBuffer {
    // TODO
    uint result_token;
    uint result_topK[1];
};

#ifdef LLAMA_HOST
}
#endif
