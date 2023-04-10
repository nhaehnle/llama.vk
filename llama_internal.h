// Internal header to be included by llama.cpp, llama-vk.cpp, and tests/benchmarks only.

#ifndef LLAMA_INTERNAL_H
#define LLAMA_INTERNAL_H

#include <vector>
#include <string>
struct ggml_tensor;

struct llama_load_tensor_shard {
    std::vector<uint32_t> ne;
    size_t size;
    enum ggml_type type;
    size_t file_idx;
    size_t file_off;

    void calc_size();
};

enum llama_split_type {
    SPLIT_NONE,
    SPLIT_BY_COLUMNS,
    SPLIT_BY_ROWS
};

struct llama_load_tensor {
    std::vector<llama_load_tensor_shard> shards;

    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    llama_split_type split_type = SPLIT_NONE;
    std::vector<uint32_t> ne;
    size_t size;
    struct ggml_tensor * ggml_tensor = NULL;
    uint8_t * data;

    llama_load_tensor(const std::string & name) : name(name) {}

    void calc_all();
    void calc_type();
    void calc_split_type();
    void calc_ne();
    void calc_size();
};

struct llama_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<llama_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum llama_file_version {
    LLAMA_FILE_VERSION_GGML,
    LLAMA_FILE_VERSION_GGMF_V1, // added version field and scores in vocab
    LLAMA_FILE_VERSION_GGJT_V1, // added padding
};

struct llama_file_loader {
    llama_file file;
    llama_file_version file_version;
    llama_hparams hparams;
    llama_vocab vocab;

    llama_file_loader(const char * fname, size_t file_idx, llama_load_tensors_map & tensors_map);
    void read_magic();
    void read_hparams();
    void read_vocab();
    void read_tensor_metadata(size_t file_idx, llama_load_tensors_map & tensors_map);
};

std::vector<std::pair<std::string, struct ggml_tensor *>>& llama_internal_get_tensor_map(struct llama_context * ctx);

#endif // LLAMA_INTERNAL_H
