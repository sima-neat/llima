#pragma once

// Backend ownership stays private; model wrappers own bindings and lease programs.
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <simaai/gst-api.h>

namespace simaai::llima {

class RuntimeContext;
class MLABuffer;
struct RuntimeBufferImport {
    RuntimeContext* context;
    uint32_t id;
    uint64_t epoch;
    ~RuntimeBufferImport();
};
struct RuntimeProgram {
    RuntimeContext* context;
    size_t index;
    ~RuntimeProgram();
    std::map<std::string, std::shared_ptr<MLABuffer>> neutral_reloc;
    std::map<std::string, MLABuffer*> active_reloc; // Empty selects the neutral adapter.
    std::vector<std::shared_ptr<RuntimeBufferImport>> reloc_imports;
};

class RuntimeContext {
  public:
    static RuntimeContext& instance();
    std::shared_ptr<RuntimeProgram> reserve(std::filesystem::path path);
    // Serializes MLA calls and the program/import ownership registry.
    std::recursive_mutex execution_mutex;
    mla_handle_p handle = nullptr;
    std::map<std::filesystem::path, size_t> path_to_index;
    std::vector<std::filesystem::path> paths;
    std::vector<mla_model_p> models;
    std::vector<std::weak_ptr<RuntimeProgram>> programs;
    // Allocation identity follows storage when recurrent state swaps buffers.
    uint64_t epoch = 0;
    std::vector<uint32_t> failed_buffer_releases;
    std::unordered_map<uint64_t, std::shared_ptr<RuntimeBufferImport>> imported_buffers;
};

} // namespace simaai::llima
