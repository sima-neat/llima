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
// Shared ownership of one DMA-buffer import, tagged with its MLA connection generation.
struct RuntimeBufferImport {
    RuntimeContext* context;
    uint32_t id;
    uint64_t epoch;
    ~RuntimeBufferImport();
};
// A submitted run retains the DMA imports for exactly the relocation ports it selected.
struct RuntimeRelocation {
    std::vector<mla_tensor> ports;
    std::vector<std::shared_ptr<RuntimeBufferImport>> imports;
    // Keep the zero binding with the candidate until its native update succeeds.
    std::shared_ptr<const RuntimeRelocation> neutral;
};
// Shared native program and relocation state.
// Unloading is deferred while queued runs still retain the program.
struct RuntimeProgram {
    RuntimeContext* context;
    size_t index;
    ~RuntimeProgram();
    void activate_relocation(const std::shared_ptr<const RuntimeRelocation>& desired);
    void unload();
    size_t queued_runs = 0;
    bool unload_requested = false;
    std::shared_ptr<const RuntimeRelocation> neutral_reloc;
    std::shared_ptr<const RuntimeRelocation> active_reloc; // Null selects the base model.
};

// Process-wide MLA connection and native program/buffer registry.
// Its execution mutex serializes hardware calls and registry changes.
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
