#ifndef _SIMA_LLIMA_MLA_MODEL_
#define _SIMA_LLIMA_MLA_MODEL_

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "mla_buffer.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

class FileProvider;  // forward declaration; full type in file_provider.hpp
class LanguageModel;

// One reloc port binding: size bytes of buffer starting at offset.
struct MLARelocBinding {
    MLABuffer* buffer = nullptr;
    uint64_t offset = 0;
    uint64_t size = 0;
};

void connect_mla_rt(const std::vector<std::string>& args);
void disconnect_mla_rt();

class MLAModelWithBuffer {
    friend class LanguageModel;

    public:
        // instance > 0 loads a separate copy of the same ELF, with its own reloc bindings
        // (streamed-MoE template slots). MLA-RT shares identical sections between copies.
        MLAModelWithBuffer(
            std::filesystem::path model_path,
            std::vector<MLABufferSlice> ifms,
            std::vector<MLABufferSlice> ofms,
            uint32_t instance = 0
        );
        ~MLAModelWithBuffer() = default;

        void load();
        void free();
        void run(
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr = nullptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr = nullptr
        );
        void add_to_queue(
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr = nullptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr = nullptr
        );
        void update_reloc(const std::map<std::string, MLABuffer*>& reloc_buffers);
        // Binds reloc ports in order. Each binding must cover the model's reloc length.
        // Only legal while no queued or in-flight job uses this model.
        void update_reloc(const std::vector<MLARelocBinding>& relocs);

        static void run_queue();
        // Submit every queued run without waiting; wait_queue() reaps them. Runs execute in
        // submission order, so later runs may depend on earlier ones.
        static void submit_queue();
        static void wait_queue();
        // Run every queued run with one batched dispatch (mla_run_model_multi_model), which
        // prepares jobs ahead and costs far less per job than submitting them one at a time.
        // Synchronous; waits for any in-flight async runs first.
        static void run_queue_batched();
        // When set, run_queue() uses the batched dispatch.
        static inline bool batched_dispatch = false;
        // Sum of MLA tile time of every queued job reaped so far (microseconds).
        static uint64_t total_tile_us();
        // SIMA_LLIMA_PROFILE_UNITS=1: MLA tile time per unit kind (layer digits folded to #); =2: per unit.
        static bool unit_profile_enabled();
        static void reset_unit_profile();
        static std::string unit_profile_report(uint64_t tokens);
        // Host time spent in add_to_queue, submitting and reaping (nanoseconds), for profiling.
        static void host_counters(uint64_t& add_ns, uint64_t& submit_ns, uint64_t& reap_ns, uint64_t& jobs);
        // Existing disk-loading overload.
        static void load_all_models(
            std::optional<std::filesystem::path> relative_dir = std::nullopt
        );
        // Loads through files (nullptr = from disk).
        static void load_all_models(
            std::optional<std::filesystem::path> relative_dir,
            FileProvider* files
        );
        static void free_all_models(
            std::optional<std::filesystem::path> relative_dir = std::nullopt
        );

        static void read_env_vars() {
            _profile = get_env_var("SIMA_LLIMA_RUN_PROFILE", _profile);
            _print_inouts = get_env_var("SIMA_LLIMA_RUN_PRINT_INOUTS", _print_inouts);
            _save_inouts = get_env_var("SIMA_LLIMA_RUN_SAVE_INOUTS", _save_inouts);
            if (_save_inouts) {
                _save_inout_dir = get_env_var(
                    "SIMA_LLIMA_RUN_SAVE_INOUT_DIR", _save_inout_dir
                );
                // Optional regex on the model file stem: save only matching models.
                _save_inout_filter = get_env_var(
                    "SIMA_LLIMA_RUN_SAVE_INOUTS_FILTER", _save_inout_filter
                );
            }
            _enable_queue = !get_env_var(
                "SIMA_LLIMA_RUN_DISABLE_QUEUE", !_enable_queue
            );
            _disable_parallel_load = get_env_var(
                "SIMA_LLIMA_RUN_DISABLE_PARALLEL_LOAD", _disable_parallel_load
            );
        }

    private:
        void _debug_inouts(
            const std::string& name,
            std::map<uint8_t, MLABufferSlice>* fm_map_ptr
        );
        void _bind_ifm(
            uint8_t index,
            MLABuffer* buffer,
            std::initializer_list<uint32_t> begins
        );
        void _bind_ofm(
            uint8_t index,
            MLABuffer* buffer,
            std::initializer_list<uint32_t> begins
        );

        std::size_t _model_idx;
        std::vector<MLABufferSlice> _ifms;
        std::vector<MLABufferSlice> _ofms;

        static inline bool _profile = false;
        static inline bool _print_inouts = false;
        static inline bool _save_inouts = false;
        static inline std::string _save_inout_filter;
        static inline std::string _save_inout_dir = "debug/model_io";
        static inline bool _enable_queue = true;
        static inline bool _disable_parallel_load = false;
};

}
}

#endif
