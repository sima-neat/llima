
#ifndef _SIMA_LLIMA_MLA_MODEL_
#define _SIMA_LLIMA_MLA_MODEL_

#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <dispatcherbase.hh>
#include <dispatcherfactory.hh>
#include <job.hh>

#include "mla_buffer.hpp"
#include "utils.hpp"


namespace simaai {
namespace llima {

class LanguageModel;

void connect_mla_rt(const std::vector<std::string>& args);
void disconnect_mla_rt();


class MLAModelWithBuffer {
    friend void connect_mla_rt(const std::vector<std::string>& args);
    friend void disconnect_mla_rt();
    friend class LanguageModel;

    public:
        MLAModelWithBuffer(
            std::filesystem::path model_path,
            std::vector<MLABufferSlice> ifms,
            std::vector<MLABufferSlice> ofms
        );
        ~MLAModelWithBuffer() {};

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
        void update_reloc(const std::map<std::string, uint64_t>& reloc_addr_map);

        static void initialize();
        static void run_queue();
        static void load_all_models(
            std::optional<std::filesystem::path> relative_dir = std::nullopt
        );
        static void free_all_models(
            std::optional<std::filesystem::path> relative_dir = std::nullopt
        );

        static void read_env_vars() {
            // Set the debug info from env variables.
            _profile = get_env_var("SIMA_LLIMA_RUN_PROFILE", _profile);
            _print_inouts = get_env_var("SIMA_LLIMA_RUN_PRINT_INOUTS", _print_inouts);
            _save_inouts = get_env_var("SIMA_LLIMA_RUN_SAVE_INOUTS", _save_inouts);
            if (_save_inouts)
                _save_inout_dir = get_env_var("SIMA_LLIMA_RUN_SAVE_INOUT_DIR", _save_inout_dir);
            _enable_queue = !get_env_var("SIMA_LLIMA_RUN_DISABLE_QUEUE", !_enable_queue);
            _disable_parallel_load = get_env_var(
                "SIMA_LLIMA_RUN_DISABLE_PARALLEL_LOAD", _disable_parallel_load
            );
        }

    private:
        void _debug_inouts(const std::string& name, std::map<uint8_t, MLABufferSlice>* fm_map_ptr);
        void _prepare_run(
            simaaidispatcher::PreparedMlaPlan& plan,
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        );
        static simaaidispatcher::PreparedMlaRunRef _make_run_ref(
            simaaidispatcher::PreparedMlaPlan& plan,
            mla_model_p handle
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
        static simaaidispatcher::DispatcherBase* _get_dispatcher();

        uint16_t _model_idx;
        std::vector<MLABufferSlice> _ifms;
        std::vector<MLABufferSlice> _ofms;
        simaaidispatcher::PreparedMlaPlan _prepared_plan;

        static std::map<std::filesystem::path, uint16_t> _unique_model_path_to_idx_map;
        static std::vector<std::filesystem::path> _unique_model_paths;
        static std::vector<mla_model_p> _unique_model_ptrs;
        static thread_local simaaidispatcher::DispatcherBase::PreparedMlaPartitionQueueRequest
            _queue_request;
        static thread_local std::vector<simaaidispatcher::PreparedMlaPlan> _queued_plans;
        static thread_local std::vector<mla_model_p> _queued_handles;
        static thread_local std::size_t _queued_plan_count;
        static simaaidispatcher::DispatcherBase* _dispatcher;
        static inline bool _profile = false;
        static inline bool _print_inouts = false;
        static inline bool _save_inouts = false;
        static inline std::string _save_inout_dir = "debug/model_io";
        static inline bool _enable_queue = true;
        static inline bool _disable_parallel_load = false;
};


}
}

#endif
