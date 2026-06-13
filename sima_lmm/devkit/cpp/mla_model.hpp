
#ifndef _SIMA_LLIMA_MLA_MODEL_
#define _SIMA_LLIMA_MLA_MODEL_

#include <cstdint>
#include <filesystem>
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


void connect_mla_rt(const std::vector<std::string>& args);
void disconnect_mla_rt();


class MLAModelWithBuffer {
    friend void connect_mla_rt(const std::vector<std::string>& args);
    friend void disconnect_mla_rt();

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
            bool do_parallel_load, std::optional<std::filesystem::path> relative_dir = std::nullopt
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
        }

    private:
        void _debug_inouts(const std::string& name, std::map<uint8_t, MLABufferSlice>* fm_map_ptr);
        simaaidispatcher::PreparedMlaRunRef _prepare_run_ref(
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        );
        static simaaidispatcher::DispatcherBase* _get_dispatcher();

        uint16_t _model_idx;
        std::vector<MLABufferSlice> _ifms;
        std::vector<MLABufferSlice> _ofms;
        simaaidispatcher::PreparedMlaPlan _prepared_plan;

        static std::map<std::filesystem::path, uint16_t> _unique_model_path_to_idx_map;
        static std::vector<std::filesystem::path> _unique_model_paths;
        static std::vector<mla_model_p> _unique_model_ptrs;
        static simaaidispatcher::DispatcherBase::PreparedMlaPartitionQueueRequest _queue_request;
        static simaaidispatcher::DispatcherBase* _dispatcher;
        static inline bool _profile = false;
        static inline bool _print_inouts = false;
        static inline bool _save_inouts = false;
        static inline std::string _save_inout_dir = "debug/model_io";
        static inline bool _enable_queue = true;
};


}
}

#endif
