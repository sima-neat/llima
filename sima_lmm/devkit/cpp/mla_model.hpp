
#ifndef _SIMA_LLIMA_MLA_MODEL_
#define _SIMA_LLIMA_MLA_MODEL_

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "mla_buffer.hpp"
#include "utils.hpp"


namespace simaai {
namespace llima {


class MlaExecutionSession;

/*
 * One LLiMa process owns one ordered /dev/mla context.  The retained argument
 * is a source-compatibility boundary for existing Python callers; dispatcher
 * and MLA-RT command-line options are no longer interpreted.
 */
void connect_mla(const std::vector<std::string>& legacy_args = {});
void disconnect_mla();


class MLAModelWithBuffer {
    friend class MlaExecutionSession;

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
        /*
         * Install a future-submission-only immutable adapter set. Already
         * queued snapshots retain their original BufferViews.
         */
        void update_reloc(
            const std::map<std::string, MLABufferSlice>& reloc_buffers
        );

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
        }

    private:
        void _debug_inouts(const std::string& name, std::map<uint8_t, MLABufferSlice>* fm_map_ptr);
        std::shared_ptr<MlaExecutionSession> _session;
        std::size_t _model_idx = 0;
        std::vector<MLABufferSlice> _ifms;
        std::vector<MLABufferSlice> _ofms;
        static inline bool _profile = false;
        static inline bool _print_inouts = false;
        static inline bool _save_inouts = false;
        static inline std::string _save_inout_dir = "debug/model_io";
};


}
}

#endif
