//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************


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
        struct QueuedRun {
            uint16_t model_idx;
            std::vector<simaaidispatcher::RuntimeBufferBinding> ifm_bindings;
            std::vector<simaaidispatcher::RuntimeBufferBinding> ofm_bindings;
        };

        void _update_buf_addrs(
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        );
        void _debug_inouts(const std::string& name, std::map<uint8_t, MLABufferSlice>* fm_map_ptr);
        simaaidispatcher::JobMLA _make_job(
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        ) const;
        std::vector<simaaidispatcher::RuntimeBufferBinding> _make_bindings(
            const std::vector<MLABufferSlice>& default_slices,
            std::map<uint8_t, MLABufferSlice>* override_map_ptr,
            simaaidispatcher::RuntimeBindingRole role
        ) const;
        static simaaidispatcher::RuntimeBufferBinding _make_binding(
            const std::vector<MLABufferSlice>& default_slices,
            const MLABufferSlice& effective_slice,
            uint8_t logical_idx,
            simaaidispatcher::RuntimeBindingRole role
        );
        static simaaidispatcher::DispatcherBase* _get_dispatcher();

        uint16_t _model_idx;
        std::vector<MLABufferSlice> _ifms;
        std::vector<uint64_t> _ifm_buf_addrs;
        std::vector<MLABufferSlice> _ofms;
        std::vector<uint64_t> _ofm_buf_addrs;

        static std::map<std::filesystem::path, uint16_t> _unique_model_path_to_idx_map;
        static std::vector<std::filesystem::path> _unique_model_paths;
        static std::vector<mla_model_p> _unique_model_ptrs;
        static std::vector<QueuedRun> _queue;
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
