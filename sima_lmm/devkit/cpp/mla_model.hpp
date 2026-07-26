
#ifndef _SIMA_LLIMA_MLA_MODEL_
#define _SIMA_LLIMA_MLA_MODEL_

#include <cstdint>
#include <filesystem>
#include <functional>
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
 * One caller-owned ordered transaction.  The first model added binds the
 * segment to that model's explicit session and acquires the session execution
 * lock; every later model must belong to the same session.  commit() submits
 * the immutable snapshots with the private queue-ahead window and drains all
 * authoritative completions. The segment intentionally retains the lock
 * after a successful commit so one logical inference transaction can cross a
 * CPU observation/mutation point and then commit another hardware subsegment
 * without allowing a competing producer to interleave its dependent state.
 * Destruction releases the lock; destruction without commit drops only
 * never-submitted snapshots.
 *
 * A small pimpl keeps Backend and SubmissionSnapshot transport details out of
 * LLiMa's installed header.
 */
class MlaExecutionSegment {
    friend class MLAModelWithBuffer;

    public:
        MlaExecutionSegment();
        explicit MlaExecutionSegment(
            std::shared_ptr<MlaExecutionSession> session
        );
        ~MlaExecutionSegment();
        MlaExecutionSegment(MlaExecutionSegment&&) noexcept;
        MlaExecutionSegment& operator=(MlaExecutionSegment&&) noexcept;
        MlaExecutionSegment(const MlaExecutionSegment&) = delete;
        MlaExecutionSegment& operator=(const MlaExecutionSegment&) = delete;

        void commit();
        [[nodiscard]] bool empty() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
        void abort() noexcept;
};

/*
 * One LLiMa process owns one ordered /dev/mla context.  The retained argument
 * is a source-compatibility boundary for existing Python callers; dispatcher
 * and MLA-RT command-line options are no longer interpreted.
 */
void connect_mla(const std::vector<std::string>& legacy_args = {});
void disconnect_mla();
std::shared_ptr<MlaExecutionSession> create_mla_execution_session();
std::shared_ptr<MlaExecutionSession> current_mla_execution_session();
void require_mla_execution_session_healthy(
    const std::shared_ptr<MlaExecutionSession>& session
);


class MLAModelWithBuffer {
    friend class MlaExecutionSession;

    public:
        MLAModelWithBuffer(
            std::filesystem::path model_path,
            std::vector<MLABufferSlice> ifms,
            std::vector<MLABufferSlice> ofms,
            bool zero_hidden_inputs = false
        );
        MLAModelWithBuffer(
            std::shared_ptr<MlaExecutionSession> session,
            std::filesystem::path model_path,
            std::vector<MLABufferSlice> ifms,
            std::vector<MLABufferSlice> ofms,
            bool zero_hidden_inputs = false
        );
        ~MLAModelWithBuffer() {};

        void load();
        void free();
        void run(
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr = nullptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr = nullptr
        );
        void add_to_segment(
            MlaExecutionSegment& segment,
            std::map<uint8_t, MLABufferSlice>* ifm_map_ptr = nullptr,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr = nullptr
        );
        using RelocBuffers = std::map<std::string, MLABufferSlice>;
        struct RelocUpdate {
            MLAModelWithBuffer* model = nullptr;
            RelocBuffers buffers;
        };

        /*
         * Atomically replace/clear the adapter bindings for a related model
         * set. The preparation callback runs while this model's complete
         * execution session is idle, so it may safely upload an inactive
         * adapter bank. Publication occurs only after every hidden input and
         * checked BufferView validates; an exception leaves the old set
         * active. The optional clear callback runs after publication but
         * before another segment may start, so storage can be scrubbed safely.
        */
        void set_reloc_set(
            const std::vector<MLAModelWithBuffer*>& replaced_models,
            const std::function<std::vector<RelocUpdate>()>& prepare
        );
        void clear_reloc_set(
            const std::vector<MLAModelWithBuffer*>& models,
            const std::function<void()>& after_clear = {}
        );

        void load_related_models(
            std::optional<std::filesystem::path> relative_dir = std::nullopt
        );
        void free_related_models(
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
