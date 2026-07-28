// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SiMa.ai

/*
 * The direct session needs the dma-buf export API added by the companion
 * Internals/kernel work.  Include the package-owned namespaced header before
 * any legacy LLiMa header: develop normally finds the older flat
 * <simaai_memory.h> through the SDK sysroot, and both revisions intentionally
 * share one include guard.  If that flat header wins first, the compiler hides
 * simaai_memory_export_dmabuf_fd even though the coherent Internals dev package
 * is installed.  This ordering makes the companion package authoritative
 * without copying another memory-library ABI into LLiMa.
 */
#include <simaai/simaai_memory.h>

#include "mla_model.hpp"

#include <simaai/neat/mla/MlaKernelBackend.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

namespace simaai {
namespace llima {

namespace mla = simaai::neat::mla;

namespace {

/*
 * Modalix DVT's 1/2/3/4/8 sweep selected three as the smallest accepted
 * window that removes CQE-to-refill misses: depth two produced 546 no-READY
 * boundaries in the representative 1,914-job LFM run, while depth three
 * reduced that to the 79 segment/tail boundaries (depth four/eight did not
 * improve it).  Keep this private and re-qualify rather than exposing a knob.
 */
constexpr std::size_t kDefaultQueueAheadDepth = 3;
constexpr std::uint32_t kJobTimeoutMs = 60000;

std::size_t configured_queue_ahead_depth() {
#if defined(SIMA_LLIMA_ENABLE_DVT_QUEUE_DEPTH_OVERRIDE)
    /*
     * This seam exists only in the separately-built DVT qualification
     * runtime.  It is intentionally not part of the LLiMa, Neat, Backend, or
     * kernel API: queue depth is an executor implementation detail and must
     * not become an application compatibility knob.  Qualification sweeps the
     * same production executor at a few bounded depths, then the normal build
     * bakes in the smallest value that hides measured steady-state refill
     * boundaries (segment and sequence tails still end without successors).
     */
    if (const char* value = std::getenv("SIMA_LLIMA_DVT_QUEUE_DEPTH")) {
        char* end = nullptr;
        errno = 0;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (errno != 0 || end == value || *end != '\0' ||
            (parsed != 1 && parsed != 2 && parsed != 3 &&
             parsed != 4 && parsed != 8)) {
            throw std::runtime_error(
                "SIMA_LLIMA_DVT_QUEUE_DEPTH must be one of 1,2,3,4,8"
            );
        }
        return static_cast<std::size_t>(parsed);
    }
#endif
    return kDefaultQueueAheadDepth;
}

std::mutex default_session_mutex;
std::shared_ptr<MlaExecutionSession> default_session;

[[noreturn]] void throw_status(
    std::string_view operation,
    const mla::Status& status
) {
    throw std::runtime_error(fmt::format(
        "{} failed: code={} detail={}",
        operation,
        status.code,
        status.message.empty() ? "none" : status.message
    ));
}

bool path_matches_family(
    const std::filesystem::path& path,
    const std::optional<std::filesystem::path>& selector
) {
    if (!selector.has_value()) {
        return true;
    }
    const auto normalized_path =
        std::filesystem::absolute(path).lexically_normal();
    const auto normalized_selector =
        std::filesystem::absolute(*selector).lexically_normal();

    /*
     * LLiMa develop uses two forms of "related model" selector:
     *
     *   * Whisper passes an actual directory containing its QMLAs.
     *   * Language/vision models pass the common filename stem, for example
     *     ".../elf_files/LFM_language", while the files are siblings named
     *     "LFM_language_n1_...elf".
     *
     * Treating every selector as a directory made the second form match
     * nothing. The direct runtime then loaded a new position bucket lazily in
     * prepare_locked(); at positions 384 and 512 that put roughly 315 ms of
     * model admission directly in the token latency.
     *
     * Preserve exact component containment for real directories. For the
     * filename-stem form, require the same parent and a separator boundary
     * after the complete stem. This admits "text_n1.elf" for "text" without
     * reviving the unsafe raw-prefix behavior where "text2_n1.elf" also
     * matched "text".
     */
    /*
     * This is a component-prefix test, not equality of two complete paths.
     * The four-iterator std::equal overload also requires equal range lengths,
     * so it rejected every ordinary file below a selector directory (for
     * example Whisper's elf_files/model.elf).  Stop when the selector is
     * exhausted instead; path components preserve the boundary guarantee that
     * a raw string-prefix comparison would lose.
     */
    const auto prefix_end = std::mismatch(
        normalized_selector.begin(), normalized_selector.end(),
        normalized_path.begin(), normalized_path.end()
    );
    if (prefix_end.first == normalized_selector.end()) {
        return true;
    }
    if (normalized_selector.parent_path() !=
        normalized_path.parent_path()) {
        return false;
    }

    const std::string stem = normalized_selector.filename().string();
    const std::string candidate = normalized_path.filename().string();
    if (candidate.size() <= stem.size() ||
        candidate.compare(0, stem.size(), stem) != 0) {
        return false;
    }
    const char boundary = candidate[stem.size()];
    return boundary == '_' || boundary == '.';
}

std::shared_ptr<MlaExecutionSession> require_default_session() {
    std::lock_guard lock(default_session_mutex);
    if (!default_session) {
        throw std::runtime_error(
            "MLA execution session is not connected; call connect() first"
        );
    }
    return default_session;
}

}  // namespace

struct __attribute__((visibility("hidden"))) MlaBindingCell {
    struct ParentGeneration {
        const MLABuffer* parent = nullptr;
        std::uint64_t allocation_cookie = 0;
    };

    std::size_t model_index = 0;
    std::vector<MLABufferSlice> ifms;
    std::vector<MLABufferSlice> ofms;
    std::vector<ParentGeneration> parent_generations;
    mla::BoundExecution exact_default;
    std::uint64_t adapter_generation = 0;
};

/*
 * A segment records immutable binding intent and materializes the complete
 * transaction at commit.  Copies of the uncommon override maps preserve the
 * old stack-local caller contract; no pointer to an add_to_segment() local
 * survives.  Backend submission starts only after every entry has validated.
 */
struct __attribute__((visibility("hidden"))) MlaPendingBinding {
    std::shared_ptr<MlaBindingCell> binding;
    std::optional<std::map<uint8_t, MLABufferSlice>> ifm_overrides;
    std::optional<std::map<uint8_t, MLABufferSlice>> ofm_overrides;
};

static std::shared_ptr<MlaBindingCell> as_binding_cell(
    const std::shared_ptr<void>& opaque
) {
    return std::static_pointer_cast<MlaBindingCell>(opaque);
}

/*
 * The complete LLiMa ownership and ordering domain.  There is one Backend,
 * hence one kernel context and one immutable Background priority.  Models,
 * dma-buf registrations, adapter selections and the queue-ahead executor are
 * all context-local; no dispatcher handle or process-global model pointer is
 * observable outside this object.
 */
class MlaExecutionSession {
  private:
    struct BuiltBinding {
        mla::BoundExecution execution;
        std::vector<MlaBindingCell::ParentGeneration> parent_generations;
    };

  public:
    struct SubmissionSnapshot {
        mla::BoundExecution execution;
        std::uint32_t model_index = 0;
    };

    /* Non-owning inputs are consumed synchronously under both session locks. */
    struct AdapterUpdate {
        std::size_t index;
        const std::map<std::string, MLABufferSlice>* adapters;
        std::shared_ptr<MlaBindingCell> binding;
    };

    static std::shared_ptr<MlaExecutionSession> create() {
        mla::Status status;
        auto backend = mla::Backend::open(
            mla::WorkloadPriority::kBackground, &status
        );
        if (!backend || !status) {
            throw_status("Backend::open", status);
        }
        return std::shared_ptr<MlaExecutionSession>(
            new MlaExecutionSession(
                std::move(backend), configured_queue_ahead_depth()
            )
        );
    }

    ~MlaExecutionSession() {
        try {
            free_models(std::nullopt);
        } catch (const std::exception& error) {
            spdlog::error("MLA model cleanup failed: {}", error.what());
        }
        if (backend_) {
            const mla::Status status = backend_->stop();
            if (!status) {
                spdlog::error(
                    "MLA Backend stop failed: {} ({})",
                    status.code, status.message
                );
            }
        }
    }

    std::size_t register_model(
        const std::filesystem::path& model_path,
        bool zero_hidden_inputs
    ) {
        const auto absolute_path =
            std::filesystem::absolute(model_path).lexically_normal();
        if (!std::filesystem::is_regular_file(absolute_path)) {
            throw std::runtime_error(
                fmt::format("Model file does not exist: {}", absolute_path)
            );
        }

        std::lock_guard lock(mutex_);
        require_healthy_locked();
        if (const auto found = path_to_index_.find(absolute_path);
            found != path_to_index_.end()) {
            if (models_[found->second].zero_hidden_inputs !=
                zero_hidden_inputs) {
                throw std::logic_error(fmt::format(
                    "MLA model {} was registered with conflicting hidden-input semantics",
                    absolute_path
                ));
            }
            return found->second;
        }
        const std::size_t index = models_.size();
        path_to_index_.emplace(absolute_path, index);
        models_.push_back(ModelEntry{
            .path = absolute_path,
            .zero_hidden_inputs = zero_hidden_inputs,
        });
        return index;
    }

    void register_binding_cell(
        const std::shared_ptr<MlaBindingCell>& binding
    ) {
        if (!binding) {
            throw std::invalid_argument("null MLA binding cell");
        }
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        validate_index_locked(binding->model_index);
        models_[binding->model_index].bindings.emplace_back(binding);
    }

    void load_model(std::size_t index) {
        std::lock_guard execution_lock(execution_mutex_);
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        validate_index_locked(index);
        if (models_[index].package) {
            return;
        }
        load_group_locked({index});
    }

    void load_models(
        std::optional<std::filesystem::path> relative_directory
    ) {
        std::lock_guard execution_lock(execution_mutex_);
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        std::vector<std::size_t> missing;
        std::size_t matched = 0;
        for (std::size_t index = 0; index < models_.size(); ++index) {
            if (!path_matches_family(
                    models_[index].path, relative_directory)) {
                continue;
            }
            ++matched;
            if (!models_[index].package) {
                missing.push_back(index);
            }
        }
        /*
         * A related-family load is a model-initialization contract, not an
         * optional prefetch hint. Silently accepting an unmatched selector
         * previously deferred every MODEL_DEFINE to the first token that used
         * that QMLA family. Fail at initialization instead: this both catches
         * malformed packages and guarantees that decode cannot acquire a new
         * model merely because it crossed a compiled position bucket.
         */
        if (relative_directory && matched == 0) {
            const std::filesystem::path first_registered =
                models_.empty() ? std::filesystem::path{} : models_.front().path;
            throw std::invalid_argument(fmt::format(
                "no registered MLA model matches family selector {} "
                "(registered_models={} first={})",
                *relative_directory, models_.size(), first_registered
            ));
        }
        if (!missing.empty()) {
            /*
             * One package publication is the all-or-nothing residency
             * transaction. The old "parallel load" switch was a dispatcher
             * implementation detail; one deterministic transaction is both
             * simpler and safer.
             */
            load_group_locked(missing);
        }
    }

    void free_models(
        std::optional<std::filesystem::path> relative_directory
    ) {
        std::lock_guard execution_lock(execution_mutex_);
        std::lock_guard lock(mutex_);

        std::vector<std::shared_ptr<PackageHold>> retained;
        retained.reserve(packages_.size());
        for (auto& package : packages_) {
            bool release = false;
            for (const std::size_t index : package->model_indices) {
                if (path_matches_family(models_[index].path, relative_directory)) {
                    release = true;
                    break;
                }
            }
            if (!release) {
                retained.push_back(package);
                continue;
            }
            /*
             * A package is the transactional ownership unit. If any member
             * matches the requested directory, release every member rather
             * than leave a partially resident package.
             */
            for (const std::size_t index : package->model_indices) {
                /*
                 * BoundExecution intentionally retains ModelState and every
                 * dma-buf registration. Drop all live position-specific
                 * handles before package/import teardown or the new fast path
                 * would turn a scoped family release into permanent DMS0
                 * residency.
                 */
                for (const auto& weak : models_[index].bindings) {
                    if (auto binding = weak.lock()) {
                        binding->exact_default = {};
                        binding->parent_generations.clear();
                        binding->adapter_generation = 0;
                    }
                }
                models_[index].package.reset();
                models_[index].package_ordinal = 0;
                models_[index].active_adapters.clear();
                ++models_[index].adapter_generation;
            }
        }
        packages_ = std::move(retained);
        /*
         * A dma-buf import is session-owned, not package-owned, so the cookie
         * cache cannot cheaply prove which package was its last user. Develop
         * relied on MLA-RT teardown while freeing the associated MLABuffers;
         * retaining these direct Backend handles would instead keep their
         * dma-bufs (and therefore DMS0 allocations) alive until disconnect.
         *
         * execution_mutex_ is held and every segment drains before releasing
         * it, so no accepted job can reference an entry here. Clear the small
         * cache on every scoped model-family release and lazily re-import any
         * buffers still needed by retained families on their next prepare.
         * This is deliberately simpler and safer than maintaining a second
         * per-package reference graph, and it adds no steady-state hot-path
         * work.
         */
        import_cache_.clear();
        import_generation_.clear();
        import_use_epoch_ = 0;
        /*
         * The neutral LoRA allocation is shared by every zero-hidden family
         * in this session, and retained binding cells keep its MLABuffer
         * address as an allocation-generation witness.  Destroying it after a
         * scoped release would leave those witnesses dangling even though
         * their BoundExecutions correctly retain the underlying dma-buf.
         * Keep the tiny owner object while any loaded family can still use it;
         * a complete session/family teardown drops every witness first and may
         * then release the allocation.
         */
        const bool neutral_binding_still_live = std::any_of(
            models_.begin(), models_.end(),
            [](const ModelEntry& entry) {
                return entry.package && entry.zero_hidden_inputs;
            }
        );
        if (!neutral_binding_still_live) {
            zero_hidden_input_.reset();
        }
    }

    std::size_t inspect_public_input_count(
        const std::filesystem::path& model_path,
        bool zero_hidden_inputs
    ) {
        /* register_model takes mutex_ itself; do it before the ordered
         * execution boundary to avoid recursive locking. */
        const std::size_t index =
            register_model(model_path, zero_hidden_inputs);
        std::lock_guard execution_lock(execution_mutex_);
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        validate_index_locked(index);
        if (!models_[index].package) {
            load_group_locked({index});
        }
        const auto& info = models_[index].package->package.info(
            models_[index].package_ordinal
        );
        return static_cast<std::size_t>(std::count_if(
            info.inputs.begin(), info.inputs.end(),
            [](const mla::TensorPortInfo& port) {
                return port.public_port;
            }
        ));
    }

    /*
     * Resolve one complete caller-owned segment under one metadata lock.
     * Develop and the first direct implementation acquired mutex_ once per
     * compiled partition (29 times for the measured LFM token), even though
     * execution_mutex_ already excludes publication changes for the segment.
     * Keeping validation here preserves the same safety boundary while
     * removing repeated lock/unlock and health checks from the serial token
     * path.
     */
    void prepare_many(
        const std::vector<MlaPendingBinding>& pending,
        std::vector<SubmissionSnapshot>& snapshots
    ) {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        snapshots.clear();
        if (snapshots.capacity() < pending.size()) {
            snapshots.reserve(pending.size());
        }
        try {
            for (const MlaPendingBinding& request : pending) {
                if (!request.binding) {
                    throw std::invalid_argument("null MLA binding cell");
                }
                snapshots.push_back(prepare_locked(
                    request.binding,
                    request.ifm_overrides ? &*request.ifm_overrides : nullptr,
                    request.ofm_overrides ? &*request.ofm_overrides : nullptr
                ));
            }
        } catch (...) {
            /*
             * No Backend::submit occurs until this method returns.  Do not
             * leave a valid prefix in reusable scratch after a validation or
             * allocation failure.
             */
            snapshots.clear();
            throw;
        }
    }

    SubmissionSnapshot prepare_locked(
        const std::shared_ptr<MlaBindingCell>& binding,
        const std::map<uint8_t, MLABufferSlice>* ifm_overrides,
        const std::map<uint8_t, MLABufferSlice>* ofm_overrides
    ) {
        if (!binding) {
            throw std::invalid_argument("null MLA binding cell");
        }
        require_healthy_locked();
        validate_index_locked(binding->model_index);
        if (!models_[binding->model_index].package) {
            load_group_locked({binding->model_index});
        }

        ModelEntry& entry = models_[binding->model_index];
        const bool exact_request =
            (!ifm_overrides || ifm_overrides->empty()) &&
            (!ofm_overrides || ofm_overrides->empty());

        if (exact_request &&
            binding->adapter_generation == entry.adapter_generation &&
            parent_generations_match_locked(*binding) &&
            binding->exact_default.valid()) {
            return {
                .execution = binding->exact_default,
                .model_index =
                    static_cast<std::uint32_t>(binding->model_index),
            };
        }

        if (exact_request) {
            BuiltBinding rebuilt = build_binding_locked(
                *binding, nullptr, nullptr, &entry.active_adapters,
                true
            );
            binding->exact_default = std::move(rebuilt.execution);
            binding->parent_generations =
                std::move(rebuilt.parent_generations);
            binding->adapter_generation = entry.adapter_generation;
            if (binding->exact_default.valid()) {
                return {
                    .execution = binding->exact_default,
                    .model_index =
                        static_cast<std::uint32_t>(binding->model_index),
                };
            }
        }

        /*
         * Only genuinely changing physical ports take this path. Ordinary
         * single-token LFM decode has one such embedding input and reuses 28
         * exact handles. There is deliberately no mutable patch API: build a
         * complete checked immutable object through the same Backend trust
         * boundary as every other caller.
         */
        BuiltBinding dynamic = build_binding_locked(
            *binding, ifm_overrides, ofm_overrides,
            &entry.active_adapters, false
        );
        if (!dynamic.execution.valid()) {
            throw std::logic_error(
                "dynamic MLA binding did not produce an execution"
            );
        }
        return {
            .execution = std::move(dynamic.execution),
            .model_index =
                static_cast<std::uint32_t>(binding->model_index),
        };
    }

    std::unique_lock<std::mutex> acquire_execution_lock() {
        return std::unique_lock<std::mutex>(execution_mutex_);
    }

    /*
     * execution_mutex_ permits one live segment per session.  The request and
     * immutable-snapshot arrays can therefore be session-owned scratch rather
     * than one malloc/free pair per generated token.
     */
    void begin_segment() {
        pending_scratch_.clear();
        snapshot_scratch_.clear();
        if (pending_scratch_.capacity() < 64) {
            pending_scratch_.reserve(64);
        }
        if (snapshot_scratch_.capacity() < 64) {
            snapshot_scratch_.reserve(64);
        }
    }

    void end_segment() noexcept {
        pending_scratch_.clear();
        snapshot_scratch_.clear();
    }

    std::vector<MlaPendingBinding>& pending_scratch() noexcept {
        return pending_scratch_;
    }

    std::vector<SubmissionSnapshot>& snapshot_scratch() noexcept {
        return snapshot_scratch_;
    }

    void run_snapshots(std::vector<SubmissionSnapshot>& snapshots) {
        require_healthy();
        if (snapshots.empty()) {
            return;
        }

        /*
         * MLA-RT's useful property was not its userspace scheduler. Its
         * multi-model path consumed completion and launched the next piece of
         * work on one execution thread.  The first direct implementation lost
         * that property: io_uring task-work woke the submitting LLiMa thread,
         * that thread woke the Backend CQ pump, and the pump woke LLiMa again
         * so it could refill this depth-three window. DVT sched tracing measured
         * that second relay at about 30 us per CQE.
         *
         * Keep the same measured depth-three set of independent JOB_EXECs, the
         * same kernel FIFO/priority arbitration, and one CQE per job. Only the
         * refill continuation moves onto the sole CQ consumer. Once the first
         * window completes, replacement SQEs are therefore submitted by the
         * pump itself; io_uring task-work and CQ consumption stay on that
         * thread instead of ping-ponging through the token producer. This is
         * progressive publication, not the rejected all-at-once sequence ABI:
         * at most three ordinary jobs remain accepted, and another context can
         * still win at every physical model boundary.
         */
        {
            std::lock_guard lock(rolling_mutex_);
            if (rolling_snapshots_ || rolling_outstanding_ != 0) {
                throw std::logic_error(
                    "MLA rolling executor still owns a prior segment"
                );
            }
            rolling_snapshots_ = &snapshots;
            rolling_next_ = 0;
            rolling_outstanding_ = 0;
            rolling_available_head_ = 0;
            rolling_available_count_ = 0;
            rolling_initialized_ = false;
            rolling_refill_active_ = false;
            rolling_failure_ = {};
            for (auto& job : job_window_) {
                job.reset();
            }
        }

        /*
         * Publish the initial jobs on the caller. While this small loop is in
         * progress callbacks only retire slots; they cannot refill and race a
         * later initial member into the FIFO. After the barrier, the pump owns
         * all progressive refills.
         */
        for (std::size_t slot = 0;
             slot < job_window_.size() && slot < snapshots.size();
             ++slot) {
            std::size_t snapshot_index = 0;
            {
                std::lock_guard lock(rolling_mutex_);
                if (rolling_failure_.kind != RollingFailureKind::kNone) {
                    break;
                }
                snapshot_index = rolling_next_++;
                ++rolling_outstanding_;
            }
            submit_rolling_slot(slot, snapshot_index);
        }
        {
            std::lock_guard lock(rolling_mutex_);
            rolling_initialized_ = true;
        }
        start_rolling_refill();

        RollingFailure failure;
        std::vector<mla::Job> terminal_jobs;
        {
            std::unique_lock lock(rolling_mutex_);
            rolling_cv_.wait(lock, [&] {
                return rolling_outstanding_ == 0 &&
                       (rolling_failure_.kind !=
                            RollingFailureKind::kNone ||
                        rolling_next_ == snapshots.size());
            });
            failure = rolling_failure_;
            rolling_snapshots_ = nullptr;
            rolling_initialized_ = false;
            rolling_refill_active_ = false;
            rolling_available_head_ = 0;
            rolling_available_count_ = 0;
            terminal_jobs.reserve(job_window_.size());
            for (const auto& job : job_window_) {
                if (job) {
                    terminal_jobs.emplace_back(*job);
                }
            }
        }
        /*
         * rolling_outstanding_ reaches zero inside the last continuation, just
         * before it returns to JobState. Join those final Job wrappers here so
         * the session cannot be destroyed while the CQ thread is still
         * returning through a callback slot owned by this object. This is
         * cold once per logical segment and does not add another kernel call:
         * Job::wait() observes the same already-terminal CQE.
         */
        for (const auto& job : terminal_jobs) {
            (void)job.wait();
        }
        {
            std::lock_guard lock(rolling_mutex_);
            for (auto& job : job_window_) {
                job.reset();
            }
        }

        /*
         * All accepted jobs are terminal here, so no Backend slot needs the
         * snapshot handles any longer. Clear logical contents but retain the
         * vector's allocation for the next subsegment/token. The previous
         * swap-and-reserve implementation performed one malloc/free pair per
         * commit despite the segment object deliberately spanning multiple
         * commits.
         */
        snapshots.clear();
        if (failure.kind != RollingFailureKind::kNone) {
            const auto path = model_path(failure.model_index);
            if (failure.kind == RollingFailureKind::kSubmit) {
                poison(fmt::format(
                    "ordered segment submit failed at {}",
                    failure.snapshot_index
                ));
                throw std::runtime_error(fmt::format(
                    "MLA segment submit {} failed: {}",
                    path, failure.status_code
                ));
            }
            poison("ordered segment terminal failure");
            throw std::runtime_error(fmt::format(
                "MLA ordered segment {} failed: status={} completion={} "
                "fault={}",
                path, failure.status_code, failure.completion_result,
                failure.fault_class
            ));
        }
    }

    /* Caller owns execution_mutex_ across CPU preparation and publication. */
    void replace_adapter_set_locked(
        const std::vector<std::size_t>& replaced_indices,
        const std::vector<AdapterUpdate>& updates
    ) {
        std::lock_guard lock(mutex_);
        require_healthy_locked();

        std::vector<std::size_t> missing;
        std::vector<std::size_t> seen;
        missing.reserve(updates.size());
        seen.reserve(updates.size());
        for (const AdapterUpdate& update : updates) {
            validate_index_locked(update.index);
            if (!update.adapters || !update.binding) {
                throw std::invalid_argument("null MLA adapter transaction input");
            }
            if (std::find(seen.begin(), seen.end(), update.index) != seen.end()) {
                throw std::invalid_argument(
                    "MLA adapter transaction contains a model twice"
                );
            }
            seen.push_back(update.index);
            if (!models_[update.index].package) {
                missing.push_back(update.index);
            }
        }
        if (!missing.empty()) {
            load_group_locked(missing);
        }

        for (const std::size_t index : replaced_indices) {
            validate_index_locked(index);
        }

        /* Validate the complete candidate set before publishing any map. */
        for (const AdapterUpdate& update : updates) {
            ModelEntry& entry = models_[update.index];
            const auto& info =
                entry.package->package.info(entry.package_ordinal);

            for (const auto& [name, slice] : *update.adapters) {
                const auto port = std::find_if(
                    info.inputs.begin(), info.inputs.end(),
                    [&](const mla::TensorPortInfo& candidate) {
                        return !candidate.public_port &&
                               candidate.name == name;
                    }
                );
                if (port == info.inputs.end() || !slice.get_buf_ptr()) {
                    throw std::invalid_argument(fmt::format(
                        "Adapter {} is not a valid hidden input of {}",
                        name, entry.path
                    ));
                }
                const std::uint64_t supplied_extent =
                    slice.get_buf_ptr()->get_buf_len(slice.get_buf_shapes());
                if (port->byte_extent > supplied_extent) {
                    throw std::out_of_range(fmt::format(
                        "Adapter {} supplies {} bytes but {} requires {}",
                        name, supplied_extent, entry.path, port->byte_extent
                    ));
                }
                (void)import_view_locked(
                    &slice, update.binding->ifms, port->physical_index,
                    port->byte_extent, "adapter"
                );
            }

            for (const mla::TensorPortInfo& port : info.inputs) {
                if (port.public_port) {
                    continue;
                }
                const bool has_default =
                    port.physical_index < update.binding->ifms.size() &&
                    update.binding->ifms[port.physical_index].get_buf_ptr();
                if (!has_default && !update.adapters->contains(port.name)) {
                    throw std::invalid_argument(fmt::format(
                        "Adapter set for {} omits hidden input {}",
                        entry.path, port.name
                    ));
                }
            }
        }

        /*
         * Build every future exact template before publishing any adapter
         * map. A late Backend extent/context failure must leave the complete
         * old generation visible rather than switch half of the token
         * positions. Already accepted snapshots retain their old immutable
         * BoundExecution independently of these cells.
         */
        using AdapterMap = std::map<std::string, MLABufferSlice>;
        /*
         * Own every candidate map before touching published state. Copying a
         * map/string can allocate and throw; doing that in the publication
         * loop would expose a mixed adapter generation if allocation N failed.
         * std::map keeps element addresses stable while later candidates are
         * inserted, so the complete handle build below can safely reference
         * these cold-path copies.
         */
        std::map<std::size_t, AdapterMap> candidate_maps;
        for (const std::size_t index : replaced_indices) {
            candidate_maps.try_emplace(index);
        }
        for (const AdapterUpdate& update : updates) {
            candidate_maps[update.index] = *update.adapters;
        }

        struct CandidateBinding {
            std::shared_ptr<MlaBindingCell> binding;
            BuiltBinding built;
            std::uint64_t generation = 0;
        };
        std::vector<CandidateBinding> candidates;
        for (const auto& [index, adapters] : candidate_maps) {
            ModelEntry& entry = models_[index];
            if (!entry.package) {
                continue;
            }
            for (const auto& weak : entry.bindings) {
                if (auto binding = weak.lock()) {
                    candidates.push_back({
                        .binding = binding,
                        .built = build_binding_locked(
                            *binding, nullptr, nullptr, &adapters, true
                        ),
                        .generation = entry.adapter_generation + 1,
                    });
                }
            }
        }

        static_assert(
            std::is_nothrow_move_assignable_v<mla::BoundExecution>
        );
        static_assert(noexcept(
            std::declval<AdapterMap&>().swap(
                std::declval<AdapterMap&>()
            )
        ));
        /*
         * Publication begins only after every allocating copy/import/bind has
         * succeeded. From here through binding-cell publication every
         * operation is no-throw: map/vector swap, shared_ptr move, and integer
         * generation update. Therefore observers see either the complete old
         * set or the complete new set, never a partially updated token.
         */
        for (auto& [index, adapters] : candidate_maps) {
            models_[index].active_adapters.swap(adapters);
            ++models_[index].adapter_generation;
        }
        for (CandidateBinding& candidate : candidates) {
            candidate.binding->exact_default =
                std::move(candidate.built.execution);
            candidate.binding->parent_generations.swap(
                candidate.built.parent_generations
            );
            candidate.binding->adapter_generation =
                candidate.generation;
        }
    }

    /* Caller owns execution_mutex_; clear is one publication boundary. */
    void clear_adapter_set_locked(const std::vector<std::size_t>& indices) {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        std::vector<std::size_t> seen;
        seen.reserve(indices.size());
        for (const std::size_t index : indices) {
            validate_index_locked(index);
            if (std::find(seen.begin(), seen.end(), index) != seen.end()) {
                throw std::invalid_argument(
                    "MLA adapter clear transaction contains a model twice"
                );
            }
            seen.push_back(index);
        }
        const std::map<std::string, MLABufferSlice> empty_adapters;
        struct CandidateBinding {
            std::shared_ptr<MlaBindingCell> binding;
            BuiltBinding built;
            std::uint64_t generation = 0;
        };
        std::vector<CandidateBinding> candidates;
        for (const std::size_t index : indices) {
            ModelEntry& entry = models_[index];
            if (!entry.package) {
                continue;
            }
            for (const auto& weak : entry.bindings) {
                if (auto binding = weak.lock()) {
                    candidates.push_back({
                        .binding = binding,
                        .built = build_binding_locked(
                            *binding, nullptr, nullptr,
                            &empty_adapters, true
                        ),
                        .generation = entry.adapter_generation + 1,
                    });
                }
            }
        }
        for (const std::size_t index : indices) {
            models_[index].active_adapters.clear();
            ++models_[index].adapter_generation;
        }
        for (CandidateBinding& candidate : candidates) {
            candidate.binding->exact_default =
                std::move(candidate.built.execution);
            candidate.binding->parent_generations.swap(
                candidate.built.parent_generations
            );
            candidate.binding->adapter_generation =
                candidate.generation;
        }
    }

    std::filesystem::path model_path(std::size_t index) const {
        std::lock_guard lock(mutex_);
        validate_index_locked(index);
        return models_[index].path;
    }

    /*
     * Public only through the opaque session handle: high-level inference
     * owners call this at every API boundary so the first terminal failure
     * permanently prevents publication of later token/KV state.
     */
    void require_healthy() const {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
    }

  private:
    enum class RollingFailureKind : std::uint8_t {
        kNone,
        kSubmit,
        kTerminal,
    };

    /*
     * Failure publication from the CQ thread must itself be allocation-free:
     * a hardware/capacity failure is exactly when relying on fmt/string/heap
     * construction is least safe. The token producer turns this fixed record
     * into the detailed exception only after every accepted job has drained.
     */
    struct RollingFailure {
        RollingFailureKind kind = RollingFailureKind::kNone;
        std::size_t snapshot_index = 0;
        std::uint32_t model_index = 0;
        int status_code = 0;
        int completion_result = 0;
        std::uint32_t fault_class = 0;
    };

    struct RollingCallbackSlot {
        MlaExecutionSession* owner = nullptr;
        std::size_t slot = 0;
        std::size_t snapshot_index = 0;
    };

    struct PackageHold {
        mla::ModelPackage package;
        std::vector<std::size_t> model_indices;
    };

    struct ModelEntry {
        std::filesystem::path path;
        std::shared_ptr<PackageHold> package;
        std::size_t package_ordinal = 0;
        bool zero_hidden_inputs = false;
        std::map<std::string, MLABufferSlice> active_adapters;
        std::uint64_t adapter_generation = 0;
        std::vector<std::weak_ptr<MlaBindingCell>> bindings;
    };

    explicit MlaExecutionSession(
        std::unique_ptr<mla::Backend> backend,
        std::size_t queue_ahead_depth
    ) : backend_(std::move(backend)),
        queue_ahead_depth_(queue_ahead_depth),
        job_window_(queue_ahead_depth),
        rolling_callback_slots_(queue_ahead_depth),
        rolling_available_slots_(queue_ahead_depth) {
        if (queue_ahead_depth_ == 0) {
            throw std::invalid_argument("MLA queue-ahead depth must be nonzero");
        }
        for (std::size_t slot = 0; slot < queue_ahead_depth_; ++slot) {
            rolling_callback_slots_[slot].owner = this;
            rolling_callback_slots_[slot].slot = slot;
        }
    }

    static void rolling_job_completed(
        void* opaque, const mla::Status& status,
        const mla::JobCompletion& completion) noexcept {
        auto* callback = static_cast<RollingCallbackSlot*>(opaque);
        callback->owner->complete_rolling_job(
            callback->slot, callback->snapshot_index, status, completion
        );
    }

    bool rolling_done_locked() const noexcept {
        return rolling_initialized_ && rolling_outstanding_ == 0 &&
               rolling_snapshots_ &&
               (rolling_failure_.kind != RollingFailureKind::kNone ||
                rolling_next_ == rolling_snapshots_->size());
    }

    void record_rolling_failure_locked(
        RollingFailureKind kind, std::size_t snapshot_index,
        int status_code, int completion_result,
        std::uint32_t fault_class) noexcept {
        if (rolling_failure_.kind != RollingFailureKind::kNone) {
            return;
        }
        const std::uint32_t model_index =
            rolling_snapshots_ &&
                    snapshot_index < rolling_snapshots_->size()
                ? (*rolling_snapshots_)[snapshot_index].model_index
                : 0;
        rolling_failure_ = {
            .kind = kind,
            .snapshot_index = snapshot_index,
            .model_index = model_index,
            .status_code = status_code,
            .completion_result = completion_result,
            .fault_class = fault_class,
        };
    }

    void fail_rolling_submit(
        std::size_t slot, std::size_t snapshot_index,
        int status_code) noexcept {
        bool notify = false;
        {
            std::lock_guard lock(rolling_mutex_);
            /*
             * The reservation is counted before Backend::submit so completion
             * and the main drain predicate cannot observe a false zero.
             */
            if (rolling_outstanding_ == 0 || job_window_[slot]) {
                std::terminate();
            }
            --rolling_outstanding_;
            record_rolling_failure_locked(
                RollingFailureKind::kSubmit, snapshot_index,
                status_code, status_code, 0
            );
            notify = rolling_done_locked();
        }
        if (notify) {
            rolling_cv_.notify_all();
        }
    }

    void submit_rolling_slot(
        std::size_t slot, std::size_t snapshot_index) noexcept {
        {
            std::lock_guard lock(rolling_mutex_);
            /*
             * The availability queue is populated only by the prior
             * continuation for this slot. Drop its wrapper before attempting
             * replacement admission; Backend::finish() still owns that prior
             * JobState until the current callback returns.
             */
            job_window_[slot].reset();
        }
        mla::Job job;
        mla::Status status;
        try {
            mla::SubmitOptions options;
            options.timeout_ms = kJobTimeoutMs;
            status = backend_->submit(
                (*rolling_snapshots_)[snapshot_index].execution,
                options, &job
            );
        } catch (...) {
            /*
             * Backend submission is specified as a Status-returning boundary,
             * but convert an unexpected allocation/standard-library exception
             * too. Let the already accepted prefix drain; never unwind through
             * the completion pump.
             */
            fail_rolling_submit(slot, snapshot_index, -ENOMEM);
            return;
        }
        if (!status || !job.valid()) {
            fail_rolling_submit(
                slot, snapshot_index, status ? -EPROTO : status.code
            );
            return;
        }

        RollingCallbackSlot* callback = nullptr;
        mla::Job callback_job;
        {
            std::lock_guard lock(rolling_mutex_);
            if (job_window_[slot]) {
                std::terminate();
            }
            /*
             * A slot becomes reusable only from its prior continuation. The
             * previous wrapper was cleared before admission while JobState was
             * still owned by Backend::finish(); an occupied slot here means
             * two producers violated the executor's single-refill invariant.
             */
            job_window_[slot].emplace(std::move(job));
            /*
             * Copy the shared Job wrapper so registration can happen after
             * dropping rolling_mutex_. A very short job may already be
             * terminal, in which case registration invokes the continuation
             * inline and that continuation needs the same mutex.
             */
            callback_job = *job_window_[slot];
            callback = &rolling_callback_slots_[slot];
            callback->snapshot_index = snapshot_index;
        }
        /*
         * The Job is fresh and has exactly one owner-side registration site.
         * A failure here would mean a local executor invariant violation, not
         * a recoverable device condition. Continuing without the callback
         * would strand an accepted job and make buffer ownership ambiguous.
         */
        const mla::Status armed =
            callback_job.set_completion_callback(
                callback, rolling_job_completed
            );
        if (!armed) {
            std::terminate();
        }
    }

    void start_rolling_refill() noexcept {
        {
            std::lock_guard lock(rolling_mutex_);
            if (!rolling_initialized_ || rolling_refill_active_) {
                return;
            }
            rolling_refill_active_ = true;
        }

        while (true) {
            std::size_t slot = 0;
            std::size_t snapshot_index = 0;
            bool have_work = false;
            bool notify = false;
            {
                std::lock_guard lock(rolling_mutex_);
                if (rolling_failure_.kind != RollingFailureKind::kNone ||
                    !rolling_snapshots_ ||
                    rolling_next_ == rolling_snapshots_->size() ||
                    rolling_available_count_ == 0) {
                    rolling_refill_active_ = false;
                    notify = rolling_done_locked();
                } else {
                    slot = rolling_available_slots_[
                        rolling_available_head_
                    ];
                    rolling_available_head_ =
                        (rolling_available_head_ + 1) %
                        rolling_available_slots_.size();
                    --rolling_available_count_;
                    snapshot_index = rolling_next_++;
                    ++rolling_outstanding_;
                    have_work = true;
                }
            }
            if (notify) {
                rolling_cv_.notify_all();
            }
            if (!have_work) {
                return;
            }
            submit_rolling_slot(slot, snapshot_index);
        }
    }

    void complete_rolling_job(
        std::size_t slot, std::size_t snapshot_index,
        const mla::Status& status,
        const mla::JobCompletion& completion) noexcept {
        bool refill = false;
        bool notify = false;
        {
            std::lock_guard lock(rolling_mutex_);
            if (!rolling_snapshots_ || slot >= job_window_.size() ||
                !job_window_[slot] ||
                rolling_callback_slots_[slot].snapshot_index !=
                    snapshot_index ||
                rolling_outstanding_ == 0 ||
                rolling_available_count_ >=
                    rolling_available_slots_.size()) {
                std::terminate();
            }
            --rolling_outstanding_;
            rolling_available_slots_[
                (rolling_available_head_ + rolling_available_count_) %
                rolling_available_slots_.size()
            ] = slot;
            ++rolling_available_count_;
            if ((!status || completion.result != 0) &&
                rolling_failure_.kind == RollingFailureKind::kNone) {
                record_rolling_failure_locked(
                    RollingFailureKind::kTerminal, snapshot_index,
                    status.code, completion.result,
                    completion.fault_class
                );
            }
            refill = rolling_initialized_ && !rolling_refill_active_;
            notify = !refill && rolling_done_locked();
        }
        if (refill) {
            start_rolling_refill();
        } else if (notify) {
            rolling_cv_.notify_all();
        }
    }

    void validate_index_locked(std::size_t index) const {
        if (index >= models_.size()) {
            throw std::out_of_range("invalid LLiMa MLA model index");
        }
    }

    void ensure_zero_hidden_input_locked(std::uint64_t required_extent) {
        if (required_extent == 0 ||
            required_extent > std::numeric_limits<std::size_t>::max()) {
            throw std::out_of_range(
                "invalid MLA neutral hidden-input extent"
            );
        }
        if (zero_hidden_input_ &&
            zero_hidden_input_->get_allocation_size() >= required_extent) {
            return;
        }

        /*
         * Callers own execution_mutex_ whenever model preparation can reach
         * this function. Consequently no accepted job can still reference
         * the old generation while a larger neutral buffer is installed.
         * Remove its cached import before freeing the userspace allocation;
         * this also keeps repeated package lifecycles at a flat DMS0 usage.
         */
        if (zero_hidden_input_) {
            erase_import_locked(zero_hidden_input_->get_allocation_cookie());
        }
        auto replacement = std::make_unique<MLABuffer>(
            "__base_zero_hidden_input",
            std::vector<std::size_t>{
                static_cast<std::size_t>(required_extent)
            },
            "int8",
            false
        );
        replacement->allocate();
        replacement->clear();
        zero_hidden_input_ = std::move(replacement);
    }

    void require_healthy_locked() const {
        if (poisoned_) {
            throw std::runtime_error(fmt::format(
                "MLA execution session is poisoned and must be reconstructed: {}",
                poison_reason_.empty() ? "unknown terminal failure"
                                       : poison_reason_
            ));
        }
    }

    void poison_locked(std::string reason) {
        if (!poisoned_) {
            poisoned_ = true;
            poison_reason_ = std::move(reason);
        }
    }

    void poison(std::string reason) {
        {
            std::lock_guard lock(mutex_);
            poison_locked(std::move(reason));
        }
        /*
         * A dependent LLM segment may already have mutated KV state. Stop the
         * entire context instead of racing per-job cancellation and pretending
         * the token state is recoverable.
         */
        (void)backend_->stop();
    }

    void load_group_locked(const std::vector<std::size_t>& indices) {
        std::vector<std::string> paths;
        paths.reserve(indices.size());
        for (const std::size_t index : indices) {
            validate_index_locked(index);
            paths.push_back(models_[index].path.string());
        }

        mla::ModelPackage package;
        const mla::Status status =
            backend_->load_package(paths, &package);
        if (!status) {
            throw_status("Backend::load_package", status);
        }
        if (!package.valid() || package.size() != indices.size()) {
            poison_locked("package publication returned inconsistent size");
            throw std::runtime_error(
                "MLA package publication returned inconsistent size"
            );
        }

        auto hold = std::make_shared<PackageHold>();
        hold->package = std::move(package);
        hold->model_indices = indices;
        for (std::size_t ordinal = 0; ordinal < indices.size(); ++ordinal) {
            ModelEntry& entry = models_[indices[ordinal]];
            entry.package = hold;
            entry.package_ordinal = ordinal;
            spdlog::info(
                "Loaded MLA model {} digest-package={}",
                entry.path, hold->package.identity()
            );
        }
        packages_.push_back(std::move(hold));
        /*
         * Every position-specific MLAModelWithBuffer already exists before
         * LanguageModel publishes its related package. Materialize all exact
         * bindings here so decode never pays port traversal/import/validation
         * for the 28 static jobs in an ordinary token. A public input with no
         * default parent is intentionally left unbound and uses the one
         * checked dynamic path when its caller supplies an override.
         */
        try {
            prebind_all_locked();
        } catch (...) {
            poison_locked("eager MLA binding materialization failed");
            throw;
        }
    }

    bool parent_generations_match_locked(
        const MlaBindingCell& binding
    ) const {
        if (binding.parent_generations.empty()) {
            return false;
        }
        return std::all_of(
            binding.parent_generations.begin(),
            binding.parent_generations.end(),
            [](const MlaBindingCell::ParentGeneration& generation) {
                return generation.parent &&
                       generation.allocation_cookie != 0 &&
                       generation.parent->get_allocation_cookie() ==
                           generation.allocation_cookie;
            }
        );
    }

    static MLABuffer* resolve_parent(
        const MLABufferSlice* selected,
        const std::vector<MLABufferSlice>& defaults,
        std::size_t port_index
    ) {
        if (!selected) {
            return nullptr;
        }
        MLABuffer* parent = selected->get_buf_ptr();
        if (!parent && port_index < defaults.size()) {
            parent = defaults[port_index].get_buf_ptr();
        }
        return parent;
    }

    static void record_parent_generation(
        MLABuffer* parent,
        std::vector<MlaBindingCell::ParentGeneration>* generations
    ) {
        if (!parent || !generations) {
            return;
        }
        if (std::find_if(
                generations->begin(), generations->end(),
                [parent](
                    const MlaBindingCell::ParentGeneration& candidate
                ) {
                    return candidate.parent == parent;
                }
            ) != generations->end()) {
            return;
        }
        generations->push_back({
            .parent = parent,
            .allocation_cookie = parent->get_allocation_cookie(),
        });
    }

    BuiltBinding build_binding_locked(
        const MlaBindingCell& binding,
        const std::map<uint8_t, MLABufferSlice>* ifm_overrides,
        const std::map<uint8_t, MLABufferSlice>* ofm_overrides,
        const std::map<std::string, MLABufferSlice>* adapters,
        bool allow_incomplete
    ) {
        validate_index_locked(binding.model_index);
        ModelEntry& entry = models_[binding.model_index];
        if (!entry.package) {
            throw std::logic_error("cannot bind an unloaded MLA model");
        }

        mla::Model model =
            entry.package->package.model(entry.package_ordinal);
        if (!model.valid()) {
            poison_locked("package returned an invalid model handle");
            throw std::runtime_error(
                "package returned an invalid model handle"
            );
        }
        const auto& info =
            entry.package->package.info(entry.package_ordinal);
        if (binding.ifms.size() > info.inputs.size()) {
            /*
             * Compiler physical order is the sole authority. Develop sized
             * Dispatcher vectors from LLiMa defaults, which could silently
             * shift later ports when a package changed cardinality.
             */
            throw std::runtime_error(fmt::format(
                "Model {} exposes {} physical inputs but LLiMa defines {}",
                entry.path, info.inputs.size(), binding.ifms.size()
            ));
        }
        if (info.outputs.size() != binding.ofms.size()) {
            throw std::runtime_error(fmt::format(
                "Model {} requires {} outputs but LLiMa defines {}",
                entry.path, info.outputs.size(), binding.ofms.size()
            ));
        }

        BuiltBinding built;
        std::vector<mla::BufferView> inputs;
        std::vector<mla::BufferView> outputs;
        inputs.reserve(info.inputs.size());
        outputs.reserve(info.outputs.size());
        std::optional<MLABufferSlice> zero_hidden_slice;

        for (std::size_t port_index = 0;
             port_index < info.inputs.size(); ++port_index) {
            const mla::TensorPortInfo& port = info.inputs[port_index];
            const MLABufferSlice* selected = select_slice(
                port_index, binding.ifms, ifm_overrides
            );
            if ((!selected || !resolve_parent(
                    selected, binding.ifms, port_index)) &&
                !port.public_port && adapters) {
                const auto adapter = adapters->find(port.name);
                if (adapter != adapters->end()) {
                    selected = &adapter->second;
                }
            }
            if ((!selected || !resolve_parent(
                    selected, binding.ifms, port_index)) &&
                !port.public_port && entry.zero_hidden_inputs) {
                /*
                 * Hidden LoRA ports require a real neutral binding. The
                 * prebind pass sizes this shared allocation before creating
                 * any handle so a later larger port cannot invalidate earlier
                 * exact templates.
                 */
                ensure_zero_hidden_input_locked(port.byte_extent);
                zero_hidden_slice.emplace(zero_hidden_input_.get());
                selected = &zero_hidden_slice.value();
            }

            MLABuffer* parent =
                resolve_parent(selected, binding.ifms, port_index);
            if (!parent) {
                if (allow_incomplete && port.public_port) {
                    return {};
                }
                throw std::invalid_argument(fmt::format(
                    "MLA input {} has no allocated parent buffer",
                    port_index
                ));
            }
            inputs.push_back(import_view_locked(
                selected, binding.ifms, port_index,
                port.byte_extent, "input"
            ));
            record_parent_generation(
                parent, &built.parent_generations
            );
        }

        for (std::size_t port_index = 0;
             port_index < info.outputs.size(); ++port_index) {
            const MLABufferSlice* selected = select_slice(
                port_index, binding.ofms, ofm_overrides
            );
            MLABuffer* parent =
                resolve_parent(selected, binding.ofms, port_index);
            if (!parent) {
                if (allow_incomplete) {
                    return {};
                }
                throw std::invalid_argument(fmt::format(
                    "MLA output {} has no allocated parent buffer",
                    port_index
                ));
            }
            outputs.push_back(import_view_locked(
                selected, binding.ofms, port_index,
                info.outputs[port_index].byte_extent, "output"
            ));
            record_parent_generation(
                parent, &built.parent_generations
            );
        }

        mla::Status status =
            backend_->bind(model, inputs, outputs, &built.execution);
        if (!status) {
            throw_status("Backend::bind", status);
        }
        return built;
    }

    void prebind_all_locked() {
        std::uint64_t largest_zero_extent = 0;
        for (const ModelEntry& entry : models_) {
            if (!entry.package || !entry.zero_hidden_inputs) {
                continue;
            }
            const auto& info =
                entry.package->package.info(entry.package_ordinal);
            for (const mla::TensorPortInfo& port : info.inputs) {
                if (!port.public_port) {
                    largest_zero_extent = std::max(
                        largest_zero_extent, port.byte_extent
                    );
                }
            }
        }
        if (largest_zero_extent != 0) {
            ensure_zero_hidden_input_locked(largest_zero_extent);
        }

        for (ModelEntry& entry : models_) {
            if (!entry.package) {
                continue;
            }
            std::vector<std::weak_ptr<MlaBindingCell>> live;
            live.reserve(entry.bindings.size());
            for (const auto& weak : entry.bindings) {
                auto binding = weak.lock();
                if (!binding) {
                    continue;
                }
                live.emplace_back(binding);
                BuiltBinding built = build_binding_locked(
                    *binding, nullptr, nullptr,
                    &entry.active_adapters, true
                );
                binding->exact_default = std::move(built.execution);
                binding->parent_generations =
                    std::move(built.parent_generations);
                binding->adapter_generation =
                    entry.adapter_generation;
            }
            entry.bindings = std::move(live);
        }
    }

    static const MLABufferSlice* select_slice(
        std::size_t index,
        const std::vector<MLABufferSlice>& defaults,
        const std::map<uint8_t, MLABufferSlice>* overrides
    ) {
        if (index >= defaults.size()) {
            return nullptr;
        }
        if (overrides && index <= std::numeric_limits<uint8_t>::max()) {
            const auto found =
                overrides->find(static_cast<uint8_t>(index));
            if (found != overrides->end()) {
                return &found->second;
            }
        }
        return &defaults[index];
    }

    mla::BufferView import_view_locked(
        const MLABufferSlice* selected,
        const std::vector<MLABufferSlice>& defaults,
        std::size_t port_index,
        std::uint64_t compiler_extent,
        std::string_view direction
    ) {
        if (!selected) {
            throw std::invalid_argument(fmt::format(
                "MLA {} {} has no buffer slice", direction, port_index
            ));
        }
        MLABuffer* parent = selected->get_buf_ptr();
        if (!parent && port_index < defaults.size()) {
            parent = defaults[port_index].get_buf_ptr();
        }
        if (!parent || !parent->get_simaai_memory() ||
            parent->get_allocation_cookie() == 0) {
            throw std::invalid_argument(fmt::format(
                "MLA {} {} has no allocated parent buffer",
                direction, port_index
            ));
        }

        const std::uint64_t offset =
            parent->get_buf_addr_offset(selected->get_buf_begins());
        const std::uint64_t declared_extent =
            parent->get_buf_len(selected->get_buf_shapes());
        if (compiler_extent > declared_extent) {
            /*
             * A slice shape is an ownership boundary, not debug metadata.
             * Develop passed this exact get_buf_len(shape) bound to Dispatcher;
             * checking only the larger parent allocation in the direct path
             * would let a short override expose adjacent rows to JOB_EXEC.
             */
            throw std::out_of_range(fmt::format(
                "MLA {} {} compiler extent {} exceeds declared slice extent {}",
                direction, port_index, compiler_extent, declared_extent
            ));
        }
        if (offset > parent->get_allocation_size() ||
            compiler_extent >
                parent->get_allocation_size() - offset) {
            throw std::out_of_range(fmt::format(
                "MLA {} {} compiler extent {} at offset {} exceeds {}",
                direction, port_index, compiler_extent, offset,
                parent->get_allocation_size()
            ));
        }

        const std::uint64_t cookie = parent->get_allocation_cookie();
        /*
         * MLABuffer::free()/allocate() gives the same logical buffer a new
         * cookie. Develop discarded the old MLA-RT binding while patching the
         * next run; a strong direct-cache entry would instead keep the prior
         * dma-buf (and DMS0 allocation) alive forever. Replace that parent's
         * previous generation before importing the new one.
         */
        if (const auto generation = import_generation_.find(parent);
            generation != import_generation_.end() &&
            generation->second != cookie) {
            erase_import_locked(generation->second);
        }

        auto imported = import_cache_.find(cookie);
        if (imported == import_cache_.end()) {
            const int fd = simaai_memory_export_dmabuf_fd(
                parent->get_simaai_memory(), 0
            );
            if (fd < 0) {
                throw std::runtime_error(fmt::format(
                    "simaai_memory_export_dmabuf_fd failed: {}",
                    std::strerror(errno)
                ));
            }
            mla::Buffer buffer;
            const mla::Status status =
                backend_->import_dmabuf(fd, &buffer);
            const int saved_errno = errno;
            (void)::close(fd);
            if (!status) {
                throw std::runtime_error(fmt::format(
                    "Backend::import_dmabuf failed: {} ({}) errno={}",
                    status.code, status.message, saved_errno
                ));
            }
            ImportEntry entry;
            entry.buffer = std::move(buffer);
            entry.parent = parent;
            entry.last_use = ++import_use_epoch_;
            imported = import_cache_.emplace(cookie, std::move(entry)).first;
            import_generation_[parent] = cookie;
            trim_import_cache_locked();
            /* trim_import_cache_locked() never selects the just-touched newest
             * entry, but re-find it rather than retaining an iterator across a
             * future unordered_map implementation change. */
            imported = import_cache_.find(cookie);
        } else {
            imported->second.last_use = ++import_use_epoch_;
            import_generation_[parent] = cookie;
        }

        mla::BufferView view =
            imported->second.buffer.view(offset, compiler_extent);
        if (!view.valid()) {
            throw std::out_of_range(fmt::format(
                "Backend rejected MLA {} {} BufferView",
                direction, port_index
            ));
        }
        return view;
    }

    struct ImportEntry {
        mla::Buffer buffer;
        const MLABuffer* parent = nullptr;
        std::uint64_t last_use = 0;
    };

    void erase_import_locked(std::uint64_t cookie) {
        const auto found = import_cache_.find(cookie);
        if (found == import_cache_.end()) {
            return;
        }
        const auto generation = import_generation_.find(found->second.parent);
        if (generation != import_generation_.end() &&
            generation->second == cookie) {
            import_generation_.erase(generation);
        }
        import_cache_.erase(found);
    }

    void trim_import_cache_locked() {
        /*
         * Generation replacement handles the normal free/allocate cycle
         * exactly. This small LRU is the final bound for callers that create
         * and destroy ever-new MLABuffer objects without freeing a model
         * family. Accepted SubmissionSnapshots retain their BufferViews, so an
         * eviction cannot invalidate queued work; a later prepare simply
         * reimports. 256 keeps normal LLM working sets resident while making
         * pathological DMS0 retention finite. The scan is cold-path only and
         * bounded, never part of CQE refill or kernel scheduling.
         */
        static constexpr std::size_t kMaxRetainedImports = 256;
        while (import_cache_.size() > kMaxRetainedImports) {
            const auto oldest = std::min_element(
                import_cache_.begin(), import_cache_.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second.last_use < rhs.second.last_use;
                }
            );
            erase_import_locked(oldest->first);
        }
    }

    std::unique_ptr<mla::Backend> backend_;
    const std::size_t queue_ahead_depth_;
    std::vector<std::optional<mla::Job>> job_window_;
    /*
     * Allocation-free rolling-executor state. All fields below are protected
     * by rolling_mutex_. The execution lock guarantees one logical segment,
     * while this smaller lock coordinates its caller with the Backend CQ
     * consumer. Fixed-size vectors are allocated once with the session; the
     * callback path performs no container growth.
     */
    std::mutex rolling_mutex_;
    std::condition_variable rolling_cv_;
    std::vector<RollingCallbackSlot> rolling_callback_slots_;
    std::vector<std::size_t> rolling_available_slots_;
    std::vector<SubmissionSnapshot>* rolling_snapshots_ = nullptr;
    std::size_t rolling_next_ = 0;
    std::size_t rolling_outstanding_ = 0;
    std::size_t rolling_available_head_ = 0;
    std::size_t rolling_available_count_ = 0;
    bool rolling_initialized_ = false;
    bool rolling_refill_active_ = false;
    RollingFailure rolling_failure_;
    std::vector<MlaPendingBinding> pending_scratch_;
    std::vector<SubmissionSnapshot> snapshot_scratch_;
    /*
     * Serializes the entire logical execution transaction, not merely access
     * to the model registry.  LLM layers mutate ordered KV/speculative state;
     * permitting two producers to interleave their segment commits on the
     * same FIFO context would be memory-safe but semantically corrupt.
     */
    std::mutex execution_mutex_;
    mutable std::mutex mutex_;
    std::map<std::filesystem::path, std::size_t> path_to_index_;
    std::vector<ModelEntry> models_;
    std::vector<std::shared_ptr<PackageHold>> packages_;
    std::unordered_map<std::uint64_t, ImportEntry> import_cache_;
    std::unordered_map<const MLABuffer*, std::uint64_t> import_generation_;
    std::uint64_t import_use_epoch_ = 0;
    /*
     * One immutable neutral LoRA binding shared by all hidden port views in
     * this context. It is absent for non-LoRA models, preserving fail-closed
     * behavior for an unexpected hidden input with no explicit semantics.
     */
    std::unique_ptr<MLABuffer> zero_hidden_input_;
    bool poisoned_ = false;
    std::string poison_reason_;
};

struct MlaExecutionSegment::Impl {
    explicit Impl(std::shared_ptr<MlaExecutionSession> value)
        : session(std::move(value)),
          execution_lock(session
              ? session->acquire_execution_lock()
              : std::unique_lock<std::mutex>{}) {
        if (!session) {
            throw std::invalid_argument("MLA execution segment has no session");
        }
        session->begin_segment();
        pending = &session->pending_scratch();
        snapshots = &session->snapshot_scratch();
    }

    ~Impl() {
        /*
         * execution_lock is still owned while reusable handles and override
         * maps are dropped.  This covers both a successful drain and the
         * transactional abort path.
         */
        if (session) {
            session->end_segment();
        }
    }

    std::shared_ptr<MlaExecutionSession> session;
    std::unique_lock<std::mutex> execution_lock;
    std::vector<MlaPendingBinding>* pending = nullptr;
    std::vector<MlaExecutionSession::SubmissionSnapshot>* snapshots = nullptr;
};

MlaExecutionSegment::MlaExecutionSegment() = default;
MlaExecutionSegment::MlaExecutionSegment(
    std::shared_ptr<MlaExecutionSession> session
) : _impl(std::make_unique<Impl>(std::move(session))) {}
MlaExecutionSegment::~MlaExecutionSegment() = default;
MlaExecutionSegment::MlaExecutionSegment(MlaExecutionSegment&&) noexcept = default;
MlaExecutionSegment&
MlaExecutionSegment::operator=(MlaExecutionSegment&&) noexcept = default;

bool MlaExecutionSegment::empty() const noexcept {
    return !_impl || !_impl->pending || _impl->pending->empty();
}

void MlaExecutionSegment::abort() noexcept {
    /*
     * No snapshot in this object has reached Backend::submit until commit().
     * Destroying the pimpl is therefore a complete transactional rollback and
     * releases the per-session execution lock without cancellation syscalls.
     */
    _impl.reset();
}

void MlaExecutionSegment::commit() {
    if (!_impl) {
        return;
    }
    if (!_impl->pending || _impl->pending->empty()) {
        return;
    }
    /*
     * Keep the execution lock after a successful drain. Callers such as the
     * LLM and Eagle executors have intentional CPU observation points between
     * hardware subsegments; retaining the same caller-owned transaction there
     * prevents another producer from interleaving KV/speculative state. On a
     * failure abort() releases the lock only after run_snapshots() has drained
     * every authoritative terminal CQE and poisoned the session.
     */
    try {
        _impl->session->prepare_many(
            *_impl->pending, *_impl->snapshots
        );
        /*
         * Overrides and binding intents are no longer needed once immutable
         * Backend handles exist. Retain capacity for the next subsegment.
         */
        _impl->pending->clear();
        _impl->session->run_snapshots(*_impl->snapshots);
    } catch (...) {
        abort();
        throw;
    }
}

void connect_mla(const std::vector<std::string>& legacy_args) {
    std::lock_guard lock(default_session_mutex);
    if (default_session) {
        return;
    }
    if (!legacy_args.empty()) {
        spdlog::warn(
            "MLA-RT arguments are deprecated and ignored by the direct "
            "kernel backend: [{}]",
            fmt::join(legacy_args, ", ")
        );
    }
    default_session = create_mla_execution_session();
    spdlog::info(
        "Connected LLiMa directly to /dev/mla with Background priority"
    );
}

std::shared_ptr<MlaExecutionSession> create_mla_execution_session() {
    return MlaExecutionSession::create();
}

std::shared_ptr<MlaExecutionSession> current_mla_execution_session() {
    return require_default_session();
}

void require_mla_execution_session_healthy(
    const std::shared_ptr<MlaExecutionSession>& session
) {
    if (!session) {
        throw std::runtime_error("MLA execution session is not available");
    }
    session->require_healthy();
}

void disconnect_mla() {
    std::shared_ptr<MlaExecutionSession> session;
    {
        std::lock_guard lock(default_session_mutex);
        session = std::move(default_session);
    }
    /*
     * Session destruction is the single no-throw shutdown path.  Calling
     * free_models() here used to throw when model preparation failed after
     * one or more snapshots had been queued, masking the useful root cause
     * and, during exception unwinding, terminating the process.  The session
     * destructor discards only never-submitted snapshots, releases packages,
     * and then stops the kernel context in that order.
     */
    session.reset();
}

MLAModelWithBuffer::MLAModelWithBuffer(
    std::filesystem::path model_path,
    std::vector<MLABufferSlice> ifms,
    std::vector<MLABufferSlice> ofms,
    bool zero_hidden_inputs
) : MLAModelWithBuffer(
        current_mla_execution_session(), std::move(model_path),
        std::move(ifms), std::move(ofms), zero_hidden_inputs
    ) {}

MLAModelWithBuffer::MLAModelWithBuffer(
    std::shared_ptr<MlaExecutionSession> session,
    std::filesystem::path model_path,
    std::vector<MLABufferSlice> ifms,
    std::vector<MLABufferSlice> ofms,
    bool zero_hidden_inputs
) : _session(std::move(session)) {
    if (!_session) {
        throw std::invalid_argument("MLAModelWithBuffer requires a session");
    }
    const std::size_t model_index =
        _session->register_model(model_path, zero_hidden_inputs);
    auto binding = std::make_shared<MlaBindingCell>();
    binding->model_index = model_index;
    binding->ifms = std::move(ifms);
    binding->ofms = std::move(ofms);
    _binding = binding;
    _session->register_binding_cell(binding);
}

MLAModelWithBuffer::~MLAModelWithBuffer() = default;

void MLAModelWithBuffer::load() {
    _session->load_model(as_binding_cell(_binding)->model_index);
}

void MLAModelWithBuffer::free() {
    /*
     * ModelPackage is the ownership unit, so selective object destruction does
     * not tear down a shared package behind other model objects. Explicit
     * family/session free calls release package groups deterministically.
     */
}

void MLAModelWithBuffer::run(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    _debug_inouts("ifm", ifm_map_ptr);
    MlaExecutionSegment segment(_session);
    add_to_segment(segment, ifm_map_ptr, ofm_map_ptr);
    segment.commit();
    _debug_inouts("ofm", ofm_map_ptr);
}

void MLAModelWithBuffer::add_to_segment(
    MlaExecutionSegment& segment,
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    try {
        if (!segment._impl) {
            segment._impl = std::make_unique<MlaExecutionSegment::Impl>(_session);
        } else if (segment._impl->session != _session) {
            throw std::invalid_argument(
                "one MLA execution segment cannot mix kernel contexts"
            );
        }
        MlaPendingBinding pending{
            .binding = as_binding_cell(_binding),
        };
        if (ifm_map_ptr && !ifm_map_ptr->empty()) {
            pending.ifm_overrides = *ifm_map_ptr;
        }
        if (ofm_map_ptr && !ofm_map_ptr->empty()) {
            pending.ofm_overrides = *ofm_map_ptr;
        }
        segment._impl->pending->push_back(std::move(pending));
    } catch (...) {
        /*
         * Segment construction is transactional.  If preparing this model
         * intent fails, none of the earlier entries in the same caller-owned
         * segment may run later by accident.  Validation/materialization at
         * commit has the same rule and happens before the first submit.
         */
        segment.abort();
        throw;
    }
}

void MLAModelWithBuffer::_add_embedding_row_to_segment(
    MlaExecutionSegment& segment,
    uint8_t port,
    MLABuffer* parent,
    uint32_t row,
    uint32_t width
) {
    try {
        if (!segment._impl) {
            segment._impl =
                std::make_unique<MlaExecutionSegment::Impl>(_session);
        } else if (segment._impl->session != _session) {
            throw std::invalid_argument(
                "one MLA execution segment cannot mix kernel contexts"
            );
        }
        MlaPendingBinding pending{
            .binding = as_binding_cell(_binding),
        };
        pending.ifm_overrides.emplace();
        pending.ifm_overrides->emplace(
            std::piecewise_construct,
            std::forward_as_tuple(port),
            std::forward_as_tuple(
                parent,
                std::vector<uint32_t>{row, 0},
                std::vector<uint32_t>{1, width}
            )
        );
        segment._impl->pending->push_back(std::move(pending));
    } catch (...) {
        segment.abort();
        throw;
    }
}

void MLAModelWithBuffer::set_reloc_set(
    const std::vector<MLAModelWithBuffer*>& replaced_models,
    const std::function<std::vector<RelocUpdate>()>& prepare
) {
    if (!prepare) {
        throw std::invalid_argument("MLA adapter preparation callback is empty");
    }

    /*
     * The callback may allocate and upload an inactive adapter bank. Holding
     * the execution lock across that CPU work, validation, and publication is
     * what makes the switch a real DMA ownership boundary rather than merely
     * a thread-safe map assignment.
     */
    auto execution_lock = _session->acquire_execution_lock();
    std::vector<RelocUpdate> updates = prepare();
    std::vector<std::size_t> replaced_indices;
    replaced_indices.reserve(replaced_models.size());
    for (const MLAModelWithBuffer* model : replaced_models) {
        if (!model || model->_session != _session) {
            throw std::invalid_argument(
                "one MLA adapter transaction cannot mix kernel contexts"
            );
        }
        const auto binding = as_binding_cell(model->_binding);
        if (std::find(replaced_indices.begin(), replaced_indices.end(),
                      binding->model_index) == replaced_indices.end()) {
            replaced_indices.push_back(binding->model_index);
        }
    }
    std::vector<MlaExecutionSession::AdapterUpdate> session_updates;
    session_updates.reserve(updates.size());
    for (const RelocUpdate& update : updates) {
        if (!update.model || update.model->_session != _session) {
            throw std::invalid_argument(
                "one MLA adapter transaction cannot mix kernel contexts"
            );
        }
        const auto binding = as_binding_cell(update.model->_binding);
        session_updates.push_back({
            .index = binding->model_index,
            .adapters = &update.buffers,
            .binding = binding,
        });
    }
    _session->replace_adapter_set_locked(replaced_indices, session_updates);
}

void MLAModelWithBuffer::clear_reloc_set(
    const std::vector<MLAModelWithBuffer*>& models,
    const std::function<void()>& after_clear
) {
    auto execution_lock = _session->acquire_execution_lock();
    std::vector<std::size_t> indices;
    indices.reserve(models.size());
    for (const MLAModelWithBuffer* model : models) {
        if (!model || model->_session != _session) {
            throw std::invalid_argument(
                "one MLA adapter transaction cannot mix kernel contexts"
            );
        }
        indices.push_back(
            as_binding_cell(model->_binding)->model_index
        );
    }
    _session->clear_adapter_set_locked(indices);
    if (after_clear) {
        after_clear();
    }
}

void MLAModelWithBuffer::load_related_models(
    std::optional<std::filesystem::path> relative_dir
) {
    /*
     * The direct backend publishes the selected QMLA set as one immutable
     * ModelPackage transaction.  The old Dispatcher choice between serial
     * and parallel loading therefore no longer exists: there is exactly one
     * admission path and failure cannot expose a partially loaded family.
     */
    _session->load_models(std::move(relative_dir));
}

void MLAModelWithBuffer::free_related_models(
    std::optional<std::filesystem::path> relative_dir
) {
    _session->free_models(std::move(relative_dir));
}

std::size_t MLAModelWithBuffer::inspect_public_input_count(
    const std::shared_ptr<MlaExecutionSession>& session,
    const std::filesystem::path& model_path,
    bool zero_hidden_inputs
) {
    if (!session) {
        throw std::invalid_argument(
            "cannot inspect MLA package without an execution session"
        );
    }
    return session->inspect_public_input_count(
        model_path, zero_hidden_inputs
    );
}

void MLAModelWithBuffer::_debug_inouts(
    const std::string& name,
    std::map<uint8_t, MLABufferSlice>* fm_map_ptr
) {
    const auto binding = as_binding_cell(_binding);
    const auto model_path =
        _session->model_path(binding->model_index);
    if (_print_inouts) {
        auto& fms =
            (name == "ifm") ? binding->ifms : binding->ofms;
        std::ostringstream print_buffer;
        print_buffer << model_path << std::endl;
        for (uint32_t i = 0; i < fms.size(); ++i) {
            print_buffer << name << i << " ";
            if (fm_map_ptr && fm_map_ptr->contains(i)) {
                auto& buf_slice = fm_map_ptr->at(i);
                auto buf_ptr = (
                    buf_slice.get_buf_ptr()
                        ? buf_slice.get_buf_ptr()
                        : fms[i].get_buf_ptr()
                );
                const auto begins = buf_slice.get_buf_begins().value();
                const auto shapes = buf_slice.get_buf_shapes().value();
                print_buffer << MLABufferSlice(buf_ptr, begins, shapes);
            } else {
                print_buffer << fms[i];
            }
            spdlog::info("{}", print_buffer.str());
            print_buffer.str("");
            print_buffer.clear();
        }
    }

    if (_save_inouts) {
        auto& fms =
            (name == "ifm") ? binding->ifms : binding->ofms;
        for (uint32_t i = 0; i < fms.size(); ++i) {
            std::filesystem::path directory = (
                _save_inout_dir
                / model_path.stem()
                / (name + std::to_string(i))
            );
            std::filesystem::create_directories(directory);
            const auto num_files = count_regular_files(directory);
            const std::filesystem::path file_name =
                directory / fmt::format("{}.bin", num_files);
            if (fm_map_ptr && fm_map_ptr->contains(i)) {
                auto& buf_slice = fm_map_ptr->at(i);
                auto buf_ptr = (
                    buf_slice.get_buf_ptr()
                        ? buf_slice.get_buf_ptr()
                        : fms[i].get_buf_ptr()
                );
                MLABufferSlice(
                    buf_ptr,
                    buf_slice.get_buf_begins().value(),
                    buf_slice.get_buf_shapes().value()
                ).to_file(file_name);
            } else {
                fms[i].to_file(file_name);
            }
        }
    }
}

}  // namespace llima
}  // namespace simaai
