// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SiMa.ai

#include "mla_model.hpp"

#include <simaai/neat/mla/MlaKernelBackend.h>
#include <simaai_memory.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <thread>
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

bool path_is_within(
    const std::filesystem::path& path,
    const std::optional<std::filesystem::path>& directory
) {
    if (!directory.has_value()) {
        return true;
    }
    const auto normalized_path =
        std::filesystem::absolute(path).lexically_normal();
    const auto normalized_directory =
        std::filesystem::absolute(*directory).lexically_normal();

    /*
     * Compare path components, not string prefixes.  `/models/text2` is not
     * inside `/models/text`; a prefix check could otherwise publish or release
     * the wrong transactional package family.
     */
    return std::equal(
        normalized_directory.begin(), normalized_directory.end(),
        normalized_path.begin(), normalized_path.end()
    );
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

/*
 * The complete LLiMa ownership and ordering domain.  There is one Backend,
 * hence one kernel context and one immutable Background priority.  Models,
 * dma-buf registrations, adapter selections and the queue-ahead executor are
 * all context-local; no dispatcher handle or process-global model pointer is
 * observable outside this object.
 */
class MlaExecutionSession {
  public:
    struct SubmissionSnapshot {
        mla::Model model;
        mla::ExecutionBindings bindings;
        std::filesystem::path model_path;
    };

    /* Non-owning inputs are consumed synchronously under both session locks. */
    struct AdapterUpdate {
        std::size_t index;
        const std::map<std::string, MLABufferSlice>* adapters;
        const std::vector<MLABufferSlice>* default_ifms;
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
        for (std::size_t index = 0; index < models_.size(); ++index) {
            if (!models_[index].package &&
                path_is_within(models_[index].path, relative_directory)) {
                missing.push_back(index);
            }
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
                if (path_is_within(models_[index].path, relative_directory)) {
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
                models_[index].package.reset();
                models_[index].package_ordinal = 0;
                models_[index].active_adapters.clear();
            }
        }
        packages_ = std::move(retained);
        if (!relative_directory.has_value()) {
            import_cache_.clear();
            zero_hidden_input_.reset();
        }
    }

    SubmissionSnapshot prepare(
        std::size_t index,
        const std::vector<MLABufferSlice>& default_ifms,
        const std::vector<MLABufferSlice>& default_ofms,
        std::map<uint8_t, MLABufferSlice>* ifm_overrides,
        std::map<uint8_t, MLABufferSlice>* ofm_overrides
    ) {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        validate_index_locked(index);
        if (!models_[index].package) {
            load_group_locked({index});
        }

        ModelEntry& entry = models_[index];
        SubmissionSnapshot snapshot;
        snapshot.model =
            entry.package->package.model(entry.package_ordinal);
        snapshot.model_path = entry.path;
        if (!snapshot.model.valid()) {
            poison_locked("package returned an invalid model handle");
            throw std::runtime_error("package returned an invalid model handle");
        }

        const auto& info =
            entry.package->package.info(entry.package_ordinal);
        if (entry.zero_hidden_inputs && entry.active_adapters.empty()) {
            std::uint64_t largest_hidden_extent = 0;
            for (const mla::TensorPortInfo& port : info.inputs) {
                if (!port.public_port) {
                    largest_hidden_extent = std::max(
                        largest_hidden_extent, port.byte_extent
                    );
                }
            }
            if (largest_hidden_extent != 0) {
                ensure_zero_hidden_input_locked(largest_hidden_extent);
            }
        }
        snapshot.bindings.inputs.reserve(info.inputs.size());
        std::optional<MLABufferSlice> zero_hidden_slice;
        for (std::size_t port_index = 0;
             port_index < info.inputs.size(); ++port_index) {
            const mla::TensorPortInfo& port = info.inputs[port_index];
            const MLABufferSlice* selected = select_slice(
                port_index, default_ifms, ifm_overrides
            );
            if ((!selected || !selected->get_buf_ptr()) &&
                !port.public_port) {
                const auto adapter =
                    entry.active_adapters.find(port.name);
                if (adapter != entry.active_adapters.end()) {
                    selected = &adapter->second;
                }
                if ((!selected || !selected->get_buf_ptr()) &&
                    entry.zero_hidden_inputs) {
                    /*
                     * LoRA QMLA images expose A/B filters and their scales as
                     * ordinary hidden read-only inputs. The neutral/base
                     * model is therefore not an omitted binding: it is a
                     * checked all-zero binding. Use one session-owned zero
                     * allocation for every hidden port instead of allocating
                     * thousands of per-layer zero tensors. The compiler
                     * extent still controls each BufferView, so sharing does
                     * not weaken bounds checking.
                     */
                    ensure_zero_hidden_input_locked(port.byte_extent);
                    zero_hidden_slice.emplace(zero_hidden_input_.get());
                    selected = &zero_hidden_slice.value();
                }
            }
            snapshot.bindings.inputs.push_back(
                import_view_locked(selected, default_ifms, port_index,
                                   port.byte_extent, "input")
            );
        }

        if (info.outputs.size() != default_ofms.size()) {
            throw std::runtime_error(fmt::format(
                "Model {} requires {} outputs but LLiMa defines {}",
                entry.path, info.outputs.size(), default_ofms.size()
            ));
        }
        snapshot.bindings.outputs.reserve(info.outputs.size());
        for (std::size_t port_index = 0;
             port_index < info.outputs.size(); ++port_index) {
            const MLABufferSlice* selected = select_slice(
                port_index, default_ofms, ofm_overrides
            );
            snapshot.bindings.outputs.push_back(
                import_view_locked(selected, default_ofms, port_index,
                                   info.outputs[port_index].byte_extent,
                                   "output")
            );
        }
        return snapshot;
    }

    std::unique_lock<std::mutex> acquire_execution_lock() {
        return std::unique_lock<std::mutex>(execution_mutex_);
    }

    void run_snapshots(std::deque<SubmissionSnapshot> snapshots) {
        require_healthy();
        if (snapshots.empty()) {
            return;
        }

        /*
         * Keep a bounded accepted window and refill it immediately after the
         * oldest terminal CQE.  Two descriptor banks cap ACTIVE+READY
         * generations for one model; they do not prove that an accepted depth
         * of two is sufficient.  A third accepted job can hide CQE-to-refill
         * jitter while the kernel advances RUNNING/READY preparation.  The
         * private depth is therefore selected by the DVT sweep, not exposed as
         * an application option.  Admission is per kernel context, so this
         * bound controls retained state and failure blast radius rather than
         * reserving another Neat context's credits.
         */
        std::deque<mla::Job> inflight;
        std::size_t next = 0;
        std::exception_ptr first_failure;
        while (next < snapshots.size() || !inflight.empty()) {
            while (!first_failure && next < snapshots.size() &&
                   inflight.size() < queue_ahead_depth_) {
                mla::Job job;
                mla::SubmitOptions options;
                options.timeout_ms = kJobTimeoutMs;
                mla::Status status = backend_->submit(
                    snapshots[next].model,
                    snapshots[next].bindings,
                    options,
                    &job
                );
                if (!status) {
                    first_failure = std::make_exception_ptr(
                        std::runtime_error(fmt::format(
                            "MLA segment submit {} failed: {} ({})",
                            snapshots[next].model_path,
                            status.code,
                            status.message
                        ))
                    );
                    poison(fmt::format(
                        "ordered segment submit failed at {}", next
                    ));
                    break;
                }
                inflight.push_back(std::move(job));
                ++next;
            }

            if (inflight.empty()) {
                break;
            }
            mla::JobCompletion completion;
            const mla::Status status =
                inflight.front().wait(&completion);
            inflight.pop_front();
            if ((!status || completion.result != 0) && !first_failure) {
                first_failure = std::make_exception_ptr(
                    std::runtime_error(fmt::format(
                        "MLA ordered segment failed: {} completion={} fault={}",
                        status.code, completion.result,
                        completion.fault_class
                    ))
                );
                poison("ordered segment terminal failure");
            }
        }
        if (first_failure) {
            std::rethrow_exception(first_failure);
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
            if (!update.adapters || !update.default_ifms) {
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
                    &slice, *update.default_ifms, port->physical_index,
                    port->byte_extent, "adapter"
                );
            }

            for (const mla::TensorPortInfo& port : info.inputs) {
                if (port.public_port) {
                    continue;
                }
                const bool has_default =
                    port.physical_index < update.default_ifms->size() &&
                    (*update.default_ifms)[port.physical_index].get_buf_ptr();
                if (!has_default && !update.adapters->contains(port.name)) {
                    throw std::invalid_argument(fmt::format(
                        "Adapter set for {} omits hidden input {}",
                        entry.path, port.name
                    ));
                }
            }
        }

        /* Clear the old family and publish the new family under one mutex. */
        for (const std::size_t index : replaced_indices) {
            models_[index].active_adapters.clear();
        }
        for (const AdapterUpdate& update : updates) {
            models_[update.index].active_adapters = *update.adapters;
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
        for (const std::size_t index : indices) {
            /*
             * Accepted SubmissionSnapshots own their imported BufferViews and
             * remain unchanged. Clearing this future-submission map restores
             * the package's immutable/default hidden inputs; if the package
             * has no valid base binding, the next prepare fails closed rather
             * than executing zeroed adapter storage.
             */
            models_[index].active_adapters.clear();
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
    };

    explicit MlaExecutionSession(
        std::unique_ptr<mla::Backend> backend,
        std::size_t queue_ahead_depth
    ) : backend_(std::move(backend)),
        queue_ahead_depth_(queue_ahead_depth) {
        if (queue_ahead_depth_ == 0) {
            throw std::invalid_argument("MLA queue-ahead depth must be nonzero");
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
            import_cache_.erase(zero_hidden_input_->get_allocation_cookie());
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
    }

    static const MLABufferSlice* select_slice(
        std::size_t index,
        const std::vector<MLABufferSlice>& defaults,
        std::map<uint8_t, MLABufferSlice>* overrides
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
            imported =
                import_cache_.emplace(cookie, std::move(buffer)).first;
        }

        mla::BufferView view =
            imported->second.view(offset, compiler_extent);
        if (!view.valid()) {
            throw std::out_of_range(fmt::format(
                "Backend rejected MLA {} {} BufferView",
                direction, port_index
            ));
        }
        return view;
    }

    std::unique_ptr<mla::Backend> backend_;
    const std::size_t queue_ahead_depth_;
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
    std::unordered_map<std::uint64_t, mla::Buffer> import_cache_;
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
    }

    std::shared_ptr<MlaExecutionSession> session;
    std::unique_lock<std::mutex> execution_lock;
    std::deque<MlaExecutionSession::SubmissionSnapshot> snapshots;
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
    return !_impl || _impl->snapshots.empty();
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
    if (_impl->snapshots.empty()) {
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
    std::deque<MlaExecutionSession::SubmissionSnapshot> snapshots;
    snapshots.swap(_impl->snapshots);
    try {
        _impl->session->run_snapshots(std::move(snapshots));
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
) : _session(std::move(session)),
    _ifms(std::move(ifms)),
    _ofms(std::move(ofms)) {
    if (!_session) {
        throw std::invalid_argument("MLAModelWithBuffer requires a session");
    }
    _model_idx = _session->register_model(model_path, zero_hidden_inputs);
}

void MLAModelWithBuffer::load() {
    _session->load_model(_model_idx);
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
        segment._impl->snapshots.push_back(_session->prepare(
            _model_idx, _ifms, _ofms, ifm_map_ptr, ofm_map_ptr
        ));
    } catch (...) {
        /*
         * Segment construction is transactional.  If preparing this model
         * fails, none of the earlier snapshots in the same caller-owned
         * segment may run later by accident.  They have not reached the
         * kernel yet, so dropping them is both safe and syscall-free.
         */
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
        if (std::find(replaced_indices.begin(), replaced_indices.end(),
                      model->_model_idx) == replaced_indices.end()) {
            replaced_indices.push_back(model->_model_idx);
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
        session_updates.push_back({
            .index = update.model->_model_idx,
            .adapters = &update.buffers,
            .default_ifms = &update.model->_ifms,
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
        indices.push_back(model->_model_idx);
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

void MLAModelWithBuffer::_debug_inouts(
    const std::string& name,
    std::map<uint8_t, MLABufferSlice>* fm_map_ptr
) {
    const auto model_path = _session->model_path(_model_idx);
    if (_print_inouts) {
        auto& fms = (name == "ifm") ? _ifms : _ofms;
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
        auto& fms = (name == "ifm") ? _ifms : _ofms;
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
