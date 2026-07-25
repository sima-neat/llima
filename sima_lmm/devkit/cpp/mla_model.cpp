// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 SiMa.ai

#include "mla_model.hpp"

#include <simaai/neat/mla/MlaKernelBackend.h>
#include <simaai_memory.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
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

constexpr std::size_t kQueueAheadDepth = 2;
constexpr std::uint32_t kJobTimeoutMs = 60000;

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

    static std::shared_ptr<MlaExecutionSession> create() {
        mla::Status status;
        auto backend = mla::Backend::open(
            mla::WorkloadPriority::kBackground, &status
        );
        if (!backend || !status) {
            throw_status("Backend::open", status);
        }
        return std::shared_ptr<MlaExecutionSession>(
            new MlaExecutionSession(std::move(backend))
        );
    }

    ~MlaExecutionSession() {
        /*
         * A producer can fail while preparing the middle of a segment (for
         * example, because one BufferView is out of range).  The snapshots
         * already in pending_ were never submitted to hardware, so teardown
         * must discard them rather than let a cleanup exception hide the
         * original preparation error.  Submitted jobs are not stored here;
         * run_segment() always drains those before it returns.
         */
        discard_pending_segment();
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

    std::size_t register_model(const std::filesystem::path& model_path) {
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
            return found->second;
        }
        const std::size_t index = models_.size();
        path_to_index_.emplace(absolute_path, index);
        models_.push_back(ModelEntry{.path = absolute_path});
        return index;
    }

    void load_model(std::size_t index) {
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
        std::lock_guard lock(mutex_);
        if (!pending_.empty()) {
            throw std::runtime_error(
                "cannot release MLA packages inside an open execution segment"
            );
        }

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
        snapshot.bindings.inputs.reserve(info.inputs.size());
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

    void run(SubmissionSnapshot snapshot) {
        require_healthy();
        mla::Job job;
        mla::SubmitOptions options;
        options.timeout_ms = kJobTimeoutMs;
        mla::Status status = backend_->submit(
            snapshot.model, snapshot.bindings, options, &job
        );
        if (!status) {
            poison(fmt::format(
                "submit failed for {}: {} ({})",
                snapshot.model_path, status.code, status.message
            ));
            throw_status("Backend::submit", status);
        }
        mla::JobCompletion completion;
        status = job.wait(&completion);
        if (!status || completion.result != 0) {
            poison(fmt::format(
                "job failed for {}: {} completion={} fault={}",
                snapshot.model_path, status.code, completion.result,
                completion.fault_class
            ));
            throw_status("Job::wait", status);
        }
        if (MLAModelWithBuffer::_profile &&
            (completion.valid & mla::kCompletionValidActive) != 0) {
            spdlog::info(
                "MLA model={} active_us={:.3f}",
                snapshot.model_path,
                completion.active_microseconds()
            );
        }
    }

    void enqueue(SubmissionSnapshot snapshot) {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        const std::thread::id caller = std::this_thread::get_id();
        if (pending_.empty()) {
            segment_owner_ = caller;
        } else if (segment_owner_ != caller) {
            throw std::runtime_error(
                "one LLiMa execution segment cannot span producer threads"
            );
        }
        pending_.push_back(std::move(snapshot));
    }

    void discard_current_segment() noexcept {
        std::lock_guard lock(mutex_);
        if (pending_.empty()) {
            return;
        }

        /*
         * Only the producer that owns the transaction may roll it back.  A
         * failing call from an unrelated thread must not silently erase work
         * assembled by the real owner.  enqueue() will report that ownership
         * violation if the unrelated caller reaches publication.
         */
        if (segment_owner_ == std::this_thread::get_id()) {
            pending_.clear();
            segment_owner_ = {};
        }
    }

    void run_segment() {
        std::deque<SubmissionSnapshot> snapshots;
        {
            std::lock_guard lock(mutex_);
            require_healthy_locked();
            if (pending_.empty()) {
                return;
            }
            if (segment_owner_ != std::this_thread::get_id()) {
                throw std::runtime_error(
                    "execution segment must be committed by its producer"
                );
            }
            snapshots.swap(pending_);
            segment_owner_ = {};
        }

        /*
         * Two jobs match the kernel's two immutable descriptor banks. Refill
         * immediately after the oldest terminal CQE: while job N executes,
         * N+1 is already prepared, but this Background context never consumes
         * an unbounded number of global admission slots. Foreground Neat can
         * therefore run at the first compiled-job boundary.
         */
        std::deque<mla::Job> inflight;
        std::size_t next = 0;
        std::exception_ptr first_failure;
        while (next < snapshots.size() || !inflight.empty()) {
            while (!first_failure && next < snapshots.size() &&
                   inflight.size() < kQueueAheadDepth) {
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

    void set_adapters(
        std::size_t index,
        const std::map<std::string, MLABufferSlice>& adapters,
        const std::vector<MLABufferSlice>& default_ifms
    ) {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
        validate_index_locked(index);
        if (!models_[index].package) {
            load_group_locked({index});
        }
        ModelEntry& entry = models_[index];
        const auto& info =
            entry.package->package.info(entry.package_ordinal);

        for (const auto& [name, slice] : adapters) {
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
            /*
             * Validate and import now. Publication is one map assignment only
             * after every adapter has passed; accepted snapshots remain
             * unchanged.
             */
            (void)import_view_locked(
                &slice, default_ifms, port->physical_index,
                port->byte_extent, "adapter"
            );
        }

        for (const mla::TensorPortInfo& port : info.inputs) {
            if (port.public_port) {
                continue;
            }
            const bool has_default =
                port.physical_index < default_ifms.size() &&
                default_ifms[port.physical_index].get_buf_ptr();
            if (!has_default && !adapters.contains(port.name)) {
                throw std::invalid_argument(fmt::format(
                    "Adapter set for {} omits hidden input {}",
                    entry.path, port.name
                ));
            }
        }
        entry.active_adapters = adapters;
    }

    std::filesystem::path model_path(std::size_t index) const {
        std::lock_guard lock(mutex_);
        validate_index_locked(index);
        return models_[index].path;
    }

  private:
    void discard_pending_segment() noexcept {
        std::lock_guard lock(mutex_);
        pending_.clear();
        segment_owner_ = {};
    }

    struct PackageHold {
        mla::ModelPackage package;
        std::vector<std::size_t> model_indices;
    };

    struct ModelEntry {
        std::filesystem::path path;
        std::shared_ptr<PackageHold> package;
        std::size_t package_ordinal = 0;
        std::map<std::string, MLABufferSlice> active_adapters;
    };

    explicit MlaExecutionSession(std::unique_ptr<mla::Backend> backend)
      : backend_(std::move(backend)) {}

    void validate_index_locked(std::size_t index) const {
        if (index >= models_.size()) {
            throw std::out_of_range("invalid LLiMa MLA model index");
        }
    }

    void require_healthy_locked() const {
        if (poisoned_) {
            throw std::runtime_error(
                "MLA execution session is poisoned and must be reconstructed"
            );
        }
    }

    void require_healthy() const {
        std::lock_guard lock(mutex_);
        require_healthy_locked();
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
    mutable std::mutex mutex_;
    std::map<std::filesystem::path, std::size_t> path_to_index_;
    std::vector<ModelEntry> models_;
    std::vector<std::shared_ptr<PackageHold>> packages_;
    std::unordered_map<std::uint64_t, mla::Buffer> import_cache_;
    std::deque<SubmissionSnapshot> pending_;
    std::thread::id segment_owner_;
    bool poisoned_ = false;
    std::string poison_reason_;
};

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
    default_session = MlaExecutionSession::create();
    spdlog::info(
        "Connected LLiMa directly to /dev/mla with Background priority"
    );
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
    std::vector<MLABufferSlice> ofms
) : _session(require_default_session()),
    _ifms(std::move(ifms)),
    _ofms(std::move(ofms)) {
    _model_idx = _session->register_model(model_path);
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
    load();
    _debug_inouts("ifm", ifm_map_ptr);
    _session->run(_session->prepare(
        _model_idx, _ifms, _ofms, ifm_map_ptr, ofm_map_ptr
    ));
    _debug_inouts("ofm", ofm_map_ptr);
}

void MLAModelWithBuffer::add_to_queue(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    load();
    try {
        _session->enqueue(_session->prepare(
            _model_idx, _ifms, _ofms, ifm_map_ptr, ofm_map_ptr
        ));
    } catch (...) {
        /*
         * Segment construction is transactional.  If preparing this model
         * fails, none of the earlier snapshots in the same caller-owned
         * segment may run later by accident.  They have not reached the
         * kernel yet, so dropping them is both safe and syscall-free.
         */
        _session->discard_current_segment();
        throw;
    }
}

void MLAModelWithBuffer::run_queue() {
    require_default_session()->run_segment();
}

void MLAModelWithBuffer::update_reloc(
    const std::map<std::string, MLABufferSlice>& reloc_buffers
) {
    load();
    _session->set_adapters(_model_idx, reloc_buffers, _ifms);
}

void MLAModelWithBuffer::load_all_models(
    std::optional<std::filesystem::path> relative_dir
) {
    /*
     * The direct backend publishes the selected QMLA set as one immutable
     * ModelPackage transaction.  The old Dispatcher choice between serial
     * and parallel loading therefore no longer exists: there is exactly one
     * admission path and failure cannot expose a partially loaded family.
     */
    require_default_session()->load_models(std::move(relative_dir));
}

void MLAModelWithBuffer::free_all_models(
    std::optional<std::filesystem::path> relative_dir
) {
    require_default_session()->free_models(std::move(relative_dir));
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
