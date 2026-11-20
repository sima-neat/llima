#include <regex>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <map>
#include <cctype>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <simaai/gst-api.h>
#include <spdlog/spdlog.h>

#include "file_provider.hpp"
#include "mla_model.hpp"

namespace simaai {
namespace llima {
namespace {

struct QueuedRun {
    std::size_t model_index = 0;
    std::vector<mla_tensor> ifms;
    std::vector<mla_tensor> ofms;
};

struct MlaRuntimeState {
    std::mutex registry_mutex;
    std::recursive_mutex execution_mutex;
    mla_handle_p handle = nullptr;
    // Keyed by ELF path, or "<path>#<instance>" for extra instances of one ELF.
    std::map<std::filesystem::path, std::size_t> path_to_index;
    std::vector<std::filesystem::path> paths;  // ELF file per index
    std::vector<mla_model_p> models;
    // Allocation identity follows storage when MLABuffer::swap_storage is used.
    std::unordered_map<uint64_t, uint32_t> imported_buffers;
};

MlaRuntimeState& runtime_state() {
    static MlaRuntimeState state;
    return state;
}

thread_local std::vector<QueuedRun> queued_runs;
thread_local std::size_t queued_run_count = 0;
// Host-side profiling counters (nanoseconds).
uint64_t add_queue_ns = 0, submit_ns = 0, reap_ns = 0, submitted_jobs = 0;
inline uint64_t elapsed_ns(std::chrono::steady_clock::time_point t) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t).count());
}

mla_handle_p require_handle() {
    auto& state = runtime_state();
    if (!state.handle) {
        throw std::runtime_error("MLA-RT is not connected; call connect() first");
    }
    return state.handle;
}

uint32_t imported_buffer_id(MLABuffer* buffer) {
    auto& state = runtime_state();
    const uint64_t generation = buffer->get_allocation_generation();
    if (generation == 0) {
        throw std::logic_error("cannot import an unallocated MLA buffer");
    }

    const auto found = state.imported_buffers.find(generation);
    if (found != state.imported_buffers.end()) return found->second;

    uint32_t id = 0;
    const int rc = mla_import_dmabuf(
        require_handle(), buffer->get_dmabuf_fd(), MLA_BUF_IFM, &id, nullptr
    );
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT failed to import DMA-BUF: rc={} ({})",
            rc, std::strerror(rc < 0 ? -rc : rc)
        ));
    }
    state.imported_buffers.emplace(generation, id);
    return id;
}

bool path_matches_family(
    const std::filesystem::path& model_path,
    const std::optional<std::filesystem::path>& selector
) {
    if (!selector) return true;
    const auto model = std::filesystem::absolute(model_path).lexically_normal();
    const auto family = std::filesystem::absolute(*selector).lexically_normal();

    // Callers use either a directory (Whisper) or a filename stem (language
    // and vision models). Match real path components first, then accept a
    // sibling whose filename begins with the complete stem at a separator.
    const auto mismatch = std::mismatch(
        family.begin(), family.end(), model.begin(), model.end()
    );
    if (mismatch.first == family.end()) {
        return true;
    }
    if (family.parent_path() != model.parent_path()) {
        return false;
    }

    const std::string stem = family.filename().string();
    const std::string candidate = model.filename().string();
    if (candidate.size() <= stem.size() ||
        candidate.compare(0, stem.size(), stem) != 0) {
        return false;
    }
    const char boundary = candidate[stem.size()];
    return boundary == '_' || boundary == '.';
}

void validate_override_indices(
    const std::map<uint8_t, MLABufferSlice>* overrides,
    std::size_t count,
    const char* kind
) {
    if (!overrides) return;
    for (const auto& [index, unused] : *overrides) {
        (void)unused;
        if (index >= count) {
            throw std::out_of_range(fmt::format(
                "{} override index {} is outside {} bindings", kind, index, count
            ));
        }
    }
}

const MLABufferSlice& effective_slice(
    const std::vector<MLABufferSlice>& defaults,
    const std::map<uint8_t, MLABufferSlice>* overrides,
    std::size_t index
) {
    if (overrides) {
        const auto override_it = overrides->find(static_cast<uint8_t>(index));
        if (override_it != overrides->end()) return override_it->second;
    }
    return defaults[index];
}

MLABuffer* effective_buffer(
    const std::vector<MLABufferSlice>& defaults,
    const MLABufferSlice& slice,
    std::size_t index
) {
    MLABuffer* buffer = slice.get_buf_ptr();
    if (!buffer) buffer = defaults[index].get_buf_ptr();
    if (!buffer) {
        throw std::invalid_argument(fmt::format(
            "MLA binding {} has no backing buffer", index
        ));
    }
    return buffer;
}

void make_bindings(
    const std::vector<MLABufferSlice>& defaults,
    const std::map<uint8_t, MLABufferSlice>* overrides,
    const char* kind,
    std::vector<mla_tensor>& bindings
) {
    validate_override_indices(overrides, defaults.size(), kind);
    if (defaults.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(fmt::format("{} binding count exceeds MLA-RT", kind));
    }
    bindings.clear();
    bindings.reserve(defaults.size());
    for (std::size_t i = 0; i < defaults.size(); ++i) {
        const auto& slice = effective_slice(defaults, overrides, i);
        MLABuffer* buffer = effective_buffer(defaults, slice, i);
        const uint64_t offset = buffer->get_buf_addr_offset(slice.get_buf_begins());
        if (offset >= buffer->get_allocation_size()) {
            throw std::out_of_range(fmt::format(
                "{} binding {} starts at {} outside its {}-byte dma-buf",
                kind, i, offset, buffer->get_allocation_size()
            ));
        }
        // The fd binding describes the accessible carrier extent from this
        // offset. The model's descriptors define which bytes are actually
        // touched; this matters for strided KV-cache slices whose ELF section
        // size is larger than their per-token access span.
        const uint64_t length = buffer->get_allocation_size() - offset;
        bindings.push_back({imported_buffer_id(buffer), 0, offset, length});
    }
}

mla_model_p model_for(std::size_t index) {
    auto& state = runtime_state();
    std::lock_guard lock(state.registry_mutex);
    if (index >= state.models.size()) {
        throw std::out_of_range("MLA model registry index is invalid");
    }
    return state.models[index];
}

namespace {
// SIMA_LLIMA_PROFILE_UNITS: per-unit MLA tile time.
struct UnitTime {
    uint64_t jobs = 0;
    uint64_t tile_us = 0;
};
// 1: per unit kind (layer digits folded); 2: per unit.
const int unit_profile_level = [] {
    const char* v = std::getenv("SIMA_LLIMA_PROFILE_UNITS");
    return v == nullptr ? 0 : std::atoi(v);
}();
const bool unit_profile_on = unit_profile_level > 0;
std::unordered_map<std::size_t, UnitTime> unit_times;  // by model index

void record_unit_time(std::size_t model_index, uint64_t tile_us) {
    if (!unit_profile_on) return;
    auto& t = unit_times[model_index];
    ++t.jobs;
    t.tile_us += tile_us;
}
}  // namespace

std::filesystem::path path_for(std::size_t index) {
    auto& state = runtime_state();
    std::lock_guard lock(state.registry_mutex);
    if (index >= state.paths.size()) {
        throw std::out_of_range("MLA model registry index is invalid");
    }
    return state.paths[index];
}

MLABufferSlice materialize_view(
    MLABuffer* buffer,
    const MLABufferSlice& slice
) {
    if (slice.get_buf_shapes()) {
        auto begins = slice.get_buf_begins().value_or(
            std::vector<uint32_t>(buffer->get_shape().size(), 0)
        );
        return MLABufferSlice(buffer, std::move(begins), *slice.get_buf_shapes());
    }
    if (slice.get_buf_begins()) {
        return MLABufferSlice(buffer, *slice.get_buf_begins());
    }
    return MLABufferSlice(buffer);
}

} // namespace

void release_mla_buffer_import(const MLABuffer* buffer) noexcept {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    const auto found = state.imported_buffers.find(buffer->get_allocation_generation());
    if (found == state.imported_buffers.end()) return;

    const int rc = mla_release_dmabuf(state.handle, found->second);
    if (rc != 0) {
        // Retain the id so disconnect can retry release.
        spdlog::error(
            "MLA-RT failed to release DMA-BUF registration {} during buffer teardown: rc={} ({})",
            found->second, rc, std::strerror(rc < 0 ? -rc : rc)
        );
        return;
    }
    state.imported_buffers.erase(found);
}

void connect_mla_rt(const std::vector<std::string>& args) {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    if (state.handle) return;

    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(mla_get_handle_argv), &info) && info.dli_fname) {
        spdlog::info("Loaded libMLArt.so: {}", info.dli_fname);
    }

    std::vector<std::string> effective_args = args;
    if (effective_args.empty()) {
        effective_args = {
            "libMLArt.so", "--connect", "kernel,ma", "-t", "-b=no", "-x", "4"
        };
    }
    std::vector<char*> argv;
    argv.reserve(effective_args.size());
    for (auto& value : effective_args) argv.push_back(value.data());

    spdlog::info("Connect MLA-RT with args: [{}]", fmt::join(effective_args, ", "));
    try {
        // LLiMa is a throughput workload. Keep latency-sensitive vision
        // processes at the platform default priority above it. MLA_CTX_PRIORITY
        // can override this request before the first connection.
        mla_ctx_cfg cfg{};
        cfg.fields = MLA_CTX_CFG_GROUP_PRIO;
        cfg.group_prio = 10;
        state.handle = mla_get_handle_cfg(
            static_cast<int>(argv.size()), argv.data(), &cfg
        );
    } catch (const std::exception& error) {
        throw std::runtime_error(fmt::format("Failed to connect MLA-RT: {}", error.what()));
    }
    if (!state.handle) throw std::runtime_error("Failed to connect MLA-RT");

    mla_ctx_cfg applied_cfg{};
    const int cfg_rc = mla_get_ctx_cfg(state.handle, &applied_cfg);
    if (cfg_rc != 0) {
        mla_free_handle(state.handle);
        state.handle = nullptr;
        throw std::runtime_error(fmt::format(
            "MLA-RT failed to read back context priority: rc={} ({})",
            cfg_rc, std::strerror(cfg_rc < 0 ? -cfg_rc : cfg_rc)
        ));
    }
    spdlog::info("MLA-RT context group priority: {}", applied_cfg.group_prio);
}

void disconnect_mla_rt() {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    for (auto& model : state.models) {
        if (model) {
            mla_free_model(model);
            model = nullptr;
        }
    }
    queued_runs.clear();
    queued_run_count = 0;
    for (const auto& [generation, id] : state.imported_buffers) {
        (void)generation;
        const int rc = mla_release_dmabuf(state.handle, id);
        if (rc != 0) {
            spdlog::error(
                "MLA-RT failed to release DMA-BUF registration {}: rc={} ({})",
                id, rc, std::strerror(rc < 0 ? -rc : rc)
            );
        }
    }
    state.imported_buffers.clear();
    if (state.handle) {
        mla_free_handle(state.handle);
        state.handle = nullptr;
    }
}

MLAModelWithBuffer::MLAModelWithBuffer(
    std::filesystem::path model_path,
    std::vector<MLABufferSlice> ifms,
    std::vector<MLABufferSlice> ofms,
    uint32_t instance
) : _ifms(std::move(ifms)), _ofms(std::move(ofms)) {
    model_path = std::filesystem::absolute(model_path).lexically_normal();
    // Deferred assets may not exist yet; load_all_models() checks them at load time.
    auto& state = runtime_state();
    std::lock_guard lock(state.registry_mutex);
    const std::filesystem::path key = instance == 0
        ? model_path
        : std::filesystem::path(fmt::format("{}#{}", model_path.string(), instance));
    const auto [it, inserted] = state.path_to_index.emplace(key, state.paths.size());
    _model_idx = it->second;
    if (inserted) {
        state.paths.push_back(std::move(model_path));
        state.models.push_back(nullptr);
    }
}

void MLAModelWithBuffer::load() {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    require_handle();
    if (state.models[_model_idx]) return;
    state.models[_model_idx] = mla_load_model(state.handle, state.paths[_model_idx].c_str());
    if (!state.models[_model_idx]) {
        throw std::runtime_error(fmt::format(
            "MLA-RT failed to load model: {}", state.paths[_model_idx]
        ));
    }
    spdlog::info("Loaded model: {}", state.paths[_model_idx]);
}

void MLAModelWithBuffer::free() {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    if (_model_idx >= state.models.size() || !state.models[_model_idx]) return;
    mla_free_model(state.models[_model_idx]);
    state.models[_model_idx] = nullptr;
}

void MLAModelWithBuffer::run(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    load();
    _debug_inouts("ifm", ifm_map_ptr);

    std::vector<mla_tensor> ifms;
    std::vector<mla_tensor> ofms;
    make_bindings(_ifms, ifm_map_ptr, "IFM", ifms);
    make_bindings(_ofms, ofm_map_ptr, "OFM", ofms);
    mla_job_h job = nullptr;
    int rc = mla_submit_async(
        model_for(_model_idx),
        static_cast<int>(ifms.size()), ifms.data(),
        static_cast<int>(ofms.size()), ofms.data(),
        &job
    );
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT submit failed for {}: rc={} ({})",
            path_for(_model_idx), rc, std::strerror(rc < 0 ? -rc : rc)
        ));
    }
    uint64_t tile_us = 0;
    uint64_t submit_to_reap_us = 0;
    rc = mla_wait(job, &tile_us, &submit_to_reap_us);
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT wait failed for {}: rc={} ({})",
            path_for(_model_idx), rc, std::strerror(rc < 0 ? -rc : rc)
        ));
    }
    record_unit_time(_model_idx, tile_us);
    if (_profile) {
        spdlog::info(
            "MLA-RT run {}: tile={} us, submit-to-reap={} us",
            path_for(_model_idx), tile_us, submit_to_reap_us
        );
    }
    _debug_inouts("ofm", ofm_map_ptr);
}

void MLAModelWithBuffer::add_to_queue(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    if (!_enable_queue) {
        run(ifm_map_ptr, ofm_map_ptr);
        return;
    }
    const auto add_start = std::chrono::steady_clock::now();
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    load();
    struct AddTimer { std::chrono::steady_clock::time_point t; ~AddTimer() { add_queue_ns += elapsed_ns(t); } } add_timer{add_start};
    const std::size_t slot = queued_run_count;
    if (slot == queued_runs.size()) queued_runs.emplace_back();
    auto& queued = queued_runs[slot];
    queued.model_index = _model_idx;
    make_bindings(_ifms, ifm_map_ptr, "IFM", queued.ifms);
    make_bindings(_ofms, ofm_map_ptr, "OFM", queued.ofms);
    ++queued_run_count;
}

namespace {

struct InflightRun {
    mla_job_h job = nullptr;
    std::size_t model_index = 0;
    std::vector<mla_tensor> ifms;  // kept alive until the job is reaped
    std::vector<mla_tensor> ofms;
};

thread_local std::deque<InflightRun> inflight_runs;

thread_local uint64_t inflight_tile_us = 0;
uint64_t lifetime_tile_us = 0;


// The public async pool is capped at 256 in-flight jobs.
constexpr std::size_t queue_ahead_depth = 256;

void reap_one(std::string& first_error) {
    const auto reap_start = std::chrono::steady_clock::now();
    auto run = std::move(inflight_runs.front());
    inflight_runs.pop_front();
    uint64_t tile_us = 0;
    const int rc = mla_wait(run.job, &tile_us, nullptr);
    if (rc != 0 && first_error.empty()) {
        first_error = fmt::format(
            "MLA-RT queued wait failed for {}: rc={} ({})",
            path_for(run.model_index), rc, std::strerror(rc < 0 ? -rc : rc)
        );
    }
    inflight_tile_us += tile_us;
    lifetime_tile_us += tile_us;
    record_unit_time(run.model_index, tile_us);
    reap_ns += elapsed_ns(reap_start);
}

}  // namespace

void MLAModelWithBuffer::submit_queue() {
    if (!_enable_queue || queued_run_count == 0) return;
    const auto submit_start = std::chrono::steady_clock::now();
    struct SubmitTimer { std::chrono::steady_clock::time_point t; ~SubmitTimer() { submit_ns += elapsed_ns(t); } } timer{submit_start};
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    const std::size_t run_count = queued_run_count;
    queued_run_count = 0;
    std::string first_error;
    for (std::size_t next = 0; next < run_count; ++next) {
        while (inflight_runs.size() >= queue_ahead_depth) reap_one(first_error);
        auto& queued = queued_runs[next];
        mla_model_p model = model_for(queued.model_index);
        if (!model) {
            throw std::runtime_error(fmt::format(
                "MLA model was released before queued execution: {}", path_for(queued.model_index)
            ));
        }
        InflightRun run{nullptr, queued.model_index, queued.ifms, queued.ofms};
        const int rc = mla_submit_async(
            model,
            static_cast<int>(run.ifms.size()), run.ifms.data(),
            static_cast<int>(run.ofms.size()), run.ofms.data(),
            &run.job
        );
        if (rc != 0) {
            throw std::runtime_error(fmt::format(
                "MLA-RT queued submit failed for {}: rc={} ({})",
                path_for(queued.model_index), rc, std::strerror(rc < 0 ? -rc : rc)
            ));
        }
        inflight_runs.push_back(std::move(run));
        ++submitted_jobs;
    }
    if (!first_error.empty()) throw std::runtime_error(first_error);
}

uint64_t MLAModelWithBuffer::total_tile_us() { return lifetime_tile_us; }

bool MLAModelWithBuffer::unit_profile_enabled() { return unit_profile_on; }

void MLAModelWithBuffer::reset_unit_profile() { unit_times.clear(); }

std::string MLAModelWithBuffer::unit_profile_report(uint64_t tokens) {
    // Group by unit kind: the ELF stem without the model-name prefix, digit runs folded to '#'.
    std::map<std::string, UnitTime> groups;
    for (const auto& [index, t] : unit_times) {
        std::string stem = path_for(index).stem().string();
        if (const auto at = stem.find("_language_"); at != std::string::npos) stem.erase(0, at + 10);
        if (const auto at = stem.find("_stage1_mla"); at != std::string::npos) stem.erase(at);
        // Keep the token-width prefix (n1_ / n128_); fold layer and cache-token indices.
        const std::size_t prefix = stem.find('_') == std::string::npos ? 0 : stem.find('_') + 1;
        std::string kind = stem.substr(0, prefix);
        for (std::size_t i = prefix; i < stem.size(); ++i) {
            if (unit_profile_level < 2 && std::isdigit(static_cast<unsigned char>(stem[i]))) {
                while (i + 1 < stem.size() && std::isdigit(static_cast<unsigned char>(stem[i + 1]))) ++i;
                kind += '#';
            } else {
                kind += stem[i];
            }
        }
        auto& g = groups[kind];
        g.jobs += t.jobs;
        g.tile_us += t.tile_us;
    }
    std::vector<std::pair<std::string, UnitTime>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second.tile_us > b.second.tile_us;
    });
    uint64_t total = 0;
    for (const auto& [kind, t] : sorted) total += t.tile_us;
    std::string out = fmt::format("[unit-profile] tokens={} total_tile={:.1f}ms\n", tokens, total / 1e3);
    for (const auto& [kind, t] : sorted) {
        out += fmt::format(
            "[unit-profile]   {:<36s} jobs={:<7d} tile={:9.1f}ms {:6.2f}ms/token {:7.1f}us/job {:5.1f}%\n",
            kind, t.jobs, t.tile_us / 1e3, tokens ? t.tile_us / 1e3 / tokens : 0.0,
            t.jobs ? double(t.tile_us) / t.jobs : 0.0, total ? 100.0 * t.tile_us / total : 0.0
        );
    }
    return out;
}

void MLAModelWithBuffer::host_counters(uint64_t& add_ns, uint64_t& sub_ns, uint64_t& rp_ns, uint64_t& jobs) {
    add_ns = add_queue_ns; sub_ns = submit_ns; rp_ns = reap_ns; jobs = submitted_jobs;
}

void MLAModelWithBuffer::wait_queue() {
    if (inflight_runs.empty()) return;
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::string first_error;
    while (!inflight_runs.empty()) reap_one(first_error);
    if (!first_error.empty()) throw std::runtime_error(first_error);
}

void MLAModelWithBuffer::run_queue_batched() {
    if (!_enable_queue) return;
    wait_queue();
    if (queued_run_count == 0) return;
    const auto submit_start = std::chrono::steady_clock::now();
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    const std::size_t run_count = queued_run_count;
    queued_run_count = 0;
    thread_local std::vector<mla_model_run> runs;
    runs.clear();
    runs.reserve(run_count);
    for (std::size_t i = 0; i < run_count; ++i) {
        const auto& queued = queued_runs[i];
        mla_model_p model = model_for(queued.model_index);
        if (!model) {
            throw std::runtime_error(fmt::format(
                "MLA model was released before queued execution: {}", path_for(queued.model_index)
            ));
        }
        runs.push_back({
            model, queued.ifms.data(), static_cast<int>(queued.ifms.size()),
            queued.ofms.data(), static_cast<int>(queued.ofms.size())
        });
    }
    const int rc = mla_run_model_multi_model(runs.data(), static_cast<int>(runs.size()));
    submitted_jobs += run_count;
    reap_ns += elapsed_ns(submit_start);  // the whole batched call, including execution
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT batched dispatch of {} runs failed: rc={} ({})",
            run_count, rc, std::strerror(rc < 0 ? -rc : rc)
        ));
    }
}

void MLAModelWithBuffer::run_queue() {
    if (!_enable_queue) return;
    if (batched_dispatch) {
        run_queue_batched();
        return;
    }
    const auto queue_start = std::chrono::steady_clock::now();
    const std::size_t run_count = queued_run_count + inflight_runs.size();
    inflight_tile_us = 0;
    submit_queue();
    wait_queue();
    if (_profile && run_count) {
        const auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - queue_start
        ).count();
        spdlog::info(
            "MLA-RT queue: {} jobs, total tile={} us, wall={} us",
            run_count, inflight_tile_us, wall_us
        );
    }
}

void MLAModelWithBuffer::update_reloc(
    const std::map<std::string, MLABuffer*>& reloc_buffers
) {
    if (reloc_buffers.empty()) return;
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    load();
    mla_model_p model = model_for(_model_idx);
    const auto lengths = mla_get_ifm_ofm_len_vector(model).reloc_len_array;
    if (lengths.size() != reloc_buffers.size()) {
        throw std::invalid_argument(fmt::format(
            "Relocation count for {} is {}, model expects {}",
            path_for(_model_idx), reloc_buffers.size(), lengths.size()
        ));
    }
    // Bind adapters by internal_buf_id, not physical address. A reusable-CMA
    // memory profile refuses the raw-phys attach path outright (the DMS pool is
    // System RAM), so import each adapter's DMA-BUF and pass its buf_id, exactly
    // as the IFM/OFM run path does in make_bindings(). Reloc ports are ordered
    // by adapter buffer name, which matches the model's reloc section order.
    std::vector<mla_tensor> relocs;
    relocs.reserve(reloc_buffers.size());
    for (const auto& [name, buffer] : reloc_buffers) {
        (void)name;
        const uint64_t offset = buffer->get_buf_addr_offset();
        const uint64_t length = buffer->get_allocation_size() - offset;
        relocs.push_back({imported_buffer_id(buffer), 0, offset, length});
    }
    const int rc = mla_update_model_reloc(
        model, static_cast<int>(relocs.size()), relocs.data()
    );
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT relocation failed for {}: rc={}", path_for(_model_idx), rc
        ));
    }
}

void MLAModelWithBuffer::update_reloc(const std::vector<MLARelocBinding>& relocs) {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    load();
    mla_model_p model = model_for(_model_idx);
    const auto lengths = mla_get_ifm_ofm_len_vector(model).reloc_len_array;
    if (lengths.size() != relocs.size()) {
        throw std::invalid_argument(fmt::format(
            "Relocation count for {} is {}, model expects {}",
            path_for(_model_idx), relocs.size(), lengths.size()
        ));
    }
    std::vector<mla_tensor> tensors;
    tensors.reserve(relocs.size());
    for (std::size_t i = 0; i < relocs.size(); ++i) {
        const auto& reloc = relocs[i];
        // A size mismatch means the record's section order differs from the model's ports.
        if (reloc.size < static_cast<uint64_t>(lengths[i])) {
            throw std::invalid_argument(fmt::format(
                "Reloc port {} of {} needs {} bytes, binding has {}",
                i, path_for(_model_idx), lengths[i], reloc.size
            ));
        }
        if (reloc.offset + reloc.size > reloc.buffer->get_allocation_size()) {
            throw std::out_of_range("Reloc binding exceeds its buffer");
        }
        tensors.push_back({imported_buffer_id(reloc.buffer), 0, reloc.offset, reloc.size});
    }
    const int rc = mla_update_model_reloc(model, static_cast<int>(tensors.size()), tensors.data());
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT relocation failed for {}: rc={} ({})",
            path_for(_model_idx), rc, std::strerror(rc < 0 ? -rc : rc)
        ));
    }
}

void MLAModelWithBuffer::load_all_models(std::optional<std::filesystem::path> relative_dir) {
    load_all_models(std::move(relative_dir), nullptr);
}

void MLAModelWithBuffer::load_all_models(
    std::optional<std::filesystem::path> relative_dir,
    FileProvider* files
) {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    require_handle();

    // Deferred assets require serial fetch -> load -> evict, not batch loading.
    const bool must_pull = files != nullptr && files->pulls_files();
    if (!_disable_parallel_load && !must_pull) {
        std::map<std::filesystem::path, uint16_t> batch_paths;
        for (const auto& [path, index] : state.path_to_index) {
            if (path != state.paths[index]) continue;  // extra instance, loaded below
            if (!state.models[index] && path_matches_family(path, relative_dir)) {
                if (batch_paths.size() > std::numeric_limits<uint16_t>::max()) {
                    throw std::overflow_error("MLA model batch exceeds MLA-RT index range");
                }
                batch_paths.emplace(path, static_cast<uint16_t>(batch_paths.size()));
            }
        }
        if (!batch_paths.empty()) {
            std::vector<mla_model_p> batch_models(batch_paths.size(), nullptr);
            mla_load_model_multi(state.handle, batch_paths, batch_models, std::nullopt);
            for (const auto& [path, batch_index] : batch_paths) {
                auto model = batch_models[batch_index];
                if (!model) {
                    throw std::runtime_error(fmt::format(
                        "MLA-RT bulk load failed for model: {}", path
                    ));
                }
                state.models[state.path_to_index.at(path)] = model;
                spdlog::info("Loaded model: {}", path);
            }
        }
        for (const auto& [key, index] : state.path_to_index) {
            const auto& path = state.paths[index];
            if (key == path || state.models[index] || !path_matches_family(path, relative_dir)) {
                continue;
            }
            state.models[index] = mla_load_model(state.handle, path.c_str());
            if (!state.models[index]) {
                throw std::runtime_error(fmt::format("MLA-RT failed to load model: {}", key));
            }
        }
        return;
    }

    for (const auto& [key, index] : state.path_to_index) {
        const auto& path = state.paths[index];
        if (state.models[index] || !path_matches_family(path, relative_dir)) continue;
        if (files) files->fetch(path);           // pull over PCIe (no-op on disk)
        state.models[index] = mla_load_model(state.handle, path.c_str());
        if (!state.models[index]) {
            // Leave the fetched file in place for debugging on failure.
            throw std::runtime_error(fmt::format("MLA-RT failed to load model: {}", path));
        }
        // Extra instances of one ELF share the file: evict after its last instance loads.
        const bool last_user = std::none_of(
            state.path_to_index.begin(), state.path_to_index.end(),
            [&](const auto& other) {
                return state.paths[other.second] == path && !state.models[other.second];
            }
        );
        if (files && last_user) files->evict(path);  // delete disk copy (no-op on disk)
        spdlog::info("Loaded model: {}", key);
    }
}

void MLAModelWithBuffer::free_all_models(
    std::optional<std::filesystem::path> relative_dir
) {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    for (std::size_t i = 0; i < state.models.size(); ++i) {
        if (!state.models[i] || !path_matches_family(state.paths[i], relative_dir)) continue;
        mla_free_model(state.models[i]);
        state.models[i] = nullptr;
    }
}

void MLAModelWithBuffer::_bind_ifm(
    uint8_t index,
    MLABuffer* buffer,
    std::initializer_list<uint32_t> begins
) {
    _ifms.at(index)._bind(buffer, begins);
}

void MLAModelWithBuffer::_bind_ofm(
    uint8_t index,
    MLABuffer* buffer,
    std::initializer_list<uint32_t> begins
) {
    _ofms.at(index)._bind(buffer, begins);
}

void MLAModelWithBuffer::_debug_inouts(
    const std::string& name,
    std::map<uint8_t, MLABufferSlice>* fm_map_ptr
) {
    if (!_print_inouts && !_save_inouts) return;
    auto& defaults = name == "ifm" ? _ifms : _ofms;
    validate_override_indices(fm_map_ptr, defaults.size(), name.c_str());
    for (std::size_t i = 0; i < defaults.size(); ++i) {
        const auto& slice = effective_slice(defaults, fm_map_ptr, i);
        MLABuffer* buffer = effective_buffer(defaults, slice, i);
        const auto view = materialize_view(buffer, slice);

        if (_print_inouts) {
            std::ostringstream output;
            output << path_for(_model_idx) << '\n' << name << i << ' ' << view;
            spdlog::info("{}", output.str());
        }
        if (_save_inouts && (
                _save_inout_filter.empty()
                || std::regex_search(
                    path_for(_model_idx).stem().string(), std::regex(_save_inout_filter)
                ))) {
            const auto directory = std::filesystem::path(_save_inout_dir)
                / path_for(_model_idx).stem() / fmt::format("{}{}", name, i);
            std::filesystem::create_directories(directory);
            const auto file = directory / fmt::format("{}.bin", count_regular_files(directory));
            try {
                view.to_file(file);
            } catch (const std::out_of_range&) {
                // Strided views (e.g. KV-cache scales) declare the parent's full shape; at a
                // nonzero offset that envelope passes the allocation end. Skip the dump only.
                std::filesystem::remove(file);
                spdlog::warn("Skipped dump of {} {}{}: strided view", path_for(_model_idx).stem().string(), name, i);
            }
        }
    }
}

}
}
