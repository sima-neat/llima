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
#include <unordered_map>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <simaai/gst-api.h>
#include <spdlog/spdlog.h>

#include "mla_model.hpp"

namespace simaai {
namespace llima {
namespace {

struct QueuedRun {
    std::size_t model_index = 0;
    std::vector<mla_tensor> ifms;
    std::vector<mla_tensor> ofms;
};

struct ImportedBuffer {
    uint64_t generation = 0;
    uint32_t id = 0;
};

struct MlaRuntimeState {
    std::mutex registry_mutex;
    std::recursive_mutex execution_mutex;
    mla_handle_p handle = nullptr;
    std::map<std::filesystem::path, std::size_t> path_to_index;
    std::vector<std::filesystem::path> paths;
    std::vector<mla_model_p> models;
    std::unordered_map<const MLABuffer*, ImportedBuffer> imported_buffers;
};

MlaRuntimeState& runtime_state() {
    static MlaRuntimeState state;
    return state;
}

thread_local std::vector<QueuedRun> queued_runs;
thread_local std::size_t queued_run_count = 0;

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

    auto found = state.imported_buffers.find(buffer);
    if (found != state.imported_buffers.end() &&
        found->second.generation == generation) {
        return found->second.id;
    }
    if (found != state.imported_buffers.end()) {
        const int rc = mla_release_dmabuf(require_handle(), found->second.id);
        if (rc != 0) {
            throw std::runtime_error(fmt::format(
                "MLA-RT failed to release stale DMA-BUF registration: rc={} ({})",
                rc, std::strerror(rc < 0 ? -rc : rc)
            ));
        }
        state.imported_buffers.erase(found);
    }

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
    state.imported_buffers.emplace(buffer, ImportedBuffer{generation, id});
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
    for (const auto& [buffer, imported] : state.imported_buffers) {
        (void)buffer;
        const int rc = mla_release_dmabuf(state.handle, imported.id);
        if (rc != 0) {
            spdlog::error(
                "MLA-RT failed to release DMA-BUF registration {}: rc={} ({})",
                imported.id, rc, std::strerror(rc < 0 ? -rc : rc)
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
    std::vector<MLABufferSlice> ofms
) : _ifms(std::move(ifms)), _ofms(std::move(ofms)) {
    model_path = std::filesystem::absolute(model_path).lexically_normal();
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error(fmt::format("Model file does not exist: {}", model_path));
    }
    auto& state = runtime_state();
    std::lock_guard lock(state.registry_mutex);
    const auto [it, inserted] = state.path_to_index.emplace(model_path, state.paths.size());
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
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    load();
    const std::size_t slot = queued_run_count;
    if (slot == queued_runs.size()) queued_runs.emplace_back();
    auto& queued = queued_runs[slot];
    queued.model_index = _model_idx;
    make_bindings(_ifms, ifm_map_ptr, "IFM", queued.ifms);
    make_bindings(_ofms, ofm_map_ptr, "OFM", queued.ofms);
    ++queued_run_count;
}

void MLAModelWithBuffer::run_queue() {
    if (!_enable_queue || queued_run_count == 0) return;
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    const std::size_t run_count = queued_run_count;
    queued_run_count = 0;
    const auto queue_start = std::chrono::steady_clock::now();
    uint64_t total_tile_us = 0;
    constexpr std::size_t queue_ahead_depth = 256;
    std::deque<std::pair<mla_job_h, std::size_t>> inflight;
    std::size_t next = 0;
    std::string first_error;

    // Submit the complete model sequence before reaping, matching MLA-RT's
    // legacy pipelined multi-model behavior while retaining one kernel
    // scheduling boundary per model. The public async pool is capped at 256;
    // longer queues advance as completed slots become available.
    while (next < run_count || !inflight.empty()) {
        while (first_error.empty() && next < run_count &&
               inflight.size() < queue_ahead_depth) {
            const auto& queued = queued_runs[next];
            mla_model_p model = model_for(queued.model_index);
            if (!model) {
                first_error = fmt::format(
                    "MLA model was released before queued execution: {}",
                    path_for(queued.model_index)
                );
                break;
            }
            mla_job_h job = nullptr;
            int rc = mla_submit_async(
                model,
                static_cast<int>(queued.ifms.size()), queued.ifms.data(),
                static_cast<int>(queued.ofms.size()), queued.ofms.data(),
                &job
            );
            if (rc != 0) {
                first_error = fmt::format(
                    "MLA-RT queued submit failed for {}: rc={} ({})",
                    path_for(queued.model_index), rc,
                    std::strerror(rc < 0 ? -rc : rc)
                );
                break;
            }
            inflight.emplace_back(job, queued.model_index);
            ++next;
        }

        if (inflight.empty()) break;
        const auto [job, model_index] = inflight.front();
        inflight.pop_front();
        uint64_t tile_us = 0;
        const int rc = mla_wait(job, &tile_us, nullptr);
        if (rc != 0 && first_error.empty()) {
            first_error = fmt::format(
                "MLA-RT queued wait failed for {}: rc={} ({})",
                path_for(model_index), rc, std::strerror(rc < 0 ? -rc : rc)
            );
        }
        total_tile_us += tile_us;
    }
    if (!first_error.empty()) throw std::runtime_error(first_error);
    if (_profile) {
        const auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - queue_start
        ).count();
        spdlog::info(
            "MLA-RT queue: {} jobs, total tile={} us, wall={} us",
            run_count, total_tile_us, wall_us
        );
    }
}

void MLAModelWithBuffer::update_reloc(
    const std::map<std::string, uint64_t>& reloc_addr_map
) {
    if (reloc_addr_map.empty()) return;
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    load();
    mla_model_p model = model_for(_model_idx);
    const auto lengths = mla_get_ifm_ofm_len_vector(model).reloc_len_array;
    if (lengths.size() != reloc_addr_map.size()) {
        throw std::invalid_argument(fmt::format(
            "Relocation count for {} is {}, model expects {}",
            path_for(_model_idx), reloc_addr_map.size(), lengths.size()
        ));
    }
    std::vector<DADDR_LEN> relocs;
    relocs.reserve(reloc_addr_map.size());
    std::size_t index = 0;
    for (const auto& [name, address] : reloc_addr_map) {
        (void)name;
        relocs.emplace_back(address, lengths[index++]);
    }
    const int rc = mla_update_model_rel(
        model, 0, nullptr, 0, nullptr,
        static_cast<int>(relocs.size()), relocs.data()
    );
    if (rc <= 0) {
        throw std::runtime_error(fmt::format(
            "MLA-RT relocation failed for {}: rc={}", path_for(_model_idx), rc
        ));
    }
}

void MLAModelWithBuffer::load_all_models(
    std::optional<std::filesystem::path> relative_dir
) {
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    std::lock_guard registry_lock(state.registry_mutex);
    require_handle();

    if (!_disable_parallel_load) {
        std::map<std::filesystem::path, uint16_t> batch_paths;
        for (const auto& [path, index] : state.path_to_index) {
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
        return;
    }

    for (const auto& [path, index] : state.path_to_index) {
        if (state.models[index] || !path_matches_family(path, relative_dir)) continue;
        state.models[index] = mla_load_model(state.handle, path.c_str());
        if (!state.models[index]) {
            throw std::runtime_error(fmt::format("MLA-RT failed to load model: {}", path));
        }
        spdlog::info("Loaded model: {}", path);
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
        if (_save_inouts) {
            const auto directory = std::filesystem::path(_save_inout_dir)
                / path_for(_model_idx).stem() / fmt::format("{}{}", name, i);
            std::filesystem::create_directories(directory);
            const auto file = directory / fmt::format("{}.bin", count_regular_files(directory));
            view.to_file(file);
        }
    }
}

}
}
