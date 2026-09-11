#include <dlfcn.h>

#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>
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
    std::filesystem::path model_path;
    std::vector<mla_fd_tensor> ifms;
    std::vector<mla_fd_tensor> ofms;
};

struct MlaRuntimeState {
    std::mutex registry_mutex;
    std::recursive_mutex execution_mutex;
    mla_handle_p handle = nullptr;
    std::map<std::filesystem::path, std::size_t> path_to_index;
    std::vector<std::filesystem::path> paths;
    std::vector<mla_model_p> models;
};

MlaRuntimeState& runtime_state() {
    static MlaRuntimeState state;
    return state;
}

thread_local std::vector<QueuedRun> queued_runs;

mla_handle_p require_handle() {
    auto& state = runtime_state();
    if (!state.handle) {
        throw std::runtime_error("MLA-RT is not connected; call connect() first");
    }
    return state.handle;
}

bool path_is_under(
    const std::filesystem::path& model_path,
    const std::optional<std::filesystem::path>& directory
) {
    if (!directory) return true;
    const auto model = std::filesystem::absolute(model_path).lexically_normal();
    const auto root = std::filesystem::absolute(*directory).lexically_normal();
    auto model_it = model.begin();
    for (auto root_it = root.begin(); root_it != root.end(); ++root_it, ++model_it) {
        if (model_it == model.end() || *model_it != *root_it) return false;
    }
    return true;
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

std::vector<mla_fd_tensor> make_fd_bindings(
    const std::vector<MLABufferSlice>& defaults,
    const std::map<uint8_t, MLABufferSlice>* overrides,
    const char* kind
) {
    validate_override_indices(overrides, defaults.size(), kind);
    if (defaults.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(fmt::format("{} binding count exceeds MLA-RT", kind));
    }
    std::vector<mla_fd_tensor> bindings;
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
        bindings.push_back({buffer->get_dmabuf_fd(), offset, length});
    }
    return bindings;
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

    auto ifms = make_fd_bindings(_ifms, ifm_map_ptr, "IFM");
    auto ofms = make_fd_bindings(_ofms, ofm_map_ptr, "OFM");
    mla_job_h job = nullptr;
    int rc = mla_submit_async_fd(
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
    auto ifms = make_fd_bindings(_ifms, ifm_map_ptr, "IFM");
    auto ofms = make_fd_bindings(_ofms, ofm_map_ptr, "OFM");
    QueuedRun queued;
    queued.model_index = _model_idx;
    queued.model_path = path_for(_model_idx);
    queued.ifms = std::move(ifms);
    queued.ofms = std::move(ofms);
    queued_runs.push_back(std::move(queued));
}

void MLAModelWithBuffer::run_queue() {
    if (!_enable_queue || queued_runs.empty()) return;
    auto& state = runtime_state();
    std::lock_guard execution_lock(state.execution_mutex);
    auto runs = std::move(queued_runs);
    queued_runs.clear();
    uint64_t total_tile_us = 0;
    // LLiMa stages form an ordered dependency chain and reuse the same
    // dma-bufs. Consume each job before submitting the next one so implicit
    // dma-buf fences cannot make an earlier stage wait on a later writer.
    // The completion boundary also gives the kernel a scheduling point for
    // latency-sensitive MLA contexts owned by other processes.
    for (const auto& queued : runs) {
        mla_model_p model = model_for(queued.model_index);
        if (!model) {
            throw std::runtime_error(fmt::format(
                "MLA model was released before queued execution: {}",
                queued.model_path
            ));
        }
        mla_job_h job = nullptr;
        int rc = mla_submit_async_fd(
            model,
            static_cast<int>(queued.ifms.size()), queued.ifms.data(),
            static_cast<int>(queued.ofms.size()), queued.ofms.data(),
            &job
        );
        if (rc != 0) {
            throw std::runtime_error(fmt::format(
                "MLA-RT queued submit failed for {}: rc={} ({})",
                queued.model_path, rc, std::strerror(rc < 0 ? -rc : rc)
            ));
        }
        uint64_t tile_us = 0;
        rc = mla_wait(job, &tile_us, nullptr);
        if (rc != 0) {
            throw std::runtime_error(fmt::format(
                "MLA-RT queued wait failed for {}: rc={} ({})",
                queued.model_path, rc, std::strerror(rc < 0 ? -rc : rc)
            ));
        }
        total_tile_us += tile_us;
    }
    if (_profile) {
        spdlog::info(
            "MLA-RT queue: {} jobs, total tile={} us", runs.size(), total_tile_us
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
        std::map<std::filesystem::path, uint16_t> selected;
        for (const auto& [path, index] : state.path_to_index) {
            if (index > std::numeric_limits<uint16_t>::max()) {
                throw std::overflow_error("MLA model registry exceeds MLA-RT index range");
            }
            if (!state.models[index] && path_is_under(path, relative_dir)) {
                selected.emplace(path, static_cast<uint16_t>(index));
            }
        }
        if (!selected.empty()) {
            mla_load_model_multi(state.handle, selected, state.models, std::nullopt);
            for (const auto& [path, index] : selected) {
                if (!state.models[index]) {
                    throw std::runtime_error(fmt::format(
                        "MLA-RT bulk load failed for model: {}", path
                    ));
                }
                spdlog::info("Loaded model: {}", path);
            }
        }
        return;
    }

    for (const auto& [path, index] : state.path_to_index) {
        if (state.models[index] || !path_is_under(path, relative_dir)) continue;
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
        if (!state.models[i] || !path_is_under(state.paths[i], relative_dir)) continue;
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
