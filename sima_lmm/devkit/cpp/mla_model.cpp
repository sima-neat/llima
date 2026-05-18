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


#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <sstream>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "mla_model.hpp"


namespace simaai {
namespace llima {

void connect_mla_rt(const std::vector<std::string>& args) {
    if (MLAModelWithBuffer::_dispatcher) return;

    if (!args.empty()) {
        spdlog::warn("MLA dispatcher mode ignores mla_rt_args: [{}]", fmt::join(args, ", "));
    }

    MLAModelWithBuffer::_dispatcher =
        simaaidispatcher::DispatcherFactory::getDispatcher(
            simaaidispatcher::DispatcherFactory::MLASHM
        );
    if (!MLAModelWithBuffer::_dispatcher) {
        throw std::runtime_error("Failed to acquire MLASHM dispatcher");
    }
    spdlog::info("Connected MLA runtime through MLASHM dispatcher");
}

void disconnect_mla_rt() {
    MLAModelWithBuffer::free_all_models();

    if (MLAModelWithBuffer::_dispatcher) {
        simaaidispatcher::DispatcherFactory::releaseDispatcher(
            simaaidispatcher::DispatcherFactory::MLASHM
        );
        MLAModelWithBuffer::_dispatcher = nullptr;
    }
}


std::map<std::filesystem::path, uint16_t> MLAModelWithBuffer::_unique_model_path_to_idx_map;
std::vector<std::filesystem::path> MLAModelWithBuffer::_unique_model_paths;
std::vector<mla_model_p> MLAModelWithBuffer::_unique_model_ptrs;
std::vector<simaaidispatcher::JobMLA> MLAModelWithBuffer::_queue;
simaaidispatcher::DispatcherBase* MLAModelWithBuffer::_dispatcher = nullptr;


MLAModelWithBuffer::MLAModelWithBuffer(
    std::filesystem::path model_path,
    std::vector<MLABufferSlice> ifms,
    std::vector<MLABufferSlice> ofms
) : _ifms(std::move(ifms)), _ofms(std::move(ofms)) {
    if (_unique_model_path_to_idx_map.contains(model_path)) {
        _model_idx = _unique_model_path_to_idx_map[model_path];
    } else if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error(fmt::format("Model file does not exist: {}", model_path));
    } else {
        _model_idx = _unique_model_ptrs.size();
        _unique_model_path_to_idx_map[model_path] = _model_idx;
        _unique_model_paths.emplace_back(model_path);
        _unique_model_ptrs.emplace_back(nullptr);
    }

    for (const auto& ifm: _ifms) {
        _ifm_buf_addrs.emplace_back(ifm.get_buf_addr());
    }
    for (const auto& ofm: _ofms) {
        _ofm_buf_addrs.emplace_back(ofm.get_buf_addr());
    }
}


void MLAModelWithBuffer::load() {
    if (_unique_model_ptrs[_model_idx]) return;
    auto* dispatcher = _get_dispatcher();
    _unique_model_ptrs[_model_idx] = dispatcher->load(_unique_model_paths[_model_idx].string());
    if (!_unique_model_ptrs[_model_idx]) {
        throw std::runtime_error(fmt::format(
            "Failed to load model through MLASHM dispatcher: {} ({})",
            _unique_model_paths[_model_idx],
            dispatcher->lastErrorString()
        ));
    }
    spdlog::info("Loaded model: {}", _unique_model_paths[_model_idx]);
}


void MLAModelWithBuffer::free() {
    if (!_unique_model_ptrs[_model_idx]) return;
    auto* dispatcher = _get_dispatcher();
    const int rc = dispatcher->release(_unique_model_ptrs[_model_idx]);
    if (rc != 0) {
        spdlog::error(
            "Failed to release model through MLASHM dispatcher: {} ({})",
            _unique_model_paths[_model_idx],
            dispatcher->lastErrorString()
        );
    }
    _unique_model_ptrs[_model_idx] = nullptr;
}


simaaidispatcher::DispatcherBase* MLAModelWithBuffer::_get_dispatcher() {
    if (!_dispatcher) {
        _dispatcher = simaaidispatcher::DispatcherFactory::getDispatcher(
            simaaidispatcher::DispatcherFactory::MLASHM
        );
    }
    if (!_dispatcher) {
        throw std::runtime_error("MLASHM dispatcher is unavailable");
    }
    return _dispatcher;
}


simaaidispatcher::RuntimeBufferBinding MLAModelWithBuffer::_make_binding(
    const std::vector<MLABufferSlice>& default_slices,
    const MLABufferSlice& effective_slice,
    uint8_t logical_idx,
    simaaidispatcher::RuntimeBindingRole role
) {
    if (logical_idx >= default_slices.size()) {
        throw std::runtime_error(fmt::format(
            "MLASHM binding index {} is out of range (count={})",
            static_cast<unsigned>(logical_idx),
            default_slices.size()
        ));
    }

    MLABuffer* base = effective_slice.get_buf_ptr()
        ? effective_slice.get_buf_ptr()
        : default_slices[logical_idx].get_buf_ptr();
    if (!base) {
        const char* kind = role == simaaidispatcher::RuntimeBindingRole::Input ? "IFM" : "OFM";
        throw std::runtime_error(fmt::format(
            "MLASHM dispatcher does not support null {} binding at logical index {}",
            kind,
            static_cast<unsigned>(logical_idx)
        ));
    }

    const uint64_t base_phys = base->get_buf_addr();
    const uint64_t slice_phys = effective_slice.get_buf_ptr()
        ? effective_slice.get_buf_addr()
        : default_slices[logical_idx].get_buf_addr(effective_slice.get_buf_begins());
    if (slice_phys < base_phys) {
        throw std::runtime_error(fmt::format(
            "MLASHM binding physical address is before base allocation: index={} slice=0x{:x} base=0x{:x}",
            static_cast<unsigned>(logical_idx),
            slice_phys,
            base_phys
        ));
    }

    simaaidispatcher::RuntimeBufferBinding binding;
    binding.role = role;
    binding.logicalIndex = logical_idx;
    binding.physicalIndex = logical_idx;
    binding.allocatorIndex = -1;
    binding.byteOffset = slice_phys - base_phys;
    binding.sizeBytes = base->get_buf_len(effective_slice.get_buf_shapes());
    binding.memory = base->get_simaai_memory();
    if (!binding.memory) {
        throw std::runtime_error(fmt::format(
            "MLASHM binding has no allocated simaai memory at logical index {}",
            static_cast<unsigned>(logical_idx)
        ));
    }
    return binding;
}


std::vector<simaaidispatcher::RuntimeBufferBinding> MLAModelWithBuffer::_make_bindings(
    const std::vector<MLABufferSlice>& default_slices,
    std::map<uint8_t, MLABufferSlice>* override_map_ptr,
    simaaidispatcher::RuntimeBindingRole role
) const {
    std::vector<simaaidispatcher::RuntimeBufferBinding> bindings;
    bindings.reserve(default_slices.size());
    for (uint32_t i = 0; i < default_slices.size(); ++i) {
        if (i > UINT8_MAX) {
            throw std::runtime_error("MLASHM dispatcher binding index exceeds uint8_t range");
        }
        const MLABufferSlice* effective_slice = &default_slices[i];
        if (override_map_ptr) {
            const auto override_it = override_map_ptr->find(static_cast<uint8_t>(i));
            if (override_it != override_map_ptr->end()) {
                effective_slice = &override_it->second;
            }
        }
        bindings.emplace_back(
            _make_binding(default_slices, *effective_slice, static_cast<uint8_t>(i), role)
        );
    }
    return bindings;
}


simaaidispatcher::JobMLA MLAModelWithBuffer::_make_job(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) const {
    simaaidispatcher::JobMLA job;
    job.handle = _unique_model_ptrs[_model_idx];
    job.batchSize = 1;
    job.batchModel = 1;
    job.cb = nullptr;
    job.timeout = std::chrono::duration<double>(0);
    job.priority = 0;
    job.bindingTable.inputBindings = _make_bindings(
        _ifms,
        ifm_map_ptr,
        simaaidispatcher::RuntimeBindingRole::Input
    );
    job.bindingTable.outputBindings = _make_bindings(
        _ofms,
        ofm_map_ptr,
        simaaidispatcher::RuntimeBindingRole::Output
    );
    return job;
}


void MLAModelWithBuffer::_update_buf_addrs(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    if (ifm_map_ptr && ifm_map_ptr->size()) {
        for (const auto& [idx, buf_slice]: *ifm_map_ptr) {
            auto buf_ptr = buf_slice.get_buf_ptr();
            if (buf_ptr) {
                _ifm_buf_addrs[idx] = buf_slice.get_buf_addr();
            } else {
                _ifm_buf_addrs[idx] = _ifms[idx].get_buf_addr(buf_slice.get_buf_begins());
            }
        }
    }

    if (ofm_map_ptr && ofm_map_ptr->size()) {
        for (const auto& [idx, buf_slice]: *ofm_map_ptr) {
            auto buf_ptr = buf_slice.get_buf_ptr();
            if (buf_ptr) {
                _ofm_buf_addrs[idx] = buf_slice.get_buf_addr();
            } else {
                _ofm_buf_addrs[idx] = _ofms[idx].get_buf_addr(buf_slice.get_buf_begins());
            }
        }
    }
}


void MLAModelWithBuffer::run(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    load();
    _update_buf_addrs(ifm_map_ptr, ofm_map_ptr);

    _debug_inouts("ifm", ifm_map_ptr);

    auto job = _make_job(ifm_map_ptr, ofm_map_ptr);
    auto* dispatcher = _get_dispatcher();
    const int rc = dispatcher->run(job);
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLASHM dispatcher run failed for {}: rc={} ({})",
            _unique_model_paths[_model_idx],
            rc,
            dispatcher->lastErrorString()
        ));
    }

    _debug_inouts("ofm", ofm_map_ptr);
}


void MLAModelWithBuffer::add_to_queue(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    if (!MLAModelWithBuffer::_enable_queue) {
        // Run queue is disabled. Run the model immediately.
        return run(ifm_map_ptr, ofm_map_ptr);
    }

    load();
    _update_buf_addrs(ifm_map_ptr, ofm_map_ptr);
    MLAModelWithBuffer::_queue.push_back(_make_job(ifm_map_ptr, ofm_map_ptr));
}


void MLAModelWithBuffer::run_queue() {
    if (!MLAModelWithBuffer::_enable_queue || MLAModelWithBuffer::_queue.empty()) return;

    auto* dispatcher = _get_dispatcher();
    try {
        const int rc = dispatcher->runQueue(MLAModelWithBuffer::_queue);
        if (rc != 0) {
            throw std::runtime_error(fmt::format(
                "MLASHM dispatcher runQueue failed: rc={} ({})",
                rc,
                dispatcher->lastErrorString()
            ));
        }
    } catch (...) {
        MLAModelWithBuffer::_queue.clear();
        throw;
    }
    MLAModelWithBuffer::_queue.clear();
}


void MLAModelWithBuffer::update_reloc(const std::map<std::string, uint64_t>& reloc_addr_map) {
    if (reloc_addr_map.empty()) return;
    spdlog::warn(
        "MLA dispatcher backend does not support update_reloc yet; ignoring relocation for {}: {}",
        _unique_model_paths[_model_idx],
        reloc_addr_map
    );
}


void MLAModelWithBuffer::load_all_models(
    bool do_parallel_load, std::optional<std::filesystem::path> relative_dir
) {
    auto* dispatcher = _get_dispatcher();
    std::vector<std::filesystem::path> file_names;
    std::vector<uint16_t> indices;
    for (const auto& [file_name, idx]: _unique_model_path_to_idx_map) {
        if (relative_dir.has_value() && !file_name.string().starts_with(relative_dir.value().string())) {
            continue;
        }
        if (_unique_model_ptrs[idx]) {
            continue;
        }
        file_names.push_back(file_name);
        indices.push_back(idx);
    }

    if (file_names.empty()) {
        return;
    }

    if (do_parallel_load) {
        std::vector<std::string> paths;
        paths.reserve(file_names.size());
        for (const auto& file_name: file_names) {
            paths.push_back(file_name.string());
        }
        auto handles = dispatcher->loadMany(paths);
        if (handles.size() != file_names.size()) {
            throw std::runtime_error(fmt::format(
                "Bulk MLASHM model load returned {} handles for {} models ({})",
                handles.size(),
                file_names.size(),
                dispatcher->lastErrorString()
            ));
        }
        for (std::size_t i = 0; i < file_names.size(); ++i) {
            _unique_model_ptrs[indices[i]] = handles[i];
            if (!_unique_model_ptrs[indices[i]]) {
                throw std::runtime_error(fmt::format(
                    "Failed to bulk load model through MLASHM dispatcher: {} ({})",
                    file_names[i],
                    dispatcher->lastErrorString()
                ));
            }
            spdlog::info("Loaded model: {}", file_names[i]);
        }
        return;
    }

    for (std::size_t i = 0; i < file_names.size(); ++i) {
        _unique_model_ptrs[indices[i]] = dispatcher->load(file_names[i].string());
        if (!_unique_model_ptrs[indices[i]]) {
            throw std::runtime_error(fmt::format(
                "Failed to load model through MLASHM dispatcher: {} ({})",
                file_names[i],
                dispatcher->lastErrorString()
            ));
        }
        spdlog::info("Loaded model: {}", file_names[i]);
    }
}


void MLAModelWithBuffer::free_all_models(
    std::optional<std::filesystem::path> relative_dir
) {
    if (relative_dir.has_value()) {
        for (size_t i = 0; i < MLAModelWithBuffer::_unique_model_paths.size(); ++i) {
            const auto& model_path = MLAModelWithBuffer::_unique_model_paths[i];
            if (
                MLAModelWithBuffer::_unique_model_ptrs[i]
                && model_path.string().starts_with(relative_dir.value().string())
            ) {
                const int rc = _get_dispatcher()->release(MLAModelWithBuffer::_unique_model_ptrs[i]);
                if (rc != 0) {
                    spdlog::error(
                        "Failed to release model through MLASHM dispatcher: {} ({})",
                        model_path,
                        _get_dispatcher()->lastErrorString()
                    );
                }
                MLAModelWithBuffer::_unique_model_ptrs[i] = nullptr;
            }
        }
    } else {
        for (size_t i = 0; i < MLAModelWithBuffer::_unique_model_paths.size(); ++i) {
            if (!MLAModelWithBuffer::_unique_model_ptrs[i]) {
                continue;
            }
            const int rc = _get_dispatcher()->release(MLAModelWithBuffer::_unique_model_ptrs[i]);
            if (rc != 0) {
                spdlog::error(
                    "Failed to release model through MLASHM dispatcher: {} ({})",
                    MLAModelWithBuffer::_unique_model_paths[i],
                    _get_dispatcher()->lastErrorString()
                );
            }
            MLAModelWithBuffer::_unique_model_ptrs[i] = nullptr;
        }
    }
}


void MLAModelWithBuffer::_debug_inouts(
    const std::string& name, std::map<uint8_t, MLABufferSlice>* fm_map_ptr
) {
    if (_print_inouts) {
        auto& fms = (name == "ifm")? _ifms : _ofms;
        std::ostringstream print_buffer;
        print_buffer << _unique_model_paths[_model_idx] << std::endl;
        for (uint32_t i = 0; i < fms.size(); ++i) {
            print_buffer << name << i << " ";
            if (fm_map_ptr && fm_map_ptr->contains(i)) {
                auto& buf_slice = fm_map_ptr->at(i);
                auto buf_ptr = (
                    buf_slice.get_buf_ptr()? buf_slice.get_buf_ptr() : fms[i].get_buf_ptr()
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
        auto& fms = (name == "ifm")? _ifms : _ofms;
        for (uint32_t i = 0; i < fms.size(); ++i) {
            std::filesystem::path d = (
                _save_inout_dir
                / _unique_model_paths[_model_idx].stem()
                / (name + std::to_string(i))
            );
            std::filesystem::create_directories(d);
            auto num_files = count_regular_files(d);
            std::filesystem::path fn = d / fmt::format("{}.bin", num_files);
            if (fm_map_ptr && fm_map_ptr->contains(i)) {
                auto& buf_slice = fm_map_ptr->at(i);
                auto buf_ptr = (
                    buf_slice.get_buf_ptr()? buf_slice.get_buf_ptr() : fms[i].get_buf_ptr()
                );
                const auto begins = buf_slice.get_buf_begins().value();
                const auto shapes = buf_slice.get_buf_shapes().value();
                MLABufferSlice(buf_ptr, begins, shapes).to_file(fn);
            } else {
                fms[i].to_file(fn);
            }
        }
    }

}


}
}
