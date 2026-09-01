
#include <algorithm>
#include <any>
#include <cstdint>
#include <future>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "mla_model.hpp"


namespace simaai {
namespace llima {

namespace {

struct MlaRunCompletion {
    int32_t rc = 0;
    std::size_t failed_index = simaaidispatcher::DispatcherBase::NoFailedQueueIndex;
    std::string detail;
};

using MlaRunPromise = std::shared_ptr<std::promise<MlaRunCompletion>>;

MlaRunCompletion make_submit_error(
    simaaidispatcher::DispatcherBase* dispatcher,
    int32_t rc
) {
    MlaRunCompletion result;
    result.rc = rc;
    result.detail = dispatcher->lastErrorString();
    return result;
}

MlaRunCompletion wait_for_result(
    simaaidispatcher::DispatcherBase* dispatcher,
    std::future<MlaRunCompletion>& future
) {
    // The completion fires as a callback on the dispatcher thread, so future.get() pays a
    // cross-thread wakeup (block + context switch) every flush. On the hot decode path the
    // main thread has nothing else to do while the MLA runs, so busy-wait for the result
    // instead -- avoids the wakeup latency, matching a synchronous mla_rt call.
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        // spin
    }
    MlaRunCompletion result = future.get();
    if (result.rc != 0) {
        result.detail = dispatcher->lastErrorString();
    }
    return result;
}

MlaRunCompletion submit_prepared_and_wait(
    simaaidispatcher::DispatcherBase* dispatcher,
    simaaidispatcher::JobMLA&& job
) {
    auto promise = std::make_shared<std::promise<MlaRunCompletion>>();
    auto future = promise->get_future();

    job.userData = promise;
    job.cb = [](
        const std::map<std::string, simaai_memory_t*>&,
        int32_t rc,
        std::any user_data
    ) {
        auto completion =
            std::any_cast<MlaRunPromise>(user_data);
        MlaRunCompletion result;
        result.rc = rc;
        completion->set_value(std::move(result));
    };

    const int32_t submit_rc = dispatcher->submitPrepared(simaaidispatcher::Job{std::move(job)});
    if (submit_rc != 0) {
        return make_submit_error(dispatcher, submit_rc);
    }
    return wait_for_result(dispatcher, future);
}

MlaRunCompletion submit_queue_and_wait(
    simaaidispatcher::DispatcherBase* dispatcher,
    simaaidispatcher::DispatcherBase::PreparedMlaPartitionQueueRequest&& request
) {
    auto promise = std::make_shared<std::promise<MlaRunCompletion>>();
    auto future = promise->get_future();

    request.userData = promise;
    request.cb = [](
        int32_t rc,
        std::size_t failed_index,
        const simaaidispatcher::DispatcherBase::ErrorSnapshot&,
        const simaaidispatcher::DispatcherBase::ProfileSnapshot&,
        std::any user_data
    ) {
        auto completion =
            std::any_cast<MlaRunPromise>(user_data);
        MlaRunCompletion result;
        result.rc = rc;
        result.failed_index = failed_index;
        completion->set_value(std::move(result));
    };

    const int32_t submit_rc = dispatcher->submitPreparedMlaPartitionQueue(std::move(request));
    if (submit_rc != 0) {
        return make_submit_error(dispatcher, submit_rc);
    }
    return wait_for_result(dispatcher, future);
}

}

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
thread_local simaaidispatcher::DispatcherBase::PreparedMlaPartitionQueueRequest
    MLAModelWithBuffer::_queue_request;
thread_local std::vector<simaaidispatcher::PreparedMlaPlan>
    MLAModelWithBuffer::_queued_plans;
thread_local std::vector<mla_model_p> MLAModelWithBuffer::_queued_handles;
thread_local std::size_t MLAModelWithBuffer::_queued_plan_count = 0;
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

    _prepared_plan.mode = simaaidispatcher::PreparedMlaPlan::DispatchMode::MultiIoSingleBatch;
    _prepared_plan.ifm_count = static_cast<int>(_ifms.size());
    _prepared_plan.ofm_count = static_cast<int>(_ofms.size());
    _prepared_plan.batch_size = 1;
    _prepared_plan.batch_model = 1;
    _prepared_plan.ifm_len.resize(_ifms.size());
    _prepared_plan.ofm_len.resize(_ofms.size());
    _prepared_plan.ifm_paddr.resize(_ifms.size());
    _prepared_plan.ofm_paddr.resize(_ofms.size());
}


void MLAModelWithBuffer::load() {
    if (_unique_model_ptrs[_model_idx]) return;
    auto* dispatcher = _get_dispatcher();
    const auto model_path = std::filesystem::absolute(_unique_model_paths[_model_idx]);
    _unique_model_ptrs[_model_idx] = dispatcher->load(model_path.string());
    if (!_unique_model_ptrs[_model_idx]) {
        throw std::runtime_error(fmt::format(
            "Failed to load model through MLASHM dispatcher: {} ({})",
            model_path,
            dispatcher->lastErrorString()
        ));
    }
    spdlog::info("Loaded model: {}", model_path);
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


void MLAModelWithBuffer::_prepare_run(
    simaaidispatcher::PreparedMlaPlan& plan,
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    auto patch_io = [](
        const std::vector<MLABufferSlice>& default_slices,
        std::map<uint8_t, MLABufferSlice>* override_map_ptr,
        std::vector<uint64_t>& paddr,
        std::vector<int>& len
    ) {
        for (uint32_t i = 0; i < default_slices.size(); ++i) {
            const MLABufferSlice* effective_slice = &default_slices[i];
            if (override_map_ptr) {
                const auto override_it = override_map_ptr->find(static_cast<uint8_t>(i));
                if (override_it != override_map_ptr->end()) {
                    effective_slice = &override_it->second;
                }
            }
            MLABuffer* base = effective_slice->get_buf_ptr()
                ? effective_slice->get_buf_ptr()
                : default_slices[i].get_buf_ptr();
            paddr[i] = effective_slice->get_buf_ptr()
                ? effective_slice->get_buf_addr()
                : default_slices[i].get_buf_addr(effective_slice->get_buf_begins());
            len[i] = static_cast<int>(base->get_buf_len(effective_slice->get_buf_shapes()));
        }
    };

    plan.mode = simaaidispatcher::PreparedMlaPlan::DispatchMode::MultiIoSingleBatch;
    plan.ifm_count = static_cast<int>(_ifms.size());
    plan.ofm_count = static_cast<int>(_ofms.size());
    plan.batch_size = 1;
    plan.batch_model = 1;
    plan.ifm_len.resize(_ifms.size());
    plan.ofm_len.resize(_ofms.size());
    plan.ifm_paddr.resize(_ifms.size());
    plan.ofm_paddr.resize(_ofms.size());

    patch_io(_ifms, ifm_map_ptr, plan.ifm_paddr, plan.ifm_len);
    patch_io(_ofms, ofm_map_ptr, plan.ofm_paddr, plan.ofm_len);
}


simaaidispatcher::PreparedMlaRunRef MLAModelWithBuffer::_make_run_ref(
    simaaidispatcher::PreparedMlaPlan& plan,
    mla_model_p handle
) {
    simaaidispatcher::PreparedMlaRunRef run;
    run.handle = handle;
    run.ifm_paddr = plan.ifm_paddr.data();
    run.ifm_len = plan.ifm_len.data();
    run.ifm_count = static_cast<uint16_t>(plan.ifm_paddr.size());
    run.ofm_paddr = plan.ofm_paddr.data();
    run.ofm_len = plan.ofm_len.data();
    run.ofm_count = static_cast<uint16_t>(plan.ofm_paddr.size());
    return run;
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


void MLAModelWithBuffer::run(
    std::map<uint8_t, MLABufferSlice>* ifm_map_ptr,
    std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    load();

    _debug_inouts("ifm", ifm_map_ptr);

    auto* dispatcher = _get_dispatcher();
    _prepare_run(_prepared_plan, ifm_map_ptr, ofm_map_ptr);
    const auto run_ref = _make_run_ref(_prepared_plan, _unique_model_ptrs[_model_idx]);

    simaaidispatcher::JobMLA job;
    job.handle = run_ref.handle;
    job.batchSize = 1;
    job.batchModel = 1;
    job.prepared = &_prepared_plan;
    auto result = submit_prepared_and_wait(dispatcher, std::move(job));
    if (result.rc != 0) {
        throw std::runtime_error(fmt::format(
            "MLASHM dispatcher run failed for {}: rc={} ({})",
            _unique_model_paths[_model_idx],
            result.rc,
            result.detail.empty() ? dispatcher->lastErrorString() : result.detail
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
    const std::size_t slot = MLAModelWithBuffer::_queued_plan_count;
    if (slot == MLAModelWithBuffer::_queued_plans.size()) {
        MLAModelWithBuffer::_queued_plans.emplace_back();
        MLAModelWithBuffer::_queued_handles.emplace_back(nullptr);
    }
    _prepare_run(MLAModelWithBuffer::_queued_plans[slot], ifm_map_ptr, ofm_map_ptr);
    MLAModelWithBuffer::_queued_handles[slot] = _unique_model_ptrs[_model_idx];
    ++MLAModelWithBuffer::_queued_plan_count;
}


void MLAModelWithBuffer::run_queue() {
    if (!MLAModelWithBuffer::_enable_queue || MLAModelWithBuffer::_queued_plan_count == 0) return;

    auto* dispatcher = _get_dispatcher();
    try {
        MLAModelWithBuffer::_queue_request.runs.clear();
        MLAModelWithBuffer::_queue_request.runs.reserve(
            MLAModelWithBuffer::_queued_plan_count
        );
        for (std::size_t i = 0; i < MLAModelWithBuffer::_queued_plan_count; ++i) {
            MLAModelWithBuffer::_queue_request.runs.push_back(
                _make_run_ref(
                    MLAModelWithBuffer::_queued_plans[i],
                    MLAModelWithBuffer::_queued_handles[i]
                )
            );
        }
        auto result = submit_queue_and_wait(
            dispatcher,
            std::move(MLAModelWithBuffer::_queue_request)
        );
        if (result.rc != 0) {
            throw std::runtime_error(fmt::format(
                "MLASHM dispatcher runQueue failed: rc={} failed_index={} ({})",
                result.rc,
                result.failed_index,
                result.detail.empty() ? dispatcher->lastErrorString() : result.detail
            ));
        }
    } catch (...) {
        MLAModelWithBuffer::_queue_request = {};
        MLAModelWithBuffer::_queued_plan_count = 0;
        throw;
    }
    MLAModelWithBuffer::_queue_request = {};
    MLAModelWithBuffer::_queued_plan_count = 0;
}


void MLAModelWithBuffer::update_reloc(const std::map<std::string, uint64_t>& reloc_addr_map) {
    if (reloc_addr_map.empty()) return;

    load();
    std::vector<DADDR_LEN> reloc_addrs;
    reloc_addrs.reserve(reloc_addr_map.size());
    for (const auto& [_, addr]: reloc_addr_map) {
        (void)_;
        reloc_addrs.emplace_back(static_cast<DADDR>(addr), 0);
    }

    auto* dispatcher = _get_dispatcher();
    const int rc = dispatcher->updateReloc(_unique_model_ptrs[_model_idx], reloc_addrs);
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "Failed to update relocations through MLASHM dispatcher: {} rc={} ({})",
            _unique_model_paths[_model_idx],
            rc,
            dispatcher->lastErrorString()
        ));
    }
}


void MLAModelWithBuffer::load_all_models(
    std::optional<std::filesystem::path> relative_dir
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

    if (!_disable_parallel_load) {
        std::vector<std::string> paths;
        paths.reserve(file_names.size());
        for (const auto& file_name: file_names) {
            paths.push_back(std::filesystem::absolute(file_name).string());
        }
        // Bulk loadMany has a per-request capacity; MoE compiles far more ELFs than one
        // request holds, so load in fixed-size batches. Lower this if 128 still overflows.
        constexpr std::size_t kLoadBatch = 128;
        for (std::size_t off = 0; off < paths.size(); off += kLoadBatch) {
            const std::size_t end = std::min(off + kLoadBatch, paths.size());
            std::vector<std::string> chunk(paths.begin() + off, paths.begin() + end);
            auto handles = dispatcher->loadMany(chunk);
            if (handles.size() != chunk.size()) {
                throw std::runtime_error(fmt::format(
                    "Bulk MLASHM model load returned {} handles for {} models ({})",
                    handles.size(),
                    chunk.size(),
                    dispatcher->lastErrorString()
                ));
            }
            for (std::size_t j = 0; j < chunk.size(); ++j) {
                const std::size_t i = off + j;
                _unique_model_ptrs[indices[i]] = handles[j];
                if (!_unique_model_ptrs[indices[i]]) {
                    throw std::runtime_error(fmt::format(
                        "Failed to bulk load model through MLASHM dispatcher: {} ({})",
                        file_names[i],
                        dispatcher->lastErrorString()
                    ));
                }
                spdlog::info("Loaded model: {}", file_names[i]);
            }
        }
        return;
    }

    for (std::size_t i = 0; i < file_names.size(); ++i) {
        const auto model_path = std::filesystem::absolute(file_names[i]);
        _unique_model_ptrs[indices[i]] = dispatcher->load(model_path.string());
        if (!_unique_model_ptrs[indices[i]]) {
            throw std::runtime_error(fmt::format(
                "Failed to load model through MLASHM dispatcher: {} ({})",
                model_path,
                dispatcher->lastErrorString()
            ));
        }
        spdlog::info("Loaded model: {}", model_path);
    }
}


void MLAModelWithBuffer::free_all_models(
    std::optional<std::filesystem::path> relative_dir
) {
    simaaidispatcher::DispatcherBase* dispatcher = nullptr;
    const auto should_release = [&](std::size_t i) {
        if (!MLAModelWithBuffer::_unique_model_ptrs[i]) {
            return false;
        }
        return !relative_dir.has_value() ||
               MLAModelWithBuffer::_unique_model_paths[i].string().starts_with(
                   relative_dir.value().string());
    };

    for (std::size_t i = 0; i < MLAModelWithBuffer::_unique_model_paths.size(); ++i) {
        if (!should_release(i)) {
            continue;
        }
        if (!dispatcher) {
            dispatcher = _get_dispatcher();
        }
        const int rc = dispatcher->release(MLAModelWithBuffer::_unique_model_ptrs[i]);
        if (rc != 0) {
            spdlog::error(
                "Failed to release model through MLASHM dispatcher: {} ({})",
                MLAModelWithBuffer::_unique_model_paths[i],
                dispatcher->lastErrorString()
            );
        }
        MLAModelWithBuffer::_unique_model_ptrs[i] = nullptr;
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
