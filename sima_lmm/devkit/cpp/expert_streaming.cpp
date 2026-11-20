#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/std.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "expert_streaming.hpp"

namespace simaai {
namespace llima {

namespace {

uint64_t env_u64(const char* name, uint64_t default_value) {
    const std::string value = get_env_var(name, std::string{});
    if (value.empty()) return default_value;
    return std::stoull(value);
}

uint64_t round_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

}  // namespace

ExpertIndex ExpertIndex::load(const std::filesystem::path& dir) {
    std::ifstream file(dir / "index.json");
    if (!file) {
        throw std::runtime_error(fmt::format("Expert index not found: {}", dir / "index.json"));
    }
    const auto json = nlohmann::json::parse(file);
    if (json.at("format") != "sima-llima-expert-records/1") {
        throw std::runtime_error(fmt::format("Unsupported expert record format in {}", dir));
    }
    ExpertIndex index;
    index.dir = dir;
    index.num_layers = json.at("num_layers");
    index.num_experts = json.at("num_experts");
    index.record_size = json.at("record_size");
    index.alignment = json.at("section_alignment");
    for (const auto& section : json.at("sections")) {
        index.sections.push_back({section.at("name"), section.at("offset"), section.at("size")});
    }
    index.layer_files.resize(index.num_layers);
    for (const auto& [layer, name] : json.at("layer_files").items()) {
        index.layer_files.at(std::stoul(layer)) = dir / name.get<std::string>();
    }
    if (index.record_size % index.alignment) {
        throw std::runtime_error("Expert record size must be a multiple of its alignment");
    }
    return index;
}

ExpertCache::ExpertCache(const ExpertIndex& index, uint64_t capacity_bytes, uint64_t slab_bytes)
    : _index(index),
      _entry_of_key(static_cast<size_t>(index.num_layers) * index.num_experts, -1) {
    const uint64_t records_per_slab = slab_bytes / index.record_size;
    if (records_per_slab == 0) {
        throw std::invalid_argument("Expert slab is smaller than one expert record");
    }
    const uint64_t num_records = std::min<uint64_t>(
        capacity_bytes / index.record_size,
        static_cast<uint64_t>(index.num_layers) * index.num_experts
    );
    if (num_records == 0) throw std::invalid_argument("Expert cache holds no records");
    const uint64_t num_slabs = (num_records + records_per_slab - 1) / records_per_slab;
    _entries.reserve(num_records);
    for (uint64_t s = 0; s < num_slabs; ++s) {
        const uint64_t count = std::min(records_per_slab, num_records - s * records_per_slab);
        auto slab = std::make_unique<MLABuffer>(
            fmt::format("expert_slab{}", s),
            std::vector<size_t>{static_cast<size_t>(count * index.record_size)}, "int8", false
        );
        slab->allocate();
        for (uint64_t r = 0; r < count; ++r) {
            _entries.push_back({slab.get(), r * index.record_size});
        }
        _slabs.push_back(std::move(slab));
    }
    spdlog::info(
        "Expert cache: {} records ({:.2f} GiB) in {} slabs, {:.1f}% of {} experts",
        num_records, num_records * index.record_size / double(1ull << 30), num_slabs,
        100.0 * num_records / (double(index.num_layers) * index.num_experts),
        uint64_t(index.num_layers) * index.num_experts
    );
}

uint32_t ExpertCache::acquire(ExpertKey key, bool& miss, bool low_priority) {
    std::lock_guard lock(_mutex);
    const int64_t k = _key(key);
    const int32_t found = _entry_of_key.at(static_cast<size_t>(k));
    if (found >= 0) {
        auto& entry = _entries[found];
        if (!low_priority) entry.last_use = ++_clock;
        entry.pinned_epoch = _epoch;
        miss = false;
        ++_hits;
        return static_cast<uint32_t>(found);
    }
    // Evict the least-recently-used entry not needed by the current batch.
    uint32_t victim = std::numeric_limits<uint32_t>::max();
    uint64_t oldest = std::numeric_limits<uint64_t>::max();
    for (uint32_t i = 0; i < _entries.size(); ++i) {
        const auto& entry = _entries[i];
        // Pinned by one of the last _pin_batches batches (0 = never acquired).
        if (entry.pinned_epoch != 0 && entry.pinned_epoch + _pin_batches > _epoch) continue;
        if (entry.key < 0) { victim = i; break; }
        if (entry.last_use < oldest) { oldest = entry.last_use; victim = i; }
    }
    if (victim == std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error(
            "Expert cache is smaller than one batch of experts; raise SIMA_LLIMA_EXPERT_CACHE_BYTES"
        );
    }
    auto& entry = _entries[victim];
    if (entry.key >= 0) _entry_of_key[static_cast<size_t>(entry.key)] = -1;
    entry.key = k;
    entry.last_use = low_priority ? 0 : ++_clock;
    entry.pinned_epoch = _epoch;
    _entry_of_key[static_cast<size_t>(k)] = static_cast<int32_t>(victim);
    miss = true;
    ++_misses;
    return victim;
}

struct ExpertStore::Impl {
    const ExpertIndex& index;
    std::vector<int> fds;
    std::vector<bool> direct;
    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable work_ready;
    std::condition_variable work_done;
    std::span<const Request> batch;
    // Each record is read as `chunks` pieces by different threads, so a single miss is not
    // limited to one read in flight (decode layers usually miss only one expert).
    uint64_t chunks = 1;
    uint64_t chunk_size = 0;
    std::atomic<size_t> next{0};
    size_t remaining = 0;
    std::vector<uint64_t> parts_left;  // per request: chunks still being read
    uint64_t generation = 0;
    bool stop = false;
    std::string error;

    explicit Impl(const ExpertIndex& idx) : index(idx) {}

    void read_one(const Request& request, uint64_t chunk, uint8_t* bounce) {
        const int fd = fds.at(request.key.layer);
        const uint64_t begin = chunk * chunk_size;
        if (begin >= index.record_size) return;
        const uint64_t size = std::min(chunk_size, index.record_size - begin);
        const off_t offset = static_cast<off_t>(request.key.expert) * index.record_size + begin;
        uint64_t done = 0;
        while (done < size) {
            const ssize_t n = pread(fd, bounce + done, size - done, offset + done);
            if (n <= 0) {
                throw std::runtime_error(fmt::format(
                    "Expert read failed: layer {} expert {}: {}", request.key.layer,
                    request.key.expert, n == 0 ? "end of file" : std::strerror(errno)
                ));
            }
            done += static_cast<uint64_t>(n);
        }
        request.entry->slab->upload_raw(bounce, request.entry->offset + begin, size, true);
    }

    void worker() {
        void* raw = nullptr;
        if (posix_memalign(&raw, index.alignment, index.record_size) != 0) {
            throw std::bad_alloc();
        }
        std::unique_ptr<uint8_t, decltype(&std::free)> bounce(static_cast<uint8_t*>(raw), &std::free);
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock lock(mutex);
                work_ready.wait(lock, [&] { return stop || generation != seen; });
                if (stop) return;
                seen = generation;
            }
            for (size_t i; (i = next.fetch_add(1)) < batch.size() * chunks;) {
                try {
                    read_one(batch[i / chunks], i % chunks, bounce.get());
                } catch (const std::exception& e) {
                    std::lock_guard lock(mutex);
                    if (error.empty()) error = e.what();
                }
                std::lock_guard lock(mutex);
                const bool request_done = --parts_left[i / chunks] == 0;
                if (--remaining == 0 || request_done) work_done.notify_all();
            }
        }
    }
};

ExpertStore::ExpertStore(const ExpertIndex& index, uint32_t io_threads)
    : _impl(std::make_unique<Impl>(index)) {
    for (const auto& path : index.layer_files) {
        int fd = path.empty() ? -1 : open(path.c_str(), O_RDONLY | O_DIRECT);
        bool direct = fd >= 0;
        if (fd < 0 && !path.empty()) fd = open(path.c_str(), O_RDONLY);  // e.g. tmpfs, NFS
        if (fd < 0 && !path.empty()) {
            throw std::runtime_error(fmt::format("Cannot open expert file {}: {}", path,
                                                 std::strerror(errno)));
        }
        _impl->fds.push_back(fd);
        _impl->direct.push_back(direct);
    }
    for (uint32_t t = 0; t < std::max<uint32_t>(1, io_threads); ++t) {
        _impl->threads.emplace_back([impl = _impl.get()] { impl->worker(); });
    }
}

ExpertStore::~ExpertStore() {
    {
        std::lock_guard lock(_impl->mutex);
        _impl->stop = true;
    }
    _impl->work_ready.notify_all();
    for (auto& thread : _impl->threads) thread.join();
    for (int fd : _impl->fds) if (fd >= 0) close(fd);
}

void ExpertStore::start(std::span<const Request> requests) {
    if (requests.empty()) return;
    std::lock_guard lock(_impl->mutex);
    if (_impl->remaining != 0) throw std::logic_error("ExpertStore: previous batch still running");
    _impl->batch = requests;
    _impl->next = 0;
    // Split records only when there are fewer records than threads to keep busy.
    const uint64_t threads = _impl->threads.size();
    _impl->chunks = requests.size() >= threads ? 1 : (threads + requests.size() - 1) / requests.size();
    _impl->chunk_size = ((_impl->index.record_size + _impl->chunks - 1) / _impl->chunks + 4095) / 4096 * 4096;
    _impl->remaining = requests.size() * _impl->chunks;
    _impl->parts_left.assign(requests.size(), _impl->chunks);
    _impl->error.clear();
    ++_impl->generation;
    _impl->work_ready.notify_all();
}

void ExpertStore::wait() {
    std::unique_lock lock(_impl->mutex);
    _impl->work_done.wait(lock, [&] { return _impl->remaining == 0; });
    if (!_impl->error.empty()) {
        std::string error = std::move(_impl->error);
        _impl->error.clear();
        throw std::runtime_error(error);
    }
}

void ExpertStore::wait_for(size_t index) {
    std::unique_lock lock(_impl->mutex);
    if (index >= _impl->parts_left.size()) throw std::out_of_range("ExpertStore::wait_for: no such request");
    _impl->work_done.wait(lock, [&] { return _impl->parts_left[index] == 0 || !_impl->error.empty(); });
    if (!_impl->error.empty()) {
        std::string error = std::move(_impl->error);
        _impl->error.clear();
        throw std::runtime_error(error);
    }
}

ExpertSlotPool::ExpertSlotPool(
    const ExpertIndex& index,
    const std::filesystem::path& template_elf,
    uint32_t num_slots,
    const std::vector<MLABufferSlice>& ifms,
    const std::vector<MLABufferSlice>& ofms
) : _index(index) {
    _slots.reserve(num_slots);
    for (uint32_t s = 0; s < num_slots; ++s) {
        _slots.push_back({std::make_unique<MLAModelWithBuffer>(template_elf, ifms, ofms, s)});
        _slots.back().lru = _lru.insert(_lru.end(), s);
    }
    _slot_of_entry.reserve(num_slots * 2);
}

FusedExpertTemplate::FusedExpertTemplate(
    const ExpertIndex& index,
    const std::filesystem::path& template_elf,
    uint32_t num_slots,
    uint32_t num_instances,
    const std::vector<MLABufferSlice>& ifms,
    const std::vector<MLABufferSlice>& ofms
) : _num_slots(num_slots) {
    // Port order is the sorted section name; single-digit slot ids keep the renamed names in
    // slot-major order under plain string comparison.
    if (num_slots == 0 || num_slots > 10) {
        throw std::invalid_argument(fmt::format("Fused expert template supports 1-10 slots, got {}", num_slots));
    }
    const std::string base = "experts.0.";
    std::vector<std::pair<std::string, Port>> named;
    for (uint32_t slot = 0; slot < num_slots; ++slot) {
        for (const auto& section : index.sections) {
            const auto at = section.name.find(base);
            if (at == std::string::npos) {
                throw std::runtime_error(fmt::format(
                    "Expert record section {} does not name expert 0", section.name
                ));
            }
            std::string name = section.name;
            name.replace(at, base.size(), fmt::format("experts.{}.", slot));
            named.push_back({std::move(name), {slot, section.offset, section.size}});
        }
    }
    std::sort(named.begin(), named.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& [name, port] : named) _ports.push_back(port);
    for (uint32_t i = 0; i < num_instances; ++i) {
        _instances.push_back(std::make_unique<MLAModelWithBuffer>(template_elf, ifms, ofms, i));
    }
}

MLAModelWithBuffer& FusedExpertTemplate::bind(std::span<const uint32_t> entry_ids, ExpertCache& cache) {
    if (entry_ids.size() != _num_slots) {
        throw std::invalid_argument(fmt::format(
            "Fused expert template needs {} entries, got {}", _num_slots, entry_ids.size()
        ));
    }
    const auto start = std::chrono::steady_clock::now();
    std::vector<MLARelocBinding> relocs;
    relocs.reserve(_ports.size());
    for (const auto& port : _ports) {
        const auto& entry = cache.entry(entry_ids[port.slot]);
        relocs.push_back({entry.slab, entry.offset + port.offset, port.size});
    }
    auto& model = *_instances[_next];
    _next = (_next + 1) % _instances.size();
    model.update_reloc(relocs);
    ++_rebinds;
    _rebind_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    return model;
}

MLAModelWithBuffer& ExpertSlotPool::slot_for(uint32_t entry_id, ExpertCache& cache) {
    // O(1): a slot already bound to this entry is reused; otherwise the least recently used
    // slot not used in this batch is rebound.
    Slot* chosen = nullptr;
    if (const auto found = _slot_of_entry.find(entry_id); found != _slot_of_entry.end()) {
        chosen = &_slots[found->second];
    } else {
        for (auto it = _lru.rbegin(); it != _lru.rend(); ++it) {
            if (_slots[*it].used_epoch != _epoch) { chosen = &_slots[*it]; break; }
        }
        if (!chosen) {
            throw std::runtime_error("Every expert template slot is in use by this batch");
        }
        if (chosen->entry_id >= 0) _slot_of_entry.erase(static_cast<uint32_t>(chosen->entry_id));
        _slot_of_entry[entry_id] = static_cast<uint32_t>(chosen - _slots.data());
        const auto rebind_start = std::chrono::steady_clock::now();
        const auto& entry = cache.entry(entry_id);
        std::vector<MLARelocBinding> relocs;
        relocs.reserve(_index.sections.size());
        for (const auto& section : _index.sections) {
            relocs.push_back({entry.slab, entry.offset + section.offset, section.size});
        }
        chosen->model->update_reloc(relocs);
        chosen->entry_id = entry_id;
        ++_rebinds;
        _rebind_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - rebind_start
        ).count();
    }
    _lru.splice(_lru.begin(), _lru, chosen->lru);
    chosen->used_epoch = _epoch;
    return *chosen->model;
}

std::unique_ptr<ExpertPredictor> ExpertPredictor::load(
    const std::filesystem::path& file, uint32_t num_layers, uint32_t num_experts, uint32_t hidden
) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return nullptr;
    auto predictor = std::make_unique<ExpertPredictor>();
    predictor->_num_layers = num_layers;
    predictor->_num_experts = num_experts;
    predictor->_hidden = hidden;
    const size_t count = static_cast<size_t>(num_layers) * num_experts * hidden;
    std::vector<uint16_t> raw(count);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(count * 2));
    if (static_cast<size_t>(in.gcount()) != count * 2) {
        throw std::runtime_error(fmt::format("Router gate file {} is truncated", file));
    }
    predictor->_gates.resize(count);
    predictor->_row_scales.resize(static_cast<size_t>(num_layers) * num_experts);
    std::vector<float> row(hidden);
    for (size_t r = 0; r < predictor->_row_scales.size(); ++r) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < hidden; ++i) {
            const uint32_t bits = static_cast<uint32_t>(raw[r * hidden + i]) << 16;  // BF16 -> FP32
            std::memcpy(&row[i], &bits, 4);
            amax = std::max(amax, std::fabs(row[i]));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        predictor->_row_scales[r] = scale;
        for (uint32_t i = 0; i < hidden; ++i) {
            predictor->_gates[r * hidden + i] = static_cast<int8_t>(std::lrint(row[i] / scale));
        }
    }
    return predictor;
}

std::vector<uint32_t> ExpertPredictor::predict(
    uint32_t layer, std::span<const float> x, uint32_t k
) const {
    // INT8 x INT8 dot products (1 MB of weights per layer), split across cores.
    float amax = 0.0f;
    for (const float v : x) amax = std::max(amax, std::fabs(v));
    const float x_scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    std::vector<int8_t> xq(_hidden);
    for (uint32_t i = 0; i < _hidden; ++i) xq[i] = static_cast<int8_t>(std::lrint(x[i] / x_scale));
    const int8_t* w = _gates.data() + static_cast<size_t>(layer) * _num_experts * _hidden;
    const float* row_scale = _row_scales.data() + static_cast<size_t>(layer) * _num_experts;
    std::vector<std::pair<float, uint32_t>> scores(_num_experts);
    const int8_t* xp = xq.data();
    const uint32_t hidden = _hidden;
#pragma omp parallel for num_threads(4) schedule(static)
    for (int e = 0; e < static_cast<int>(_num_experts); ++e) {
        const int8_t* r = w + static_cast<size_t>(e) * hidden;
        int32_t acc = 0;
        for (uint32_t i = 0; i < hidden; ++i) acc += int32_t(r[i]) * int32_t(xp[i]);
        scores[e] = {acc * row_scale[e], static_cast<uint32_t>(e)};
    }
    k = std::min(k, _num_experts);
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<uint32_t> top(k);
    for (uint32_t i = 0; i < k; ++i) top[i] = scores[i].second;
    return top;
}

ExpertStreamingOptions ExpertStreamingOptions::from_env(
    const std::filesystem::path& default_dir, uint32_t top_k
) {
    ExpertStreamingOptions options;
    const std::string dir = get_env_var("SIMA_LLIMA_EXPERT_DIR", std::string{});
    options.dir = dir.empty() ? default_dir : std::filesystem::path(dir);
    options.cache_bytes = env_u64("SIMA_LLIMA_EXPERT_CACHE_BYTES", std::numeric_limits<uint64_t>::max());
    options.prefill_cache_bytes = env_u64(
        "SIMA_LLIMA_EXPERT_PREFILL_CACHE_BYTES", options.prefill_cache_bytes
    );
    options.slab_bytes = env_u64("SIMA_LLIMA_EXPERT_SLAB_BYTES", options.slab_bytes);
    options.io_threads = static_cast<uint32_t>(env_u64("SIMA_LLIMA_EXPERT_IO_THREADS", options.io_threads));
    options.slots = static_cast<uint32_t>(env_u64("SIMA_LLIMA_EXPERT_SLOTS", 0));  // 0 = auto
    options.prefetch_ahead = static_cast<uint32_t>(
        std::clamp<uint64_t>(env_u64("SIMA_LLIMA_EXPERT_PREFETCH_AHEAD", 1), 1, 2)
    );
    options.prefetch_io_threads = static_cast<uint32_t>(env_u64("SIMA_LLIMA_EXPERT_PREFETCH_IO_THREADS", 0));
    options.prefetch_low_priority = env_u64("SIMA_LLIMA_EXPERT_PREFETCH_LOW_PRIORITY", 1) != 0;
    options.prefetch_topk = static_cast<uint32_t>(
        env_u64("SIMA_LLIMA_EXPERT_PREFETCH_TOPK", options.prefetch_topk)
    );
    options.prefill_slots = static_cast<uint32_t>(
        env_u64("SIMA_LLIMA_EXPERT_PREFILL_SLOTS", options.prefill_slots)
    );
    options.fused = env_u64("SIMA_LLIMA_EXPERT_FUSED", 0) != 0;
    options.prefill_batch = static_cast<uint32_t>(
        env_u64("SIMA_LLIMA_EXPERT_PREFILL_BATCH", options.prefill_batch)
    );
    if (options.prefill_batch == 0) {
        throw std::invalid_argument("SIMA_LLIMA_EXPERT_PREFILL_BATCH must be at least 1");
    }
    if (options.slots != 0 && options.slots < top_k) {
        throw std::invalid_argument(fmt::format(
            "SIMA_LLIMA_EXPERT_SLOTS={} is below top_k={}; one token's experts must fit", options.slots, top_k
        ));
    }
    return options;
}

}
}
