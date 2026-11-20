#ifndef _SIMA_LLIMA_EXPERT_STREAMING_
#define _SIMA_LLIMA_EXPERT_STREAMING_

// Streamed Mixture-of-Experts runtime support (MixtureOfExpertsConfig::streamed_experts).
//
// Expert weights live on storage as fixed-size records written by
// sima_lmm/host/gen_expert_blobs.py: layer_<L>.bin holds num_experts records, each holding the
// expert template's reloc sections in reloc order. At run time:
//   - ExpertCache keeps recently used records in a few large MLA buffers (slabs) under a byte
//     budget, evicting least-recently-used records. One dma-buf per slab, not per record:
//     a process can hold only about as many dma-bufs as open file descriptors.
//   - ExpertStore reads missing records with a thread pool. O_DIRECT cannot target the MLA
//     carve-out (no struct pages, EFAULT), so each thread reads into an aligned host bounce
//     buffer and copies into the slab.
//   - ExpertSlotPool owns S loaded instances of the expert template ELF for one token width.
//     A slot stays bound (mla_update_model_reloc) to one cache entry; running an expert reuses
//     a slot already bound to its entry or rebinds the least-recently-used slot. Rebinding is
//     only legal with no queued jobs on that slot, so callers rebind between queue flushes.
//     S = top_k rebinds most jobs; S = cache entries binds each entry once.

#include <cstdint>
#include <filesystem>
#include <list>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <utility>
#include <exception>
#include <thread>
#include <functional>
#include <condition_variable>
#include <span>
#include <string>
#include <vector>

#include "mla_buffer.hpp"
#include "mla_model.hpp"

namespace simaai {
namespace llima {

struct ExpertSection {
    std::string name;
    uint64_t offset = 0;  // inside a record
    uint64_t size = 0;
};

struct ExpertIndex {
    std::filesystem::path dir;
    uint32_t num_layers = 0;
    uint32_t num_experts = 0;
    uint64_t record_size = 0;
    uint64_t alignment = 4096;
    std::vector<ExpertSection> sections;              // reloc order
    std::vector<std::filesystem::path> layer_files;   // indexed by layer

    static ExpertIndex load(const std::filesystem::path& dir);
};

// Identifies one routed expert.
struct ExpertKey {
    uint32_t layer = 0;
    uint32_t expert = 0;
    bool operator==(const ExpertKey&) const = default;
};

class ExpertCache {
    public:
        // capacity_bytes is rounded down to whole records; slabs are allocated lazily.
        ExpertCache(const ExpertIndex& index, uint64_t capacity_bytes, uint64_t slab_bytes);

        struct Entry {
            MLABuffer* slab = nullptr;
            uint64_t offset = 0;      // record offset in the slab
            int64_t key = -1;         // layer * num_experts + expert, -1 when empty
            uint64_t last_use = 0;
            uint64_t pinned_epoch = 0;
        };

        // Returns the entry holding key, or claims an evictable entry and reports miss = true;
        // the caller must fill it before use. Entries acquired in the current or the previous
        // batch are never evicted, so the next batch can be read while the current one runs.
        // low_priority (prefetch): a filled entry starts as the least recently used, so a
        // misprediction is evicted first once unpinned; a hit is not promoted. The prediction
        // thread and the main thread may call acquire/contains/begin_batch concurrently.
        uint32_t acquire(ExpertKey key, bool& miss, bool low_priority = false);
        void begin_batch() { std::lock_guard lock(_mutex); ++_epoch; }
        Entry& entry(uint32_t id) { return _entries.at(id); }
        uint32_t num_entries() const { return static_cast<uint32_t>(_entries.size()); }
        bool contains(ExpertKey key) const {
            std::lock_guard lock(_mutex);
            return _entry_of_key.at(static_cast<size_t>(_key(key))) >= 0;
        }
        // Entries stay unevictable for this many batches after their last acquire (default 2:
        // the current and the previous batch). Two-layer-ahead prefetch needs 3.
        void set_pin_batches(uint32_t n) { _pin_batches = std::max<uint32_t>(n, 1); }
        uint64_t hits() const { return _hits; }
        uint64_t misses() const { return _misses; }

    private:
        int64_t _key(ExpertKey key) const {
            return static_cast<int64_t>(key.layer) * _index.num_experts + key.expert;
        }
        const ExpertIndex& _index;
        std::vector<std::unique_ptr<MLABuffer>> _slabs;
        std::vector<Entry> _entries;
        std::vector<int32_t> _entry_of_key;  // key -> entry id or -1
        uint64_t _clock = 0;
        uint64_t _epoch = 1;
        uint64_t _hits = 0;
        uint64_t _misses = 0;
        uint32_t _pin_batches = 2;
        mutable std::mutex _mutex;
};

class ExpertStore {
    public:
        ExpertStore(const ExpertIndex& index, uint32_t io_threads);
        ~ExpertStore();
        ExpertStore(const ExpertStore&) = delete;
        ExpertStore& operator=(const ExpertStore&) = delete;

        struct Request {
            ExpertKey key;
            ExpertCache::Entry* entry;
        };
        // Reads every request into its entry (parallel across io_threads) and flushes the
        // written slab ranges for the MLA. Blocks until all reads finish.
        void read(std::span<const Request> requests) { start(requests); wait(); }
        // Asynchronous form: start() returns at once, wait() blocks until the batch is done.
        // The requests must stay alive until wait() returns; one batch at a time.
        void start(std::span<const Request> requests);
        void wait();
        // Blocks until request `index` of the current batch is read (and flushed).
        void wait_for(size_t index);

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
};

class ExpertSlotPool {
    public:
        // template_elf: the expert template for one token width. ifms/ofms: its default
        // bindings (norm_hidden, weight, acc) -> (acc_out); callers override per job.
        ExpertSlotPool(
            const ExpertIndex& index,
            const std::filesystem::path& template_elf,
            uint32_t num_slots,
            const std::vector<MLABufferSlice>& ifms,
            const std::vector<MLABufferSlice>& ofms
        );

        uint32_t num_slots() const { return static_cast<uint32_t>(_slots.size()); }
        // Slot bound to entry_id, rebinding the least-recently-used slot not used in the current
        // batch if needed. Throws if every slot is already used in this batch.
        MLAModelWithBuffer& slot_for(uint32_t entry_id, ExpertCache& cache);
        void begin_batch() { ++_epoch; }
        uint64_t rebinds() const { return _rebinds; }
        uint64_t rebind_ns() const { return _rebind_ns; }

    private:
        struct Slot {
            std::unique_ptr<MLAModelWithBuffer> model;
            int64_t entry_id = -1;
            uint64_t used_epoch = 0;
            std::list<uint32_t>::iterator lru;  // position in _lru (front = most recent)
        };
        const ExpertIndex& _index;
        std::vector<Slot> _slots;
        std::list<uint32_t> _lru;                           // slot ids, most recent first
        std::unordered_map<uint32_t, uint32_t> _slot_of_entry;  // cache entry -> bound slot
        uint64_t _clock = 0;
        uint64_t _epoch = 1;
        uint64_t _rebinds = 0;
        uint64_t _rebind_ns = 0;
};

// Fused decode template: all top_k routed experts of a layer in one job. Slot j's reloc sections
// are the single template's sections with "experts.0." renamed "experts.<j>.", so every slot uses
// the same record layout and any cache entry can be bound to any slot. Each job rebinds all slots
// of one instance (one mla_update_model_reloc call); instances alternate so a rebind never
// touches the instance of a job still in flight.
class FusedExpertTemplate {
    public:
        FusedExpertTemplate(
            const ExpertIndex& index,
            const std::filesystem::path& template_elf,
            uint32_t num_slots,
            uint32_t num_instances,
            const std::vector<MLABufferSlice>& ifms,
            const std::vector<MLABufferSlice>& ofms
        );

        uint32_t num_slots() const { return _num_slots; }
        // Instance bound to entry_ids[j] in slot j (entry_ids.size() == num_slots()).
        MLAModelWithBuffer& bind(std::span<const uint32_t> entry_ids, ExpertCache& cache);
        uint64_t rebinds() const { return _rebinds; }
        uint64_t rebind_ns() const { return _rebind_ns; }

    private:
        struct Port {
            uint32_t slot = 0;
            uint64_t offset = 0;  // inside a record
            uint64_t size = 0;
        };
        uint32_t _num_slots = 0;
        std::vector<Port> _ports;  // in reloc port order (sorted fused section names)
        std::vector<std::unique_ptr<MLAModelWithBuffer>> _instances;
        uint32_t _next = 0;
        uint64_t _rebinds = 0;
        uint64_t _rebind_ns = 0;
};

// The compiler tiles the template per token width, so each width has its own record set in
// <dir>/n<N>/ and its own cache, reader and slot pool.
//
// Runtime knobs (environment):
//   SIMA_LLIMA_EXPERT_DIR          record directory (default <model>/expert_files)
//   SIMA_LLIMA_EXPERT_CACHE_BYTES  decode (n1) expert cache budget (default: every record)
//   SIMA_LLIMA_EXPERT_PREFILL_CACHE_BYTES
//                                  prefill expert cache budget (default 256 MiB: prefill hits
//                                  ~0% because each prompt touches thousands of experts once,
//                                  so the bytes are better spent on the decode cache)
//   SIMA_LLIMA_EXPERT_SLAB_BYTES   slab size (default 256 MiB)
//   SIMA_LLIMA_EXPERT_IO_THREADS   storage read threads (default 8: measured on dvt57, 8 beat 4
//                                  for prefill throughput; 12 gave no further gain)
//   SIMA_LLIMA_EXPERT_SLOTS        decode template instances (default: one per decode cache
//                                  entry, so an entry is bound once and only misses copy data;
//                                  each rebind costs ~50-80 us of mla_update_model_reloc)
//   SIMA_LLIMA_EXPERT_PREFETCH_TOPK
//                                  next-layer experts predicted and prefetched per decode layer
//                                  (default 10; 0 disables; needs router_gates.bin)
//   SIMA_LLIMA_EXPERT_PREFETCH_AHEAD
//                                  decode layers predicted ahead: 1 = next layer (default),
//                                  2 = also the layer after it (earlier reads, more mispredictions)
//   SIMA_LLIMA_EXPERT_PREFETCH_LOW_PRIORITY
//                                  1 (default): prefetched records enter the cache as least
//                                  recently used, so mispredictions do not displace used records
//   SIMA_LLIMA_EXPERT_PREFILL_SLOTS
//                                  prefill template instances (default: one per prefill cache
//                                  entry, so each entry stays bound and is never rebound)
//   SIMA_LLIMA_EXPERT_FUSED        1: decode runs each layer's top_k experts as one job on the fused
//                                  template, if its ELF is deployed (default 0). Measured slower
//                                  on dvt57 (80B: 7.0 vs 8.6 tok/s): MLA time only drops 88.6 ->
//                                  85.7 ms/token, while rebinding its 60 reloc sections costs
//                                  ~470 us per layer. Ignored with SIMA_LLIMA_EXPERT_MIN_WEIGHT.
//   SIMA_LLIMA_EXPERT_PREFILL_BATCH
//                                  experts per prefill queue flush (default 32)
struct ExpertStreamingOptions {
    std::filesystem::path dir;
    uint64_t cache_bytes = 0;
    uint64_t prefill_cache_bytes = 256ull << 20;
    uint64_t slab_bytes = 256ull << 20;
    uint32_t io_threads = 8;
    uint32_t slots = 0;
    uint32_t prefill_slots = 0;     // 0 = one per prefill cache entry (bind once)
    uint32_t prefill_batch = 32;    // SIMA_LLIMA_EXPERT_PREFILL_BATCH: experts per prefill flush
    bool fused = false;             // SIMA_LLIMA_EXPERT_FUSED: one decode job per layer
    uint32_t prefetch_topk = 10;    // SIMA_LLIMA_EXPERT_PREFETCH_TOPK, 0 disables prefetch
    uint32_t prefetch_ahead = 1;    // SIMA_LLIMA_EXPERT_PREFETCH_AHEAD
    uint32_t prefetch_io_threads = 0;  // SIMA_LLIMA_EXPERT_PREFETCH_IO_THREADS (0 = io_threads)
    bool prefetch_low_priority = true;  // SIMA_LLIMA_EXPERT_PREFETCH_LOW_PRIORITY

    static ExpertStreamingOptions from_env(
        const std::filesystem::path& default_dir, uint32_t top_k
    );
};

// Predicts the next layer's experts from the current layer's normalized hidden state with the
// next layer's router weights (router_gates.bin beside the decode records: num_layers x
// num_experts x hidden BF16, from the source checkpoint). Consecutive layers see similar hidden
// states, so the predicted experts can be read from storage while the current layer runs.
class ExpertPredictor {
    public:
        // Returns nullptr when the gate file is absent.
        static std::unique_ptr<ExpertPredictor> load(
            const std::filesystem::path& file, uint32_t num_layers, uint32_t num_experts,
            uint32_t hidden
        );
        // Top-k expert indices of `layer` for hidden state x (length hidden).
        std::vector<uint32_t> predict(uint32_t layer, std::span<const float> x, uint32_t k) const;

    private:
        uint32_t _num_layers = 0;
        uint32_t _num_experts = 0;
        uint32_t _hidden = 0;
        // INT8 gates with one scale per expert row: ranking needs only approximate scores, and
        // the CPU's memory bandwidth (not compute) bounds the per-layer product.
        std::vector<int8_t> _gates;        // [layer][expert][hidden]
        std::vector<float> _row_scales;    // [layer][expert]
};

// Counters for one phase (prefill or decode) of streamed MoE, printed with SIMA_LLIMA_EXPERT_STATS.
struct ExpertPhaseStats {
    uint64_t layers = 0;          // router calls
    uint64_t expert_jobs = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t rebinds = 0;
    uint64_t read_ns = 0;         // NVMe reads (blocking part)
    uint64_t rebind_ns = 0;       // mla_update_model_reloc
    uint64_t mla_wait_ns = 0;     // run_queue inside streamed MoE (router + batch flushes)
    uint64_t host_ns = 0;         // rest of the streamed-MoE host step
    uint64_t tile_start_us = 0;   // MLA tile-time counter at phase start
    uint64_t prefetches = 0;      // records read ahead from a prediction
    uint64_t prefetch_used = 0;   // ... that the next layer then selected
    uint64_t predict_ns = 0;
    uint64_t prefetch_wait_ns = 0;      // main thread: waiting for this layer's prefetched records
    uint64_t prefetch_reuse_wait_ns = 0;  // helper thread: previous prefetch batch before reuse
    uint64_t first_submit_ns = 0; // router done -> first expert job submitted (MLA idle)
    uint64_t route_ns = 0;        // host routing: logits download, top-k, weight uploads
    uint64_t add0 = 0, submit0 = 0, reap0 = 0, jobs0 = 0;  // MLA wrapper host counters at start
    void reset() {
        *this = ExpertPhaseStats{};
        tile_start_us = MLAModelWithBuffer::total_tile_us();
        MLAModelWithBuffer::host_counters(add0, submit0, reap0, jobs0);
        MLAModelWithBuffer::reset_unit_profile();
    }
};

// One persistent thread running one task at a time (the decode read-ahead prediction). A thread
// per layer from std::async cost ~150 us of MLA idle time per layer on dvt57.
class BackgroundWorker {
    public:
        BackgroundWorker() : _thread([this] { loop(); }) {}
        ~BackgroundWorker() {
            {
                std::lock_guard lock(_mutex);
                _stop = true;
            }
            _cv.notify_all();
            _thread.join();
        }
        // Runs task on the worker; the previous task must have been joined.
        void submit(std::function<void()> task) {
            {
                std::lock_guard lock(_mutex);
                if (_busy) throw std::logic_error("BackgroundWorker: previous task not joined");
                _task = std::move(task);
                _busy = true;
            }
            _cv.notify_all();
        }
        // Waits for the submitted task, if any, and rethrows its exception.
        void join() {
            std::unique_lock lock(_mutex);
            _cv.wait(lock, [&] { return !_busy; });
            if (_error) std::rethrow_exception(std::exchange(_error, nullptr));
        }
        bool busy() const { std::lock_guard lock(_mutex); return _busy; }

    private:
        void loop() {
            std::unique_lock lock(_mutex);
            for (;;) {
                _cv.wait(lock, [&] { return _stop || (_busy && _task); });
                if (_stop) return;
                auto task = std::move(_task);
                _task = nullptr;
                lock.unlock();
                std::exception_ptr error;
                try { task(); } catch (...) { error = std::current_exception(); }
                lock.lock();
                _error = error;
                _busy = false;
                _cv.notify_all();
            }
        }
        mutable std::mutex _mutex;
        std::condition_variable _cv;
        std::function<void()> _task;
        std::exception_ptr _error;
        bool _busy = false;
        bool _stop = false;
        std::thread _thread;
};

// Everything streamed MoE keeps for one token width.
struct ExpertWidthState {
    std::unique_ptr<ExpertIndex> index;
    std::unique_ptr<ExpertCache> cache;
    std::unique_ptr<ExpertStore> store;
    std::unique_ptr<ExpertSlotPool> slots;
    // Decode only: all top_k experts of a layer as one job (SIMA_LLIMA_EXPERT_FUSED=1).
    std::unique_ptr<FusedExpertTemplate> fused;
    // Decode read-ahead: one batch per layer in a ring indexed by layer % size, so the reads
    // for layer L+1 (and L+2) can be issued while layer L still consumes its own.
    struct PrefetchBatch {
        std::unique_ptr<ExpertStore> store;
        uint32_t layer = UINT32_MAX;
        std::vector<ExpertStore::Request> reads;  // in flight on store; keys[i] is reads[i]
        std::vector<ExpertKey> keys;
    };
    std::vector<PrefetchBatch> prefetch;
};

}
}

#endif
