#pragma once

// Private resource, cache and workspace owners; not an installed API header.
#include "attention_cache.hpp"
#include "language_model.hpp"
#include "embedding_offload.hpp"
#include "model_resources.hpp"

namespace simaai::llima {

struct LoRAAdapter {
    std::string name;
    std::map<std::string, MLABuffer> buffers;
    std::map<size_t, std::map<std::string, MLABuffer*>> bindings; // Runtime program index -> sections.
};

struct LanguageAssets {
    explicit LanguageAssets(std::shared_ptr<ModelResources<VlmConfig>> value)
        : package(std::move(value)), cfg(package->cfg) {}
    static std::shared_ptr<LanguageAssets> acquire(const std::filesystem::path& path,
                                                   std::shared_ptr<FileProvider> files);
    std::shared_ptr<ModelResources<VlmConfig>> package;
    const VlmConfig& cfg;
    void prepare_embeddings();
    void load();
    void load_per_layer_embeddings();
    bool ready = false;
    std::map<std::string, std::weak_ptr<LoRAAdapter>> adapters;
    std::map<std::string, MLABuffer> buffers;
    std::shared_ptr<EmbeddingOffload> embedding_offload;
    RopeTable master_rope_table;
    RopeTable global_freq_host; // pristine CPU copy;
    RopeTable local_freq_host;
    size_t per_layer_embedding_rows_per_shard = 0;
    std::vector<MLABuffer*> per_layer_embedding_shards;
    std::vector<int32_t> d2t;
    std::vector<uint32_t> gemma4_token_ordering;
};

struct SequenceCache {
    void invalidate_reuse();
    std::map<std::string, MLABuffer> buffers;
    std::vector<uint32_t> cached_token_ids;
    uint32_t cached_first_generated_token = 0;
    // Restore the prior EAGLE3 tree only when the complete prompt matches.
    std::vector<uint32_t> cached_draft_tokens;
    std::vector<std::vector<int32_t>> cached_retrieve_indices;
    eagle_helpers::EagleTreeMask cached_tree_mask;
    std::vector<int32_t> cached_tree_position_ids;
    uint16_t cached_eagle3_prompt_len = 0;
    size_t cached_eagle3_stable_kv = 0;
    uint16_t kv_cache_len = 0; // tokens in KV cache;
    std::vector<uint16_t> checkpoint_boundaries;
    std::vector<LanguageModel::RecurrentCheckpoints> cached_states;
    // Zero marks an unused slot. Slot 0 is reserved for a stable system/tool prefix.
    std::array<uint16_t, LanguageModel::checkpoint_slot_count> state_checkpoint_positions{};
    std::string image_identity;
};

struct LanguageModel::DFlashScratch {
    std::vector<uint32_t> draft_input_ids;
    std::vector<uint32_t> proposals;
    std::vector<uint32_t> verify_input;
    std::vector<uint32_t> emitted;
    std::vector<LMHeadLogitView> logit_views;
    std::vector<Eigen::bfloat16> attention_mask;
};

struct LanguageWorkspace {
    void clear_cached_scratch() {
        dflash_resolved_prefix = 0;
        gemma4_mtp_prefill_hidden_state.reset();
        eagle3_tree_mask.reset();
        eagle3_stable_kv = 0;
    }
    std::shared_ptr<LoRAAdapter> bound_adapter;
    // A target and its current draft share this identity. Reusing either runner with a
    // different partner requires a cold prefill on both sides.
    struct SpeculativePair {};
    std::shared_ptr<SpeculativePair> speculative_pair;
    std::vector<CacheLayerBuffers> mtp_target_kv;
    LanguageModel::DFlashScratch dflash;
    std::vector<uint32_t> offload_prompt_ids;
    bool offload_chunk_staging = false;
    std::map<std::string, MLABuffer> buffers;
    LanguageModelMap pre_model_map;
    LanguageModelMap cache_model_map;
    std::map<LanguageModelMapKey, CachePorts> cache_ports;
    LanguageModelMap post_model_map;
    LanguageModelMap conv_model_map;
    LanguageModelMap conv_final_model_map;
    LanguageModelMap per_layer_model_map;
    LanguageModelMap linear_model_map;
    LanguageModelMap router_model_map;
    LanguageModelMap expert_model_map;
    LanguageModelMap weightedsum_model_map;
    std::map<uint16_t, MLAModelWithBuffer> fc_model_map;
    LanguageModelMap dflash_context_model_map;
    LanguageModelMap dflash_state_resolver_model_map;
    uint16_t dflash_resolved_prefix = 0;
    std::vector<CacheLayerBuffers> attention_binding_buffers;
    MLABuffer* global_freq_real = nullptr;
    MLABuffer* global_freq_imag = nullptr;
    MLABuffer* local_freq_real = nullptr;
    MLABuffer* local_freq_imag = nullptr;
    bool has_image_token = false;
    std::vector<uint32_t> prompt_per_layer_token_ids;
    bool capture_state_checkpoints = false;
    uint8_t writable_checkpoint_slots = 0;
    uint16_t system_checkpoint_position = 0;
    size_t rolling_checkpoint_slot = 0;
    std::vector<std::vector<Eigen::bfloat16>> eagle3_intermediate_hidden_states;
    std::optional<std::vector<Eigen::bfloat16>> gemma4_mtp_prefill_hidden_state;
    std::shared_ptr<eagle_helpers::EagleTreeMask> eagle3_tree_mask;
    eagle_helpers::EagleTreeMask eagle3_tree_mask_init;
    std::vector<int32_t> eagle3_position_ids;
    size_t eagle3_stable_kv = 0;
};

struct TokenLimits {
    std::set<uint32_t> stop_token_ids;
    uint16_t max_num_tokens;
};

// Legacy stop_model() is called by the CLI's SIGINT handler. A request observes
// only interrupts delivered during its lifetime. Caller-owned tokens can be stopped earlier.
using InterruptCounter = std::atomic<uint64_t>;
static_assert(InterruptCounter::is_always_lock_free);

struct RequestState {
    struct Interrupt {
        explicit Interrupt(const InterruptCounter& counter)
            : counter(counter), initial(counter.load(std::memory_order_relaxed)) {}
        bool fired() const { return counter.load(std::memory_order_relaxed) != initial; }
        const InterruptCounter& counter;
        uint64_t initial;
    };

    TokenLimits limits;
    std::stop_token stop;
    Interrupt target_interrupt;
    std::optional<Interrupt> draft_interrupt;
    bool length_limit_reached = false;

    bool cancelled() const {
        return stop.stop_requested() || target_interrupt.fired() ||
               (draft_interrupt && draft_interrupt->fired());
    }
};

// Holds request defaults, the streamer, interrupts, and the active request;
// also tracks cancellation across standalone prefill/decode calls.
struct RequestControl {
    RequestControl(std::set<uint32_t> stops, uint16_t limit, TextStreamer& streamer)
        : defaults{std::move(stops), limit}, text_streamer(streamer) {}

    TokenLimits& limits() { return active ? active->limits : defaults; }
    bool cancelled() const {
        return active ? active->cancelled() : interrupts.load(std::memory_order_relaxed) != standalone_start;
    }

    TokenLimits defaults;
    TextStreamer& text_streamer;
    InterruptCounter interrupts{0};
    uint64_t standalone_start = 0; // Shared by a standalone prefill/decode pair.
    // Only the serialized execution thread uses this pointer. stop_model() touches only interrupts.
    RequestState* active = nullptr;
};

} // namespace simaai::llima
