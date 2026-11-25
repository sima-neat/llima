#pragma once

// Private resource, cache and workspace owners; not an installed API header.
#include "language/attention_cache.hpp"
#include "language/language_model.hpp"
#include "language/embedding_offload.hpp"
#include "package/model_resources.hpp"
#include <stdexcept>

namespace simaai::llima {

// Loaded adapter buffers and their per-program relocation bindings.
// Retained by a workspace while bound to its programs.
struct LoRAAdapter {
    std::string name;
    std::map<std::string, MLABuffer> buffers;
    std::map<size_t, std::map<std::string, MLABuffer*>> bindings; // Runtime program index -> sections.
};

// Shared loaded language-model data, including embeddings and original position tables.
// Compatible model instances reuse these assets while keeping separate caches and workspaces.
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

// Retains KV data, recurrent checkpoints and prompt-reuse information
// between requests on one language model.
struct SequenceCache {
    void invalidate_reuse();
    void exchange(SequenceCache& other);
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

// Initial EAGLE3 proposal tree, captured target features and first-token timing.
struct LanguageModel::InitTreeResult {
    std::vector<uint32_t> draft_tokens;
    std::vector<std::vector<int32_t>> retrieve_indices;
    eagle_helpers::EagleTreeMask tree_mask;  // 4D (1, 1, N, N)
    std::vector<int32_t> tree_position_ids;
    std::vector<std::vector<Eigen::bfloat16>> hidden_states;  // 3 captured layers
    uint32_t token;                                            // root token
    std::chrono::steady_clock::time_point root_ready_time;
    double time_to_first_token;
};

// Draft proposal tree and the masks/indices needed to verify its candidate paths.
struct LanguageModel::TopkGenerateResult {
    std::vector<uint32_t> draft_tokens;
    std::vector<std::vector<int32_t>> retrieve_indices;
    eagle_helpers::EagleTreeMask tree_mask;  // 4D (1, 1, N, N)
    std::vector<int32_t> tree_position_ids;
};

// Hidden states and vocabulary logits produced by one EAGLE3 draft forward pass.
struct LanguageModel::DraftForwardResult {
    std::vector<Eigen::bfloat16> hidden_states;  // (seq_length, hidden_size)
    std::vector<Eigen::bfloat16> logits;         // (num_tokens, draft_vocab_size)
};

// Target predictions and hidden states for a batch of Gemma 4 MTP candidate tokens.
struct LanguageModel::Gemma4MtpTargetBatchResult {
    std::vector<uint32_t> next_token_ids;
    std::vector<Eigen::bfloat16> hidden_states;
};

// One MTP draft token and its projected hidden state for the next draft step.
struct LanguageModel::Gemma4MtpDraftStepResult {
    uint32_t token_id;
    std::vector<Eigen::bfloat16> projected_hidden_state;  // (backbone_hidden_size)
};

// Target predictions and captured features for all nodes in an EAGLE3 proposal tree.
struct LanguageModel::TargetVerifyResult {
    std::vector<uint32_t> next_token_ids;                     // target argmax per node
    std::vector<std::vector<Eigen::bfloat16>> hidden_states;   // 3 captures, each (num_tokens, hidden_size)
};

// Target predictions rearranged by candidate path, with the captured tree hidden states.
struct LanguageModel::TreeDecodingResult {
    std::vector<int32_t> next_token_ids;          // flat (n_paths × max_depth)
    std::vector<Eigen::bfloat16> hidden_states;   // flat (K × 3 × hidden_size)
};

// Accepted sequence, bonus token and prepared proposal tree for the next EAGLE3 round.
struct LanguageModel::UpdateInferenceInputsResult {
    std::vector<uint32_t> input_ids;                       // old + accepted + bonus
    std::vector<uint32_t> draft_tokens;                    // next round's tree tokens
    std::vector<std::vector<int32_t>> retrieve_indices;    // next round
    eagle_helpers::EagleTreeMask tree_mask;                // next round
    std::vector<int32_t> tree_position_ids;                // next round
    int32_t new_token;                                     // accumulated generated-token count
    uint32_t bonus_token;
};

// References one LM-head output buffer and records which vocabulary entries it covers.
struct LanguageModel::LMHeadLogitView {
    MLABuffer* buffer;
    const uint8_t* data;
    size_t row_bytes;
    uint32_t vocab_begin;
    uint32_t width;
};

// Reuses router/expert buffer references and CPU working arrays for one compiled token count.
struct LanguageModel::MoeHostCache {
    uint16_t num_tokens = 0;             // the moe_nt this entry serves
    MLABuffer* values = nullptr;         // n{nt}_router_values
    MLABuffer* indices = nullptr;        // n{nt}_router_indices
    MLABuffer* weights = nullptr;        // n{nt}_router_weights
    std::vector<MLABuffer*> expert_out;  // top_k decode slots or num_experts prefill outputs
    // Decode only: top_k OFM overrides routing each expert into its combine slot.
    std::vector<std::map<uint8_t, MLABufferSlice>> slot_ofm;
    // Host staging, sized once; never reallocated on the critical path.
    std::vector<Eigen::bfloat16> values_scratch;   // nt * top_k
    std::vector<int32_t> indices_scratch;          // nt * top_k
    std::vector<Eigen::bfloat16> weights_scratch;  // nt * num_experts
    std::vector<uint8_t> activated_scratch;        // num_experts (prefill only)
};

// Reusable proposal, verification, mask and logit-view storage for DFlash decoding.
struct LanguageModel::DFlashScratch {
    std::vector<uint32_t> draft_input_ids;
    std::vector<uint32_t> proposals;
    std::vector<uint32_t> verify_input;
    std::vector<uint32_t> emitted;
    std::vector<LMHeadLogitView> logit_views;
    std::vector<Eigen::bfloat16> attention_mask;
};

// Owns reusable working buffers, program bindings and temporary data for one language model.
// SequenceCache stores the sequence history kept between requests.
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

// Stop-token IDs and total token limit, including the prompt.
struct TokenLimits {
    std::set<uint32_t> stop_token_ids;
    uint16_t max_num_tokens;
};

// Legacy stop_model() is called by the CLI's SIGINT handler. A request observes
// only interrupts delivered during its lifetime. Caller-owned tokens can be stopped earlier.
using InterruptCounter = std::atomic<uint64_t>;
static_assert(InterruptCounter::is_always_lock_free);

// Holds limits, cancellation and interrupt-counter snapshots for one operation.
// The target and optional draft use this state until the request ends.
struct RequestState {
    // References a counter and remembers its starting value to detect new interrupts.
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
    bool cache_created = false;
    uint32_t cached_prompt_tokens = 0;
    const LanguageModel* metrics_model = nullptr;

    bool cancelled() const {
        return stop.stop_requested() || target_interrupt.fired() ||
               (draft_interrupt && draft_interrupt->fired());
    }
};

// Stores default limits and links a model to its streamer and current request.
// RequestScope owns the request state. The interrupt counter belongs to this model.
struct RequestControl {
    RequestControl(std::set<uint32_t> stops, uint16_t limit, TextStreamer& streamer)
        : defaults{std::move(stops), limit}, text_streamer(streamer) {}

    RequestState& require_active_request() const {
        if (!active) throw std::logic_error("Model execution requires an active request");
        return *active;
    }
    TokenLimits& limits() { return require_active_request().limits; }
    bool cancelled() const { return require_active_request().cancelled(); }

    TokenLimits defaults;
    TextStreamer& text_streamer;
    InterruptCounter interrupts{0};
    // Only the serialized execution thread uses this pointer. stop_model() touches only interrupts.
    RequestState* active = nullptr;
};

} // namespace simaai::llima
