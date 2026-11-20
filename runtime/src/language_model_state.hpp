#pragma once

// Private resource, session and workspace owners; not an installed API header.
#include "language_model.hpp"
#include "embedding_offload.hpp"
#include "model_resources.hpp"

namespace simaai::llima {

struct LanguageAdapter {
    std::string name;
    std::map<std::string, MLABuffer> buffers;
    std::map<size_t, std::map<std::string, MLABuffer*>> bindings; // Runtime program index -> sections.
};

struct LanguageResources {
    explicit LanguageResources(std::shared_ptr<ModelResources<VlmConfig>> value)
        : package(std::move(value)), cfg(package->cfg) {}
    static std::shared_ptr<LanguageResources> acquire(const std::filesystem::path& path,
                                                      std::shared_ptr<FileProvider> files);
    std::shared_ptr<ModelResources<VlmConfig>> package;
    const VlmConfig& cfg;
    void prepare_embeddings();
    void load();
    void load_per_layer_embeddings();
    bool ready = false;
    std::map<std::string, std::weak_ptr<LanguageAdapter>> adapters;
    struct BufferSpec {
        std::vector<size_t> shape;
        std::string dtype;
        bool align_last_dim;
    };
    std::map<std::string, BufferSpec> session_layout;
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

struct LanguageSession {
    void init_layout(const LanguageSession& prototype);
    size_t memory_bytes() const;
    std::shared_ptr<LanguageSession> draft;
    void reset();
    void invalidate_reuse();
    void allocate(const LanguageResources& resources);
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
    std::vector<LanguageModel::CachedState> cached_states;
    // Zero marks an unused slot. Slot 0 is reserved for a stable system/tool prefix.
    std::array<uint16_t, LanguageModel::checkpoint_slot_count> state_checkpoint_positions{};
    std::shared_ptr<LanguageAdapter> adapter;
    std::string image_identity;
};

// Bounded contexts with stable addresses; callers serialize through the model workspace.
class SessionManager {
  public:
    SessionManager();
    const std::shared_ptr<LanguageSession>& get(SessionId id) const;
    SessionId create(const LanguageSession& prototype);
    void release(SessionId id);
    void set_limit(size_t limit);
    size_t limit() const { return _limit; }
    std::vector<SessionId> ids() const;
    SessionId active = 0;

  private:
    size_t _limit = 4;
    SessionId _next = 1;
    std::map<SessionId, std::shared_ptr<LanguageSession>> _sessions;
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
    std::shared_ptr<LanguageAdapter> bound_adapter;
    std::weak_ptr<LanguageSession> paired_draft_identity; // The draft runner owns its public session 0.
    std::vector<uint8_t> mtp_shared_kv_layers;
    LanguageModel::DFlashScratch dflash;
    std::vector<uint32_t> offload_prompt_ids;
    bool offload_chunk_staging = false;
    std::map<std::string, MLABuffer> buffers;
    // Stable pointer slots collected once while constructing execution bindings.
    std::map<std::string, std::vector<MLABuffer**>> session_bindings;
    LanguageModelMap pre_model_map;
    LanguageModelMap cache_model_map;
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
    std::vector<LanguageModel::AttentionBindingBuffers> attention_binding_buffers;
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

struct LanguageRunState {
    LanguageRunState(std::set<uint32_t> stops, uint16_t limit, TextStreamer& streamer)
        : stop_token_ids(std::move(stops)), max_num_tokens(limit), text_streamer(streamer),
          is_running(false) {}
    std::set<uint32_t> stop_token_ids;
    uint16_t max_num_tokens;
    TextStreamer& text_streamer;
    std::atomic<bool> is_running; // Request activity; cancellation is never inferred from this flag.
    std::atomic<bool> cancelled = false;
};

} // namespace simaai::llima
