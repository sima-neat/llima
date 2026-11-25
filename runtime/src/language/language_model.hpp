#ifndef _SIMA_LLIMA_LANGUAGE_MODEL_
#define _SIMA_LLIMA_LANGUAGE_MODEL_

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "file_provider.hpp"
#include "mla_model.hpp"
#include "vlm_config.hpp"
#include <spdlog/logger.h>
#include "language/speculative/eagle_helpers.hpp"
#include "generation_types.hpp"
#include "language/rope_utils.hpp"
#include "text_streamer.hpp"
#include "language/cache_slots.hpp"

namespace simaai {
namespace llima {

struct EmbeddingOffload;
struct LanguageAssets;
struct LoRAAdapter;
struct SequenceCache;
struct LanguageWorkspace;
struct RequestControl;
class LayerExecutor;
class RequestScope;
// Pre/post maps use (num_tokens, layer_idx, 0). Cache maps use
// (num_tokens, cache_kind, aligned_cache_token_idx).
using LanguageModelMapKey = std::tuple<uint16_t, uint8_t, uint16_t>;
using LanguageModelMap = std::map<LanguageModelMapKey, MLAModelWithBuffer>;

// Options for raw-token execution. The token limit includes the input prompt.
struct TokenRunOptions {
    std::optional<uint16_t> max_total_tokens;
    std::optional<std::set<uint32_t>> stop_token_ids;
    std::stop_token stop;
    std::optional<ChronoTimer> timer_ttft;
    uint16_t stable_prefix_token_count = 0;
    GenerationPerformanceResult* performance = nullptr;
    std::optional<std::string> cache_id;
};

// Runs token prefill, decoding and scoring with shared model data.
// Owns one cache and working buffers. The caller must prevent simultaneous requests.
class LanguageModel {
    public:
        // Reads the model files through file_provider (nullptr = from disk).
        LanguageModel(
            std::filesystem::path model_path,
            std::set<uint32_t> stop_token_ids,
            std::optional<uint32_t> image_token_id,
            std::optional<uint32_t> pad_token_id,
            TextStreamer& text_streamer,
            std::shared_ptr<FileProvider> file_provider = nullptr,
            size_t max_kv_cache_slots = 1
        );
        ~LanguageModel();

        // Both entry points stage their raw input tokens. Callers serialize access.
        std::optional<std::vector<uint32_t>> run_model(
            std::span<const uint32_t> input_token_ids, const TokenRunOptions& options = {}
        );
        std::optional<std::vector<uint32_t>> run_model_speculative_decoding(
            LanguageModel& draft_lm, std::span<const uint32_t> input_token_ids,
            const TokenRunOptions& options = {}
        );
        LogLikelihoodResult run_model_for_loglikelihood(
            std::span<const uint32_t> input_token_ids,
            size_t continuation_start,
            std::span<const uint32_t> continuation_token_ids,
            bool use_group_prefill = true
        );
        void attach_draft(LanguageModel& draft);
        size_t shared_resource_bytes() const;
        size_t workspace_memory_bytes() const;
        size_t kv_cache_count() const;
        size_t kv_cache_bytes_per_slot() const;
        bool remove_kv_cache(const std::string& id);
        void clear_kv_caches();

        void stop_model();

        void set_reloc(const std::string& reloc_name);
        void unset_reloc();

    private:
        friend struct LanguageAssets;
        friend struct SequenceCache;
        friend struct LanguageWorkspace;
        friend class VisionLanguageModel;
        friend class RequestScope;
        friend class LayerExecutor;
        friend struct LanguageModelTestAccess;

        std::vector<std::map<uint8_t, MLABufferSlice>> create_input_buffers(
            std::span<const uint32_t> input_token_ids
        );
        uint32_t run_model_once(
            uint16_t num_tokens,
            uint16_t token_idx,
            uint16_t num_input_tokens,
            uint32_t token_id,
            std::vector<Eigen::bfloat16>* logits_ptr = nullptr
        );
        const std::set<uint32_t>& get_stop_token_ids() const;
        void set_kv_cache_len(uint16_t len);

        // Result of initialize_tree.
        struct InitTreeResult;

        // Result of topk_generate.
        struct TopkGenerateResult;

        // Draft-side tree expansion. `hidden_states` is (seq_length × 3*hidden_size)
        // target captures for prefill, or (K × hidden_size) per-iter for decode.
        TopkGenerateResult topk_generate(
            LanguageModel& target_lm,
            std::vector<uint32_t> input_ids,
            std::vector<Eigen::bfloat16> hidden_states,
            int num_cached_tokens,
            bool is_prefill
        );

        // Result of run_eagle3_draft_model (lm_head is fused into post_model).
        struct DraftForwardResult;

        struct Gemma4MtpTargetBatchResult;

        struct Gemma4MtpDraftStepResult;

        // Draft model forward with lm_head fused. is_prefill picks n128 (with FC
        // fusion on 3*hidden_size captures) vs n5 (single-layer hidden_size input).
        DraftForwardResult run_eagle3_draft_model(
            LanguageModel& target_lm,
            std::vector<Eigen::bfloat16> hidden_states,
            std::vector<uint32_t> input_ids,
            std::optional<std::vector<uint8_t>> attention_mask,
            std::optional<std::vector<int32_t>> position_ids,
            int num_cached_tokens,
            bool is_prefill
        );

        // Result of run_eagle3_target_verify.
        struct TargetVerifyResult;

        // Target verify (n16). position_ids are absolute. Captures hidden states
        // at layers 2, N/2, N-3. KV write offset is baked via model_key.
        TargetVerifyResult run_eagle3_target_verify(
            std::vector<uint32_t> input_ids,
            std::vector<int32_t> position_ids
        );

        // Result of tree_decoding.
        struct TreeDecodingResult;

        // Tree decoding: target verify + retrieve_indices logit gather +
        // concat of the 3 layer captures.
        TreeDecodingResult tree_decoding(
            std::vector<uint32_t> tree_candidates,
            std::vector<int32_t> tree_position_ids,
            std::span<const uint32_t> input_ids,
            const std::vector<std::vector<int32_t>>& retrieve_indices
        );

        // Target prefill + initial topk_generate. input_ids is by-value so the
        // body can append the root token before passing to topk_generate.
        InitTreeResult initialize_tree(
            LanguageModel& draft_lm,
            std::vector<uint32_t> input_ids,
            int num_cached_tokens,
            ChronoTimer timer_ttft
        );

        // Gather the accepted path's scattered KVs (at select_indices) and move
        // them to contiguous positions starting at prev_input_len. Updates
        // _cache->kv_cache_len. Operates on all layers' cache_{key,val}_l{i} buffers.
        void compact_kv_after_accept(
            std::span<const uint16_t> select_indices,
            uint16_t prev_input_len
        );

        // Per-iter update after target verify + accept/reject: compacts KV to the
        // accepted path and builds the next tree from the target bonus token.
        struct UpdateInferenceInputsResult;

        UpdateInferenceInputsResult update_inference_inputs(
            LanguageModel& draft_lm,
            std::vector<uint32_t> input_ids,
            const std::vector<std::vector<int32_t>>& candidates,
            size_t best_candidate,
            int32_t accept_length,
            const std::vector<std::vector<int32_t>>& retrieve_indices,
            int32_t new_token,
            const std::vector<Eigen::bfloat16>& hidden_state_new,
            uint32_t bonus_token
        );

        std::optional<std::vector<uint32_t>> _generate_tokens(
            RequestScope& request, std::span<const uint32_t> ids,
            std::optional<uint16_t> limit, std::optional<ChronoTimer> timer,
            GenerationPerformanceResult* performance, uint16_t stable_prefix);
        std::optional<std::vector<uint32_t>> _run_model_ordinary(
            std::span<const uint32_t> ids, std::optional<ChronoTimer> timer,
            uint16_t stable_prefix, GenerationPerformanceResult* performance);
        std::optional<std::vector<uint32_t>> _run_model_gemma4_mtp(
            LanguageModel& draft, std::span<const uint32_t> ids, uint16_t limit,
            std::optional<ChronoTimer> timer, GenerationPerformanceResult* performance);
        LogLikelihoodResult _score(std::span<const uint32_t> ids, size_t continuation_start,
            std::span<const uint32_t> continuation_ids, bool use_group_prefill);
        void invalidate_prompt_reuse();
        void invalidate_all_prompt_reuse();
        bool _select_cache(const std::optional<std::string>& id);
        void _validate_eagle3_target() const;
        void _bind_adapter(const std::shared_ptr<LoRAAdapter>& adapter);
        void set_image_identity(std::string identity);

        std::shared_ptr<LanguageAssets> _resources;
        const VlmConfig& _cfg;
        const std::filesystem::path& _elf_dir;
        const std::filesystem::path& _devkit_dir;
        const std::shared_ptr<FileProvider>& _files;
        std::shared_ptr<spdlog::logger> _logger;
        std::unique_ptr<SequenceCache> _cache;
        // Programs retain pointers to _cache's buffer objects. Inactive slots
        // exchange allocation handles with those stable objects, never KV bytes.
        std::vector<std::unique_ptr<SequenceCache>> _cache_slots;
        CacheSlots _cache_ids;
        size_t _active_cache_slot = 0;
        std::unique_ptr<LanguageWorkspace> _workspace;
        std::unique_ptr<RequestControl> _run;

        void define_buffer(const std::string& name, const std::vector<size_t>& shape,
                           const std::string& dtype = "bfloat16", bool align_last_dim = true);
        void _define_resource_buffer(const std::string& name, const std::vector<size_t>& shape,
                                     const std::string& dtype = "bfloat16", bool align_last_dim = true);
        void _define_cache_buffer(const std::string& name, const std::vector<size_t>& shape,
                                    const std::string& dtype = "bfloat16", bool align_last_dim = true);
        MLABuffer& get_buffer(const std::string& name);
        bool has_buffer(const std::string& name) const;
        void _init_attention_bindings();
        void _gather_embedding_rows(
            std::span<const uint32_t> ids, MLABuffer& destination,
            MLABuffer* scales, size_t row_begin = 0
        );
        void _prepare_offloaded_prompt(uint16_t num_tokens, uint16_t token_idx);
        void _define_draft_fc_models();
        void _define_dflash_models();
        void _append_dflash_context(
            LanguageModel& target_lm, uint16_t num_tokens,
            uint16_t token_idx, uint16_t valid_tokens
        );
        struct DFlashScratch;
        MLABuffer* _get_dflash_hidden_buffer(uint16_t num_tokens, uint8_t layer_idx);
        void _bind_dflash_linear_conv_state(uint16_t prefix_tokens);
        void _resolve_dflash_linear_state(uint16_t prefix_tokens);
        void _commit_dflash_linear_state(uint16_t prefix_tokens);
        void _save_dflash_state_checkpoint(
            uint16_t token_count, uint16_t prefix_tokens
        );
        void _upload_dflash_attention_mask(
            uint16_t num_tokens, uint16_t token_idx,
            uint8_t layer_idx, bool bidirectional,
            std::vector<Eigen::bfloat16>& mask
        );
        std::pair<uint16_t, uint32_t> _run_dflash_target_verify(
            std::span<const uint32_t> input_ids, uint16_t token_idx,
            DFlashScratch& scratch
        );
        const std::vector<uint32_t>& _run_dflash_draft(
            LanguageModel& target_lm, uint32_t anchor_token, uint16_t token_idx,
            DFlashScratch& scratch
        );
        std::optional<std::vector<uint32_t>> _run_model_dflash_speculative_decoding(
            LanguageModel& draft_lm,
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens,
            std::optional<ChronoTimer> timer_ttft,
            GenerationPerformanceResult* performance_result,
            uint16_t stable_prefix_token_count
        );
        std::optional<std::vector<uint32_t>> _run_model_eagle3_speculative_decoding(
            LanguageModel& draft_lm,
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens,
            std::optional<ChronoTimer> timer_ttft,
            GenerationPerformanceResult* performance_result
        );
        // Describes which recurrent or convolution layers to save and their state layout in memory.
        struct CheckpointLayout {
            // Hidden-layer indices belonging to this stateful family.
            std::vector<uint8_t> layer_indices;
            // MLA buffer name prefix; full name is `{prefix}{layer_idx}`.
            std::string buffer_name_prefix;
            // Sequence positions in one tail snapshot (e.g. conv_L_cache - 1).
            uint16_t tail_len;
            // Elements per sequence position (e.g. hidden_size).
            uint32_t num_elems;
            // Bytes per element.
            size_t elem_size;
            // Bytes per tail snapshot = tail_len * num_elems * elem_size.
            size_t tail_bytes;
            // Group prefill models output only one final state row, not per-token state history.
            bool prefill_single_output = false;
        };
        static constexpr size_t checkpoint_slot_count = 4;
        // Owns per-layer state snapshots at saved sequence positions for prompt-prefix reuse.
        struct RecurrentCheckpoints {
            explicit RecurrentCheckpoints(const CheckpointLayout& description)
                : layout(description), checkpoints(layout.layer_indices.size()) {}
            CheckpointLayout layout;
            // Each occupied slot owns the snapshot for one layer at one sequence position.
            std::vector<std::array<std::vector<uint8_t>, checkpoint_slot_count>> checkpoints;
        };
        void _initialize();
        void _finalize();
        void _define_buffer_freq_table(const std::string& name, uint32_t rope_dimension_count);
        bool _has_linear_attention_layers() const;
        const LinearAttentionConfig& _linear_attn_cfg() const;
        void _define_buffers();
        void _define_model(
            LanguageModelMap& models,
            const LanguageModelMapKey& key,
            const std::filesystem::path& model_path,
            const std::vector<MLABufferSlice>& ifms,
            const std::vector<MLABufferSlice>& ofms
        );
        void _define_attn_models_iter(uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx);
        // Mixture-of-Experts post block: router + experts + weighted-sum (replaces post).
        void _define_moe_post_models(
            uint16_t num_tokens, uint8_t layer_idx,
            const std::vector<MLABufferSlice>& post_ifms,
            const std::vector<MLABufferSlice>& post_ofms
        );
        // Runtime for one MoE post block: router -> host weight scatter -> experts ->
        // weighted-sum. Gated on _cfg.lm_cfg.is_moe().
        void _run_moe_post(
            std::map<uint8_t, MLABufferSlice>* ifm_map,
            uint16_t num_tokens, uint8_t layer_idx
        );
        LanguageModelMapKey _get_cache_model_key(
            uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
        ) const;
        uint16_t _get_cache_token_idx_begin(
            uint16_t num_tokens, uint16_t token_idx, LayerType layer_type
        ) const;
        void _define_state_models_iter(uint16_t num_tokens, uint8_t layer_idx);
        void _define_conv_models_iter(uint16_t num_tokens, uint8_t layer_idx);
        void _define_linear_models_iter(uint16_t num_tokens, uint8_t layer_idx);
        void _define_models();
        void _define_per_layer_models();
        std::filesystem::path _get_elf_path_pre(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_cache(
            uint16_t num_tokens,
            uint16_t token_idx,
            bool use_sliding_cache
        );
        std::filesystem::path _get_elf_path_post(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_router(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_expert(
            uint16_t num_tokens, uint8_t layer_idx, uint16_t expert_idx
        );
        std::filesystem::path _get_elf_path_weightedsum(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_conv(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_conv_final(uint8_t layer_idx);
        std::filesystem::path _get_elf_path_per_layer(uint16_t num_tokens);
        std::filesystem::path _get_elf_path_linear(uint16_t num_tokens, uint8_t layer_idx);
        static constexpr uint16_t LONG_CONTEXT_MIN_TOKENS = 2048;
        static constexpr uint16_t MAX_NUM_TOKENS_ALIGNMENT = 1024;
        uint16_t _get_cache_mask_size(
            LayerType layer_type, uint16_t context_length, bool is_group
        ) const {
            if (
                layer_type != LayerType::SlidingAttention
                && _cfg.pipeline_cfg.long_context_future_token_mask_size.has_value()
                && context_length > LONG_CONTEXT_MIN_TOKENS
            ) {
                return _cfg.pipeline_cfg.long_context_future_token_mask_size.value();
            }
            return is_group
                ? _cfg.pipeline_cfg.input_token_group_size
                : _cfg.pipeline_cfg.future_token_mask_size;
        }
        uint16_t _get_max_future_token_mask_size() const {
            return std::max(
                _cfg.pipeline_cfg.future_token_mask_size,
                _cfg.pipeline_cfg.long_context_future_token_mask_size.value_or(0)
            );
        }
        bool _uses_per_layer_inputs() const {
            return _cfg.model_type == "vlm-gemma4" && _cfg.lm_cfg.hidden_size_per_layer_input > 0;
        }
        void _init_state_checkpoint_capture(
            size_t input_token_count, uint16_t stable_prefix_token_count
        );
        uint16_t _prepare_state_checkpoints_for_prefill(uint16_t num_cached_tokens);
        void _upload_group_future_token_masks(uint16_t num_tokens, uint16_t token_idx);
        void _save_state_checkpoint(
            uint16_t token_count, uint16_t num_tokens, uint16_t valid_tokens, bool is_prefill
        );
        std::optional<size_t> _select_state_checkpoint_slot(
            uint16_t token_count, bool is_prefill
        );
        void _move_state_tail_for_decode(uint16_t valid_tokens);

        uint16_t _set_input_text_embeds(std::span<const uint32_t> input_token_ids);
        void _stage_embedding_rows(
            LanguageModel& source_model,
            std::span<const uint32_t> token_ids,
            MLABuffer& destination,
            MLABuffer* destination_scales = nullptr
        );
        void _run_model_once_for_loglikelihood_logits(
            uint16_t token_idx, uint32_t input_token_id
        );
        LogLikelihoodResult _run_model_once_for_loglikelihood(
            uint16_t token_idx, uint32_t input_token_id, uint32_t target_token_id
        );
        std::vector<uint32_t> _get_per_layer_token_ids(
            std::span<const uint32_t> input_token_ids
        ) const;
        void _upload_per_layer_embedding_rows(
            std::span<const uint32_t> token_ids, uint16_t num_tokens,
            uint16_t valid_tokens = 0
        );
        void _init_cache_layout();
        void _compute_and_upload_per_layer_inputs_prefill(
            uint16_t num_tokens, uint16_t token_idx, uint16_t num_input_tokens
        );
        uint32_t _calc_next_token_id(MLABuffer* buf_ptr);
        struct LMHeadLogitView;
        void _prepare_lm_head_logit_views(
            uint16_t num_tokens, uint16_t output_rows, std::vector<LMHeadLogitView>& views
        );
        static uint32_t _argmax_lm_head_row(
            std::span<const LMHeadLogitView> views, uint16_t row
        );
        uint32_t _argmax_gemma4_mtp_masked_row(uint16_t num_tokens, uint16_t row);
        std::vector<uint32_t> _argmax_lm_head_rows(
            uint16_t num_tokens, uint16_t valid_tokens,
            std::span<const uint32_t> expected_draft_tokens
        );
        std::vector<Eigen::bfloat16> _read_embedding_row_bf16(uint32_t token_id);
        std::vector<Eigen::bfloat16> _read_gemma4_mtp_target_hidden_rows(
            uint16_t num_tokens, uint16_t valid_tokens
        );
        void _upload_gemma4_mtp_freq_rows(
            uint16_t num_tokens, uint16_t position_id, uint16_t valid_tokens
        );
        void _upload_gemma4_mtp_causal_mask(
            MLABuffer& buffer,
            uint16_t num_tokens,
            uint16_t first_visible_token_count,
            uint16_t valid_tokens,
            uint16_t cache_token_idx_begin,
            uint16_t context_length,
            uint16_t sliding_window = 0
        );
        Gemma4MtpTargetBatchResult _run_gemma4_mtp_target_batch(
            uint16_t token_idx, std::span<const uint32_t> token_ids
        );
        Gemma4MtpDraftStepResult _run_gemma4_mtp_draft_step(
            LanguageModel& target_lm,
            uint32_t token_id,
            const std::vector<Eigen::bfloat16>& hidden_state,
            uint16_t query_position_id,
            uint16_t shared_kv_len
        );

        uint32_t _prefill(std::span<const uint32_t> input_token_ids, uint16_t num_cached_tokens,
                          std::optional<ChronoTimer> timer_ttft, std::vector<double>* durations);
        void _decode(uint16_t num_input_tokens, uint32_t token_id, std::vector<double>* durations);
        void _notify_first_token(uint32_t token_id, double duration);
        void _notify_new_token(
            uint32_t token_id, double duration, bool from_draft = false
        );
        void _notify_cache_full() const;
        void _notify_stop() const;
        void _notify_interrupt() const;

        std::optional<uint32_t> _image_token_id;
        std::optional<uint32_t> _pad_token_id;
        bool _use_group_token_models;
        bool _need_argmax;

        // Mixture-of-Experts maps. router/weightedsum are keyed like post
        // ({num_tokens, layer_idx, 0}); the expert map repurposes the third key slot
        // as the expert index ({num_tokens, layer_idx, expert_idx}).
        // Preresolved handles + staging for the MoE host round-trip, one per MoE variant.
        struct MoeHostCache;
        std::vector<MoeHostCache> _moe_host;
        void _init_moe_host_cache();
        MoeHostCache& _get_moe_host(uint16_t moe_nt);

};

}
}

#endif
