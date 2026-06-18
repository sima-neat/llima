#ifndef _SIMA_LLIMA_LANGUAGE_MODEL_
#define _SIMA_LLIMA_LANGUAGE_MODEL_

#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "base_model.hpp"
#include "eagle_helpers.hpp"
#include "rope_utils.hpp"
#include "text_streamer.hpp"

namespace simaai {
namespace llima {

struct LogLikelihoodResult {
    double logprob;
    bool is_greedy;
};

// Key to access the language model map: (num_tokens, layer_idx, token_idx).
using LanguageModelMapKey = std::tuple<uint16_t, uint8_t, uint16_t>;
using LanguageModelMap = std::map<LanguageModelMapKey, MLAModelWithBuffer>;

class LanguageModel : public BaseModel<VlmConfig> {
    public:
        LanguageModel(
            std::filesystem::path model_path,
            std::set<uint32_t> stop_token_ids,
            std::optional<uint32_t> image_token_id,
            std::optional<uint32_t> pad_token_id,
            TextStreamer& text_streamer
        );
        virtual ~LanguageModel() override { _finalize(); }

        std::vector<std::map<uint8_t, MLABufferSlice>> create_input_buffers(
            std::span<const uint32_t> input_token_ids
        );
        std::optional<std::vector<uint32_t>> run_model(
            std::span<const uint32_t> input_token_ids,
            std::optional<ChronoTimer> timer_ttft = std::nullopt,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt
        );
        uint32_t run_model_prefill(
            std::span<const uint32_t> input_token_ids,
            uint16_t num_cached_tokens,
            std::optional<ChronoTimer> = std::nullopt
        );
        void run_model_decode(uint16_t num_input_tokens, uint32_t token_id);
        uint32_t run_model_once(
            uint16_t num_tokens,
            uint16_t token_idx,
            uint16_t num_input_tokens,
            uint32_t token_id,
            std::vector<Eigen::bfloat16>* logits_ptr = nullptr,
            bool skip_output = false
        );
        LogLikelihoodResult run_model_for_loglikelihood(
            std::span<const uint32_t> input_token_ids,
            size_t continuation_start,
            std::span<const uint32_t> continuation_token_ids,
            bool use_group_prefill = true
        );
        // Speculative-decoding entry point: target invokes this, passing the
        // draft as a reference. Returns the newly generated token IDs, or
        // std::nullopt when generation aborted (e.g. cache full pre-init).
        // Caller is responsible for serializing concurrent invocations.
        std::optional<std::vector<uint32_t>> run_model_speculative_decoding(
            LanguageModel& draft_lm,
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<ChronoTimer> timer_ttft = std::nullopt
        );
        void stop_model() { _is_running = false; }

        void set_reloc(const std::string& reloc_name);
        void unset_reloc();

        LanguageModelMap& get_model_map(const std::string& model_type);
        uint16_t set_max_num_tokens(std::optional<uint16_t> max_num_tokens);
        auto get_max_num_tokens() const { return _max_num_tokens; }
        std::set<uint32_t> set_stop_token_ids(std::optional<std::set<uint32_t>> stop_token_ids);
        const auto& get_stop_token_ids() const { return _stop_token_ids; }
        void clear_cached_token_ids() { _cached_token_ids.clear(); }

        // Captured hidden states from layers 2, N/2, N-3 during prefill (spec decoding).
        // Populated by run_model_once when queue is disabled and spec mode is active.
        const std::vector<std::vector<Eigen::bfloat16>>& get_eagle3_intermediate_hidden_states() const {
            return _eagle3_intermediate_hidden_states;
        }

        // Result of initialize_tree.
        struct InitTreeResult {
            std::vector<uint32_t> draft_tokens;
            std::vector<std::vector<int32_t>> retrieve_indices;
            eagle_helpers::EagleTreeMask tree_mask;  // 4D (1, 1, N, N)
            std::vector<int32_t> tree_position_ids;
            std::vector<std::vector<Eigen::bfloat16>> hidden_states;  // 3 captured layers
            uint32_t token;                                            // root token
            std::chrono::steady_clock::time_point root_ready_time;
        };

        // Result of topk_generate.
        struct TopkGenerateResult {
            std::vector<uint32_t> draft_tokens;
            std::vector<std::vector<int32_t>> retrieve_indices;
            eagle_helpers::EagleTreeMask tree_mask;  // 4D (1, 1, N, N)
            std::vector<int32_t> tree_position_ids;
        };

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
        struct DraftForwardResult {
            std::vector<Eigen::bfloat16> hidden_states;  // (seq_length, hidden_size)
            std::vector<Eigen::bfloat16> logits;         // (num_tokens, draft_vocab_size)
        };

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
        struct TargetVerifyResult {
            std::vector<Eigen::bfloat16> logits;                       // (num_tokens, lm_head_output)
            std::vector<std::vector<Eigen::bfloat16>> hidden_states;   // 3 captures, each (num_tokens, hidden_size)
        };

        // Target verify (n16). position_ids are absolute. Captures hidden states
        // at layers 2, N/2, N-3. KV write offset is baked via model_key.
        TargetVerifyResult run_eagle3_target_verify(
            std::vector<uint32_t> input_ids,
            std::vector<int32_t> position_ids
        );

        // Result of tree_decoding.
        struct TreeDecodingResult {
            std::vector<Eigen::bfloat16> logits;          // flat (n_paths × max_depth × vocab_size)
            size_t vocab_size;
            std::vector<Eigen::bfloat16> hidden_states;   // flat (K × 3 × hidden_size)
        };

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
        // _kv_cache_len. Operates on all layers' cache_{key,val}_l{i} buffers.
        void compact_kv_after_accept(
            std::span<const uint16_t> select_indices,
            uint16_t prev_input_len
        );

        // Per-iter update after target verify + accept/reject: compacts KV to the
        // accepted path, samples bonus_token from sample_p, builds next tree.
        struct UpdateInferenceInputsResult {
            std::vector<uint32_t> input_ids;                       // old + accepted + bonus
            std::vector<uint32_t> draft_tokens;                    // next round's tree tokens
            std::vector<std::vector<int32_t>> retrieve_indices;    // next round
            eagle_helpers::EagleTreeMask tree_mask;                // next round
            std::vector<int32_t> tree_position_ids;                // next round
            int32_t new_token;                                     // accumulated generated-token count
            uint32_t bonus_token;                                  // = argmax(sample_p)
        };

        UpdateInferenceInputsResult update_inference_inputs(
            LanguageModel& draft_lm,
            std::vector<uint32_t> input_ids,
            const std::vector<std::vector<int32_t>>& candidates,
            size_t best_candidate,
            int32_t accept_length,
            const std::vector<std::vector<int32_t>>& retrieve_indices,
            int32_t new_token,
            const std::vector<Eigen::bfloat16>& hidden_state_new,
            const std::vector<Eigen::bfloat16>& sample_p
        );

        // Number of tokens currently in the KV cache (set by prefill, advanced by decode).
        uint16_t get_kv_cache_len() const { return _kv_cache_len; }
        void set_kv_cache_len(uint16_t len) { _kv_cache_len = len; }

        // Current spec round's tree mask, shared between target and draft via
        // shared_ptr. nullptr = "no tree active". Cleared on both at round start.
        std::shared_ptr<eagle_helpers::EagleTreeMask> _eagle3_tree_mask;

        // eye(topk) identity, built once (draft only). Depth-loop seed.
        eagle_helpers::EagleTreeMask _eagle3_tree_mask_init;

        // zeros(topk); used in depth loop as position_ids = len_posi + offsets.
        std::vector<int32_t> _eagle3_position_ids;

        // Cumulative draft KV position. Reset per round; advanced by hidden_rows
        // after each first draft_forward in topk_generate.
        size_t _eagle3_stable_kv = 0;

    private:
        void _define_draft_fc_models();
        struct CachedState {
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
            // Tail snapshots indexed as [layer_slot][boundary_idx][byte_offset].
            std::vector<std::vector<std::vector<uint8_t>>> checkpoints;
        };

        virtual void _initialize() override;
        virtual void _finalize() override;
        void _define_buffer_freq_table(const std::string& name, uint32_t rope_dimension_count);
        virtual void _define_buffers() override;
        void _define_model(
            const std::string& model_type,
            const LanguageModelMapKey& key,
            const std::filesystem::path& model_path,
            const std::vector<MLABufferSlice>& ifms,
            const std::vector<MLABufferSlice>& ofms
        );
        void _define_attn_models_iter(uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx);
        void _define_state_models_iter(uint16_t num_tokens, uint8_t layer_idx);
        void _define_conv_models_iter(uint16_t num_tokens, uint8_t layer_idx);
        void _define_models();
        void _define_per_layer_models();
        std::filesystem::path _get_elf_path_pre(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_cache(
            uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
        );
        std::filesystem::path _get_elf_path_post(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_conv(uint16_t num_tokens, uint8_t layer_idx);
        std::filesystem::path _get_elf_path_conv_final(uint8_t layer_idx);
        std::filesystem::path _get_elf_path_per_layer(uint16_t num_tokens);
        bool _uses_per_layer_inputs() const {
            return _cfg.model_type == "vlm-gemma4" && _cfg.lm_cfg.hidden_size_per_layer_input > 0;
        }
        bool _uses_cpu_dequantized_embeddings() const {
            return _cfg.pipeline_cfg.quantize_embeddings
                && _cfg.vm_cfg.has_value() && _cfg.mm_cfg.has_value();
        }
        uint16_t _prepare_state_checkpoints_for_prefill(uint16_t num_cached_tokens);
        void _save_state_checkpoint(
            size_t boundary_idx, uint16_t num_tokens, uint16_t valid_tokens
        );
        void _move_state_tail_for_decode(uint16_t valid_tokens);

        uint16_t _set_input_text_embeds(std::span<const uint32_t> input_token_ids);
        void _dequantize_embedding_row(
            uint32_t token_id, MLABuffer& dst, size_t dst_row = 0
        );
        LogLikelihoodResult _run_model_once_for_loglikelihood(
            uint16_t token_idx, uint32_t input_token_id, uint32_t target_token_id
        );
        std::vector<uint32_t> _get_per_layer_token_ids(
            std::span<const uint32_t> input_token_ids
        ) const;
        void _upload_per_layer_embedding_rows(
            std::span<const uint32_t> token_ids, uint16_t num_tokens
        );
        void _load_per_layer_embeddings();
        void _compute_and_upload_per_layer_inputs_prefill(
            uint16_t num_tokens, uint16_t token_idx, uint16_t num_input_tokens
        );
        uint32_t _calc_next_token_id(MLABuffer* buf_ptr);

        void _notify_first_token(uint32_t token_id, double duration);
        void _notify_new_token(uint32_t token_id, double duration);
        void _notify_cache_full() const;
        void _notify_stop() const;
        void _notify_interrupt() const;

        std::set<uint32_t> _stop_token_ids;
        std::optional<uint32_t> _image_token_id;
        std::optional<uint32_t> _pad_token_id;
        uint16_t _max_num_tokens;
        TextStreamer& _text_streamer;
        bool _use_group_token_models;
        bool _need_argmax;

        LanguageModelMap _pre_model_map;
        LanguageModelMap _cache_model_map;
        LanguageModelMap _post_model_map;
        LanguageModelMap _conv_model_map;
        LanguageModelMap _conv_final_model_map;
        LanguageModelMap _per_layer_model_map;
        // Draft-only: FC fusion models indexed by num_tokens (128 prefill, 5 decode).
        std::map<uint16_t, MLAModelWithBuffer> _fc_model_map;

        RopeTable _master_rope_table;
        RopeTable _global_freq_host;  // pristine CPU copy; source for tree-RoPE rows
        RopeTable _local_freq_host;   // empty if SWA not enabled

        bool _has_image_token;

        Eigen::bfloat16* _embeddings_tensor_ptr;
        size_t _per_layer_embedding_rows_per_shard = 0;
        std::vector<MLABuffer*> _per_layer_embedding_shards;
        std::vector<uint32_t> _prompt_per_layer_token_ids;
        std::vector<uint32_t> _cached_token_ids;
        uint32_t _cached_first_generated_token;
        // Full-match cache: prior turn's tree state. Restored only when the
        // matching token prefix also has the same prompt length.
        std::vector<uint32_t> _cached_draft_tokens;
        std::vector<std::vector<int32_t>> _cached_retrieve_indices;
        eagle_helpers::EagleTreeMask _cached_tree_mask;
        std::vector<int32_t> _cached_tree_position_ids;
        uint16_t _cached_eagle3_prompt_len = 0;
        size_t _cached_eagle3_stable_kv = 0;
        uint16_t _kv_cache_len = 0;  // tokens in KV cache; set by prefill, advanced by decode
        std::vector<int32_t> _d2t;   // draft-to-target vocab offsets (draft only)
        std::vector<uint16_t> _checkpoint_boundaries;
        std::vector<CachedState> _cached_states;

        std::vector<std::vector<Eigen::bfloat16>> _eagle3_intermediate_hidden_states;

        std::atomic<bool> _is_running;
        std::optional<std::string> _reloc_name;
};

}
}

#endif
