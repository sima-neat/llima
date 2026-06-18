#ifndef _SIMA_LLIMA_LANGUAGE_MODEL_
#define _SIMA_LLIMA_LANGUAGE_MODEL_

#include <atomic>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include "base_model.hpp"
#include "rope_utils.hpp"
#include "text_streamer.hpp"

namespace simaai {
namespace llima {

struct LogLikelihoodResult {
    double logprob;
    bool is_greedy;
    double model_token_seconds = 0.0;
    double score_logits_seconds = 0.0;
    double prefill_seconds = 0.0;
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
            TextStreamer& text_streamer,
            bool do_parallel_load
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
        void stop_model() { _is_running = false; }

        void set_reloc(const std::string& reloc_name);
        void unset_reloc();

        LanguageModelMap& get_model_map(const std::string& model_type);
        uint16_t set_max_num_tokens(std::optional<uint16_t> max_num_tokens);
        auto get_max_num_tokens() const { return _max_num_tokens; }
        std::set<uint32_t> set_stop_token_ids(std::optional<std::set<uint32_t>> stop_token_ids);
        const auto& get_stop_token_ids() const { return _stop_token_ids; }
        void clear_cached_token_ids() { _cached_token_ids.clear(); }

    private:
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
        uint16_t _prepare_state_checkpoints_for_prefill(uint16_t num_cached_tokens);
        void _save_state_checkpoint(
            size_t boundary_idx, uint16_t num_tokens, uint16_t valid_tokens
        );
        void _move_state_tail_for_decode(uint16_t valid_tokens);

        uint16_t _set_input_text_embeds(std::span<const uint32_t> input_token_ids);
        LogLikelihoodResult _run_model_once_for_loglikelihood(
            uint16_t token_idx, uint32_t input_token_id, uint32_t target_token_id
        );
        std::vector<uint32_t> _get_per_layer_token_ids(
            std::span<const uint32_t> input_token_ids
        ) const;
        void _upload_per_layer_embedding_rows(
            std::span<const uint32_t> token_ids, uint16_t num_tokens
        );
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
        bool _do_parallel_load;
        bool _use_group_token_models;
        bool _need_argmax;

        LanguageModelMap _pre_model_map;
        LanguageModelMap _cache_model_map;
        LanguageModelMap _post_model_map;
        LanguageModelMap _conv_model_map;
        LanguageModelMap _conv_final_model_map;
        LanguageModelMap _per_layer_model_map;

        RopeTable _master_rope_table;
        bool _has_image_token;

        Eigen::bfloat16* _embeddings_tensor_ptr;
        std::vector<Eigen::bfloat16> _per_layer_embeddings_tensor;
        std::vector<uint32_t> _prompt_per_layer_token_ids;
        std::vector<uint32_t> _cached_token_ids;
        uint32_t _cached_first_generated_token;
        std::vector<uint16_t> _checkpoint_boundaries;
        std::vector<CachedState> _cached_states;

        std::atomic<bool> _is_running;
        std::optional<std::string> _reloc_name;
};

}
}

#endif
