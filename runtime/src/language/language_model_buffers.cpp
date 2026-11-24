#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <limits>

#include "language_model.hpp"
#include "language/language_model_state.hpp"

namespace simaai {
namespace llima {
namespace {
constexpr size_t PER_LAYER_EMBEDDING_MAX_SHARD_SIZE = 1024ULL * 1024 * 1024;
}



void LanguageModel::define_buffer(const std::string& name, const std::vector<size_t>& shape,
                                  const std::string& dtype, bool align_last_dim) {
    _workspace->buffers.erase(name);
    _workspace->buffers.try_emplace(name, name, shape, dtype, align_last_dim);
}

void LanguageModel::_define_resource_buffer(const std::string& name,
                                            const std::vector<size_t>& shape,
                                            const std::string& dtype, bool align_last_dim) {
    _resources->buffers.try_emplace(name, name, shape, dtype, align_last_dim);
}

void LanguageModel::_define_cache_buffer(const std::string& name,
                                           const std::vector<size_t>& shape,
                                           const std::string& dtype, bool align_last_dim) {
    _cache->buffers.try_emplace(name, name, shape, dtype, align_last_dim);
}

MLABuffer& LanguageModel::get_buffer(const std::string& name) {
    if (auto it = _workspace->buffers.find(name); it != _workspace->buffers.end())
        return it->second;
    if (auto it = _cache->buffers.find(name); it != _cache->buffers.end())
        return it->second;
    return _resources->buffers.at(name);
}

bool LanguageModel::has_buffer(const std::string& name) const {
    return _workspace->buffers.contains(name) || _cache->buffers.contains(name) ||
           _resources->buffers.contains(name);
}

void LanguageModel::_define_buffer_freq_table(const std::string& name, uint32_t rope_dimension_count) {
    define_buffer(
        fmt::format("{}_freq_real", name),
        {_cfg.pipeline_cfg.max_num_tokens, rope_dimension_count / 2}
    );
    define_buffer(
        fmt::format("{}_freq_imag", name),
        {_cfg.pipeline_cfg.max_num_tokens, rope_dimension_count / 2}
    );
}


void LanguageModel::_define_buffers() {
    // Embedding table. Drafts use the target's embeddings, so skip.
    const bool is_draft = _cfg.lm_cfg.is_speculative_draft();
    const bool is_eagle3_draft = _cfg.lm_cfg.is_eagle3_draft();
    const bool is_gemma4_mtp_draft = _cfg.lm_cfg.is_gemma4_mtp_draft();
    if (!is_draft) {
        if (!_resources->embedding_offload) _define_resource_buffer(
            "embeddings",
            {_cfg.lm_cfg.token_cfg.vocab_size, _cfg.lm_cfg.hidden_size},
            (_cfg.pipeline_cfg.quantize_embeddings)? "int8" : "bfloat16"
        );
        if (_resources->embedding_offload) {
            define_buffer("offload_decode_embeds", {1, _cfg.lm_cfg.hidden_size},
                _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16");
            if (_cfg.pipeline_cfg.quantize_embeddings) define_buffer("offload_decode_scales", {1, 1});
        }
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            _define_resource_buffer("embedding_scales", {_cfg.lm_cfg.token_cfg.vocab_size, 1});
        }
    }
    if (
        _cfg.lm_cfg.is_spec_decode()
        && _cfg.pipeline_cfg.quantize_embeddings
    ) {
        const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
        define_buffer(
            fmt::format("eagle3_input_embeds_n{}", single_num_tokens),
            {single_num_tokens, _cfg.lm_cfg.hidden_size},
            "int8"
        );
        define_buffer(
            fmt::format("eagle3_input_embedding_scales_n{}", single_num_tokens),
            {single_num_tokens, 1}
        );
        const uint16_t group_num_tokens = _cfg.pipeline_cfg.input_token_group_size;
        if (is_draft && _use_group_token_models && group_num_tokens != single_num_tokens) {
            define_buffer(
                fmt::format("eagle3_input_embeds_n{}", group_num_tokens),
                {group_num_tokens, _cfg.lm_cfg.hidden_size},
                "int8"
            );
            define_buffer(
                fmt::format("eagle3_input_embedding_scales_n{}", group_num_tokens),
                {group_num_tokens, 1}
            );
        }
    }

    // Frequency tables.
    _define_buffer_freq_table(
        "global", _cfg.lm_cfg.rope_cfg.get_rope_dimension_count(LayerType::FullAttention)
    );
    if (_cfg.lm_cfg.attn_cfg.swa_enable)
        _define_buffer_freq_table(
            "local", _cfg.lm_cfg.rope_cfg.get_rope_dimension_count(LayerType::SlidingAttention)
        );

    if (_has_linear_attention_layers()) {
        define_buffer(
            "linear_valid_mask",
            {
                std::max(
                    _cfg.pipeline_cfg.input_token_group_size,
                    _cfg.lm_cfg.get_single_num_tokens()
                ),
                1
            },
            "bfloat16",
            true
        );
    }

    // KV caches.
    std::vector<size_t> cache_shape;
    const uint16_t conv_working_len = _cfg.pipeline_cfg.input_token_group_size + _cfg.lm_cfg.conv_L_cache - 2;
    std::vector<size_t> conv_cache_shape{conv_working_len, _cfg.lm_cfg.hidden_size};
    for (uint8_t i = 0; i < _cfg.lm_cfg.num_hidden_layers; ++i) {
        if (_cfg.lm_cfg.layer_types[i] == LayerType::Conv) {
            _define_cache_buffer(fmt::format("conv_cache_history_l{}", i), conv_cache_shape);
        } else if (_cfg.lm_cfg.layer_types[i] == LayerType::LinearAttention) {
            const auto& linear_cfg = _linear_attn_cfg();
            const uint16_t linear_conv_working_len = static_cast<uint16_t>(
                _cfg.pipeline_cfg.input_token_group_size + linear_cfg.conv_kernel_dim - 2
            );
            _define_cache_buffer(
                fmt::format("linear_conv_cache_history_l{}", i),
                {linear_conv_working_len, linear_cfg.get_conv_dim()}
            );
            _define_cache_buffer(
                fmt::format("linear_delta_state_history_l{}", i),
                {
                    1,
                    linear_cfg.get_recurrent_state_size()
                }
            );
            // Ping-pong state avoids MLA read-after-write corruption during grouped prefill.
            _define_cache_buffer(
                fmt::format("linear_delta_state_history_alt_l{}", i),
                {1, linear_cfg.get_recurrent_state_size()}
            );
            if (
                _cfg.lm_cfg.is_dflash()
                && !_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft
            ) {
                const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
                define_buffer(
                    fmt::format("linear_conv_prefix_states_l{}", i),
                    {block_size + linear_cfg.conv_kernel_dim - 2, linear_cfg.get_conv_dim()}
                );
                define_buffer(
                    fmt::format("linear_delta_resolver_output_l{}", i),
                    {1, linear_cfg.get_recurrent_state_size()}
                );
                define_buffer(
                    fmt::format("linear_delta_resolver_key_l{}", i),
                    {
                        linear_cfg.num_value_heads,
                        block_size,
                        linear_cfg.key_head_dim
                    }
                );
                define_buffer(
                    fmt::format("linear_delta_resolver_value_l{}", i),
                    {
                        linear_cfg.num_value_heads,
                        block_size,
                        linear_cfg.value_head_dim
                    }
                );
                define_buffer(
                    fmt::format("linear_delta_resolver_decay_mask_l{}", i),
                    {linear_cfg.num_value_heads, block_size, block_size}
                );
            }
        } else {
            if (_cfg.pipeline_cfg.use_strided_kv_cache) {
                cache_shape = {
                    _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                    _cfg.pipeline_cfg.max_num_tokens,
                    _cfg.lm_cfg.attn_cfg.get_head_dim(_cfg.lm_cfg.layer_types[i])
                };
            } else {
                cache_shape = {
                    _cfg.pipeline_cfg.max_num_tokens,
                    _cfg.lm_cfg.attn_cfg.get_kv_size(_cfg.lm_cfg.layer_types[i])
                };
            }
            if (!_cfg.lm_cfg.is_kv_shared_layer(i)) {
                std::string kv_dtype = _cfg.pipeline_cfg.quantize_kv_cache ? "int8" : "bfloat16";
                _define_cache_buffer(fmt::format("cache_key_l{}", i), cache_shape, kv_dtype);
                _define_cache_buffer(fmt::format("cache_val_l{}", i), cache_shape, kv_dtype);
                if (_cfg.pipeline_cfg.quantize_kv_cache) {
                    // Keep the logical C=1 scale shape. MLABuffer aligns each BF16 row to
                    // the 16-byte MLA row required by the physical HWC16 layout.
                    std::vector<size_t> scale_shape = {
                        _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                        _cfg.pipeline_cfg.max_num_tokens,
                        1
                    };
                    _define_cache_buffer(fmt::format("cache_key_scale_l{}", i), scale_shape);
                    _define_cache_buffer(fmt::format("cache_val_scale_l{}", i), scale_shape);
                }
            }
        }
    }

    // gpt_oss attention sinks: per-head logit for the (shared) cache model, one buffer per
    // layer broadcast across tokens; the cache model slices num_tokens rows.
    if (_cfg.lm_cfg.uses_attention_sinks()) {
        uint16_t max_sink_tokens = _cfg.lm_cfg.get_single_num_tokens();
        if (_use_group_token_models) {
            max_sink_tokens = std::max(max_sink_tokens, _cfg.pipeline_cfg.input_token_group_size);
        }
        for (uint8_t i = 0; i < _cfg.lm_cfg.num_hidden_layers; ++i) {
            _define_resource_buffer(
                fmt::format("sinks_l{}", i),
                {max_sink_tokens, _cfg.lm_cfg.attn_cfg.num_attention_heads}
            );
        }
    }

    // Deepstack features for qwen3.
    if (_cfg.vm_cfg.has_value()) {
        for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
            define_buffer(
                fmt::format("deepstack_feature_l{}_cache", i),
                {_cfg.pipeline_cfg.max_num_tokens, _cfg.lm_cfg.hidden_size}
            );
        }
    }

    // Other buffers.
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec{single_num_tokens};
    if (
        _use_group_token_models
        && _cfg.pipeline_cfg.input_token_group_size != single_num_tokens
    ) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    const uint16_t max_future_token_mask_size = _get_max_future_token_mask_size();
    if (max_future_token_mask_size > 1) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            define_buffer(
                "future_token_mask",
                {_cfg.lm_cfg.get_single_num_tokens(), _cfg.pipeline_cfg.max_num_tokens}
            );
        } else {
            uint32_t full_token_mask_size = round_up_to_row(
                _cfg.pipeline_cfg.max_num_tokens + max_future_token_mask_size - 1
            );
            define_buffer(
                "future_token_mask",
                {full_token_mask_size}
            );
        }
    }
    if (
        _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg->is_gemma4_mtp()
        && _cfg.lm_cfg.attn_cfg.swa_enable
    ) {
        // Full and sliding attention can use different compiled mask strides.
        // They need separate storage because the entire transformer is queued.
        define_buffer(
            "gemma4_mtp_sliding_future_token_mask",
            {_cfg.lm_cfg.get_single_num_tokens(), _cfg.pipeline_cfg.max_num_tokens}
        );
    }
    const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
    if (
        _use_group_token_models
        && _get_cache_mask_size(
            LayerType::FullAttention, _cfg.pipeline_cfg.max_num_tokens, true
        ) > group_size
    ) {
        define_buffer(
            "group_future_token_mask",
            {
                static_cast<size_t>(group_size) * _cfg.pipeline_cfg.max_num_tokens
            },
            "bfloat16",
            false
        );
    }
    if (
        _use_group_token_models
        && _cfg.lm_cfg.attn_cfg.swa_enable
        && _get_cache_mask_size(
            LayerType::SlidingAttention, _cfg.pipeline_cfg.max_num_tokens, true
        ) > group_size
    ) {
        define_buffer(
            "group_sliding_future_token_mask",
            {static_cast<size_t>(group_size) * _cfg.pipeline_cfg.max_num_tokens},
            "bfloat16",
            false
        );
    }

    for (const auto& num_tokens: num_tokens_vec) {
        // Pre input (for layer_idx > 0), post input and post output.
        define_buffer(
            fmt::format("n{}_buffer1", num_tokens), {num_tokens, _cfg.lm_cfg.hidden_size}
        );
        if (_cfg.lm_cfg.is_dflash() && !is_draft) {
            const auto capture_count =
                _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids.size();
            for (size_t index = 0; index < capture_count; ++index) {
                define_buffer(
                    fmt::format("dflash_target_hidden_n{}_{}", num_tokens, index),
                    {num_tokens, _cfg.lm_cfg.hidden_size}
                );
            }
        }

        // Pre output and cache input.
        define_buffer(
            fmt::format("n{}_buffer2", num_tokens),
            {_cfg.lm_cfg.attn_cfg.num_attention_heads, num_tokens, _cfg.lm_cfg.attn_cfg.head_dim}
        );

        // Cache output and post input.
        define_buffer(
            fmt::format("n{}_buffer3", num_tokens),
            {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(LayerType::FullAttention)}
        );

        if (_uses_per_layer_inputs()) {
            // Output of the standalone per-layer model, consumed by each post layer.
            define_buffer(
                fmt::format("n{}_per_layer_input", num_tokens),
                {_cfg.lm_cfg.num_hidden_layers * num_tokens, _cfg.lm_cfg.hidden_size_per_layer_input}
            );
        }

        if (_cfg.lm_cfg.is_gemma4_mtp_target()) {
            define_buffer(
                fmt::format("n{}_target_hidden_states", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
        }

        // Mixture-of-Experts working buffers (router outputs + per-expert outputs).
        if (_cfg.lm_cfg.is_moe()) {
            const uint16_t num_experts = _cfg.lm_cfg.moe_cfg.value().num_experts;
            const uint16_t top_k = _cfg.lm_cfg.moe_cfg.value().num_experts_per_tok;
            // Router outputs: top-k weights + selected indices (int32 from the MLA), residual
            // h, and norm(h) (computed once by the router; every expert consumes it).
            define_buffer(fmt::format("n{}_router_values", num_tokens), {num_tokens, top_k});
            define_buffer(
                fmt::format("n{}_router_indices", num_tokens), {num_tokens, top_k}, "int32"
            );
            define_buffer(
                fmt::format("n{}_residual", num_tokens), {num_tokens, _cfg.lm_cfg.hidden_size}
            );
            define_buffer(
                fmt::format("n{}_norm_hidden", num_tokens), {num_tokens, _cfg.lm_cfg.hidden_size}
            );
            // Dense per-expert routing weights (host scatters the k values here).
            define_buffer(fmt::format("n{}_router_weights", num_tokens), {num_tokens, num_experts});
            // Decode routes selected experts into top-k slots; prefill combines all experts.
            const uint16_t num_outputs = num_tokens == 1 ? top_k : num_experts;
            for (uint16_t e = 0; e < num_outputs; ++e) {
                define_buffer(
                    fmt::format("n{}_expert{}", num_tokens, e),
                    {num_tokens, _cfg.lm_cfg.hidden_size}
                );
            }
        }

        // Qwen3.5: transient gate buffer, written by Pre and consumed by Post within the same layer.
        if (_cfg.lm_cfg.attn_cfg.attn_output_gate) {
            define_buffer(
                fmt::format("n{}_buffer_gate", num_tokens),
                {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(LayerType::FullAttention)}
            );
        }

        // EAGLE3 draft-only buffers: second pre input, hidden-state output, FC fusion.
        if (is_eagle3_draft) {
            define_buffer(
                fmt::format("n{}_buffer1a", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
            define_buffer(
                fmt::format("n{}_buffer5", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
            define_buffer(
                fmt::format("fc_n{}_input", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size * 3}
            );
        } else if (is_gemma4_mtp_draft) {
            define_buffer(
                fmt::format("gemma4_mtp_input_n{}", num_tokens),
                {num_tokens, 2 * _cfg.lm_cfg.speculative_decoding_cfg->backbone_hidden_size}
            );
            define_buffer(
                fmt::format("n{}_buffer5", num_tokens),
                {num_tokens, _cfg.lm_cfg.speculative_decoding_cfg->backbone_hidden_size}
            );
            if (_cfg.lm_cfg.speculative_decoding_cfg->use_ordered_embeddings) {
                define_buffer(
                    fmt::format("n{}_gemma4_mtp_centroid_logits", num_tokens),
                    {num_tokens, _cfg.lm_cfg.speculative_decoding_cfg->num_centroids}
                );
            }
        }
        if (is_eagle3_draft || (_cfg.lm_cfg.is_dflash() && is_draft)) {
            define_buffer(
                fmt::format("fc_n{}_output", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
        }
    }

    if (_uses_per_layer_inputs()) {
        const std::string dtype = _cfg.pipeline_cfg.quantize_embeddings
            ? "int8" : "bfloat16";
        // Vocabulary sharding keeps every token's complete L*H row contiguous.
        const size_t out_dim = static_cast<size_t>(_cfg.lm_cfg.num_hidden_layers)
                             * _cfg.lm_cfg.hidden_size_per_layer_input;
        const size_t elem_size = _cfg.pipeline_cfg.quantize_embeddings
            ? sizeof(int8_t) : sizeof(Eigen::bfloat16);
        if (!_resources->ready) {
            _resources->per_layer_embedding_rows_per_shard = std::max<size_t>(
                1, PER_LAYER_EMBEDDING_MAX_SHARD_SIZE / (out_dim * elem_size)
            );
        }
        const size_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
        const size_t num_shards = (
            vocab_size + _resources->per_layer_embedding_rows_per_shard - 1
        ) / _resources->per_layer_embedding_rows_per_shard;
        if (!_resources->ready) {
            _resources->per_layer_embedding_shards.clear();
            _resources->per_layer_embedding_shards.reserve(num_shards);
        }
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            _define_resource_buffer("per_layer_embedding_scales", {vocab_size, 1});
        }
        for (size_t shard_idx = 0; !_resources->embedding_offload && shard_idx < num_shards; ++shard_idx) {
            const size_t row_begin = shard_idx * _resources->per_layer_embedding_rows_per_shard;
            const size_t num_rows = std::min(
                _resources->per_layer_embedding_rows_per_shard, vocab_size - row_begin
            );
            const auto name = fmt::format("per_layer_embeddings_s{}", shard_idx);
            _define_resource_buffer(
                name,
                {num_rows, out_dim},
                dtype
            );
            if (!_resources->ready) {
                _resources->per_layer_embedding_shards.emplace_back(&get_buffer(name));
            }
        }
        // MTP target verification consumes a native speculative-width projection.
        const auto speculative_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
        std::vector<uint16_t> per_layer_num_tokens_vec{
            (_cfg.lm_cfg.is_gemma4_mtp_target() || _cfg.lm_cfg.is_dflash())
                ? speculative_num_tokens : uint16_t{1}
        };
        if (_use_group_token_models) {
            const auto group_num_tokens = _cfg.pipeline_cfg.input_token_group_size;
            if (
                std::find(
                    per_layer_num_tokens_vec.begin(),
                    per_layer_num_tokens_vec.end(),
                    group_num_tokens
                ) == per_layer_num_tokens_vec.end()
            ) {
                per_layer_num_tokens_vec.emplace_back(group_num_tokens);
            }
        }
        for (const auto num_tokens: per_layer_num_tokens_vec) {
            define_buffer(
                fmt::format("per_layer_emb_staging_n{}", num_tokens),
                {num_tokens, out_dim},
                dtype
            );
            if (_cfg.pipeline_cfg.quantize_embeddings) {
                define_buffer(
                    fmt::format("per_layer_emb_staging_scale_n{}", num_tokens),
                    {num_tokens, 1}
                );
            }
        }
    }

    // Post output for last layer.
    if (!_cfg.lm_cfg.is_spec_decode()){
        if (_cfg.lm_cfg.lm_head_num_splits == 1 && !_cfg.pipeline_cfg.return_logits)
            define_buffer("n1_buffer4", {1}, "int32");
        else
            define_buffer("n1_buffer4", {_cfg.lm_cfg.token_cfg.vocab_size});
    } else {
        const auto lm_head_output_size = _cfg.lm_cfg.get_lm_head_output_size();
        const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
        for (const auto& num_tokens: num_tokens_vec) {
            // Target prefill reuses the single-token-group final post model
            // (n16 for EAGLE3). Only drafts need group-width final-post outputs.
            if (
                !is_draft
                && num_tokens != _cfg.lm_cfg.get_single_num_tokens()
            ) {
                continue;
            }
            if (
                _cfg.lm_cfg.lm_head_num_splits == 1
                && !_cfg.pipeline_cfg.return_logits
                && !(is_draft && _cfg.lm_cfg.is_dflash())
            ) {
                define_buffer(
                    fmt::format("n{}_buffer4", num_tokens), {num_tokens, 1}, "int32"
                );
            } else if (_cfg.lm_cfg.lm_head_num_splits == 1) {
                const uint16_t output_rows = is_draft && _cfg.lm_cfg.is_dflash()
                    ? num_tokens - 1 : num_tokens;
                define_buffer(
                    fmt::format("n{}_buffer4", num_tokens),
                    {output_rows, lm_head_output_size}
                );
            } else {
                uint32_t i = 0;
                for(uint32_t split_begin = 0; split_begin < lm_head_output_size; split_begin += split_dim, ++i) {
                    uint32_t split_size = std::min(lm_head_output_size, split_begin + split_dim) - split_begin;
                    const uint16_t output_rows = is_draft && _cfg.lm_cfg.is_dflash()
                        ? num_tokens - 1 : num_tokens;
                    define_buffer(
                        fmt::format("n{}_lm_split{}", num_tokens, i),
                        {output_rows, split_size}
                    );
                }
            }
        }
    }

    _workspace->global_freq_real = &get_buffer("global_freq_real");
    _workspace->global_freq_imag = &get_buffer("global_freq_imag");
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        _workspace->local_freq_real = &get_buffer("local_freq_real");
        _workspace->local_freq_imag = &get_buffer("local_freq_imag");
    }
    _init_attention_bindings();
}


void LanguageModel::_init_attention_bindings() {
    auto& layers = _workspace->attention_binding_buffers;
    layers.reserve(_cfg.lm_cfg.num_hidden_layers);
    for (uint8_t layer = 0; layer < _cfg.lm_cfg.num_hidden_layers; ++layer) {
        layers.push_back(CacheLayerBuffers::for_layer(_cfg, layer, [&](const std::string& name) {
            return &get_buffer(name);
        }));
    }
}

}
}
