#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>
#include <limits>

#include "language_model.hpp"
#include "language_model_state.hpp"

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

void LanguageModel::_initialize() {
    _logger->info("Language model initialize starting ...");
    _resources->prepare_embeddings();
    _define_buffers();
    for (auto* buffers : {&_resources->buffers, &_workspace->buffers}) {
        for (auto& [name, buffer] : *buffers) buffer.try_allocate();
    }

    for (auto& [name, buffer] : _cache->buffers) {
        buffer.allocate();
        buffer.clear();
    }

    // Define and load the models in parallel.
    _define_models();
    MLAModelWithBuffer::load_all_models(_elf_dir / _cfg.language_model_name, _files.get());

    // Resolve the MoE host round-trip handles once (buffers allocated above).
    if (_cfg.lm_cfg.is_moe()) {
        _init_moe_host_cache();
    }

    _resources->load();

    get_buffer("global_freq_real").upload(_resources->global_freq_host.re.data());
    get_buffer("global_freq_imag").upload(_resources->global_freq_host.im.data());
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        get_buffer("local_freq_real").upload(_resources->local_freq_host.re.data());
        get_buffer("local_freq_imag").upload(_resources->local_freq_host.im.data());
    }

    // Upload the future token mask.
    const uint16_t max_future_token_mask_size = _get_max_future_token_mask_size();
    if (max_future_token_mask_size > 1) {
        std::vector<Eigen::bfloat16> future_token_mask(
            max_future_token_mask_size - 1,
            std::numeric_limits<Eigen::bfloat16>::lowest()
        );
        MLABuffer& future_token_mask_buf = get_buffer("future_token_mask");
        future_token_mask_buf.clear();
        // The pointwise Gemma4 MTP draft has exactly one max_num_tokens-wide
        // mask row, with no legacy tail space. Its runtime mask is populated
        // immediately before every draft execution.
        if (!_cfg.lm_cfg.is_gemma4_mtp_draft()) {
            future_token_mask_buf.upload_raw(
                future_token_mask.data(),
                _cfg.pipeline_cfg.max_num_tokens * 2,
                future_token_mask.size() * 2
            );
        }
    }
    _logger->info("Language model initialize completed");
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


void LanguageModel::_define_model(
    LanguageModelMap& models,
    const LanguageModelMapKey& key,
    const std::filesystem::path& model_path,
    const std::vector<MLABufferSlice>& ifms,
    const std::vector<MLABufferSlice>& ofms
) {
    models.emplace(key, MLAModelWithBuffer(model_path, ifms, ofms));
}


void LanguageModel::_define_attn_models_iter(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    const LanguageModelMapKey pre_post_key{num_tokens, layer_idx, 0};
    const LanguageModelMapKey cache_key = _get_cache_model_key(
        num_tokens, token_idx, layer_idx
    );
    const bool define_pre = !_workspace->pre_model_map.contains(pre_post_key);
    const bool define_cache = !_workspace->cache_model_map.contains(cache_key);
    const auto& post_model_map = _cfg.lm_cfg.is_moe() ? _workspace->router_model_map : _workspace->post_model_map;
    const bool define_post = !post_model_map.contains(pre_post_key);
    if (!define_pre && !define_cache && !define_post) {
        return;
    }

    const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const std::string freq_prefix = layer_type == LayerType::SlidingAttention ? "local" : "global";
    const uint16_t cache_token_idx_begin = _get_cache_token_idx_begin(
        num_tokens, token_idx, layer_type
    );

    // Pre model.
    std::vector<uint32_t> pre_kv_cache_shape;
    std::vector<uint32_t> pre_kv_cache_offset;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        pre_kv_cache_offset = {0, token_idx, 0};
        pre_kv_cache_shape = {
            _cfg.lm_cfg.attn_cfg.num_key_value_heads,
            _cfg.pipeline_cfg.max_num_tokens,
            _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
        };
    } else {
        pre_kv_cache_offset = {token_idx, 0};
        pre_kv_cache_shape = {num_tokens, _cfg.lm_cfg.attn_cfg.get_kv_size(layer_type)};
    }

    // Draft pre takes an extra IFM (buffer1a) for the FC fusion output / target hidden state.
    const bool is_draft = _cfg.lm_cfg.is_speculative_draft();
    const bool is_eagle3_draft = _cfg.lm_cfg.is_eagle3_draft();
    const bool is_gemma4_mtp_draft = _cfg.lm_cfg.is_gemma4_mtp_draft();
    const bool pre_uses_embedding_scale = (
        _cfg.pipeline_cfg.quantize_embeddings && layer_idx == 0 && !is_gemma4_mtp_draft
    );
    const bool post_uses_embedding_scale = pre_uses_embedding_scale && !is_eagle3_draft;

    std::vector<MLABufferSlice> pre_ifms;
    std::vector<MLABufferSlice> pre_ofms;
    if (layer_idx) {
        pre_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
        if (is_eagle3_draft) {
            pre_ifms.emplace_back(
                MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1a", num_tokens))}
            );
        }
    } else {
        pre_ifms.emplace_back(MLABufferSlice{});
        if (pre_uses_embedding_scale) {
            pre_ifms.emplace_back(
                MLABufferSlice{nullptr, {0, 0}, {num_tokens, 1}}
            );
        }
        if (is_eagle3_draft) {
            pre_ifms.emplace_back(MLABufferSlice{});
        }
    }
    auto buf_name = fmt::format("{}_freq_real", freq_prefix);
    pre_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(buf_name),
            {token_idx, 0},
            {num_tokens, (uint32_t)get_buffer(buf_name).get_shape().back()}
        }
    );
    buf_name = fmt::format("{}_freq_imag", freq_prefix);
    pre_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(buf_name),
            {token_idx, 0},
            {num_tokens, (uint32_t)get_buffer(buf_name).get_shape().back()}
        }
    );
    pre_ofms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("n{}_buffer2", num_tokens)),
            {0, 0, 0},
            {
                _cfg.lm_cfg.attn_cfg.num_attention_heads,
                num_tokens,
                _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
            }
        }
    );
    if (!_cfg.lm_cfg.is_kv_shared_layer(layer_idx)) {
        pre_ofms.emplace_back(
            MLABufferSlice(
                &get_buffer(fmt::format("cache_key_l{}", layer_idx)),
                pre_kv_cache_offset,
                pre_kv_cache_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            pre_ofms.emplace_back(
                MLABufferSlice(
                    &get_buffer(fmt::format("cache_key_scale_l{}", layer_idx)),
                    {0, token_idx, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
                )
            );
        }
        pre_ofms.emplace_back(
            MLABufferSlice(
                &get_buffer(fmt::format("cache_val_l{}", layer_idx)),
                pre_kv_cache_offset,
                pre_kv_cache_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            pre_ofms.emplace_back(
                MLABufferSlice(
                    &get_buffer(fmt::format("cache_val_scale_l{}", layer_idx)),
                    {0, token_idx, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
                )
            );
        }
    }
    if (_cfg.lm_cfg.attn_cfg.attn_output_gate) {
        pre_ofms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer_gate", num_tokens))}
        );
    }
    if (define_pre) {
        _define_model(
            _workspace->pre_model_map,
            pre_post_key,
            _get_elf_path_pre(num_tokens, layer_idx),
            pre_ifms,
            pre_ofms
        );
    }

    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    MLABufferSlice cache_output{&get_buffer(fmt::format("n{}_buffer3", num_tokens)),
                                {0, 0}, {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(layer_type)}};
    if (define_cache) {
        const bool single = num_tokens == single_num_tokens;
        const bool sliding = std::get<1>(cache_key) != 0;
        const LayerType cache_type = sliding ? LayerType::SlidingAttention : LayerType::FullAttention;
        const uint16_t effective_length = token_idx + num_tokens - cache_token_idx_begin;
        const auto mask_size = _get_cache_mask_size(cache_type, effective_length, !single);
        const auto& ports = _workspace->cache_ports.try_emplace(cache_key, _cfg, num_tokens, mask_size).first->second;
        const uint16_t aligned_idx = std::get<2>(cache_key);
        const CacheWindow window{cache_token_idx_begin, ports.context_width(aligned_idx, num_tokens),
                                 effective_length, num_tokens};
        std::vector<MLABufferSlice> inputs(ports.count);
        // The cache consumes the query slice produced by the pre stage.
        inputs[ports.query] = pre_ofms.front();
        auto bind = [&](uint8_t port, MLABuffer* buffer, CachePorts::Dimensions begin,
                        CachePorts::Dimensions shape) {
            if (buffer) inputs[port] = MLABufferSlice{buffer, begin, shape};
        };
        ports.bind_layer(_workspace->attention_binding_buffers.at(layer_idx), _cfg, layer_type, window, bind);
        if (ports.mask) {
            const char* name = "future_token_mask";
            if (ports.mask_kind == CacheMask::Group)
                name = sliding ? "group_sliding_future_token_mask" : "group_future_token_mask";
            ports.bind_mask(&get_buffer(name), window, _cfg.pipeline_cfg.max_num_tokens, false, bind);
        }
        _define_model(
            _workspace->cache_model_map, cache_key,
            _get_elf_path_cache(num_tokens, aligned_idx, sliding), inputs, {cache_output}
        );
    }

    if (!define_post) {
        return;
    }

    // Draft post consumes the BF16 FC-fused hidden state. Target post consumes the same
    // embedding input as pre and therefore also needs its per-row scale.
    const size_t pre_hidden_state_idx = 1 + static_cast<size_t>(pre_uses_embedding_scale);
    std::vector<MLABufferSlice> post_ifms{
        is_eagle3_draft ? pre_ifms[pre_hidden_state_idx] : pre_ifms[0]
    };
    if (post_uses_embedding_scale) {
        post_ifms.emplace_back(
            MLABufferSlice{nullptr, {0, 0}, {num_tokens, 1}}
        );
    }
    const size_t post_self_attn_idx = post_ifms.size();
    post_ifms.emplace_back(cache_output);
    const bool is_group_model = (
        _use_group_token_models
        && num_tokens == _cfg.pipeline_cfg.input_token_group_size
    );
    const bool use_single_post_for_target_group = (
        _cfg.lm_cfg.is_spec_decode()
        && !is_draft
        && is_group_model
        && num_tokens != single_num_tokens
        && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
    );
    if (use_single_post_for_target_group) {
        post_ifms[0] = MLABufferSlice{
            post_ifms[0].get_buf_ptr(),
            {0, 0},
            {single_num_tokens, _cfg.lm_cfg.hidden_size}
        };
        post_ifms[post_self_attn_idx] = MLABufferSlice{
            post_ifms[post_self_attn_idx].get_buf_ptr(),
            {0, 0},
            {single_num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(layer_type)}
        };
    }
    if (_uses_per_layer_inputs()) {
        post_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(fmt::format("n{}_per_layer_input", num_tokens)),
                std::vector<uint32_t>{static_cast<uint32_t>(layer_idx) * num_tokens, 0},
                std::vector<uint32_t>{
                    use_single_post_for_target_group ? single_num_tokens : num_tokens,
                    _cfg.lm_cfg.hidden_size_per_layer_input
                }
            }
        );
    }
    if (_cfg.lm_cfg.attn_cfg.attn_output_gate) {
        post_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer_gate", num_tokens))}
        );
    }
    if (
        _cfg.vm_cfg.has_value()
        && layer_idx < _cfg.vm_cfg.value().deepstack_visual_indexes.size()
        && num_tokens > 1
    ) {
        buf_name = fmt::format("deepstack_feature_l{}_cache", layer_idx);
        post_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(buf_name),
                {token_idx, 0},
                {
                    use_single_post_for_target_group ? single_num_tokens : num_tokens,
                    (uint32_t)get_buffer(buf_name).get_shape().back()
                }
            }
        );
    }
    std::vector<MLABufferSlice> post_ofms;
    std::string post_elf_path;
    if (layer_idx < _cfg.lm_cfg.num_hidden_layers - 1) {
        post_ofms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
        post_elf_path = _get_elf_path_post(num_tokens, layer_idx);
    } else {
        // Target speculative prefill only needs the final valid prompt row's
        // logits, so reuse the n16 verification post instead of loading n128.
        const uint16_t post_num_tokens = use_single_post_for_target_group
            ? single_num_tokens
            : (_cfg.lm_cfg.is_spec_decode() ? num_tokens : 1);
        post_elf_path = _get_elf_path_post(post_num_tokens, layer_idx);

        if (_cfg.lm_cfg.lm_head_num_splits == 1) {
            const std::string buf_name = _cfg.lm_cfg.is_spec_decode()
                ? fmt::format("n{}_buffer4", post_num_tokens)
                : std::string("n1_buffer4");
            post_ofms.emplace_back(MLABufferSlice{&get_buffer(buf_name)});
        } else {
            const auto lm_head_output_size = _cfg.lm_cfg.get_lm_head_output_size();
            const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;

            if (_cfg.lm_cfg.is_spec_decode()) {
                uint32_t i = 0;
                for (uint32_t split_begin = 0;
                     split_begin < lm_head_output_size;
                     split_begin += split_dim, ++i)
                {
                    post_ofms.emplace_back(
                        MLABufferSlice{
                            &get_buffer(fmt::format("n{}_lm_split{}", post_num_tokens, i))
                        }
                    );
                }
            } else {
                for (uint32_t split_begin = 0;
                     split_begin < lm_head_output_size;
                     split_begin += split_dim)
                {
                    auto split_size = std::min(lm_head_output_size, split_begin + split_dim) - split_begin;
                    post_ofms.emplace_back(
                        MLABufferSlice{&get_buffer("n1_buffer4"), {split_begin}, {split_size}}
                    );
                }
            }
        }

        if (
            _cfg.lm_cfg.speculative_decoding_cfg
            && _cfg.lm_cfg.speculative_decoding_cfg->use_ordered_embeddings
        ) {
            post_ofms.emplace_back(
                MLABufferSlice{
                    &get_buffer(fmt::format(
                        "n{}_gemma4_mtp_centroid_logits", post_num_tokens
                    ))
                }
            );
        }

        if (_cfg.lm_cfg.is_gemma4_mtp_target()) {
            post_ofms.emplace_back(
                MLABufferSlice{
                    &get_buffer(fmt::format("n{}_target_hidden_states", post_num_tokens))
                }
            );
        }

        // Draft post produces an additional output: hidden states for next iteration.
        if (is_eagle3_draft || is_gemma4_mtp_draft) {
            post_ofms.emplace_back(
                MLABufferSlice{&get_buffer(fmt::format("n{}_buffer5", num_tokens))}
            );
        }
    }
    if (_cfg.lm_cfg.is_moe()) {
        _define_moe_post_models(num_tokens, layer_idx, post_ifms, post_ofms);
    } else {
        _define_model(_workspace->post_model_map, pre_post_key, post_elf_path, post_ifms, post_ofms);
    }
}


void LanguageModel::_define_moe_post_models(
    uint16_t num_tokens, uint8_t layer_idx,
    const std::vector<MLABufferSlice>& post_ifms,
    const std::vector<MLABufferSlice>& post_ofms
) {
    const auto& moe = _cfg.lm_cfg.moe_cfg.value();
    const uint16_t num_experts = moe.num_experts;
    const uint16_t top_k = moe.num_experts_per_tok;
    const bool is_last = (layer_idx == _cfg.lm_cfg.num_hidden_layers - 1);

    // Last layer runs single-token (n1 ELFs/buffers), like the dense post.
    const uint16_t moe_nt = (is_last && !_cfg.lm_cfg.is_spec_decode()) ? 1 : num_tokens;

    // The router consumes the same residual, embedding scales and attention as dense post.
    std::vector<MLABufferSlice> router_ofms{
        MLABufferSlice{&get_buffer(fmt::format("n{}_router_values", moe_nt))},
        MLABufferSlice{&get_buffer(fmt::format("n{}_router_indices", moe_nt))},
        MLABufferSlice{&get_buffer(fmt::format("n{}_residual", moe_nt))},
        MLABufferSlice{&get_buffer(fmt::format("n{}_norm_hidden", moe_nt))},
    };
    const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    _define_model(
        _workspace->router_model_map, model_key, _get_elf_path_router(moe_nt, layer_idx), post_ifms, router_ofms
    );

    // Experts consume norm(h), scaled by their routing-weight column (runtime runs the top-k).
    for (uint16_t e = 0; e < num_experts; ++e) {
        std::vector<MLABufferSlice> expert_ifms{
            MLABufferSlice{&get_buffer(fmt::format("n{}_norm_hidden", moe_nt))},
            MLABufferSlice{&get_buffer(fmt::format("n{}_router_weights", moe_nt))},
        };
        // Decode always overrides OFM 0 with the selected combine slot when queuing.
        const uint16_t output_slot = moe_nt == 1 ? 0 : e;
        std::vector<MLABufferSlice> expert_ofms{
            MLABufferSlice{&get_buffer(fmt::format("n{}_expert{}", moe_nt, output_slot))},
        };
        LanguageModelMapKey expert_key{num_tokens, layer_idx, e};
        _define_model(
            _workspace->expert_model_map, expert_key, _get_elf_path_expert(moe_nt, layer_idx, e),
            expert_ifms, expert_ofms
        );
    }

    // Weighted sum of the routing-weighted expert outputs + residual (decode sums the top-k).
    const uint16_t n_combine = (moe_nt == 1) ? top_k : num_experts;
    std::vector<MLABufferSlice> ws_ifms;
    for (uint16_t e = 0; e < n_combine; ++e) {
        ws_ifms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_expert{}", moe_nt, e))});
    }
    ws_ifms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_residual", moe_nt))});

    _define_model(
        _workspace->weightedsum_model_map, model_key, _get_elf_path_weightedsum(moe_nt, layer_idx), ws_ifms, post_ofms
    );
}


void LanguageModel::_define_state_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] == LayerType::Conv) {
        _define_conv_models_iter(num_tokens, layer_idx);
    } else if (_cfg.lm_cfg.layer_types[layer_idx] == LayerType::LinearAttention) {
        _define_linear_models_iter(num_tokens, layer_idx);
    }
}


void LanguageModel::_define_conv_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    const uint16_t tail_size = std::max<uint16_t>(1, _cfg.lm_cfg.conv_L_cache - 1);
    const uint16_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;

    // Conv model.
    std::vector<MLABufferSlice> conv_ifms;
    std::vector<MLABufferSlice> conv_ofms;
    if (layer_idx) {
        conv_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
    } else {
        conv_ifms.emplace_back(MLABufferSlice{});
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            conv_ifms.emplace_back(
                MLABufferSlice{nullptr, {0, 0}, {num_tokens, 1}}
            );
        }
    }
    conv_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {tail_begin, 0},
            {tail_size, _cfg.lm_cfg.hidden_size}
        }
    );
    conv_ofms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))});
    conv_ofms.emplace_back(
        MLABufferSlice(
            &get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {num_tokens > 1 ? 0 : tail_begin, 0},
            {
                static_cast<uint32_t>(num_tokens + tail_size - 1),
                _cfg.lm_cfg.hidden_size
            }
        )
    );
    _define_model(
        _workspace->conv_model_map,
        model_key,
        _get_elf_path_conv(num_tokens, layer_idx),
        conv_ifms,
        conv_ofms
    );

    // Conv final model.
    std::vector<MLABufferSlice> conv_final_ifms{conv_ofms[0]};
    std::vector<MLABufferSlice> conv_final_ofms;
    std::string post_elf_path;
    if (layer_idx < _cfg.lm_cfg.num_hidden_layers - 1) {
        return;
    } else if (_cfg.lm_cfg.lm_head_num_splits == 1) {
        conv_final_ofms.emplace_back(MLABufferSlice{&get_buffer("n1_buffer4")});
    } else {
        const auto& vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
        const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
        for (uint32_t split_begin = 0; split_begin < vocab_size; split_begin += split_dim) {
            auto split_size = std::min(vocab_size, split_begin + split_dim) - split_begin;
            conv_final_ofms.emplace_back(
                MLABufferSlice{&get_buffer("n1_buffer4"), {split_begin}, {split_size}}
            );
        }
    }
    _define_model(
        _workspace->conv_final_model_map,
        model_key,
        _get_elf_path_conv_final(layer_idx),
        conv_final_ifms,
        conv_final_ofms
    );
}


void LanguageModel::_define_linear_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    const auto& linear_cfg = _linear_attn_cfg();
    LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    const uint16_t conv_tail_size = static_cast<uint16_t>(linear_cfg.conv_kernel_dim - 1);
    const uint16_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    const bool is_dflash_target_verify = (
        _cfg.lm_cfg.is_dflash()
        && !_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft
        && num_tokens == _cfg.lm_cfg.get_single_num_tokens()
    );
    const bool uses_linear_valid_mask = (
        num_tokens > 1
        && !is_dflash_target_verify
    );

    std::vector<MLABufferSlice> linear_ifms;
    std::vector<MLABufferSlice> linear_ofms;
    if (layer_idx) {
        linear_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
    } else {
        linear_ifms.emplace_back(MLABufferSlice{});
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            linear_ifms.emplace_back(
                MLABufferSlice{nullptr, {0, 0}, {num_tokens, 1}}
            );
        }
    }
    linear_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx)),
            {tail_begin, 0},
            {conv_tail_size, linear_cfg.get_conv_dim()}
        }
    );
    if (uses_linear_valid_mask) {
        linear_ifms.emplace_back(MLABufferSlice{
            &get_buffer("linear_valid_mask"), {0, 0}, {num_tokens, 1}
        });
    }
    linear_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)),
            {0, 0},
            {1, linear_cfg.get_recurrent_state_size()}
        }
    );

    linear_ofms.emplace_back(
        MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
    );
    linear_ofms.emplace_back(
        MLABufferSlice{
            &get_buffer(
                is_dflash_target_verify
                    ? fmt::format("linear_conv_prefix_states_l{}", layer_idx)
                    : fmt::format("linear_conv_cache_history_l{}", layer_idx)
            ),
            {
                is_dflash_target_verify || num_tokens > 1
                    ? uint32_t{0}
                    : static_cast<uint32_t>(tail_begin),
                0
            },
            {
                static_cast<uint32_t>(num_tokens + conv_tail_size - 1),
                linear_cfg.get_conv_dim()
            }
        }
    );
    linear_ofms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx)),
            {0, 0},
            {
                1,
                linear_cfg.get_recurrent_state_size()
            }
        }
    );
    if (is_dflash_target_verify) {
        linear_ofms.emplace_back(MLABufferSlice{
            &get_buffer(fmt::format("linear_delta_resolver_key_l{}", layer_idx))
        });
        linear_ofms.emplace_back(MLABufferSlice{
            &get_buffer(fmt::format("linear_delta_resolver_value_l{}", layer_idx))
        });
        linear_ofms.emplace_back(MLABufferSlice{
            &get_buffer(fmt::format(
                "linear_delta_resolver_decay_mask_l{}", layer_idx
            ))
        });
    }
    _define_model(
        _workspace->linear_model_map,
        model_key,
        _get_elf_path_linear(num_tokens, layer_idx),
        linear_ifms,
        linear_ofms
    );
}


void LanguageModel::_define_models() {
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {single_num_tokens};
    if (
        _use_group_token_models
        && _cfg.pipeline_cfg.input_token_group_size != single_num_tokens
    ) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens: num_tokens_vec) {
        const auto& max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
        const auto& num_hidden_layers = _cfg.lm_cfg.num_hidden_layers;

        const bool is_dflash_draft = _cfg.lm_cfg.is_dflash()
            && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
        if (is_dflash_draft && num_tokens != single_num_tokens) {
            continue;
        }
        if (num_tokens == single_num_tokens) {
            for (uint16_t token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
                if (_cfg.lm_cfg.is_dflash() && token_idx + num_tokens > max_num_tokens) {
                    break;
                }
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    if (
                        _cfg.lm_cfg.layer_types[layer_idx] == LayerType::FullAttention
                        || _cfg.lm_cfg.layer_types[layer_idx] == LayerType::SlidingAttention
                    ) {
                        _define_attn_models_iter(num_tokens, token_idx, layer_idx);
                    }
                }
            }
        } else {
            for (const auto& token_idx: _cfg.pipeline_cfg.input_token_group_offsets.value()) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    if (
                        _cfg.lm_cfg.layer_types[layer_idx] == LayerType::FullAttention
                        || _cfg.lm_cfg.layer_types[layer_idx] == LayerType::SlidingAttention
                    ) {
                        _define_attn_models_iter(num_tokens, token_idx, layer_idx);
                    }
                }
            }
        }

        for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            _define_state_models_iter(num_tokens, layer_idx);
        }
    }
    _define_per_layer_models();

    // Draft-only: FC fusion models.
    if (_cfg.lm_cfg.is_eagle3_draft()
        || (_cfg.lm_cfg.is_dflash() && _cfg.lm_cfg.is_speculative_draft())) {
        _define_draft_fc_models();
    }
    if (_cfg.lm_cfg.is_dflash()) {
        _define_dflash_models();
    }
}


void LanguageModel::_define_draft_fc_models() {
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {single_num_tokens};
    if (
        _use_group_token_models
        && _cfg.pipeline_cfg.input_token_group_size != single_num_tokens
    ) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens : num_tokens_vec) {
        // Reserve ELF paths now; deferred providers fetch them at MLA load time.
        auto elf_path = _files->reserve(fmt::format(
            "elf_files/{}_n{}_draft_fc_stage1_mla.elf",
            _cfg.language_model_name, num_tokens
        ));
        std::vector<MLABufferSlice> ifms;
        if (_cfg.lm_cfg.is_dflash()) {
            for (size_t index = 0;
                 index < _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids.size();
                 ++index) {
                ifms.emplace_back(MLABufferSlice{});
            }
        } else {
            ifms.emplace_back(MLABufferSlice{
                &get_buffer(fmt::format("fc_n{}_input", num_tokens))
            });
        }
        std::vector<MLABufferSlice> ofms{
            MLABufferSlice{&get_buffer(fmt::format("fc_n{}_output", num_tokens))}
        };
        _workspace->fc_model_map.emplace(
            num_tokens, MLAModelWithBuffer(elf_path, ifms, ofms)
        );
    }
}


void LanguageModel::_define_dflash_models() {
    const auto& spec_cfg = _cfg.lm_cfg.speculative_decoding_cfg.value();
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    if (!spec_cfg.is_draft) {
        if (!_has_linear_attention_layers()) {
            return;
        }
        const auto& linear_cfg = _linear_attn_cfg();
        const auto elf_path = _files->reserve(fmt::format(
            "elf_files/{}_n{}_dflash_state_resolver_stage1_mla.elf",
            _cfg.language_model_name, single_num_tokens
        ));
        for (uint8_t layer_idx = 0;
             layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            if (_cfg.lm_cfg.layer_types[layer_idx] != LayerType::LinearAttention) {
                continue;
            }
            _workspace->dflash_state_resolver_model_map.emplace(
                LanguageModelMapKey{single_num_tokens, layer_idx, 0},
                MLAModelWithBuffer(
                    elf_path,
                    {
                        MLABufferSlice{&get_buffer(fmt::format(
                            "linear_delta_state_history_alt_l{}", layer_idx
                        ))},
                        MLABufferSlice{&get_buffer(fmt::format(
                            "linear_delta_resolver_key_l{}", layer_idx
                        ))},
                        MLABufferSlice{&get_buffer(fmt::format(
                            "linear_delta_resolver_value_l{}", layer_idx
                        ))},
                        MLABufferSlice{
                            &get_buffer(fmt::format(
                                "linear_delta_resolver_decay_mask_l{}", layer_idx
                            )),
                            {0, 0, 0},
                            {
                                linear_cfg.num_value_heads,
                                1,
                                single_num_tokens
                            }
                        },
                    },
                    {
                        MLABufferSlice{&get_buffer(fmt::format(
                            "linear_delta_resolver_output_l{}", layer_idx
                        ))}
                    }
                )
            );
        }
        return;
    }

    std::vector<uint16_t> widths{single_num_tokens};
    if (
        _use_group_token_models
        && _cfg.pipeline_cfg.input_token_group_size != single_num_tokens
    ) {
        widths.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto num_tokens : widths) {
        for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
            const char* freq_prefix = layer_type == LayerType::SlidingAttention ? "local" : "global";
            const auto freq_dim = _cfg.lm_cfg.rope_cfg.get_rope_dimension_count(layer_type) / 2;
            std::vector<MLABufferSlice> outputs;
            auto& key = get_buffer(fmt::format("cache_key_l{}", layer_idx));
            auto& value = get_buffer(fmt::format("cache_val_l{}", layer_idx));
            const std::vector<uint32_t> kv_begin = _cfg.pipeline_cfg.use_strided_kv_cache
                ? std::vector<uint32_t>{0, 0, 0}
                : std::vector<uint32_t>{0, 0};
            const std::vector<uint32_t> kv_shape = _cfg.pipeline_cfg.use_strided_kv_cache
                ? std::vector<uint32_t>{
                    _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                    num_tokens,
                    _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
                }
                : std::vector<uint32_t>{
                    num_tokens, _cfg.lm_cfg.attn_cfg.get_kv_size(layer_type)
                };
            outputs.emplace_back(MLABufferSlice{&key, kv_begin, kv_shape});
            if (_cfg.pipeline_cfg.quantize_kv_cache) {
                outputs.emplace_back(MLABufferSlice{
                    &get_buffer(fmt::format("cache_key_scale_l{}", layer_idx)),
                    {0, 0, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, num_tokens, 1}
                });
            }
            outputs.emplace_back(MLABufferSlice{&value, kv_begin, kv_shape});
            if (_cfg.pipeline_cfg.quantize_kv_cache) {
                outputs.emplace_back(MLABufferSlice{
                    &get_buffer(fmt::format("cache_val_scale_l{}", layer_idx)),
                    {0, 0, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, num_tokens, 1}
                });
            }
            const LanguageModelMapKey key_id{num_tokens, layer_idx, 0};
            _workspace->dflash_context_model_map.emplace(
                key_id,
                MLAModelWithBuffer(
                    _files->reserve(fmt::format(
                        "elf_files/{}_n{}_dflash_context_layer{}_stage1_mla.elf",
                        _cfg.language_model_name, num_tokens, layer_idx
                    )),
                    {
                        MLABufferSlice{&get_buffer(fmt::format("fc_n{}_output", num_tokens))},
                        MLABufferSlice{
                            &get_buffer(fmt::format("{}_freq_real", freq_prefix)),
                            {0, 0}, {num_tokens, freq_dim}
                        },
                        MLABufferSlice{
                            &get_buffer(fmt::format("{}_freq_imag", freq_prefix)),
                            {0, 0}, {num_tokens, freq_dim}
                        },
                    },
                    outputs
                )
            );
        }
    }
}


void LanguageModel::_define_per_layer_models() {
    if (!_uses_per_layer_inputs())
        return;

    const auto speculative_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {
        (_cfg.lm_cfg.is_gemma4_mtp_target() || _cfg.lm_cfg.is_dflash())
            ? speculative_num_tokens : uint16_t{1}
    };
    if (_use_group_token_models) {
        const auto group_num_tokens = _cfg.pipeline_cfg.input_token_group_size;
        if (std::find(num_tokens_vec.begin(), num_tokens_vec.end(), group_num_tokens)
            == num_tokens_vec.end()) {
            num_tokens_vec.emplace_back(group_num_tokens);
        }
    }

    for (auto num_tokens : num_tokens_vec) {
        LanguageModelMapKey key{num_tokens, 0, 0};
        std::vector<MLABufferSlice> ifms{
            MLABufferSlice{&get_buffer(fmt::format("per_layer_emb_staging_n{}", num_tokens))}
        };
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            ifms.emplace_back(
                MLABufferSlice{
                    &get_buffer(
                        fmt::format("per_layer_emb_staging_scale_n{}", num_tokens)
                    )
                }
            );
        }
        ifms.emplace_back(MLABufferSlice{});
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            ifms.emplace_back(MLABufferSlice{});
        }
        std::vector<MLABufferSlice> ofms{
            MLABufferSlice{&get_buffer(fmt::format("n{}_per_layer_input", num_tokens))}
        };
        _define_model(_workspace->per_layer_model_map, key, _get_elf_path_per_layer(num_tokens), ifms, ofms);
    }
}


std::filesystem::path LanguageModel::_get_elf_path_pre(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_pre_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_cache(
    uint16_t num_tokens,
    uint16_t token_idx,
    bool use_sliding_cache
) {
    const std::string cache_name = use_sliding_cache ? "sliding_cache" : "cache";
    auto elf_file_name = fmt::format(
        "{}_n{}_{}_token{}_stage1_mla.elf",
        _cfg.language_model_name,
        num_tokens,
        cache_name,
        token_idx
    );
    if (use_sliding_cache
        && (num_tokens != _cfg.lm_cfg.get_single_num_tokens() || _cfg.lm_cfg.is_spec_decode())
        && token_idx + num_tokens > _cfg.lm_cfg.attn_cfg.sliding_window.value()
        && !_files->exists("elf_files/" + elf_file_name)) {
        throw std::runtime_error(fmt::format(
            "Incompatible compiled sliding-attention model. Recompile the affected "
            "sliding-cache stages and redeploy. Missing: {}.",
            elf_file_name
        ));
    }
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_post(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_post_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_router(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_router_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_expert(
    uint16_t num_tokens, uint8_t layer_idx, uint16_t expert_idx
) {
    // Experts keep the factory post_layer..expert.. naming.
    auto elf_file_name = fmt::format(
        "{}_n{}_post_layer{}_expert{}_stage1_mla.elf",
        _cfg.language_model_name, num_tokens, layer_idx, expert_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_weightedsum(
    uint16_t num_tokens, uint8_t layer_idx
) {
    auto elf_file_name = fmt::format(
        "{}_n{}_moe_weightedsum_layer{}_stage1_mla.elf",
        _cfg.language_model_name, num_tokens, layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_conv(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_layer{}_conv_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_conv_final(uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n1_post_layer{}_conv_final_stage1_mla.elf", _cfg.language_model_name, layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_per_layer(uint16_t num_tokens) {
    auto elf_file_name = fmt::format(
        "{}_n{}_per_layer_stage1_mla.elf", _cfg.language_model_name, num_tokens
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


std::filesystem::path LanguageModel::_get_elf_path_linear(
    uint16_t num_tokens, uint8_t layer_idx
) {
    auto elf_file_name = fmt::format(
        "{}_n{}_layer{}_linear_stage1_mla.elf",
        _cfg.language_model_name,
        num_tokens,
        layer_idx
    );
    return _files->reserve("elf_files/" + elf_file_name);
}


} // namespace llima
} // namespace simaai
