#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fmt/format.h>

#include "language_model.hpp"

namespace simaai {
namespace llima {


void LanguageModel::_define_model(
    const std::string& model_type,
    const LanguageModelMapKey& key,
    const std::filesystem::path& model_path,
    const std::vector<MLABufferSlice>& ifms,
    const std::vector<MLABufferSlice>& ofms
) {
    LanguageModelMap& map = get_model_map(model_type);
    map.emplace(key, MLAModelWithBuffer(model_path, ifms, ofms));
}


void LanguageModel::_define_attn_models_iter(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    const LanguageModelMapKey pre_post_key{num_tokens, layer_idx, 0};
    const LanguageModelMapKey cache_key = _get_cache_model_key(
        num_tokens, token_idx, layer_idx
    );
    const bool define_pre = !_pre_model_map.contains(pre_post_key);
    const bool define_cache = !_cache_model_map.contains(cache_key);
    const bool define_post = !_post_model_map.contains(pre_post_key);
    if (!define_pre && !define_cache && !define_post) {
        return;
    }

    const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const auto kv_source_layer = _cfg.lm_cfg.get_kv_source_layer(layer_idx);
    const auto& kv_layer_type = _cfg.lm_cfg.layer_types[kv_source_layer];
    std::string freq_prefix;
    uint16_t cache_token_idx_begin;
    if (layer_type == "sliding_attention") {
        freq_prefix = "local";
        cache_token_idx_begin = std::max(
            0,
            token_idx + num_tokens - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
        );
    } else {
        freq_prefix = "global";
        cache_token_idx_begin = 0;
    }

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
    const bool is_draft = _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
    const bool pre_uses_embedding_scale = (
        _cfg.pipeline_cfg.quantize_embeddings && layer_idx == 0
    );
    const bool post_uses_embedding_scale = pre_uses_embedding_scale && !is_draft;

    std::vector<MLABufferSlice> pre_ifms;
    std::vector<MLABufferSlice> pre_ofms;
    if (layer_idx) {
        pre_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
        if (is_draft) {
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
        if (is_draft) {
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
                &_cache_buffer(fmt::format("cache_key_l{}", layer_idx)),
                pre_kv_cache_offset,
                pre_kv_cache_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            pre_ofms.emplace_back(
                MLABufferSlice(
                    &_cache_buffer(fmt::format("cache_key_scale_l{}", layer_idx)),
                    {0, token_idx, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
                )
            );
        }
        pre_ofms.emplace_back(
            MLABufferSlice(
                &_cache_buffer(fmt::format("cache_val_l{}", layer_idx)),
                pre_kv_cache_offset,
                pre_kv_cache_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            pre_ofms.emplace_back(
                MLABufferSlice(
                    &_cache_buffer(fmt::format("cache_val_scale_l{}", layer_idx)),
                    {0, token_idx, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
                )
            );
        }
    }
    if (define_pre) {
        _define_model(
            "pre",
            pre_post_key,
            _get_elf_path_pre(num_tokens, layer_idx),
            pre_ifms,
            pre_ofms
        );
    }

    // Cache model.
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    const uint16_t eff_token_idx = token_idx - cache_token_idx_begin;
    const uint16_t eff_num_cached_tokens = token_idx + num_tokens - cache_token_idx_begin;
    const bool is_single_model = num_tokens == single_num_tokens;
    const bool use_sliding_cache = std::get<1>(cache_key) != 0;
    const std::string_view cache_layer_type = (
        use_sliding_cache ? "sliding_attention" : "full_attention"
    );
    const uint16_t cache_mask_size = _get_cache_mask_size(
        cache_layer_type, eff_num_cached_tokens, !is_single_model
    );
    const bool use_single_future_token_mask = (
        is_single_model && cache_mask_size > 1
    );
    const bool use_group_future_token_mask = (
        !is_single_model
        && cache_mask_size > num_tokens
    );
    const uint16_t aligned_eff_token_idx = std::get<2>(cache_key);
    const uint16_t aligned_eff_num_cached_tokens = use_single_future_token_mask
        ? aligned_eff_token_idx + 1
        : aligned_eff_token_idx + num_tokens;
    std::vector<uint32_t> cache_kv_cache_shape;
    std::vector<uint32_t> cache_kv_cache_offset;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        cache_kv_cache_offset = {0, cache_token_idx_begin, 0};
        cache_kv_cache_shape = {
            _cfg.lm_cfg.attn_cfg.num_key_value_heads,
            _cfg.pipeline_cfg.max_num_tokens,
            _cfg.lm_cfg.attn_cfg.get_head_dim(kv_layer_type)
        };
    } else {
        cache_kv_cache_offset = {cache_token_idx_begin, 0};
        cache_kv_cache_shape = {
            aligned_eff_num_cached_tokens,
            _cfg.lm_cfg.attn_cfg.get_kv_size(kv_layer_type)
        };
    }

    std::vector<MLABufferSlice> cache_ifms{
        MLABufferSlice{
            &get_buffer(fmt::format("n{}_buffer2", num_tokens)),
            {0, 0, 0},
            {
                _cfg.lm_cfg.attn_cfg.num_attention_heads,
                num_tokens,
                _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
            }
        },
        MLABufferSlice{
            &_cache_buffer(fmt::format("cache_key_l{}", kv_source_layer)),
            cache_kv_cache_offset,
            cache_kv_cache_shape
        },
    };
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &_cache_buffer(fmt::format("cache_key_scale_l{}", kv_source_layer)),
                {0, cache_token_idx_begin, 0},
                {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
            }
        );
    }

    if (use_group_future_token_mask) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(
                    cache_layer_type == "sliding_attention"
                        ? "group_sliding_future_token_mask"
                        : "group_future_token_mask"
                ),
                {0},
                {static_cast<uint32_t>(num_tokens) * aligned_eff_num_cached_tokens}
            }
        );
    } else if (use_single_future_token_mask) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            // Shift the col begin to cache_token_idx_begin so that buffer col
            // p ↔ K-row p for any layer type: full (cache_token_idx_begin = 0)
            // reads cols [0, aligned_eff); sliding reads
            // [cache_token_idx_begin, cache_token_idx_begin + aligned_eff).
            // The mask data layout (col c = mask for K[c]) is the same for
            // both, so no second buffer or layout change is needed.
            cache_ifms.emplace_back(
                MLABufferSlice{
                    &get_buffer("future_token_mask"),
                    {0, cache_token_idx_begin},
                    {num_tokens, aligned_eff_num_cached_tokens}
                }
            );
        } else {
            cache_ifms.emplace_back(
                MLABufferSlice{
                    &get_buffer("future_token_mask"),
                    {(uint32_t)_cfg.pipeline_cfg.max_num_tokens - eff_num_cached_tokens},
                    {aligned_eff_num_cached_tokens}
                }
            );
        }
    }
    cache_ifms.emplace_back(
        MLABufferSlice{
            &_cache_buffer(fmt::format("cache_val_l{}", kv_source_layer)),
            cache_kv_cache_offset,
            cache_kv_cache_shape
        }
    );
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &_cache_buffer(fmt::format("cache_val_scale_l{}", kv_source_layer)),
                {0, cache_token_idx_begin, 0},
                {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
            }
        );
    }
    std::vector<MLABufferSlice> cache_ofms{
        MLABufferSlice{
            &get_buffer(fmt::format("n{}_buffer3", num_tokens)),
            {0, 0},
            {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(layer_type)}
        }
    };
    if (define_cache) {
        _define_model(
            "cache",
            cache_key,
            _get_elf_path_cache(num_tokens, aligned_eff_token_idx, use_sliding_cache),
            cache_ifms,
            cache_ofms
        );
    }

    // Draft post consumes the BF16 FC-fused hidden state. Target post consumes the same
    // embedding input as pre and therefore also needs its per-row scale.
    const size_t pre_hidden_state_idx = 1 + static_cast<size_t>(pre_uses_embedding_scale);
    std::vector<MLABufferSlice> post_ifms{
        is_draft ? pre_ifms[pre_hidden_state_idx] : pre_ifms[0]
    };
    if (post_uses_embedding_scale) {
        post_ifms.emplace_back(
            MLABufferSlice{nullptr, {0, 0}, {num_tokens, 1}}
        );
    }
    const size_t post_self_attn_idx = post_ifms.size();
    post_ifms.emplace_back(cache_ofms[0]);
    const bool use_single_post_for_target_group = (
        _cfg.lm_cfg.is_spec_decode()
        && !is_draft
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

        // Draft post produces an additional output: hidden states for next iteration.
        if (is_draft) {
            post_ofms.emplace_back(
                MLABufferSlice{&get_buffer(fmt::format("n{}_buffer5", num_tokens))}
            );
        }
    }
    if (define_post) {
        _define_model("post", pre_post_key, post_elf_path, post_ifms, post_ofms);
    }
}


void LanguageModel::_define_state_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
        _define_conv_models_iter(num_tokens, layer_idx);
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
            &_cache_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {tail_begin, 0},
            {tail_size, _cfg.lm_cfg.hidden_size}
        }
    );
    conv_ofms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))});
    conv_ofms.emplace_back(
        MLABufferSlice(
            &_cache_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {num_tokens > 1 ? 0 : tail_begin, 0},
            {
                static_cast<uint32_t>(num_tokens + tail_size - 1),
                _cfg.lm_cfg.hidden_size
            }
        )
    );
    _define_model(
        "conv",
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
        "conv_final",
        model_key,
        _get_elf_path_conv_final(layer_idx),
        conv_final_ifms,
        conv_final_ofms
    );
}


void LanguageModel::_define_models() {
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {single_num_tokens};
    if (_use_group_token_models) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens: num_tokens_vec) {
        const auto& max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
        const auto& num_hidden_layers = _cfg.lm_cfg.num_hidden_layers;

        if (num_tokens == single_num_tokens) {
            for (uint16_t token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    if (
                        _cfg.lm_cfg.layer_types[layer_idx] == "full_attention"
                        || _cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention"
                    ) {
                        _define_attn_models_iter(num_tokens, token_idx, layer_idx);
                    }
                }
            }
        } else {
            for (const auto& token_idx: _cfg.pipeline_cfg.input_token_group_offsets.value()) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    if (
                        _cfg.lm_cfg.layer_types[layer_idx] == "full_attention"
                        || _cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention"
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
    const bool is_draft = _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
    if (is_draft) {
        _define_draft_fc_models();
    }
}


void LanguageModel::_define_draft_fc_models() {
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {single_num_tokens};
    if (_use_group_token_models) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens : num_tokens_vec) {
        auto elf_path = _elf_dir / fmt::format(
            "{}_n{}_draft_fc_stage1_mla.elf",
            _cfg.language_model_name, num_tokens
        );
        std::vector<MLABufferSlice> ifms{
            MLABufferSlice{&get_buffer(fmt::format("fc_n{}_input", num_tokens))}
        };
        std::vector<MLABufferSlice> ofms{
            MLABufferSlice{&get_buffer(fmt::format("fc_n{}_output", num_tokens))}
        };
        _fc_model_map.emplace(
            num_tokens, MLAModelWithBuffer(elf_path, ifms, ofms)
        );
    }
}


void LanguageModel::_define_per_layer_models() {
    if (!_uses_per_layer_inputs())
        return;

    std::vector<uint16_t> num_tokens_vec = {_cfg.lm_cfg.get_single_num_tokens()};
    if (_use_group_token_models)
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);

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
        _define_model("per_layer", key, _get_elf_path_per_layer(num_tokens), ifms, ofms);
    }
}


LanguageModelMap& LanguageModel::get_model_map(const std::string& model_type) {
    if (model_type == "pre") {
        return _pre_model_map;
    } else if (model_type == "cache") {
        return _cache_model_map;
    } else if (model_type == "post") {
        return _post_model_map;
    } else if (model_type == "conv") {
        return _conv_model_map;
    } else if (model_type == "conv_final") {
        return _conv_final_model_map;
    } else if (model_type == "per_layer") {
        return _per_layer_model_map;
    } else {
        throw std::runtime_error(std::string("Invalid model type: ") + model_type);
    }
}


std::filesystem::path LanguageModel::_get_elf_path_pre(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_pre_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _elf_dir / elf_file_name;
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
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_post(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_post_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_conv(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_layer{}_conv_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_conv_final(uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n1_post_layer{}_conv_final_stage1_mla.elf", _cfg.language_model_name, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_per_layer(uint16_t num_tokens) {
    auto elf_file_name = fmt::format(
        "{}_n{}_per_layer_stage1_mla.elf", _cfg.language_model_name, num_tokens
    );
    return _elf_dir / elf_file_name;
}


} // namespace llima
} // namespace simaai
