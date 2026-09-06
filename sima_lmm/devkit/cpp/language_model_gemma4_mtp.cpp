#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include "language_model.hpp"

namespace simaai {
namespace llima {

namespace {

uint16_t checked_u16(size_t value, std::string_view name) {
    if (value > std::numeric_limits<uint16_t>::max()) {
        throw std::runtime_error(fmt::format("{} exceeds uint16_t range", name));
    }
    return static_cast<uint16_t>(value);
}

} // namespace

uint32_t LanguageModel::_argmax_lm_head_row(uint16_t num_tokens, uint16_t row) {
    if (row >= num_tokens) {
        throw std::runtime_error(fmt::format(
            "lm_head row {} is outside n{} output", row, num_tokens
        ));
    }

    const uint32_t output_size = _cfg.lm_cfg.get_lm_head_output_size();
    const uint16_t num_splits = _cfg.lm_cfg.lm_head_num_splits;

    if (num_splits == 1 && !_cfg.pipeline_cfg.return_logits) {
        auto& buf = get_buffer(fmt::format("n{}_buffer4", num_tokens));
        std::vector<uint32_t> token_ids(num_tokens, 0);
        buf.download(token_ids.data());
        return token_ids[row];
    }

    uint32_t best_idx = 0;
    float best_val = -std::numeric_limits<float>::infinity();
    const uint32_t split_dim = _cfg.lm_cfg.lm_head_split_dim;
    for (uint16_t split_idx = 0; split_idx < num_splits; ++split_idx) {
        const uint32_t split_begin = static_cast<uint32_t>(split_idx) * split_dim;
        const uint32_t split_size = (num_splits == 1)
            ? output_size
            : std::min<uint32_t>(split_dim, output_size - split_begin);
        const std::string buf_name = (num_splits == 1)
            ? fmt::format("n{}_buffer4", num_tokens)
            : fmt::format("n{}_lm_split{}", num_tokens, split_idx);
        auto& buf = get_buffer(buf_name);
        std::vector<Eigen::bfloat16> logits(
            static_cast<size_t>(num_tokens) * split_size
        );
        buf.download(logits.data());
        const auto* row_ptr = logits.data() + static_cast<size_t>(row) * split_size;
        for (uint32_t i = 0; i < split_size; ++i) {
            const float value = static_cast<float>(row_ptr[i]);
            if (value > best_val) {
                best_val = value;
                best_idx = split_begin + i;
            }
        }
    }
    return best_idx;
}

std::vector<uint32_t> LanguageModel::_argmax_lm_head_rows(
    uint16_t num_tokens, uint16_t valid_tokens
) {
    if (valid_tokens == 0 || valid_tokens > num_tokens) {
        throw std::runtime_error(fmt::format(
            "Requested {} lm_head rows from n{} output", valid_tokens, num_tokens
        ));
    }

    const uint32_t output_size = _cfg.lm_cfg.get_lm_head_output_size();
    const uint16_t num_splits = _cfg.lm_cfg.lm_head_num_splits;
    if (num_splits == 1 && !_cfg.pipeline_cfg.return_logits) {
        auto& buf = get_buffer(fmt::format("n{}_buffer4", num_tokens));
        std::vector<uint32_t> token_ids(num_tokens, 0);
        buf.download(token_ids.data());
        token_ids.resize(valid_tokens);
        return token_ids;
    }

    std::vector<uint32_t> best_indices(valid_tokens, 0);
    std::vector<float> best_values(
        valid_tokens, -std::numeric_limits<float>::infinity()
    );
    const uint32_t split_dim = _cfg.lm_cfg.lm_head_split_dim;
    for (uint16_t split_idx = 0; split_idx < num_splits; ++split_idx) {
        const uint32_t split_begin = static_cast<uint32_t>(split_idx) * split_dim;
        const uint32_t split_size = (num_splits == 1)
            ? output_size
            : std::min<uint32_t>(split_dim, output_size - split_begin);
        const std::string buf_name = (num_splits == 1)
            ? fmt::format("n{}_buffer4", num_tokens)
            : fmt::format("n{}_lm_split{}", num_tokens, split_idx);
        auto& buf = get_buffer(buf_name);
        std::vector<Eigen::bfloat16> logits(
            static_cast<size_t>(num_tokens) * split_size
        );
        buf.download(logits.data());
        for (uint16_t row = 0; row < valid_tokens; ++row) {
            const auto* row_ptr = logits.data() + static_cast<size_t>(row) * split_size;
            for (uint32_t i = 0; i < split_size; ++i) {
                const float value = static_cast<float>(row_ptr[i]);
                if (value > best_values[row]) {
                    best_values[row] = value;
                    best_indices[row] = split_begin + i;
                }
            }
        }
    }
    return best_indices;
}

std::vector<Eigen::bfloat16> LanguageModel::_read_embedding_row_bf16(uint32_t token_id) {
    auto& embeddings = get_buffer("embeddings");
    const auto& shape = embeddings.get_shape();
    if (shape.size() != 2 || token_id >= shape.front()) {
        throw std::runtime_error(fmt::format(
            "Embedding token id {} is outside embedding table", token_id
        ));
    }

    const uint32_t hidden_size = static_cast<uint32_t>(shape.back());
    const size_t row_bytes = embeddings.get_buf_len(
        std::vector<uint32_t>{1, hidden_size}
    );
    const auto* row_base = reinterpret_cast<const uint8_t*>(
        embeddings.get_virtual_addr()
    ) + static_cast<size_t>(token_id) * row_bytes;

    std::vector<Eigen::bfloat16> row(hidden_size);
    if (embeddings.get_dtype() == "bfloat16") {
        std::memcpy(row.data(), row_base, hidden_size * sizeof(Eigen::bfloat16));
        return row;
    }

    if (embeddings.get_dtype() != "int8") {
        throw std::runtime_error(fmt::format(
            "Unsupported embedding dtype for Gemma4 MTP: {}", embeddings.get_dtype()
        ));
    }

    auto& scales = get_buffer("embedding_scales");
    const size_t scale_row_bytes = scales.get_buf_len(std::vector<uint32_t>{1, 1});
    const auto* scale_base = reinterpret_cast<const uint8_t*>(
        scales.get_virtual_addr()
    ) + static_cast<size_t>(token_id) * scale_row_bytes;
    const auto scale = *reinterpret_cast<const Eigen::bfloat16*>(scale_base);
    const float scale_f = static_cast<float>(scale) / 127.0f;
    const auto* quantized = reinterpret_cast<const int8_t*>(row_base);
    for (uint32_t i = 0; i < hidden_size; ++i) {
        row[i] = Eigen::bfloat16(static_cast<float>(quantized[i]) * scale_f);
    }
    return row;
}

std::vector<Eigen::bfloat16> LanguageModel::_read_gemma4_mtp_target_hidden_rows(
    uint16_t num_tokens, uint16_t valid_tokens
) {
    if (valid_tokens == 0 || valid_tokens > num_tokens) {
        throw std::runtime_error(fmt::format(
            "Requested {} Gemma4 MTP hidden rows from n{} output",
            valid_tokens, num_tokens
        ));
    }
    auto& buf = get_buffer(fmt::format("n{}_target_hidden_states", num_tokens));
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    std::vector<Eigen::bfloat16> hidden_states(
        static_cast<size_t>(num_tokens) * hidden_size
    );
    buf.download(hidden_states.data());
    hidden_states.resize(static_cast<size_t>(valid_tokens) * hidden_size);
    return hidden_states;
}

void LanguageModel::_upload_gemma4_mtp_freq_rows(
    uint16_t num_tokens, uint16_t position_id, uint16_t valid_tokens
) {
    if (valid_tokens == 0 || valid_tokens > num_tokens) {
        throw std::runtime_error("Invalid Gemma4 MTP RoPE row count");
    }
    auto upload_one = [&](const RopeTable& host, const std::string& real_name,
                          const std::string& imag_name) {
        if (!has_buffer(real_name) || !has_buffer(imag_name)) {
            return;
        }
        auto& real_buf = get_buffer(real_name);
        auto& imag_buf = get_buffer(imag_name);
        const uint32_t freq_dim = static_cast<uint32_t>(real_buf.get_shape().back());
        std::vector<Eigen::bfloat16> real(
            static_cast<size_t>(num_tokens) * freq_dim, Eigen::bfloat16{0.0f}
        );
        std::vector<Eigen::bfloat16> imag(
            static_cast<size_t>(num_tokens) * freq_dim, Eigen::bfloat16{0.0f}
        );
        for (uint16_t row = 0; row < valid_tokens; ++row) {
            const size_t offset = static_cast<size_t>(position_id + row) * freq_dim;
            if (offset + freq_dim > host.re.size() || offset + freq_dim > host.im.size()) {
                throw std::runtime_error(fmt::format(
                    "Gemma4 MTP position {} exceeds RoPE table for {}",
                    position_id + row, real_name
                ));
            }
            std::memcpy(
                real.data() + static_cast<size_t>(row) * freq_dim,
                host.re.data() + offset,
                freq_dim * sizeof(Eigen::bfloat16)
            );
            std::memcpy(
                imag.data() + static_cast<size_t>(row) * freq_dim,
                host.im.data() + offset,
                freq_dim * sizeof(Eigen::bfloat16)
            );
        }
        const size_t bytes = real.size() * sizeof(Eigen::bfloat16);
        real_buf.upload_raw(real.data(), 0, bytes);
        imag_buf.upload_raw(imag.data(), 0, bytes);
    };

    upload_one(_global_freq_host, "global_freq_real", "global_freq_imag");
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        upload_one(_local_freq_host, "local_freq_real", "local_freq_imag");
    }
}

void LanguageModel::_upload_gemma4_mtp_causal_mask(
    uint16_t num_tokens, uint16_t first_visible_token_count, uint16_t valid_tokens
) {
    if (!has_buffer("future_token_mask")) {
        return;
    }
    if (
        valid_tokens == 0 || valid_tokens > num_tokens
        || first_visible_token_count == 0
        || static_cast<size_t>(first_visible_token_count) + valid_tokens - 1
            > _cfg.pipeline_cfg.max_num_tokens
    ) {
        throw std::runtime_error("Invalid Gemma4 MTP causal-mask range");
    }

    auto& mask_buf = get_buffer("future_token_mask");
    const auto& shape = mask_buf.get_shape();
    if (shape.size() != 2 || shape.front() < num_tokens) {
        throw std::runtime_error(fmt::format(
            "Unexpected Gemma4 MTP future_token_mask shape for n{}", num_tokens
        ));
    }

    const size_t cols = shape.back();
    const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
    std::vector<Eigen::bfloat16> mask(static_cast<size_t>(num_tokens) * cols, neg_inf);
    for (uint16_t row = 0; row < valid_tokens; ++row) {
        const size_t visible_tokens = static_cast<size_t>(first_visible_token_count) + row;
        if (visible_tokens > cols) {
            throw std::runtime_error("Gemma4 MTP visible KV length exceeds mask capacity");
        }
        std::fill_n(
            mask.begin() + static_cast<size_t>(row) * cols,
            visible_tokens,
            Eigen::bfloat16{0.0f}
        );
    }
    mask_buf.upload_raw(mask.data(), 0, mask.size() * sizeof(Eigen::bfloat16));
}

uint8_t LanguageModel::_find_gemma4_mtp_target_kv_layer(std::string_view layer_type) const {
    for (int idx = static_cast<int>(_cfg.lm_cfg.num_hidden_layers) - 1; idx >= 0; --idx) {
        const auto layer_idx = static_cast<uint8_t>(idx);
        if (
            !_cfg.lm_cfg.is_kv_shared_layer(layer_idx)
            && _cfg.lm_cfg.layer_types[layer_idx] == layer_type
        ) {
            return layer_idx;
        }
    }
    throw std::runtime_error(fmt::format(
        "No target KV layer found for Gemma4 MTP layer type {}", layer_type
    ));
}

void LanguageModel::_bind_gemma4_mtp_shared_kv_cache(
    LanguageModel& target_lm,
    std::map<uint8_t, MLABufferSlice>& cache_ifm_map,
    uint8_t cache_ifm_idx,
    uint16_t cache_token_idx_begin,
    uint16_t aligned_eff_num_cached_tokens,
    bool has_future_token_mask,
    std::string_view layer_type
) {
    if (_cfg.pipeline_cfg.use_strided_kv_cache != target_lm._cfg.pipeline_cfg.use_strided_kv_cache) {
        throw std::runtime_error("Gemma4 MTP target/draft strided-KV settings must match");
    }
    if (_cfg.pipeline_cfg.quantize_kv_cache != target_lm._cfg.pipeline_cfg.quantize_kv_cache) {
        throw std::runtime_error("Gemma4 MTP target/draft KV quantization settings must match");
    }
    if (_cfg.pipeline_cfg.max_num_tokens != target_lm._cfg.pipeline_cfg.max_num_tokens) {
        throw std::runtime_error("Gemma4 MTP target/draft max_num_tokens must match");
    }

    const std::string layer_type_str(layer_type);
    const uint8_t source_layer = target_lm._find_gemma4_mtp_target_kv_layer(layer_type);
    const uint32_t source_head_dim = target_lm._cfg.lm_cfg.attn_cfg.get_head_dim(layer_type_str);
    const uint32_t consumer_head_dim = _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type_str);
    const uint32_t source_kv_heads = target_lm._cfg.lm_cfg.attn_cfg.num_key_value_heads;
    const uint32_t consumer_kv_heads = _cfg.lm_cfg.attn_cfg.num_key_value_heads;
    if (source_head_dim != consumer_head_dim || source_kv_heads != consumer_kv_heads) {
        throw std::runtime_error(fmt::format(
            "Gemma4 MTP shared KV shape mismatch for {}: target {}x{}, draft {}x{}",
            layer_type_str, source_kv_heads, source_head_dim,
            consumer_kv_heads, consumer_head_dim
        ));
    }

    std::vector<uint32_t> kv_begin;
    std::vector<uint32_t> kv_shape;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        kv_begin = {0, cache_token_idx_begin, 0};
        kv_shape = {
            consumer_kv_heads,
            _cfg.pipeline_cfg.max_num_tokens,
            consumer_head_dim
        };
    } else {
        kv_begin = {cache_token_idx_begin, 0};
        kv_shape = {aligned_eff_num_cached_tokens, consumer_kv_heads * consumer_head_dim};
    }

    cache_ifm_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(cache_ifm_idx++),
        std::forward_as_tuple(
            &target_lm._cache_buffer(fmt::format("cache_key_l{}", source_layer)),
            kv_begin,
            kv_shape
        )
    );
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cache_ifm_idx++),
            std::forward_as_tuple(
                &target_lm._cache_buffer(fmt::format("cache_key_scale_l{}", source_layer)),
                std::vector<uint32_t>{0, cache_token_idx_begin, 0},
                std::vector<uint32_t>{consumer_kv_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
            )
        );
    }
    if (has_future_token_mask) {
        ++cache_ifm_idx;
    }
    cache_ifm_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(cache_ifm_idx),
        std::forward_as_tuple(
            &target_lm._cache_buffer(fmt::format("cache_val_l{}", source_layer)),
            kv_begin,
            kv_shape
        )
    );
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cache_ifm_idx + 1),
            std::forward_as_tuple(
                &target_lm._cache_buffer(fmt::format("cache_val_scale_l{}", source_layer)),
                std::vector<uint32_t>{0, cache_token_idx_begin, 0},
                std::vector<uint32_t>{consumer_kv_heads, _cfg.pipeline_cfg.max_num_tokens, 1}
            )
        );
    }
}

LanguageModel::Gemma4MtpTargetBatchResult LanguageModel::_run_gemma4_mtp_target_batch(
    uint16_t token_idx, std::span<const uint32_t> token_ids
) {
    const uint16_t verification_width = _cfg.lm_cfg.get_single_num_tokens();
    const uint16_t valid_tokens = checked_u16(token_ids.size(), "Gemma4 MTP target batch");
    const uint16_t num_tokens = valid_tokens == 1 ? 1 : verification_width;
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    if (
        valid_tokens == 0
        || (valid_tokens != 1 && valid_tokens != verification_width)
    ) {
        throw std::runtime_error(fmt::format(
            "Gemma4 MTP target expects either n1 or n{} input, got n{}",
            verification_width, valid_tokens
        ));
    }
    if (
        static_cast<size_t>(token_idx) + valid_tokens
        > _cfg.pipeline_cfg.max_num_tokens
    ) {
        throw std::runtime_error("Gemma4 MTP target batch exceeds cache capacity");
    }

    _upload_gemma4_mtp_causal_mask(
        num_tokens, static_cast<uint16_t>(token_idx + 1), valid_tokens
    );
    _upload_gemma4_mtp_freq_rows(num_tokens, token_idx, valid_tokens);

    std::vector<uint32_t> staged_token_ids(token_ids.begin(), token_ids.end());
    const bool use_int8_embedding_staging = _cfg.pipeline_cfg.quantize_embeddings;
    auto& input_embeds_buf = use_int8_embedding_staging
        ? get_buffer(fmt::format("eagle3_input_embeds_n{}", num_tokens))
        : get_buffer(fmt::format("n{}_buffer1", num_tokens));
    MLABuffer* input_embedding_scales_buf = use_int8_embedding_staging
        ? &get_buffer(fmt::format("eagle3_input_embedding_scales_n{}", num_tokens))
        : nullptr;
    _stage_embedding_rows(
        *this, staged_token_ids, input_embeds_buf, input_embedding_scales_buf
    );

    if (_uses_per_layer_inputs()) {
        const auto per_layer_token_ids = _get_per_layer_token_ids(staged_token_ids);
        _upload_per_layer_embedding_rows(per_layer_token_ids, num_tokens);

        std::map<uint8_t, MLABufferSlice> per_layer_ifm_map;
        const uint8_t input_ifm_idx = _cfg.pipeline_cfg.quantize_embeddings ? 2 : 1;
        per_layer_ifm_map.emplace(
            input_ifm_idx,
            MLABufferSlice{
                &input_embeds_buf,
                {0, 0},
                {num_tokens, hidden_size}
            }
        );
        if (_cfg.pipeline_cfg.quantize_embeddings) {
            per_layer_ifm_map.emplace(
                3,
                MLABufferSlice{
                    input_embedding_scales_buf,
                    {0, 0},
                    {num_tokens, 1}
                }
            );
        }
        const LanguageModelMapKey per_layer_key{num_tokens, 0, 0};
        _per_layer_model_map.at(per_layer_key).run(&per_layer_ifm_map);
    }

    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
        if (layer_type != "full_attention" && layer_type != "sliding_attention") {
            throw std::runtime_error(
                "Gemma4 MTP target supports attention-only language layers"
            );
        }

        const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
        const LanguageModelMapKey cache_key = _get_cache_model_key(
            num_tokens, token_idx, layer_idx
        );
        auto& pre_model = _pre_model_map.at(model_key);
        auto& cache_model = _cache_model_map.at(cache_key);
        auto& post_model = _post_model_map.at(model_key);

        const bool pre_uses_embedding_scale = use_int8_embedding_staging && layer_idx == 0;
        const uint8_t freq_ifm_idx = 1 + static_cast<uint8_t>(pre_uses_embedding_scale);
        const bool is_sliding = layer_type == "sliding_attention";
        auto& freq_real = get_buffer(is_sliding ? "local_freq_real" : "global_freq_real");
        auto& freq_imag = get_buffer(is_sliding ? "local_freq_imag" : "global_freq_imag");
        pre_model._bind_ifm(freq_ifm_idx, &freq_real, {0, 0});
        pre_model._bind_ifm(freq_ifm_idx + 1, &freq_imag, {0, 0});
        if (pre_uses_embedding_scale) {
            pre_model._bind_ifm(1, input_embedding_scales_buf, {0, 0});
            post_model._bind_ifm(1, input_embedding_scales_buf, {0, 0});
        }

        uint8_t ofm_idx = 1;
        if (!_cfg.lm_cfg.is_kv_shared_layer(layer_idx)) {
            auto& key_buffer = _cache_buffer(fmt::format("cache_key_l{}", layer_idx));
            auto& val_buffer = _cache_buffer(fmt::format("cache_val_l{}", layer_idx));
            if (_cfg.pipeline_cfg.use_strided_kv_cache) {
                pre_model._bind_ofm(ofm_idx++, &key_buffer, {0, token_idx, 0});
                if (_cfg.pipeline_cfg.quantize_kv_cache) {
                    auto& scale = _cache_buffer(fmt::format("cache_key_scale_l{}", layer_idx));
                    pre_model._bind_ofm(ofm_idx++, &scale, {0, token_idx, 0});
                }
                pre_model._bind_ofm(ofm_idx++, &val_buffer, {0, token_idx, 0});
            } else {
                pre_model._bind_ofm(ofm_idx++, &key_buffer, {token_idx, 0});
                if (_cfg.pipeline_cfg.quantize_kv_cache) {
                    auto& scale = _cache_buffer(fmt::format("cache_key_scale_l{}", layer_idx));
                    pre_model._bind_ofm(ofm_idx++, &scale, {0, token_idx, 0});
                }
                pre_model._bind_ofm(ofm_idx++, &val_buffer, {token_idx, 0});
            }
            if (_cfg.pipeline_cfg.quantize_kv_cache) {
                auto& scale = _cache_buffer(fmt::format("cache_val_scale_l{}", layer_idx));
                pre_model._bind_ofm(ofm_idx, &scale, {0, token_idx, 0});
            }
        }

        const uint16_t cache_token_idx_begin = is_sliding
            ? static_cast<uint16_t>(std::max(
                0,
                static_cast<int>(token_idx) + num_tokens
                    - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
            ))
            : 0;
        const uint16_t eff_num_cached_tokens = static_cast<uint16_t>(
            token_idx + num_tokens - cache_token_idx_begin
        );
        const bool use_sliding_cache = std::get<1>(cache_key) != 0;
        const std::string_view cache_layer_type = use_sliding_cache
            ? std::string_view("sliding_attention")
            : std::string_view("full_attention");
        const bool is_single_model = (
            num_tokens == _cfg.lm_cfg.get_single_num_tokens() || num_tokens == 1
        );
        const uint16_t cache_mask_size = _get_cache_mask_size(
            cache_layer_type, eff_num_cached_tokens, !is_single_model
        );
        const bool use_single_future_token_mask = is_single_model && cache_mask_size > 1;
        const bool use_group_future_token_mask = !is_single_model && cache_mask_size > num_tokens;
        const uint16_t aligned_eff_token_idx = std::get<2>(cache_key);
        const uint16_t aligned_eff_num_cached_tokens = use_single_future_token_mask
            ? static_cast<uint16_t>(aligned_eff_token_idx + 1)
            : static_cast<uint16_t>(aligned_eff_token_idx + num_tokens);

        std::map<uint8_t, MLABufferSlice> cache_ifm_map;
        uint8_t cache_ifm_idx = 1;
        std::vector<uint32_t> kv_begin;
        std::vector<uint32_t> kv_shape;
        if (_cfg.pipeline_cfg.use_strided_kv_cache) {
            kv_begin = {0, cache_token_idx_begin, 0};
            kv_shape = {
                _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                _cfg.pipeline_cfg.max_num_tokens,
                _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
            };
        } else {
            kv_begin = {cache_token_idx_begin, 0};
            kv_shape = {
                aligned_eff_num_cached_tokens,
                _cfg.lm_cfg.attn_cfg.get_kv_size(layer_type)
            };
        }
        const uint8_t kv_source_layer = _cfg.lm_cfg.get_kv_source_layer(layer_idx);
        cache_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cache_ifm_idx++),
            std::forward_as_tuple(
                &_cache_buffer(fmt::format("cache_key_l{}", kv_source_layer)), kv_begin, kv_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            cache_ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(cache_ifm_idx++),
                std::forward_as_tuple(
                    &_cache_buffer(fmt::format("cache_key_scale_l{}", kv_source_layer)),
                    std::vector<uint32_t>{0, cache_token_idx_begin, 0},
                    std::vector<uint32_t>{
                        _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                        _cfg.pipeline_cfg.max_num_tokens,
                        1
                    }
                )
            );
        }
        if (use_group_future_token_mask || use_single_future_token_mask) {
            ++cache_ifm_idx;
        }
        cache_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(cache_ifm_idx++),
            std::forward_as_tuple(
                &_cache_buffer(fmt::format("cache_val_l{}", kv_source_layer)), kv_begin, kv_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            cache_ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(cache_ifm_idx),
                std::forward_as_tuple(
                    &_cache_buffer(fmt::format("cache_val_scale_l{}", kv_source_layer)),
                    std::vector<uint32_t>{0, cache_token_idx_begin, 0},
                    std::vector<uint32_t>{
                        _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                        _cfg.pipeline_cfg.max_num_tokens,
                        1
                    }
                )
            );
        }

        std::map<uint8_t, MLABufferSlice> ifm_map;
        if (layer_idx == 0) {
            ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    &input_embeds_buf,
                    std::vector<uint32_t>{0, 0},
                    std::vector<uint32_t>{num_tokens, hidden_size}
                )
            );
        }

        pre_model.add_to_queue(&ifm_map);
        cache_model.add_to_queue(&cache_ifm_map);
        post_model.add_to_queue(&ifm_map);
    }

    MLAModelWithBuffer::run_queue();
    _active_cache().metadata.kv_cache_len = static_cast<uint16_t>(
        token_idx + valid_tokens
    );
    return Gemma4MtpTargetBatchResult{
        _argmax_lm_head_rows(num_tokens, valid_tokens),
        _read_gemma4_mtp_target_hidden_rows(num_tokens, valid_tokens)
    };
}

LanguageModel::Gemma4MtpTargetStepResult LanguageModel::_run_gemma4_mtp_target_step(
    uint16_t token_idx, uint32_t token_id
) {
    const std::array<uint32_t, 1> token_ids{token_id};
    auto result = _run_gemma4_mtp_target_batch(token_idx, token_ids);
    return Gemma4MtpTargetStepResult{
        result.next_token_ids.front(),
        std::move(result.hidden_states)
    };
}

LanguageModel::Gemma4MtpDraftStepResult LanguageModel::_run_gemma4_mtp_draft_step(
    LanguageModel& target_lm,
    uint32_t token_id,
    const std::vector<Eigen::bfloat16>& hidden_state,
    uint16_t shared_kv_len,
    uint16_t draft_depth
) {
    if (shared_kv_len == 0) {
        throw std::runtime_error("Gemma4 MTP draft requires at least one target KV row");
    }
    const uint32_t backbone_hidden_size = _cfg.lm_cfg.assistant_backbone_hidden_size;
    if (hidden_state.size() != backbone_hidden_size) {
        throw std::runtime_error(fmt::format(
            "Gemma4 MTP hidden state size {} does not match backbone hidden size {}",
            hidden_state.size(), backbone_hidden_size
        ));
    }

    const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    const uint16_t query_position_id = checked_u16(
        static_cast<size_t>(shared_kv_len - 1) + draft_depth,
        "Gemma4 MTP draft position"
    );
    if (query_position_id >= _cfg.pipeline_cfg.max_num_tokens) {
        throw std::runtime_error("Gemma4 MTP draft position exceeds cache capacity");
    }
    const uint16_t cache_model_token_idx = shared_kv_len > num_tokens
        ? static_cast<uint16_t>(shared_kv_len - num_tokens)
        : 0;

    const auto embedding_row = target_lm._read_embedding_row_bf16(token_id);
    if (embedding_row.size() != backbone_hidden_size) {
        throw std::runtime_error("Gemma4 MTP target embedding width does not match assistant backbone width");
    }

    auto& mtp_input_buf = get_buffer(fmt::format("gemma4_mtp_input_n{}", num_tokens));
    std::vector<Eigen::bfloat16> mtp_input(
        static_cast<size_t>(num_tokens) * 2 * backbone_hidden_size,
        Eigen::bfloat16{0.0f}
    );
    std::memcpy(
        mtp_input.data(), embedding_row.data(),
        backbone_hidden_size * sizeof(Eigen::bfloat16)
    );
    std::memcpy(
        mtp_input.data() + backbone_hidden_size,
        hidden_state.data(),
        backbone_hidden_size * sizeof(Eigen::bfloat16)
    );
    mtp_input_buf.upload(mtp_input.data());

    _upload_gemma4_mtp_causal_mask(num_tokens, shared_kv_len, 1);
    _upload_gemma4_mtp_freq_rows(num_tokens, query_position_id, 1);

    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
        if (layer_type != "full_attention" && layer_type != "sliding_attention") {
            throw std::runtime_error(
                "Gemma4 MTP draft supports attention-only assistant layers"
            );
        }

        const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
        const LanguageModelMapKey cache_key = _get_cache_model_key(
            num_tokens, cache_model_token_idx, layer_idx
        );
        auto& pre_model = _pre_model_map.at(model_key);
        auto& cache_model = _cache_model_map.at(cache_key);
        auto& post_model = _post_model_map.at(model_key);

        const bool is_sliding = layer_type == "sliding_attention";
        auto& freq_real = get_buffer(is_sliding ? "local_freq_real" : "global_freq_real");
        auto& freq_imag = get_buffer(is_sliding ? "local_freq_imag" : "global_freq_imag");
        pre_model._bind_ifm(1, &freq_real, {0, 0});
        pre_model._bind_ifm(2, &freq_imag, {0, 0});

        const uint16_t cache_token_idx_begin = is_sliding
            ? static_cast<uint16_t>(std::max(
                0,
                static_cast<int>(cache_model_token_idx) + num_tokens
                    - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
            ))
            : 0;
        const uint16_t eff_num_cached_tokens = static_cast<uint16_t>(
            cache_model_token_idx + num_tokens - cache_token_idx_begin
        );
        const bool use_sliding_cache = std::get<1>(cache_key) != 0;
        const std::string_view cache_layer_type = use_sliding_cache
            ? std::string_view("sliding_attention")
            : std::string_view("full_attention");
        const bool is_single_model = num_tokens == _cfg.lm_cfg.get_single_num_tokens();
        const uint16_t cache_mask_size = _get_cache_mask_size(
            cache_layer_type, eff_num_cached_tokens, !is_single_model
        );
        const bool use_single_future_token_mask = is_single_model && cache_mask_size > 1;
        const bool use_group_future_token_mask = !is_single_model && cache_mask_size > num_tokens;
        const uint16_t aligned_eff_token_idx = std::get<2>(cache_key);
        const uint16_t aligned_eff_num_cached_tokens = use_single_future_token_mask
            ? static_cast<uint16_t>(aligned_eff_token_idx + 1)
            : static_cast<uint16_t>(aligned_eff_token_idx + num_tokens);

        std::map<uint8_t, MLABufferSlice> cache_ifm_map;
        _bind_gemma4_mtp_shared_kv_cache(
            target_lm,
            cache_ifm_map,
            /*cache_ifm_idx=*/1,
            cache_token_idx_begin,
            aligned_eff_num_cached_tokens,
            use_group_future_token_mask || use_single_future_token_mask,
            layer_type
        );

        std::map<uint8_t, MLABufferSlice> layer0_ifm_map;
        std::map<uint8_t, MLABufferSlice>* ifm_map_ptr = nullptr;
        if (layer_idx == 0) {
            layer0_ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    &mtp_input_buf,
                    std::vector<uint32_t>{0, 0},
                    std::vector<uint32_t>{num_tokens, 2 * backbone_hidden_size}
                )
            );
            ifm_map_ptr = &layer0_ifm_map;
        }

        pre_model.add_to_queue(ifm_map_ptr);
        cache_model.add_to_queue(&cache_ifm_map);
        post_model.add_to_queue(ifm_map_ptr);
    }

    MLAModelWithBuffer::run_queue();

    auto& hidden_buf = get_buffer(fmt::format("n{}_buffer5", num_tokens));
    std::vector<Eigen::bfloat16> projected(
        static_cast<size_t>(num_tokens) * backbone_hidden_size
    );
    hidden_buf.download(projected.data());
    projected.resize(backbone_hidden_size);

    return Gemma4MtpDraftStepResult{
        _argmax_lm_head_row(num_tokens, 0),
        std::move(projected)
    };
}

std::optional<std::vector<uint32_t>> LanguageModel::run_model_gemma4_mtp(
    LanguageModel& draft_lm,
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult* performance_result,
    std::optional<std::string> cache_id
) {
    if (!_cfg.lm_cfg.is_gemma4_mtp_target()) {
        throw std::runtime_error("gemma4_mtp speculative decoding must run on the target model");
    }
    if (!draft_lm._cfg.lm_cfg.is_gemma4_assistant()) {
        throw std::runtime_error(
            "gemma4_mtp speculative decoding requires a gemma4_assistant draft model"
        );
    }
    if (!draft_lm._cfg.lm_cfg.is_gemma4_mtp_draft()) {
        throw std::runtime_error("gemma4_mtp draft model is missing draft speculative config");
    }
    if (_cfg.lm_cfg.hidden_size != draft_lm._cfg.lm_cfg.assistant_backbone_hidden_size) {
        throw std::runtime_error("Gemma4 MTP target hidden size must match assistant backbone size");
    }
    if (input_token_ids.empty()) {
        throw std::runtime_error("Gemma4 MTP generation requires at least one input token");
    }

    auto target_lease = _acquire_kv_cache(cache_id);
    std::optional<KVCacheLease> draft_lease;
    try {
        draft_lease.emplace(draft_lm._acquire_kv_cache(cache_id));
    } catch (...) {
        const bool remove_target = target_lease.cache_created();
        target_lease.reset();
        if (remove_target) {
            _remove_kv_cache(cache_id);
        }
        throw;
    }

    if (target_lease.cache_created() != draft_lease->cache_created()) {
        target_lease.reset();
        draft_lease->reset();
        _remove_kv_cache(cache_id);
        draft_lm._remove_kv_cache(cache_id);
        throw std::runtime_error("Gemma4 MTP target and draft KV cache pools diverged");
    }

    ScopedActiveCache target_active(*this, target_lease.slot());
    ScopedActiveCache draft_active(draft_lm, draft_lease->slot());
    _text_streamer.push(
        DecodeCallbackType::CACHE_CREATED,
        0,
        target_lease.cache_created() || draft_lease->cache_created() ? 1.0 : 0.0
    );

    _is_running = true;
    draft_lm._is_running = true;
    if (performance_result != nullptr) {
        *performance_result = GenerationPerformanceResult{};
        performance_result->accepted_draft_tokens = 0;
    }
    if (!timer_ttft.has_value()) {
        timer_ttft = ChronoTimer{true};
    }

    const uint16_t original_max_num_tokens = set_max_num_tokens(override_max_num_tokens);
    const uint16_t max_length = _max_num_tokens;
    std::vector<uint32_t> input_ids(input_token_ids.begin(), input_token_ids.end());
    std::vector<uint32_t> output_token_ids;
    bool cache_full = false;
    bool stopped = false;

    try {
        _active_cache().metadata.token_ids.clear();
        draft_lm._active_cache().metadata.token_ids.clear();
        draft_lm._active_cache().metadata.kv_cache_len = 0;

        if (input_ids.size() >= max_length) {
            cache_full = true;
        } else {
            _set_input_text_embeds(input_token_ids);
            const auto first_begin = std::chrono::steady_clock::now();
            auto first_token = run_model_prefill(
                input_token_ids,
                /*num_cached_tokens=*/0,
                timer_ttft
            );
            const double first_duration = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - first_begin
            ).count();
            if (_is_running.load(std::memory_order_relaxed)) {
                _active_cache().metadata.kv_cache_len = checked_u16(
                    input_ids.size(), "Gemma4 MTP prompt length"
                );

                output_token_ids.push_back(first_token);
                input_ids.push_back(first_token);
                if (performance_result != nullptr) {
                    performance_result->token_durations.emplace_back(first_duration);
                    performance_result->generated_tokens = 1;
                }

                if (_stop_token_ids.contains(first_token)) {
                    stopped = true;
                }
            }
        }

        const uint16_t draft_budget = draft_lm._cfg.lm_cfg.speculative_decoding_cfg
            .value().speculative_budget;
        const uint16_t verification_width = _cfg.lm_cfg.get_single_num_tokens();
        if (draft_budget == 0 || verification_width != draft_budget + 1) {
            throw std::runtime_error(fmt::format(
                "Gemma4 MTP target width n{} must equal draft budget {} plus one",
                verification_width, draft_budget
            ));
        }
        if (draft_lm._cfg.lm_cfg.get_single_num_tokens() != 1) {
            throw std::runtime_error(
                "Gemma4 MTP assistant must use pointwise n1 execution"
            );
        }

        auto commit_processed_prefix = [&](size_t processed_tokens) {
            const auto processed_end = input_ids.begin()
                + static_cast<std::ptrdiff_t>(processed_tokens);
            _active_cache().metadata.token_ids.assign(input_ids.begin(), processed_end);
            draft_lm._active_cache().metadata.token_ids.assign(
                input_ids.begin(), processed_end
            );
            _active_cache().metadata.kv_cache_len = checked_u16(
                processed_tokens, "Gemma4 MTP committed KV length"
            );
            draft_lm._active_cache().metadata.kv_cache_len = 0;
        };

        std::optional<Gemma4MtpTargetStepResult> carried_target_step;
        size_t speculative_rounds = 0;
        size_t proposed_tokens = 0;
        size_t accepted_tokens = 0;
        bool use_target_only = false;
        constexpr size_t ADAPTATION_WINDOW = 4;
        constexpr size_t MIN_ACCEPTANCE_DENOMINATOR = 4;

        while (
            !cache_full && !stopped
            && _is_running.load(std::memory_order_relaxed)
            && input_ids.size() < max_length
        ) {
            const auto round_begin = std::chrono::steady_clock::now();
            const uint16_t current_pos = checked_u16(
                input_ids.size() - 1, "Gemma4 MTP current position"
            );
            const uint32_t current_token = input_ids.back();

            auto take_target_step = [&]() {
                if (carried_target_step.has_value()) {
                    auto result = std::move(carried_target_step.value());
                    carried_target_step.reset();
                    return result;
                }
                return _run_gemma4_mtp_target_step(current_pos, current_token);
            };

            const bool verification_fits = (
                input_ids.size() + draft_budget <= max_length
                && static_cast<size_t>(current_pos) + verification_width
                    <= _cfg.pipeline_cfg.max_num_tokens
            );
            if (use_target_only || !verification_fits) {
                auto target_step = take_target_step();
                const uint32_t emitted = target_step.next_token_id;
                output_token_ids.emplace_back(emitted);
                input_ids.emplace_back(emitted);
                commit_processed_prefix(static_cast<size_t>(current_pos) + 1);

                stopped = _stop_token_ids.contains(emitted);
                cache_full = input_ids.size() >= max_length;
                const double duration = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - round_begin
                ).count();
                _text_streamer.push(DecodeCallbackType::TPS, emitted, duration, false);
                if (performance_result != nullptr) {
                    performance_result->token_durations.emplace_back(duration);
                    ++performance_result->generated_tokens;
                }
                continue;
            }

            auto target_step = take_target_step();
            std::vector<Gemma4MtpDraftStepResult> draft_steps;
            draft_steps.reserve(draft_budget);
            uint32_t draft_input_token = current_token;
            std::vector<Eigen::bfloat16> draft_hidden = std::move(
                target_step.hidden_state
            );
            const uint16_t shared_kv_len = static_cast<uint16_t>(current_pos + 1);
            for (uint16_t depth = 0; depth < draft_budget; ++depth) {
                auto draft_step = draft_lm._run_gemma4_mtp_draft_step(
                    *this,
                    draft_input_token,
                    draft_hidden,
                    shared_kv_len,
                    depth
                );
                draft_input_token = draft_step.token_id;
                draft_hidden = draft_step.projected_hidden_state;
                draft_steps.emplace_back(std::move(draft_step));
            }

            std::vector<uint32_t> verification_tokens;
            verification_tokens.reserve(verification_width);
            verification_tokens.emplace_back(current_token);
            for (const auto& draft_step : draft_steps) {
                verification_tokens.emplace_back(draft_step.token_id);
            }
            auto verification = _run_gemma4_mtp_target_batch(
                current_pos, verification_tokens
            );
            if (
                verification.next_token_ids.size() != verification_width
                || verification.hidden_states.size()
                    != static_cast<size_t>(verification_width) * _cfg.lm_cfg.hidden_size
            ) {
                throw std::runtime_error(
                    "Invalid Gemma4 MTP target verification output"
                );
            }

            std::vector<std::pair<uint32_t, bool>> emitted_tokens;
            emitted_tokens.reserve(draft_budget);
            uint16_t accepted_this_round = 0;
            uint32_t expected_token = verification.next_token_ids.front();
            for (uint16_t depth = 0; depth < draft_budget; ++depth) {
                const bool accepted = draft_steps[depth].token_id == expected_token;
                const uint32_t emitted = accepted
                    ? draft_steps[depth].token_id
                    : expected_token;
                emitted_tokens.emplace_back(emitted, accepted);
                output_token_ids.emplace_back(emitted);
                input_ids.emplace_back(emitted);
                if (accepted) {
                    ++accepted_this_round;
                    expected_token = verification.next_token_ids[depth + 1];
                }

                if (_stop_token_ids.contains(emitted)) {
                    stopped = true;
                    break;
                }
                if (input_ids.size() >= max_length) {
                    cache_full = true;
                    break;
                }
                if (!accepted) {
                    break;
                }
            }

            const size_t processed_tokens = static_cast<size_t>(current_pos) + 1
                + accepted_this_round;
            commit_processed_prefix(processed_tokens);

            const bool accepted_full_batch = (
                accepted_this_round == draft_budget
                && emitted_tokens.size() == draft_budget
            );
            if (accepted_full_batch && !stopped && !cache_full) {
                const size_t last_hidden_begin = static_cast<size_t>(
                    verification_width - 1
                ) * _cfg.lm_cfg.hidden_size;
                carried_target_step = Gemma4MtpTargetStepResult{
                    verification.next_token_ids.back(),
                    std::vector<Eigen::bfloat16>(
                        verification.hidden_states.begin() + last_hidden_begin,
                        verification.hidden_states.end()
                    )
                };
            }

            ++speculative_rounds;
            proposed_tokens += emitted_tokens.size();
            accepted_tokens += accepted_this_round;
            if (
                !use_target_only
                && speculative_rounds >= ADAPTATION_WINDOW
                && accepted_tokens * MIN_ACCEPTANCE_DENOMINATOR < proposed_tokens
            ) {
                use_target_only = true;
                carried_target_step.reset();
                _logger->info(
                    "Gemma4 MTP acceptance {}/{} is below 25%; "
                    "switching to pointwise target decode",
                    accepted_tokens,
                    proposed_tokens
                );
            }

            const double round_duration = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - round_begin
            ).count();
            const double per_token = round_duration
                / static_cast<double>(emitted_tokens.size());
            for (const auto& [token, from_draft] : emitted_tokens) {
                _text_streamer.push(
                    DecodeCallbackType::TPS, token, per_token, from_draft
                );
                if (performance_result != nullptr) {
                    performance_result->token_durations.emplace_back(per_token);
                    ++performance_result->generated_tokens;
                    if (from_draft) {
                        ++performance_result->accepted_draft_tokens.value();
                    }
                }
            }
        }
    } catch (...) {
        _invalidate_active_kv_cache();
        draft_lm._invalidate_active_kv_cache();
        _is_running = false;
        draft_lm._is_running = false;
        set_max_num_tokens(original_max_num_tokens);
        throw;
    }

    if (!_is_running.load(std::memory_order_relaxed)) {
        _active_cache().metadata.token_ids.clear();
        draft_lm._active_cache().metadata.token_ids.clear();
        _notify_interrupt();
        _text_streamer.wait_streaming();
        draft_lm._is_running = false;
        set_max_num_tokens(original_max_num_tokens);
        return std::nullopt;
    }

    if (cache_full) {
        _notify_cache_full();
    } else {
        _notify_stop();
    }
    _text_streamer.wait_streaming();

    _is_running = false;
    draft_lm._is_running = false;
    set_max_num_tokens(original_max_num_tokens);
    return output_token_ids;
}

} // namespace llima
} // namespace simaai
