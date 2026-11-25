#include "language/embedding_offload.hpp"
#include "language/language_model_state.hpp"

#include <algorithm>
#include <cstring>
#include <fmt/ranges.h>
#include <stdexcept>

namespace simaai::llima {

std::vector<std::map<uint8_t, MLABufferSlice>> LanguageModel::create_input_buffers(
    std::span<const uint32_t> input_token_ids
) {
    // Log.
    _logger->info("Input token ids: [{}]", fmt::join(input_token_ids, ", "));

    // Determine the shape of the buffer for input embeds.
    const uint16_t num_input_tokens = input_token_ids.size();
    uint16_t num_padded_input_tokens = num_input_tokens;
    if (_use_group_token_models) {
        for (const auto& offset: _cfg.pipeline_cfg.input_token_group_offsets.value()) {
            if (offset + _cfg.pipeline_cfg.input_token_group_size > num_input_tokens) {
                num_padded_input_tokens = offset + _cfg.pipeline_cfg.input_token_group_size;
                break;
            }
        }
    }

    // Allocate the input embeds with padding to the buffer.
    if (_resources->embedding_offload) {
        _workspace->offload_chunk_staging = !_image_token_id.has_value()
            || std::find(input_token_ids.begin(), input_token_ids.end(), *_image_token_id) == input_token_ids.end();
        if (_workspace->offload_chunk_staging) {
            num_padded_input_tokens = std::max<uint16_t>(
                _cfg.lm_cfg.get_single_num_tokens(),
                _use_group_token_models ? _cfg.pipeline_cfg.input_token_group_size : 1
            );
        }
    }
    define_buffer(
        "input_embeds",
        {num_padded_input_tokens, _cfg.lm_cfg.hidden_size},
        _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16"
    );
    MLABuffer& input_embeds_buf = get_buffer("input_embeds");
    input_embeds_buf.allocate();
    input_embeds_buf.clear();
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        define_buffer("input_embedding_scales", {num_padded_input_tokens, 1});
        auto& scale_buf = get_buffer("input_embedding_scales");
        scale_buf.allocate();
        scale_buf.clear();
    }

    std::vector<std::map<uint8_t, MLABufferSlice>> vision_ofm_maps;
    if (_cfg.vm_cfg.has_value()) {
        // Allocate the deepstack feature caches for qwen3 with the same size as input embeds.
        for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
            auto& buf = get_buffer(fmt::format("deepstack_feature_l{}_cache", i));
            buf.clear();
        }

        // Find all the start idx of the image tokens and create the ofm_maps for each image.
        auto mm_tokens_per_image = _cfg.mm_cfg.value().mm_tokens_per_image;
        uint32_t token_idx = 0;
        do {
            auto it = std::find(
                input_token_ids.begin() + token_idx, input_token_ids.end(), _image_token_id
            );
            if (it == input_token_ids.end())
                break;

            // Move the token_idx to the start of an image.
            token_idx = std::distance(input_token_ids.begin(), it);

            // Fill the ofm map.
            std::map<uint8_t, MLABufferSlice> ofm_map;
            ofm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    &get_buffer("input_embeds"),
                    std::vector<uint32_t>{token_idx, 0},
                    std::vector<uint32_t>{mm_tokens_per_image, _cfg.lm_cfg.hidden_size}
                )
            );
            const uint8_t deepstack_ofm_begin = _cfg.pipeline_cfg.quantize_embeddings ? 2 : 1;
            if (_cfg.pipeline_cfg.quantize_embeddings) {
                ofm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(1),
                    std::forward_as_tuple(
                        &get_buffer("input_embedding_scales"),
                        std::vector<uint32_t>{token_idx, 0},
                        std::vector<uint32_t>{mm_tokens_per_image, 1}
                    )
                );
            }
            for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
                ofm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(i + deepstack_ofm_begin),
                    std::forward_as_tuple(
                        &get_buffer(fmt::format("deepstack_feature_l{}_cache", i)),
                        std::vector<uint32_t>{token_idx, 0},
                        std::vector<uint32_t>{mm_tokens_per_image, _cfg.lm_cfg.hidden_size}
                    )
                );
            }
            vision_ofm_maps.emplace_back(ofm_map);
            token_idx += _cfg.mm_cfg.value().mm_tokens_per_image;
        } while (token_idx < input_token_ids.size());
    }
    return vision_ofm_maps;
}

void LanguageModel::_prepare_offloaded_prompt(uint16_t num_tokens, uint16_t token_idx) {
    auto& state = *_workspace;
    auto& destination = get_buffer("input_embeds");
    auto* scales = _cfg.pipeline_cfg.quantize_embeddings ? &get_buffer("input_embedding_scales") : nullptr;
    if (state.offload_chunk_staging) {
        destination.clear(false);
        if (scales) scales->clear(false);
    }
    const size_t end = std::min<size_t>(state.offload_prompt_ids.size(), static_cast<size_t>(token_idx) + num_tokens);
    size_t begin = token_idx;
    while (begin < end) {
        if (_image_token_id && state.offload_prompt_ids[begin] == *_image_token_id) {
            ++begin; // Vision has already written this position.
            continue;
        }
        size_t next = begin + 1;
        while (next < end && (!_image_token_id || state.offload_prompt_ids[next] != *_image_token_id)) ++next;
        _gather_embedding_rows(
            std::span<const uint32_t>(state.offload_prompt_ids).subspan(begin, next - begin),
            destination, scales, state.offload_chunk_staging ? begin - token_idx : begin
        );
        begin = next;
    }
    if (state.offload_chunk_staging) {
        destination.flush_cache();
        if (scales) scales->flush_cache();
    }
}




uint16_t LanguageModel::_set_input_text_embeds(std::span<const uint32_t> input_token_ids) {
    // Log.
    _logger->info("Cached token ids: [{}]", fmt::join(_cache->cached_token_ids, ", "));

    const uint16_t num_input_tokens = input_token_ids.size();
    if (_resources->embedding_offload) _workspace->offload_prompt_ids.assign(input_token_ids.begin(), input_token_ids.end());
    MLABuffer& buf = get_buffer("input_embeds");
    MLABuffer* scales_buf = _cfg.pipeline_cfg.quantize_embeddings
        ? &get_buffer("input_embedding_scales") : nullptr;
    uint32_t token_idx = 0;
    uint32_t num_images = 0;
    uint16_t num_cached_tokens = 0;
    while (token_idx < num_input_tokens) {
        const auto& token_id = input_token_ids[token_idx];
        if (_image_token_id.has_value() && token_id == _image_token_id.value()) {
            auto next_token_idx = token_idx + _cfg.mm_cfg.value().mm_tokens_per_image;
            ++num_images;
            token_idx = next_token_idx;
        } else {
            auto next_token_idx = token_idx + 1;
            while (
                next_token_idx < num_input_tokens
                && (!_image_token_id
                    || input_token_ids[next_token_idx] != *_image_token_id)
            )
                ++next_token_idx;
            if (!_resources->embedding_offload)
                _gather_embedding_rows(
                    input_token_ids.subspan(token_idx, next_token_idx - token_idx),
                    buf, scales_buf, token_idx
                );
            for (auto idx = token_idx; idx < next_token_idx; ++idx) {
                if (
                    num_images == 0
                    && idx == num_cached_tokens
                    && idx < _cache->cached_token_ids.size()
                    && input_token_ids[idx] == _cache->cached_token_ids[idx]
                )
                    ++num_cached_tokens;
            }
            token_idx = next_token_idx;
        }
    }
    // Update rope table if needed.
    if (
        _cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "mrope"
        && (_workspace->has_image_token || num_images > 0)
    ) {
        // Previous or current input has image.
        _logger->info("Update rope table");
        auto rope_table = calc_mrope_with_image(
            _cfg, _resources->master_rope_table, _image_token_id.value(), input_token_ids
        );
        get_buffer("global_freq_real").upload(rope_table.re.data());
        get_buffer("global_freq_imag").upload(rope_table.im.data());
    }
    _workspace->has_image_token = num_images > 0;

    auto& request = _run->require_active_request();
    if (request.metrics_model == this) {
        _run->text_streamer.push(
            DecodeCallbackType::CACHED_PROMPT_TOKENS, num_cached_tokens, 0.0
        );
        request.cached_prompt_tokens = num_cached_tokens;
    }
    _logger->info("Number of tokens cached: {:d}", num_cached_tokens);
    return num_cached_tokens;
}


std::vector<uint32_t> LanguageModel::_get_per_layer_token_ids(
    std::span<const uint32_t> input_token_ids
) const {
    // Text-only Gemma4 uses per-layer inputs without defining an image token.
    if (!_image_token_id.has_value())
        return std::vector<uint32_t>(input_token_ids.begin(), input_token_ids.end());

    const uint32_t img_id = _image_token_id.value();
    if (std::find(input_token_ids.begin(), input_token_ids.end(), img_id) == input_token_ids.end()) {
        return std::vector<uint32_t>(input_token_ids.begin(), input_token_ids.end());
    }
    std::vector<uint32_t> token_ids(input_token_ids.begin(), input_token_ids.end());
    const uint32_t pad_id = _pad_token_id.value();
    for (auto& token_id : token_ids) {
        if (token_id == img_id)
            token_id = pad_id;
    }
    return token_ids;
}


void LanguageModel::_upload_per_layer_embedding_rows(
    std::span<const uint32_t> token_ids, uint16_t num_tokens,
    uint16_t valid_tokens
) {
    if (!_uses_per_layer_inputs())
        return;
    if (valid_tokens == 0)
        valid_tokens = num_tokens;
    if (valid_tokens > num_tokens || token_ids.size() < valid_tokens)
        throw std::runtime_error("Invalid per-layer embedding staging row count");
    auto& staging = get_buffer(fmt::format("per_layer_emb_staging_n{}", num_tokens));
    const size_t row_size = staging.get_shape().back() * staging.get_elem_size();
    auto* dst = reinterpret_cast<uint8_t*>(staging.get_virtual_addr());
    MLABuffer* scale_staging = nullptr;
    const uint8_t* scale_src_base = nullptr;
    size_t scale_row_size = 0;
    uint8_t* scale_dst = nullptr;
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        scale_staging = &get_buffer(
            fmt::format("per_layer_emb_staging_scale_n{}", num_tokens)
        );
        scale_row_size = scale_staging->get_buf_len(std::vector<uint32_t>{1, 1});
        scale_dst = reinterpret_cast<uint8_t*>(scale_staging->get_virtual_addr());
        scale_src_base = reinterpret_cast<const uint8_t*>(
            get_buffer("per_layer_embedding_scales").get_virtual_addr()
        );
    }
    if (_resources->embedding_offload) {
        _resources->embedding_offload->per_layer->gather(
            token_ids.first(valid_tokens), dst, row_size
        );
    }
    for (uint16_t i = 0; i < valid_tokens; ++i) {
        if (!_resources->embedding_offload) {
            const size_t shard_idx = token_ids[i] / _resources->per_layer_embedding_rows_per_shard;
            const size_t row_in_shard = token_ids[i] % _resources->per_layer_embedding_rows_per_shard;
            const auto* src = reinterpret_cast<const uint8_t*>(
                _resources->per_layer_embedding_shards[shard_idx]->get_virtual_addr()
            ) + row_in_shard * row_size;
            std::memcpy(dst + i * row_size, src, row_size);
        }
        if (scale_staging != nullptr) {
            const auto* scale_src = scale_src_base
                + static_cast<size_t>(token_ids[i]) * scale_row_size;
            std::memcpy(scale_dst + i * scale_row_size, scale_src, scale_row_size);
        }
    }
    staging.flush_cache();
    if (scale_staging != nullptr) {
        scale_staging->flush_cache();
    }
}


void LanguageModel::_compute_and_upload_per_layer_inputs_prefill(
    uint16_t num_tokens, uint16_t token_idx, uint16_t num_input_tokens
) {
    std::vector<uint32_t> token_ids(num_tokens, 0);
    const auto num_prompt_tokens = std::min<uint16_t>(num_tokens, num_input_tokens - token_idx);
    for (uint16_t i = 0; i < num_prompt_tokens; ++i) {
        token_ids[i] = _workspace->prompt_per_layer_token_ids[token_idx + i];
    }
    _upload_per_layer_embedding_rows(token_ids, num_tokens);
}

} // namespace simaai::llima
