#include "language_model_executor.hpp"
#include "language_model_state.hpp"
#include <fmt/format.h>
#include <exception>
#include <cstring>
#include <limits>
#include "utils.hpp"

namespace simaai::llima {

LanguageExecutor::EmbeddingInput LanguageExecutor::stage_tokens(LanguageModel& model,
                                                                LanguageModel& source,
                                                                std::span<const uint32_t> ids,
                                                                uint16_t width,
                                                                MLABuffer* floating_input) {
    const bool quantized = model._cfg.pipeline_cfg.quantize_embeddings;
    auto* values = quantized
                       ? &model.get_buffer(fmt::format("eagle3_input_embeds_n{}", width))
                       : (floating_input ? floating_input
                                         : &model.get_buffer(fmt::format("n{}_buffer1", width)));
    auto* scales = quantized
                       ? &model.get_buffer(fmt::format("eagle3_input_embedding_scales_n{}", width))
                       : nullptr;
    model._stage_embedding_rows(source, ids, *values, scales);
    return {values, scales};
}

LanguageExecutor::Overrides LanguageExecutor::per_layer_inputs(LanguageModel& model,
                                                               MLABuffer* values, MLABuffer* scales,
                                                               uint32_t row, uint16_t rows,
                                                               std::optional<uint32_t> token) {
    const bool quantized = model._cfg.pipeline_cfg.quantize_embeddings;
    Overrides inputs;
    if (token) {
        const uint32_t canonical = model._image_token_id && *token == *model._image_token_id
                                       ? model._pad_token_id.value()
                                       : *token;
        const bool offloaded = model._resources->embedding_offload != nullptr;
        if (offloaded)
            model._upload_per_layer_embedding_rows({&canonical, 1}, 1);
        auto* shard = offloaded
                          ? &model.get_buffer("per_layer_emb_staging_n1")
                          : model._resources->per_layer_embedding_shards.at(
                                canonical / model._resources->per_layer_embedding_rows_per_shard);
        const auto shard_row =
            offloaded ? 0 : canonical % model._resources->per_layer_embedding_rows_per_shard;
        inputs.emplace(0, MLABufferSlice{shard,
                                         {static_cast<uint32_t>(shard_row), 0},
                                         {1, static_cast<uint32_t>(shard->get_shape().back())}});
        if (quantized)
            inputs.emplace(1, MLABufferSlice{&model.get_buffer("per_layer_embedding_scales"),
                                             {canonical, 0},
                                             {1, 1}});
    }
    inputs.emplace(quantized ? 2 : 1,
                   MLABufferSlice{values, {row, 0}, {rows, model._cfg.lm_cfg.hidden_size}});
    if (quantized)
        inputs.emplace(3, MLABufferSlice{scales, {row, 0}, {rows, 1}});
    return inputs;
}

void LanguageExecutor::bind_attention_inputs(LanguageModel& model, MLAModelWithBuffer& pre,
                                             MLAModelWithBuffer& post, uint8_t layer,
                                             MLABuffer* scales, uint32_t scale_row,
                                             uint16_t rope_row) {
    const bool pre_scale = model._cfg.pipeline_cfg.quantize_embeddings && layer == 0 &&
                           !model._cfg.lm_cfg.is_gemma4_mtp_draft();
    const bool eagle_draft = model._cfg.lm_cfg.is_eagle3_draft();
    if (pre_scale) {
        if (!scales)
            throw std::runtime_error("Quantized embedding input is missing its scale buffer");
        pre._bind_ifm(1, scales, {scale_row, 0});
        if (!eagle_draft)
            post._bind_ifm(1, scales, {scale_row, 0});
    }
    const uint8_t port = 1 + static_cast<uint8_t>(pre_scale) + static_cast<uint8_t>(eagle_draft);
    const bool sliding = model._cfg.lm_cfg.layer_types[layer] == "sliding_attention";
    pre._bind_ifm(port,
                  sliding ? model._workspace->local_freq_real : model._workspace->global_freq_real,
                  {rope_row, 0});
    pre._bind_ifm(port + 1,
                  sliding ? model._workspace->local_freq_imag : model._workspace->global_freq_imag,
                  {rope_row, 0});
}

void LanguageExecutor::bind_kv_outputs(LanguageModel& model, MLAModelWithBuffer& pre, uint8_t layer,
                                       uint16_t position) {
    if (model._cfg.lm_cfg.is_kv_shared_layer(layer))
        return;
    const auto& buffers = model._workspace->attention_binding_buffers.at(layer);
    uint8_t port = 1;
    if (model._cfg.pipeline_cfg.use_strided_kv_cache) {
        pre._bind_ofm(port++, buffers.key, {0, position, 0});
        if (model._cfg.pipeline_cfg.quantize_kv_cache)
            pre._bind_ofm(port++, buffers.key_scale, {0, position, 0});
        pre._bind_ofm(port++, buffers.value, {0, position, 0});
    } else {
        pre._bind_ofm(port++, buffers.key, {position, 0});
        if (model._cfg.pipeline_cfg.quantize_kv_cache)
            pre._bind_ofm(port++, buffers.key_scale, {0, position, 0});
        pre._bind_ofm(port++, buffers.value, {position, 0});
    }
    if (model._cfg.pipeline_cfg.quantize_kv_cache)
        pre._bind_ofm(port, buffers.value_scale, {0, position, 0});
}

LanguageExecutor::Overrides LanguageExecutor::post_rows(LanguageModel& model, uint16_t width,
                                                        uint8_t layer, uint16_t position,
                                                        uint32_t first, uint16_t rows,
                                                        MLABuffer* hidden) {
    const auto& cfg = model._cfg;
    const auto q_size = cfg.lm_cfg.attn_cfg.get_q_size(cfg.lm_cfg.layer_types[layer]);
    Overrides inputs{
        {0, MLABufferSlice{hidden, {first, 0}, {rows, cfg.lm_cfg.hidden_size}}},
        {1, MLABufferSlice{nullptr, {first, 0}, {rows, q_size}}},
    };
    uint8_t port = 2;
    if (model._uses_per_layer_inputs()) {
        inputs.emplace(port++, MLABufferSlice{nullptr,
                                              {static_cast<uint32_t>(layer) * width + first, 0},
                                              {rows, cfg.lm_cfg.hidden_size_per_layer_input}});
    }
    if (cfg.lm_cfg.attn_cfg.attn_output_gate) {
        inputs.emplace(port++, MLABufferSlice{nullptr, {first, 0}, {rows, q_size}});
    }
    if (cfg.vm_cfg && layer < cfg.vm_cfg->deepstack_visual_indexes.size()) {
        auto& deepstack = model.get_buffer(fmt::format("deepstack_feature_l{}_cache", layer));
        inputs.emplace(port,
                       MLABufferSlice{nullptr,
                                      {position + first, 0},
                                      {rows, static_cast<uint32_t>(deepstack.get_shape().back())}});
    }
    return inputs;
}

LanguageExecutor::LogitsWindow LanguageExecutor::logits_window(const LanguageModelConfig& cfg,
                                                              uint16_t width, uint16_t row) {
    const uint16_t post_width = cfg.is_speculative_draft() || width == 1 ? width : cfg.get_single_num_tokens();
    const uint16_t first = row >= post_width ? row + 1 - post_width : 0;
    return {first, post_width};
}

void LanguageExecutor::queue_attention(AttentionPrograms models, LayerBindings bindings) {
    models.pre.add_to_queue(bindings.input);
    models.cache.add_to_queue(bindings.cache);
    models.post.add_to_queue(bindings.post_input, bindings.output);
}

void LanguageExecutor::queue_layer(LanguageModel& model, uint16_t width, uint16_t position,
                                   uint8_t layer, MLABuffer* scales, uint32_t scale_row,
                                   LayerBindings bindings, LayerOutput output, uint16_t logit_row) {
    const auto& cfg = model._cfg.lm_cfg;
    const auto& type = cfg.layer_types[layer];
    if (type == "full_attention" || type == "sliding_attention") {
        auto programs = bind_attention(model, width, position, layer, scales, scale_row);
        if (output == LayerOutput::CacheOnly) {
            programs.pre.add_to_queue(bindings.input);
            return;
        }

        // The final post consumes one row for ordinary generation or a contiguous
        // verification-width window containing the requested row for speculative targets.
        Overrides sliced_inputs;
        if (output == LayerOutput::LogitsRow) {
            const auto window = logits_window(cfg, width, logit_row);
            if (window.width != width) {
                MLABuffer* hidden = nullptr;
                if (bindings.post_input && bindings.post_input->contains(0))
                    hidden = bindings.post_input->at(0).get_buf_ptr();
                sliced_inputs = post_rows(model, width, layer, position, window.first, window.width, hidden);
                bindings.post_input = &sliced_inputs;
            }
        }
        if (cfg.is_moe()) {
            programs.pre.add_to_queue(bindings.input);
            programs.cache.add_to_queue(bindings.cache);
            model._run_moe_post(bindings.post_input, width, layer);
        } else {
            queue_attention(programs, bindings);
        }
        return;
    }

    if (type != "linear_attention" && type != "conv")
        throw std::runtime_error("Unsupported layer type: " + type);
    const LanguageModelMapKey key{width, layer, 0};
    auto& program = type == "linear_attention" ? model._workspace->linear_model_map.at(key)
                                               : model._workspace->conv_model_map.at(key);
    if (model._cfg.pipeline_cfg.quantize_embeddings && layer == 0)
        program._bind_ifm(1, scales, {scale_row, 0});
    program.add_to_queue(bindings.input, bindings.output);
    if (type == "conv" && output == LayerOutput::LogitsRow) {
        Overrides row;
        if (width > 1)
            row.emplace(0, MLABufferSlice{nullptr, {logit_row, 0}, {1, cfg.hidden_size}});
        model._workspace->conv_final_model_map.at(key).add_to_queue(&row);
    }
}

LanguageExecutor::AttentionPrograms LanguageExecutor::bind_attention(
    LanguageModel& model, uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx,
    MLABuffer* embedding_scale_buf, uint32_t embedding_scale_row) {
    const auto& cfg = model._cfg;
    const LanguageModelMapKey pre_post_key{num_tokens, layer_idx, 0};
    const LanguageModelMapKey cache_key = model._get_cache_model_key(num_tokens, token_idx, layer_idx);
    auto& pre_model = model._workspace->pre_model_map.at(pre_post_key);
    auto& cache_model = model._workspace->cache_model_map.at(cache_key);
    // MoE has no post model; bind these inputs to the router (it consumes the raw hidden).
    auto& post_model = cfg.lm_cfg.is_moe() ? model._workspace->router_model_map.at(pre_post_key)
                                            : model._workspace->post_model_map.at(pre_post_key);

    const auto& layer_type = cfg.lm_cfg.layer_types[layer_idx];
    const auto& binding_buffers = model._workspace->attention_binding_buffers.at(layer_idx);
    const uint16_t single_num_tokens = cfg.lm_cfg.get_single_num_tokens();
    const bool is_single_model = num_tokens == single_num_tokens;

    const uint16_t cache_token_idx_begin =
        model._get_cache_token_idx_begin(num_tokens, token_idx, layer_type);

    const bool is_eagle3_draft = cfg.lm_cfg.is_eagle3_draft();
    const bool is_gemma4_mtp_draft = cfg.lm_cfg.is_gemma4_mtp_draft();
    bind_attention_inputs(model, pre_model, post_model, layer_idx,
                                            embedding_scale_buf, embedding_scale_row, token_idx);

    if (is_gemma4_mtp_draft && cfg.lm_cfg.is_kv_shared_layer(layer_idx)) {
        throw std::runtime_error("Gemma4 MTP shared-KV binding must use run_model_gemma4_mtp");
    }

    bind_kv_outputs(model, pre_model, layer_idx, token_idx);

    const uint16_t eff_num_cached_tokens = token_idx + num_tokens - cache_token_idx_begin;
    const bool use_sliding_cache = std::get<1>(cache_key) != 0;
    const char* cache_layer_type = use_sliding_cache ? "sliding_attention" : "full_attention";
    const uint16_t cache_mask_size =
        model._get_cache_mask_size(cache_layer_type, eff_num_cached_tokens, !is_single_model);
    const bool use_single_future_token_mask = is_single_model && cache_mask_size > 1;
    const bool use_group_future_token_mask = (!is_single_model && cache_mask_size > num_tokens);
    // IFM 0 is the query buffer. Its pointer, begin, and shape are invariant
    // for a compact cache key, so only the layer-owned cache inputs are rebound.
    uint8_t cache_ifm_idx = 1;
    if (cfg.pipeline_cfg.use_strided_kv_cache) {
        cache_model._bind_ifm(cache_ifm_idx++, binding_buffers.key, {0, cache_token_idx_begin, 0});
    } else {
        cache_model._bind_ifm(cache_ifm_idx++, binding_buffers.key, {cache_token_idx_begin, 0});
    }
    if (cfg.pipeline_cfg.quantize_kv_cache) {
        cache_model._bind_ifm(cache_ifm_idx++, binding_buffers.key_scale,
                              {0, cache_token_idx_begin, 0});
    }
    if (use_group_future_token_mask) {
        // Group-mask buffer, begin, and shape are fixed by the compact key.
        ++cache_ifm_idx;
    } else if (use_single_future_token_mask) {
        auto& mask = model.get_buffer("future_token_mask");
        if (cfg.lm_cfg.is_dflash()) {
            // DFlash uploads the compact [B, aligned_context] mask layout.
            cache_model._bind_ifm(cache_ifm_idx++, &mask, {0, 0});
        } else if (cfg.lm_cfg.is_spec_decode()) {
            cache_model._bind_ifm(cache_ifm_idx++, &mask, {0, cache_token_idx_begin});
        } else {
            cache_model._bind_ifm(
                cache_ifm_idx++, &mask,
                {static_cast<uint32_t>(cfg.pipeline_cfg.max_num_tokens - eff_num_cached_tokens)});
        }
    }
    // gpt_oss sinks: cache model is layer-shared, so bind this layer's sinks before cached_values.
    if (cfg.lm_cfg.uses_attention_sinks()) {
        auto& sinks_buf = model.get_buffer(fmt::format("sinks_l{}", layer_idx));
        cache_model._bind_ifm(cache_ifm_idx++, &sinks_buf, {0, 0});
    }
    if (cfg.pipeline_cfg.use_strided_kv_cache) {
        cache_model._bind_ifm(cache_ifm_idx++, binding_buffers.value,
                              {0, cache_token_idx_begin, 0});
    } else {
        cache_model._bind_ifm(cache_ifm_idx++, binding_buffers.value, {cache_token_idx_begin, 0});
    }
    if (cfg.pipeline_cfg.quantize_kv_cache) {
        cache_model._bind_ifm(cache_ifm_idx, binding_buffers.value_scale,
                              {0, cache_token_idx_begin, 0});
    }

    // Per-layer post input is invariant because post wrappers remain per layer.
    const uint8_t post_ifm_idx =
        (2 +
         static_cast<uint8_t>(cfg.pipeline_cfg.quantize_embeddings && layer_idx == 0 &&
                              !is_gemma4_mtp_draft && !is_eagle3_draft) +
         static_cast<uint8_t>(model._uses_per_layer_inputs()) +
         static_cast<uint8_t>(cfg.lm_cfg.attn_cfg.attn_output_gate));
    if (cfg.vm_cfg.has_value() &&
        layer_idx < cfg.vm_cfg.value().deepstack_visual_indexes.size() && num_tokens > 1) {
        auto& deepstack = model.get_buffer(fmt::format("deepstack_feature_l{}_cache", layer_idx));
        post_model._bind_ifm(post_ifm_idx, &deepstack, {token_idx, 0});
    }
    return {pre_model, cache_model, post_model};
}

void LanguageModel::_gather_embedding_rows(std::span<const uint32_t> ids, MLABuffer& destination,
                                           MLABuffer* scales, size_t row_begin) {
    const auto& shape = destination.get_shape();
    const auto dtype = _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16";
    if (shape.size() != 2 || shape.back() != _cfg.lm_cfg.hidden_size || row_begin > shape.front() ||
        ids.size() > shape.front() - row_begin || destination.get_dtype() != dtype ||
        (_cfg.pipeline_cfg.quantize_embeddings != (scales != nullptr)))
        throw std::runtime_error("Invalid embedding staging shape or dtype");
    const size_t stride =
        destination.get_buf_len(std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size});
    auto* dst = static_cast<uint8_t*>(destination.get_virtual_addr()) + row_begin * stride;
    if (_resources->embedding_offload) {
        _resources->embedding_offload->normal->gather(ids, dst, stride);
    } else {
        const auto& source = get_buffer("embeddings");
        if (source.get_shape().size() != 2 || source.get_shape().back() != shape.back() ||
            source.get_dtype() != destination.get_dtype() ||
            source.get_buf_len(std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size}) != stride)
            throw std::runtime_error("Embedding staging source shape, dtype, or stride mismatch");
        for (auto id : ids) {
            if (id >= source.get_shape().front())
                throw std::out_of_range("Embedding token ID exceeds vocabulary");
        }
        const auto* src = static_cast<const uint8_t*>(source.get_virtual_addr());
        for (size_t i = 0; i < ids.size(); ++i)
            std::memcpy(dst + i * stride, src + ids[i] * stride, stride);
    }
    if (scales) {
        if (scales->get_shape().size() != 2 || scales->get_shape().back() != 1 ||
            row_begin > scales->get_shape().front() ||
            ids.size() > scales->get_shape().front() - row_begin)
            throw std::runtime_error("Invalid embedding scale staging shape");
        auto& source = get_buffer("embedding_scales");
        const size_t scale_stride = source.get_buf_len(std::vector<uint32_t>{1, 1});
        if (scales->get_buf_len(std::vector<uint32_t>{1, 1}) != scale_stride)
            throw std::runtime_error("Invalid embedding scale stride");
        for (size_t i = 0; i < ids.size(); ++i) {
            std::memcpy(
                static_cast<uint8_t*>(scales->get_virtual_addr()) + (row_begin + i) * scale_stride,
                static_cast<const uint8_t*>(source.get_virtual_addr()) + ids[i] * scale_stride,
                scale_stride);
        }
        if (!ids.empty())
            scales->flush_cache(row_begin * scale_stride, ids.size() * scale_stride);
    }
    if (!ids.empty())
        destination.flush_cache(row_begin * stride, ids.size() * stride);
}

void LanguageModel::_stage_embedding_rows(LanguageModel& source_model,
                                          std::span<const uint32_t> token_ids,
                                          MLABuffer& destination, MLABuffer* destination_scales) {
    const bool fills_destination =
        (destination.get_shape().size() == 2 &&
         token_ids.size() == destination.get_shape().front() &&
         (destination_scales == nullptr ||
          (destination_scales->get_shape().size() == 2 &&
           token_ids.size() == destination_scales->get_shape().front())));
    if (!fills_destination) {
        destination.clear(false);
        if (destination_scales)
            destination_scales->clear(false);
    }
    source_model._gather_embedding_rows(token_ids, destination, destination_scales);
    // Speculative batches can have fewer valid rows than their MLA shape.
    // Publish the cleared padding as well as the gathered rows.
    if (!fills_destination) {
        destination.flush_cache();
        if (destination_scales)
            destination_scales->flush_cache();
    }
}

void LanguageModel::_prepare_lm_head_logit_views(uint16_t num_tokens, uint16_t output_rows,
                                                 std::vector<LMHeadLogitView>& views) {
    const uint32_t output_size = _cfg.lm_cfg.get_lm_head_output_size();
    const uint16_t num_splits = _cfg.lm_cfg.lm_head_num_splits;
    views.clear();
    views.reserve(num_splits);
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
        if (buf.get_dtype() != "bfloat16" || buf.get_shape().size() != 2 ||
            buf.get_shape().front() != output_rows || buf.get_shape().back() != split_size) {
            throw std::runtime_error("Invalid lm_head logit buffer");
        }
        views.emplace_back(LMHeadLogitView{
            &buf, static_cast<const uint8_t*>(buf.get_virtual_addr()),
            buf.get_buf_len(std::vector<uint32_t>{1, split_size}), split_begin, split_size});
    }
}

uint32_t LanguageModel::_argmax_lm_head_row(std::span<const LMHeadLogitView> views, uint16_t row) {
    uint32_t best_index = 0;
    float best_value = -std::numeric_limits<float>::infinity();
    for (const auto& view : views) {
        const auto* row_ptr = reinterpret_cast<const Eigen::bfloat16*>(
            view.data + static_cast<size_t>(row) * view.row_bytes);
        const auto [value, offset] = argmax_bf16(row_ptr, view.width);
        const uint32_t index = view.vocab_begin + offset;
        if (value > best_value || (value == best_value && index < best_index)) {
            best_value = value;
            best_index = index;
        }
    }
    return best_index;
}

std::vector<uint32_t>
LanguageModel::_argmax_lm_head_rows(uint16_t num_tokens, uint16_t valid_tokens,
                                    std::span<const uint32_t> expected_draft_tokens) {
    if (valid_tokens == 0 || valid_tokens > num_tokens ||
        expected_draft_tokens.size() + 1 != valid_tokens) {
        throw std::runtime_error(
            fmt::format("Requested {} lm_head rows from n{} output", valid_tokens, num_tokens));
    }

    const uint16_t num_splits = _cfg.lm_cfg.lm_head_num_splits;
    if (num_splits == 1 && !_cfg.pipeline_cfg.return_logits) {
        auto& buf = get_buffer(fmt::format("n{}_buffer4", num_tokens));
        buf.invalidate_cache();
        const auto* output = static_cast<const uint32_t*>(buf.get_virtual_addr());
        std::vector<uint32_t> token_ids;
        token_ids.reserve(valid_tokens);
        for (uint16_t row = 0; row < valid_tokens; ++row) {
            token_ids.emplace_back(output[row]);
            if (row < expected_draft_tokens.size() && output[row] != expected_draft_tokens[row]) {
                break;
            }
        }
        return token_ids;
    }

    std::vector<LMHeadLogitView> views;
    _prepare_lm_head_logit_views(num_tokens, num_tokens, views);

    std::vector<uint32_t> best_indices;
    best_indices.reserve(valid_tokens);
    for (uint16_t row = 0; row < valid_tokens; ++row) {
        for (const auto& view : views) {
            view.buffer->invalidate_cache(static_cast<size_t>(row) * view.row_bytes,
                                          view.row_bytes);
        }
        const uint32_t best_index = _argmax_lm_head_row(views, row);
        best_indices.emplace_back(best_index);
        if (row < expected_draft_tokens.size() && best_index != expected_draft_tokens[row]) {
            break;
        }
    }
    return best_indices;
}

void LanguageModel::_bind_kv_cache_inputs(LanguageModel& kv_owner, uint8_t source_layer,
                                          std::map<uint8_t, MLABufferSlice>& cache_ifm_map,
                                          uint8_t cache_ifm_idx, uint16_t cache_token_idx_begin,
                                          uint16_t aligned_eff_num_cached_tokens,
                                          bool has_future_token_mask) {
    const auto& buffers = kv_owner._workspace->attention_binding_buffers.at(source_layer);
    const auto& attn_cfg = kv_owner._cfg.lm_cfg.attn_cfg;
    const uint32_t kv_heads = attn_cfg.num_key_value_heads;
    const uint32_t head_dim = attn_cfg.get_head_dim(kv_owner._cfg.lm_cfg.layer_types[source_layer]);
    std::vector<uint32_t> kv_begin;
    std::vector<uint32_t> kv_shape;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        kv_begin = {0, cache_token_idx_begin, 0};
        kv_shape = {kv_heads, _cfg.pipeline_cfg.max_num_tokens, head_dim};
    } else {
        kv_begin = {cache_token_idx_begin, 0};
        kv_shape = {aligned_eff_num_cached_tokens, kv_heads * head_dim};
    }

    cache_ifm_map.emplace(std::piecewise_construct, std::forward_as_tuple(cache_ifm_idx++),
                          std::forward_as_tuple(buffers.key, kv_begin, kv_shape));
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifm_map.emplace(
            std::piecewise_construct, std::forward_as_tuple(cache_ifm_idx++),
            std::forward_as_tuple(
                buffers.key_scale, std::vector<uint32_t>{0, cache_token_idx_begin, 0},
                std::vector<uint32_t>{kv_heads, _cfg.pipeline_cfg.max_num_tokens, 1}));
    }
    if (has_future_token_mask) {
        ++cache_ifm_idx;
    }
    cache_ifm_map.emplace(std::piecewise_construct, std::forward_as_tuple(cache_ifm_idx),
                          std::forward_as_tuple(buffers.value, kv_begin, kv_shape));
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifm_map.emplace(
            std::piecewise_construct, std::forward_as_tuple(cache_ifm_idx + 1),
            std::forward_as_tuple(
                buffers.value_scale, std::vector<uint32_t>{0, cache_token_idx_begin, 0},
                std::vector<uint32_t>{kv_heads, _cfg.pipeline_cfg.max_num_tokens, 1}));
    }
}

std::unique_lock<std::recursive_mutex> LanguageExecutor::lock_execution() {
    return std::unique_lock(RuntimeContext::instance().execution_mutex);
}

LanguageExecutor::Run::Run(LanguageModel& target, LanguageModel* draft)
    : _target(target), _draft(draft), _limit(target._run->max_num_tokens),
      _stop_ids(target._run->stop_token_ids), _exceptions(std::uncaught_exceptions()),
      _outer(!target._run->is_running.load(std::memory_order_relaxed)) {
    if (!_outer)
        return;
    try {
        target._run->cancelled = false;
        target._run->is_running = true;
        if (draft) {
            draft->_run->cancelled = false;
            draft->_run->is_running = true;
        }
        target._run->text_streamer.begin();
        auto execution = lock_execution();
        target.bind_session_adapter();
        if (draft)
            draft->bind_session_adapter();
    } catch (...) {
        target._run->is_running = false;
        if (draft)
            draft->_run->is_running = false;
        try {
            target._run->text_streamer.finish();
        } catch (...) {
        }
        throw;
    }
}

void LanguageExecutor::Run::complete() {
    if (_outer && !_stream_finished) {
        _stream_finished = true;
        _target._run->text_streamer.finish();
    }
}

LanguageExecutor::Run::~Run() {
    _target._workspace->capture_state_checkpoints = false;
    _target._workspace->rolling_checkpoint_slot = 0;
    _target._workspace->writable_checkpoint_slots = 0;
    _target._run->max_num_tokens = _limit;
    _target._run->stop_token_ids = std::move(_stop_ids);
    if (_outer) {
        bool failed = std::uncaught_exceptions() > _exceptions;
        try {
            complete();
        } catch (...) {
            // Cleanup must not replace the original inference exception.
            failed = true;
        }
        if (failed)
            MLAModelWithBuffer::discard_queue();
        if (failed || _target._run->cancelled.load(std::memory_order_relaxed)) {
            for (auto* model : {&_target, _draft}) {
                if (model)
                    model->_session->invalidate_reuse();
            }
        }
        _target._run->is_running = false;
        if (_draft)
            _draft->_run->is_running = false;
    }
}

} // namespace simaai::llima
