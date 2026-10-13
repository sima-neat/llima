#include <cstring>

#include "language_model.hpp"
#include "gemma4_unified_helpers.hpp"

namespace simaai::llima {

void LanguageModel::_prepare_unified_input(std::span<const uint32_t> ids, uint16_t begin,
                                         uint16_t end, const MediaLoader& media_loader) {
    auto& input = get_buffer("input_embeds");
    auto* scales = _cfg.pipeline_cfg.quantize_embeddings ? &get_buffer("input_embedding_scales") : nullptr;
    input.clear(false);
    if (scales) scales->clear(false);
    const uint16_t valid_end = std::min<size_t>(end, ids.size());
    bool has_media = false;
    auto is_media = [&](uint32_t token) {
        return (_image_token_id && token == *_image_token_id)
            || (_cfg.support_audio() && token == _cfg.audio_cfg->token_id);
    };
    for (uint16_t at = begin; at < valid_end;) {
        if (is_media(ids[at])) { has_media = true; ++at; continue; }
        uint16_t next = at + 1;
        while (next < valid_end && !is_media(ids[next])) ++next;
        _gather_embedding_rows(ids.subspan(at, next - at), input, scales, at - begin);
        at = next;
    }
    if (has_media) {
        if (!media_loader) throw std::invalid_argument("Media tokens require image/audio inputs");
        const MLABufferSlice values{&input, {0, 0}, {static_cast<uint32_t>(valid_end - begin), _cfg.lm_cfg.hidden_size}};
        const MLABufferSlice scale_rows{scales, {0, 0}, {static_cast<uint32_t>(valid_end - begin), 1}};
        media_loader(begin, valid_end, values, scales ? &scale_rows : nullptr);
    }
    input.flush_cache();
    if (scales) scales->flush_cache();
}

void LanguageModel::_prepare_image_windows(uint16_t prompt_length) {
    const auto group = _cfg.pipeline_cfg.input_token_group_size;
    _image_windows = gemma4_image_windows(_image_spans, group);
    uint16_t rows = 0;
    for (const auto& [begin, end] : _image_windows) {
        if (end > _cfg.pipeline_cfg.max_num_tokens || begin >= prompt_length) {
            throw std::invalid_argument("Image window exceeds the compiled context");
        }
        if (end - begin > group) rows = std::max<uint16_t>(rows, end - begin);
    }
    if (rows <= _image_scratch_rows) return;
    const auto heads = _cfg.lm_cfg.attn_cfg.num_attention_heads;
    const auto max_dim = std::max(_cfg.lm_cfg.attn_cfg.get_head_dim("full_attention"),
                                  _cfg.lm_cfg.attn_cfg.get_head_dim("sliding_attention"));
    _image_hidden = std::make_unique<MLABuffer>("image_hidden", std::vector<size_t>{rows, _cfg.lm_cfg.hidden_size}, "bfloat16", true);
    _image_queries = std::make_unique<MLABuffer>("image_queries", std::vector<size_t>{rows * heads * max_dim}, "bfloat16", false);
    if (_get_cache_mask_size("full_attention", _cfg.pipeline_cfg.max_num_tokens, true) > group) {
        _image_global_masks = std::make_unique<MLABuffer>("image_global_masks",
            std::vector<size_t>{static_cast<size_t>(rows) * _cfg.pipeline_cfg.max_num_tokens}, "bfloat16", false);
    }
    _image_sliding_masks = std::make_unique<MLABuffer>("image_sliding_masks",
        std::vector<size_t>{static_cast<size_t>(rows) * _cfg.get_prefill_cache_capacity("sliding_attention")}, "bfloat16", false);
    for (auto* buffer : {_image_hidden.get(), _image_queries.get(), _image_global_masks.get(), _image_sliding_masks.get()}) {
        if (buffer) buffer->allocate();
    }
    _image_scratch_rows = rows;
}

void LanguageModel::_upload_image_attention_masks(uint16_t begin, uint16_t valid_end, uint16_t slot) {
    const uint16_t group = _cfg.pipeline_cfg.input_token_group_size;
    const bool staged = std::any_of(_image_windows.begin(), _image_windows.end(), [&](const auto& window) {
        return window.first <= begin && begin < window.second && window.second - window.first > group;
    });
    for (const std::string type : {"full_attention", "sliding_attention"}) {
        const bool sliding = type == "sliding_attention";
        uint16_t capacity = _cfg.get_prefill_cache_capacity(type);
        if (sliding) {
            capacity = gemma4_cache_view(begin, group, _cfg.pipeline_cfg.max_num_tokens,
                _cfg.lm_cfg.attn_cfg.sliding_window.value(), _image_spans).second;
        } else {
            const uint16_t mask_size = _get_cache_mask_size(type, begin + group, true);
            if (mask_size <= group) continue; // The causal mask is compiled into this bucket.
            capacity = std::min<uint16_t>(round_up_to(begin + group, mask_size), capacity);
        }
        const auto mask = gemma4_attention_mask(begin, group,
            _get_cache_token_idx_begin(group, begin, type), capacity, valid_end, _image_spans,
            sliding ? _cfg.lm_cfg.attn_cfg.sliding_window.value() : 0);
        auto* buffer = staged ? (sliding ? _image_sliding_masks.get() : _image_global_masks.get())
            : &get_buffer(sliding ? "group_sliding_future_token_mask" : "group_future_token_mask");
        const uint16_t stride = _cfg.get_prefill_cache_capacity(type);
        buffer->upload_raw(mask.data(), static_cast<size_t>(slot) * group * stride * sizeof(Eigen::bfloat16),
                           mask.size() * sizeof(Eigen::bfloat16));
    }
}

uint32_t LanguageModel::_run_image_window(uint16_t begin, uint16_t end, uint16_t prompt_length) {
    const uint16_t group = _cfg.pipeline_cfg.input_token_group_size;
    const uint16_t chunks = (end - begin) / group;
    const uint32_t hidden = _cfg.lm_cfg.hidden_size;
    auto& input = get_buffer("input_embeds");
    const bool quantize_embeddings = _cfg.pipeline_cfg.quantize_embeddings;
    auto* scales = quantize_embeddings ? &get_buffer("input_embedding_scales") : nullptr;
    if (!quantize_embeddings) {
        const size_t bytes = static_cast<size_t>(end - begin) * hidden * sizeof(Eigen::bfloat16);
        input.invalidate_cache(0, bytes);
        _image_hidden->upload_raw(input.get_virtual_addr(), 0, bytes);
    }
    for (uint16_t slot = 0; slot < chunks; ++slot) {
        _upload_image_attention_masks(begin + slot * group, std::min(end, prompt_length), slot);
    }
    for (uint8_t layer = 0; layer < _cfg.lm_cfg.num_hidden_layers; ++layer) {
        const bool quantized_input = layer == 0 && quantize_embeddings;
        auto* residual_input = quantized_input ? &input : _image_hidden.get();
        const auto& type = _cfg.lm_cfg.layer_types[layer];
        const uint32_t query_elements = _cfg.lm_cfg.attn_cfg.num_attention_heads * group
            * _cfg.lm_cfg.attn_cfg.get_head_dim(type);
        for (uint16_t slot = 0; slot < chunks; ++slot) {
            const auto models = _bind_attn_models(group, begin + slot * group, layer,
                                                 scales, slot * group);
            std::map<uint8_t, MLABufferSlice> ifms{{0, {residual_input,
                {static_cast<uint32_t>(slot) * group, 0}, {group, hidden}}}};
            std::map<uint8_t, MLABufferSlice> ofms{{0, {_image_queries.get(), {slot * query_elements}, {query_elements}}}};
            models.pre->add_to_queue(&ifms, &ofms);
        }
        MLAModelWithBuffer::run_queue();
        if (!_is_running.load(std::memory_order_relaxed)) return 0;
        const bool final_layer = layer + 1 == _cfg.lm_cfg.num_hidden_layers;
        if (final_layer && end < prompt_length) break;
        for (uint16_t slot = 0; slot < chunks; ++slot) {
            if (final_layer && slot + 1 < chunks) continue;
            const uint16_t token_idx = begin + slot * group;
            const auto models = _bind_attn_models(group, token_idx, layer,
                                                 scales, slot * group);
            const bool sliding = type == "sliding_attention";
            std::map<uint8_t, MLABufferSlice> cache_ifms{{0, {_image_queries.get(), {slot * query_elements}, {query_elements}}}};
            if (sliding || _get_cache_mask_size(type, token_idx + group, true) > group) {
                const uint16_t capacity = std::get<2>(_get_cache_model_key(group, token_idx, layer)) + group;
                const uint16_t stride = _cfg.get_prefill_cache_capacity(type);
                cache_ifms.emplace(2 + _cfg.pipeline_cfg.quantize_kv_cache, MLABufferSlice{
                    sliding ? _image_sliding_masks.get() : _image_global_masks.get(),
                    {static_cast<uint32_t>(slot) * group * stride}, {static_cast<uint32_t>(group) * capacity}});
            }
            models.cache->add_to_queue(&cache_ifms);
            if (final_layer) {
                const uint32_t row = prompt_length - 1 - begin - slot * group;
                std::map<uint8_t, MLABufferSlice> post_ifms{
                    {0, {residual_input, {slot * group + row, 0}, {1, hidden}}},
                    {static_cast<uint8_t>(1 + quantized_input),
                        {&get_buffer(fmt::format("n{}_buffer3", group)), {row, 0}, {1, _cfg.lm_cfg.attn_cfg.get_q_size(type)}}},
                };
                if (quantized_input) post_ifms.emplace(1, MLABufferSlice{scales, {slot * group + row, 0}, {1, 1}});
                models.post->add_to_queue(&post_ifms);
            } else {
                std::map<uint8_t, MLABufferSlice> residual{{0, {residual_input,
                    {static_cast<uint32_t>(slot) * group, 0}, {group, hidden}}}};
                std::map<uint8_t, MLABufferSlice> output{{0, {_image_hidden.get(),
                    {static_cast<uint32_t>(slot) * group, 0}, {group, hidden}}}};
                models.post->add_to_queue(&residual, &output);
            }
        }
        MLAModelWithBuffer::run_queue();
        if (!_is_running.load(std::memory_order_relaxed)) return 0;
    }
    if (end < prompt_length) return 0;
    auto& result = get_buffer("n1_buffer4");
    if (_need_argmax) {
        result.invalidate_cache(0, static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) * sizeof(Eigen::bfloat16));
        return _calc_next_token_id(&result);
    }
    result.invalidate_cache(0, sizeof(uint32_t));
    return *static_cast<uint32_t*>(result.get_virtual_addr());
}

}  // namespace simaai::llima
