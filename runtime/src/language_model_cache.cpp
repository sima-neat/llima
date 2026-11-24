#include "language_model_state.hpp"
#include <algorithm>
#include <cstring>
#include <utility>
#include <fmt/format.h>

namespace simaai::llima {

void SequenceCache::invalidate_reuse() {
    cached_token_ids.clear();
    cached_first_generated_token = 0;
    cached_draft_tokens.clear();
    cached_retrieve_indices.clear();
    cached_tree_mask = {};
    cached_tree_position_ids.clear();
    cached_eagle3_prompt_len = 0;
    cached_eagle3_stable_kv = 0;
    kv_cache_len = 0;
    state_checkpoint_positions.fill(0);
}

void LanguageModel::_init_cache_layout() {
    if (!_use_group_token_models)
        return;
    std::vector<uint8_t> conv_layers;
    std::vector<uint8_t> linear_layers;
    for (uint8_t layer = 0; layer < _cfg.lm_cfg.num_hidden_layers; ++layer) {
        if (_cfg.lm_cfg.layer_types[layer] == LayerType::Conv)
            conv_layers.push_back(layer);
        else if (_cfg.lm_cfg.layer_types[layer] == LayerType::LinearAttention)
            linear_layers.push_back(layer);
    }
    if (conv_layers.empty() && linear_layers.empty())
        return;

    // Each grouped boundary records the state tail before that block.
    const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
    for (const auto offset : offsets) {
        if (offset)
            _cache->checkpoint_boundaries.push_back(offset);
    }
    if (!offsets.empty()) {
        const uint32_t after_last = offsets.back() + _cfg.pipeline_cfg.input_token_group_size;
        if (after_last < _cfg.pipeline_cfg.max_num_tokens)
            _cache->checkpoint_boundaries.push_back(after_last);
    }

    const auto add_state = [&](std::string prefix, uint16_t tail_len, uint32_t num_elems,
                               std::vector<uint8_t> layers, bool single_output = false) {
        const size_t element_size = sizeof(Eigen::bfloat16);
        _cache->cached_states.emplace_back(CheckpointLayout{
            std::move(layers), std::move(prefix), tail_len, num_elems, element_size,
            static_cast<size_t>(tail_len) * num_elems * element_size, single_output});
    };
    if (!conv_layers.empty()) {
        add_state("conv_cache_history_l", std::max<uint16_t>(1, _cfg.lm_cfg.conv_L_cache - 1),
                  _cfg.lm_cfg.hidden_size, std::move(conv_layers));
    }
    if (!linear_layers.empty()) {
        const auto& cfg = _linear_attn_cfg();
        add_state("linear_conv_cache_history_l", static_cast<uint16_t>(cfg.conv_kernel_dim - 1),
                  cfg.get_conv_dim(), linear_layers);
        add_state("linear_delta_state_history_l", 1, cfg.get_recurrent_state_size(),
                  std::move(linear_layers), true);
    }
}

void LanguageModel::_init_state_checkpoint_capture(
    size_t input_token_count, uint16_t stable_prefix_token_count
) {
    if (_cache->cached_states.empty())
        return;
    _workspace->capture_state_checkpoints = true;
    _workspace->rolling_checkpoint_slot = 0;
    _workspace->writable_checkpoint_slots = 0;
    const auto system_boundary = std::upper_bound(
        _cache->checkpoint_boundaries.begin(), _cache->checkpoint_boundaries.end(),
        std::min<size_t>(stable_prefix_token_count, input_token_count)
    );
    _workspace->system_checkpoint_position = system_boundary == _cache->checkpoint_boundaries.begin()
        ? 0 : *std::prev(system_boundary);
    if (_workspace->system_checkpoint_position && _cache->state_checkpoint_positions[0] != _workspace->system_checkpoint_position)
        _cache->state_checkpoint_positions[0] = 0;
}


// Restore the latest checkpoint, or clear state when checkpoints are unavailable.
uint16_t LanguageModel::_prepare_state_checkpoints_for_prefill(uint16_t num_cached_tokens) {
    if (_cache->cached_states.empty()) {
        bool cleared_state = false;
        for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            if (_cfg.lm_cfg.layer_types[layer_idx] == LayerType::Conv) {
                get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)).clear();
                cleared_state = true;
            } else if (_cfg.lm_cfg.layer_types[layer_idx] == LayerType::LinearAttention) {
                get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx)).clear();
                get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)).clear();
                get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx)).clear();
                cleared_state = true;
            }
        }
        return cleared_state ? 0 : num_cached_tokens;
    }

    // Manual clamp semantics: if cache reuse asks past the last grouped block,
    // clamp to last_offset + group_size before selecting a checkpoint boundary.
    const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
    const uint16_t clamp_token_count =
        static_cast<uint16_t>(offsets.back() + _cfg.pipeline_cfg.input_token_group_size);
    if (num_cached_tokens > clamp_token_count) {
        num_cached_tokens = clamp_token_count;
    }

    // Invalidate checkpoints from the old conversation branch and select the newest match.
    size_t checkpoint_slot = 0;
    for (size_t i = 0; i < _cache->state_checkpoint_positions.size(); ++i) {
        if (_cache->state_checkpoint_positions[i] > num_cached_tokens)
            _cache->state_checkpoint_positions[i] = 0;
        if (_cache->state_checkpoint_positions[i] > _cache->state_checkpoint_positions[checkpoint_slot])
            checkpoint_slot = i;
    }
    if (!_cache->state_checkpoint_positions[checkpoint_slot]) {
        for (const auto& state: _cache->cached_states) {
            for (const auto& layer_idx: state.layout.layer_indices) {
                get_buffer(fmt::format("{}{}", state.layout.buffer_name_prefix, layer_idx)).clear();
            }
        }
        return 0;
    }
    const auto restored_token_count = _cache->state_checkpoint_positions[checkpoint_slot];
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    for (auto& state: _cache->cached_states) {
        const uint32_t restore_row_offset = state.layout.prefill_single_output ? 0 : tail_begin;
        const size_t dst_offset = static_cast<size_t>(
            restore_row_offset * state.layout.num_elems * state.layout.elem_size
        );
        for (size_t layer_slot = 0; layer_slot < state.layout.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layout.layer_indices[layer_slot];
            auto& checkpoints = state.checkpoints[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.layout.buffer_name_prefix, layer_idx));
            buf.upload_raw(
                checkpoints[checkpoint_slot].data(),
                dst_offset,
                state.layout.tail_bytes
            );
        }
    }
    return restored_token_count;
}


std::optional<size_t> LanguageModel::_select_state_checkpoint_slot(
    uint16_t token_count, bool is_prefill
) {
    if (token_count < _workspace->system_checkpoint_position)
        return std::nullopt;

    size_t checkpoint_slot = 0;
    if (token_count != _workspace->system_checkpoint_position) {
        if (!_workspace->writable_checkpoint_slots) {
            const size_t first_slot = _workspace->system_checkpoint_position ? 1 : 0;
            _workspace->rolling_checkpoint_slot = first_slot;
            for (size_t i = first_slot; i < _cache->state_checkpoint_positions.size(); ++i) {
                if (!_cache->state_checkpoint_positions[i])
                    _workspace->writable_checkpoint_slots |= 1u << i;
                if (_cache->state_checkpoint_positions[i] < _cache->state_checkpoint_positions[_workspace->rolling_checkpoint_slot])
                    _workspace->rolling_checkpoint_slot = i;
            }
            _workspace->writable_checkpoint_slots |= 1u << _workspace->rolling_checkpoint_slot;
        }
        // Prefill keeps the latest boundaries in spare slots; decode advances only the newest one.
        if (is_prefill) {
            for (size_t i = 0; i < _cache->state_checkpoint_positions.size(); ++i) {
                if ((_workspace->writable_checkpoint_slots & (1u << i))
                    && _cache->state_checkpoint_positions[i] < _cache->state_checkpoint_positions[_workspace->rolling_checkpoint_slot])
                    _workspace->rolling_checkpoint_slot = i;
            }
        }
        checkpoint_slot = _workspace->rolling_checkpoint_slot;
    }
    return checkpoint_slot;
}


void LanguageModel::_save_state_checkpoint(
    uint16_t token_count, uint16_t num_tokens, uint16_t valid_tokens, bool is_prefill
) {
    const auto checkpoint_slot = _select_state_checkpoint_slot(token_count, is_prefill);
    if (!checkpoint_slot.has_value())
        return;

    // Save the L-1 tail of each stateful layer's MLA buffer
    // so a future prefill can resume from this boundary without replay.
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;

    for (auto& state: _cache->cached_states) {
        const uint32_t tail_row_offset = state.layout.prefill_single_output
            ? 0
            : (
                num_tokens > 1
                    ? static_cast<uint32_t>(valid_tokens - 1)
                    : tail_begin
            );
        const size_t src_offset_bytes = tail_row_offset * state.layout.num_elems * state.layout.elem_size;
        for (size_t layer_slot = 0; layer_slot < state.layout.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layout.layer_indices[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.layout.buffer_name_prefix, layer_idx));
            buf.invalidate_cache(src_offset_bytes, state.layout.tail_bytes);
            auto* ptr = static_cast<const uint8_t*>(buf.get_virtual_addr());
            auto& snapshot = state.checkpoints[layer_slot][*checkpoint_slot];
            snapshot.assign(ptr + src_offset_bytes, ptr + src_offset_bytes + state.layout.tail_bytes);
        }
    }
    _cache->state_checkpoint_positions[*checkpoint_slot] = token_count;
}


void LanguageModel::_move_state_tail_for_decode(uint16_t valid_tokens) {
    // After a partial last group, move the valid tail to the end of the conv buffer,
    //  where the decode MLAModel expects it.
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    for (auto& state: _cache->cached_states) {
        if (state.layout.prefill_single_output)
            continue;
        const size_t src_offset_bytes = (valid_tokens - 1) * state.layout.num_elems * state.layout.elem_size;
        const size_t dst_offset_bytes = tail_begin * state.layout.num_elems * state.layout.elem_size;
        for (size_t layer_slot = 0; layer_slot < state.layout.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layout.layer_indices[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.layout.buffer_name_prefix, layer_idx));
            const size_t access_begin = std::min(src_offset_bytes, dst_offset_bytes);
            const size_t access_end = std::max(
                src_offset_bytes + state.layout.tail_bytes,
                dst_offset_bytes + state.layout.tail_bytes
            );
            buf.invalidate_cache(access_begin, access_end - access_begin);
            auto* ptr = static_cast<uint8_t*>(buf.get_virtual_addr());
            std::memmove(
                ptr + dst_offset_bytes,
                ptr + src_offset_bytes,
                state.layout.tail_bytes
            );
            buf.flush_cache(dst_offset_bytes, state.layout.tail_bytes);
        }
    }
}


} // namespace simaai::llima
