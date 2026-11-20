#include "language_model_state.hpp"
#include <algorithm>
#include <cstring>
#include <utility>
#include <fmt/format.h>

namespace simaai::llima {

void LanguageSession::init_layout(const LanguageSession& prototype) {
    checkpoint_boundaries = prototype.checkpoint_boundaries;
    cached_states.reserve(prototype.cached_states.size());
    for (const auto& state : prototype.cached_states)
        cached_states.emplace_back(state.layout);
}

void LanguageSession::allocate(const LanguageResources& resources) {
    if (buffers_allocated) return;
    // Descriptors can already be bound to compiled stages. Keep their addresses
    // stable even when rolling back a partial allocation.
    for (const auto& [name, spec] : resources.session_layout)
        buffers.try_emplace(name, name, spec.shape, spec.dtype, spec.align_last_dim);
    try {
        for (auto& [name, buffer] : buffers) {
            buffer.allocate();
            buffer.clear();
        }
    } catch (...) {
        for (auto& [name, buffer] : buffers) buffer.free();
        throw;
    }
    buffers_allocated = true;
    publish_memory_bytes();
}

void LanguageSession::reset() {
    if (buffers_allocated) {
        for (auto& [name, buffer] : buffers) buffer.clear();
    }
    invalidate_reuse();
    image_identity.clear();
    if (draft)
        draft->reset();
}

void LanguageSession::invalidate_reuse() {
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

SessionManager::SessionManager() {
    _sessions.emplace(0, std::make_shared<LanguageSession>());
}

std::shared_ptr<LanguageSession> SessionManager::get(SessionId id) const {
    std::lock_guard lock(_mutex);
    const auto found = _sessions.find(id);
    if (found == _sessions.end())
        throw std::invalid_argument("Unknown language session");
    return found->second;
}

SessionId SessionManager::create(const LanguageSession& prototype) {
    std::lock_guard lock(_mutex);
    if (_sessions.size() >= _limit)
        throw std::runtime_error("Language session limit reached");
    auto session = std::make_shared<LanguageSession>();
    session->init_layout(prototype);
    const auto id = _next;
    _sessions.emplace(id, std::move(session));
    ++_next;
    return id;
}

void SessionManager::release(SessionId id) {
    if (id == 0)
        throw std::invalid_argument("The default session cannot be released");
    if (id == active)
        throw std::logic_error("Select another session before releasing the active session");
    std::lock_guard lock(_mutex);
    if (!_sessions.erase(id)) throw std::invalid_argument("Unknown language session");
}

std::vector<SessionId> SessionManager::ids() const {
    std::lock_guard lock(_mutex);
    std::vector<SessionId> result;
    for (const auto& [id, session] : _sessions)
        result.push_back(id);
    return result;
}

void SessionManager::set_limit(size_t limit) {
    std::lock_guard lock(_mutex);
    if (limit < _sessions.size())
        throw std::invalid_argument("Session limit is smaller than the live session count");
    _limit = limit;
}

SessionId LanguageModel::create_session() {
    if (_run->active)
        throw std::logic_error("Cannot create a session during generation");
    return _sessions->create(*_sessions->get(0));
}

void LanguageModel::select_session(SessionId id) {
    // Standalone prefill/decode calls can follow immediately without a request guard.
    _select_request_sessions(id, nullptr, true);
}

void LanguageModel::_select_request_sessions(SessionId id, LanguageModel* draft, bool bind_adapter) {
    if (_run->active || (draft && draft->_run->active))
        throw std::logic_error("Cannot select a session during generation");

    const auto& target_session = _sessions->get(id);
    auto draft_session = target_session->draft;
    if (draft && !draft_session) {
        draft_session = std::make_shared<LanguageSession>();
        draft_session->init_layout(*draft->_sessions->get(0));
    }

    // Complete old submissions and allocate both sides before changing either workspace.
    if (_session != target_session || !target_session->buffers_allocated ||
        (draft && (draft->_session != draft_session || !draft_session->buffers_allocated))) {
        MLAModelWithBuffer::run_queue();
        target_session->allocate(*_resources);
        if (draft)
            draft_session->allocate(*draft->_resources);
    }
    if (bind_adapter)
        _bind_adapter(target_session->adapter);
    _bind_session(target_session);
    if (draft) {
        draft->_bind_session(draft_session);
        if (target_session->draft != draft_session) {
            target_session->draft = std::move(draft_session);
            target_session->publish_memory_bytes();
        }
    }
    _sessions->active = id;
}

void LanguageModel::_bind_session(const std::shared_ptr<LanguageSession>& selected) {
    if (_session == selected)
        return;
    for (auto& [name, slots] : _workspace->session_bindings) {
        auto* buffer = &selected->buffers.at(name);
        for (auto* slot : slots)
            *slot = buffer;
    }
    _session = selected;
    _workspace->clear_session_scratch();
}

void LanguageModel::_detach_draft_sessions(LanguageModel* draft) {
    _workspace->paired_draft_identity.reset();
    for (auto id : _sessions->ids()) {
        auto& session = *_sessions->get(id);
        if (draft && session.draft && draft->_session == session.draft)
            draft->_bind_session(draft->_sessions->get(draft->active_session()));
        session.draft.reset();
        session.publish_memory_bytes();
        session.invalidate_reuse();
    }
}

void LanguageModel::_release_paired_session(SessionId id, LanguageModel* draft) {
    const auto& session = _sessions->get(id);
    if (id != 0 && draft && session->draft && draft->_session == session->draft)
        draft->_bind_session(draft->_sessions->get(draft->active_session()));
    release_session(id);
}

void LanguageModel::reset_session(SessionId id) { _reset_session(id, nullptr); }

void LanguageModel::_reset_session(SessionId id, LanguageModel* draft) {
    if (_run->active)
        throw std::logic_error("Cannot reset a session during generation");
    MLAModelWithBuffer::run_queue();
    const auto& session = _sessions->get(id);
    session->reset();
    if (_session == session) _workspace->clear_session_scratch();
    if (draft && session->draft && draft->_session == session->draft)
        draft->_workspace->clear_session_scratch();
}

void LanguageModel::release_session(SessionId id) {
    if (_run->active)
        throw std::logic_error("Cannot release a session during generation");
    if (_sessions->active == id && id != 0)
        select_session(0);
    bind_session_adapter();
    _sessions->release(id);
}

void LanguageModel::set_session_limit(size_t limit) {
    _sessions->set_limit(limit);
}
SessionId LanguageModel::active_session() const {
    return _sessions->active;
}
std::vector<SessionId> LanguageModel::session_ids() const {
    return _sessions->ids();
}
size_t LanguageModel::session_limit() const {
    return _sessions->limit();
}

void LanguageSession::publish_memory_bytes() {
    size_t bytes = 0;
    if (buffers_allocated) {
        for (const auto& [name, buffer] : buffers) bytes += buffer.get_allocation_size();
    }
    for (const auto& state : cached_states) {
        for (const auto& layer : state.checkpoints) {
            for (const auto& snapshot : layer)
                bytes += snapshot.size();
        }
    }
    published_bytes.store(bytes + (draft ? draft->published_bytes.load() : 0));
}

size_t LanguageModel::session_memory_bytes(SessionId id) const {
    return _sessions->get(id)->published_bytes.load();
}

void LanguageModel::_init_session_layout() {
    if (!_use_group_token_models)
        return;
    std::vector<uint8_t> conv_layers;
    std::vector<uint8_t> linear_layers;
    for (uint8_t layer = 0; layer < _cfg.lm_cfg.num_hidden_layers; ++layer) {
        if (_cfg.lm_cfg.layer_types[layer] == "conv")
            conv_layers.push_back(layer);
        else if (_cfg.lm_cfg.layer_types[layer] == "linear_attention")
            linear_layers.push_back(layer);
    }
    if (conv_layers.empty() && linear_layers.empty())
        return;

    // Each grouped boundary records the state tail before that block.
    const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
    for (const auto offset : offsets) {
        if (offset)
            _session->checkpoint_boundaries.push_back(offset);
    }
    if (!offsets.empty()) {
        const uint32_t after_last = offsets.back() + _cfg.pipeline_cfg.input_token_group_size;
        if (after_last < _cfg.pipeline_cfg.max_num_tokens)
            _session->checkpoint_boundaries.push_back(after_last);
    }

    const auto add_state = [&](std::string prefix, uint16_t tail_len, uint32_t num_elems,
                               std::vector<uint8_t> layers, bool single_output = false) {
        const size_t element_size = sizeof(Eigen::bfloat16);
        _session->cached_states.emplace_back(CheckpointLayout{
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
    if (_session->cached_states.empty())
        return;
    _workspace->capture_state_checkpoints = true;
    _workspace->rolling_checkpoint_slot = 0;
    _workspace->writable_checkpoint_slots = 0;
    const auto system_boundary = std::upper_bound(
        _session->checkpoint_boundaries.begin(), _session->checkpoint_boundaries.end(),
        std::min<size_t>(stable_prefix_token_count, input_token_count)
    );
    _workspace->system_checkpoint_position = system_boundary == _session->checkpoint_boundaries.begin()
        ? 0 : *std::prev(system_boundary);
    if (_workspace->system_checkpoint_position && _session->state_checkpoint_positions[0] != _workspace->system_checkpoint_position)
        _session->state_checkpoint_positions[0] = 0;
}


// Restore the latest checkpoint, or clear state when checkpoints are unavailable.
uint16_t LanguageModel::_prepare_state_checkpoints_for_prefill(uint16_t num_cached_tokens) {
    if (_session->cached_states.empty()) {
        bool cleared_state = false;
        for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
                get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)).clear();
                cleared_state = true;
            } else if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
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
    for (size_t i = 0; i < _session->state_checkpoint_positions.size(); ++i) {
        if (_session->state_checkpoint_positions[i] > num_cached_tokens)
            _session->state_checkpoint_positions[i] = 0;
        if (_session->state_checkpoint_positions[i] > _session->state_checkpoint_positions[checkpoint_slot])
            checkpoint_slot = i;
    }
    if (!_session->state_checkpoint_positions[checkpoint_slot]) {
        for (const auto& state: _session->cached_states) {
            for (const auto& layer_idx: state.layout.layer_indices) {
                get_buffer(fmt::format("{}{}", state.layout.buffer_name_prefix, layer_idx)).clear();
            }
        }
        return 0;
    }
    const auto restored_token_count = _session->state_checkpoint_positions[checkpoint_slot];
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    for (auto& state: _session->cached_states) {
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
            for (size_t i = first_slot; i < _session->state_checkpoint_positions.size(); ++i) {
                if (!_session->state_checkpoint_positions[i])
                    _workspace->writable_checkpoint_slots |= 1u << i;
                if (_session->state_checkpoint_positions[i] < _session->state_checkpoint_positions[_workspace->rolling_checkpoint_slot])
                    _workspace->rolling_checkpoint_slot = i;
            }
            _workspace->writable_checkpoint_slots |= 1u << _workspace->rolling_checkpoint_slot;
        }
        // Prefill keeps the latest boundaries in spare slots; decode advances only the newest one.
        if (is_prefill) {
            for (size_t i = 0; i < _session->state_checkpoint_positions.size(); ++i) {
                if ((_workspace->writable_checkpoint_slots & (1u << i))
                    && _session->state_checkpoint_positions[i] < _session->state_checkpoint_positions[_workspace->rolling_checkpoint_slot])
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

    for (auto& state: _session->cached_states) {
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
    _session->state_checkpoint_positions[*checkpoint_slot] = token_count;
}


void LanguageModel::_move_state_tail_for_decode(uint16_t valid_tokens) {
    // After a partial last group, move the valid tail to the end of the conv buffer,
    //  where the decode MLAModel expects it.
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    for (auto& state: _session->cached_states) {
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
