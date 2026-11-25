#include "language/request_scope.hpp"

namespace simaai::llima {

RequestScope::RequestScope(LanguageModel& target, LanguageModel* draft,
                           TextStreamer::RequestOptions stream, std::stop_token stop,
                           std::optional<std::set<uint32_t>> stop_ids)
    : _target(target), _draft(draft),
      _request{{stop_ids ? std::move(*stop_ids) : target._run->defaults.stop_token_ids,
                target._run->defaults.max_num_tokens}, std::move(stop),
               RequestState::Interrupt(target._run->interrupts), std::nullopt} {
    if (target._run->active || (draft && draft->_run->active))
        throw std::logic_error("Language model already has an active request");
    if (draft) {
        _request.draft_interrupt.emplace(draft->_run->interrupts);
        target.attach_draft(*draft);
    } else if (target._workspace->speculative_pair) {
        target.invalidate_prompt_reuse();
    }
    target._run->text_streamer.begin_request(std::move(stream));
    target._run->active = &_request;
    if (draft) {
        draft->_run->active = &_request;
    }
}

void RequestScope::finish() {
    _target._run->text_streamer.finish();
    _cancelled_at_finish = _request.cancelled();
    _completed = true;
}

RequestScope::~RequestScope() {
    if (!_completed) {
        MLAModelWithBuffer::discard_queue();
        // Preserve the original exception while releasing request callbacks.
        try { _target._run->text_streamer.finish(); } catch (...) {}
    }
    if (!_completed || _cancelled_at_finish) {
        _target.invalidate_prompt_reuse();
        if (_draft) _draft->invalidate_prompt_reuse();
    }
    _target._workspace->capture_state_checkpoints = false;
    _target._workspace->rolling_checkpoint_slot = 0;
    _target._workspace->writable_checkpoint_slots = 0;
    _target._run->active = nullptr;
    if (_draft) _draft->_run->active = nullptr;
}

} // namespace simaai::llima
