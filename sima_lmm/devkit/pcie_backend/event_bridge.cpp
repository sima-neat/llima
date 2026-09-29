// EventBridge: sends tokens as they come, sends TTFT once, and sends one
// genai.final (with average TPS) at the end of each run. It also makes a
// cancel stick, even if it arrives before LLiMa has really started.
#include "event_bridge.hpp"

#include <exception>
#include <utility>

#include <spdlog/spdlog.h>

#include "genai_protocol.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {

EventBridge::EventBridge(SvcClient& out, std::function<void()> stop_fn)
  : _out(out), _stop_fn(std::move(stop_fn)) {}

void EventBridge::begin(const std::string& id) {
    std::lock_guard<std::mutex> lock(_mutex);
    _id = id;
    _ttft_s = 0.0;
    _tokens = 0;
    _tps_count = 0;
    _tps_seconds = 0.0;
    _saw_full = false;
    _token_seq = 0;
    _cancel.store(false);
}

void EventBridge::on_text(const std::string& text, bool /*stream_end*/, bool /*from_draft*/) {
    // Send any non-empty text, also when stream_end is true:
    // TextStreamer::end() flushes the last cached text with stream_end=true,
    // so that call can carry the final words of the answer.
    // We do not use stream_end to end the run: finish() does that, after
    // run_model has returned and the streamer has drained.
    // After a cancel, text is dropped: the host no longer wants it.
    if (text.empty() || _cancel.load()) return;
    std::lock_guard<std::mutex> lock(_mutex);
    _send_locked(kTagToken, encode_token(_token_seq, text));
    ++_token_seq;   // only advances for a token we actually sent
}

void EventBridge::on_info(const std::string& metric, double value) {
    if (_cancel.load()) {
        // Why repeat the stop: stop_model() only sets _is_running = false, and
        // run_model sets _is_running = true when it starts. So a cancel that
        // comes just before the run starts is lost. Each later event is a new
        // chance to stop.
        // Repeat the stop on every event after a cancel, EXCEPT "ttft": the
        // post-prefill "Do nothing" branch in LanguageModel::run_model sends no
        // end signal, so a stop during "ttft" would hang wait_streaming. The
        // decode loop re-checks _is_running after each token, so the stop is
        // repeated from the first "tps".
        if (metric != "ttft") {
            _stop_fn();
        }
        return;
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (metric == "ttft") {
        _ttft_s = value;
        ++_tokens;
        _send_locked(kTagMetrics, encode_metric("ttft", value));
    } else if (metric == "tps") {
        // LLiMa reports one "tps" per token, as 1 / (that token's duration).
        // Sending each one would be one notification per token for no gain.
        // So we sum the token times here and send one average in genai.final:
        // tps = tokens / total decode seconds.
        ++_tokens;
        if (value > 0.0) {
            ++_tps_count;
            _tps_seconds += 1.0 / value;
        }
    } else if (metric == "FULL") {
        _saw_full = true;
    }
    // "END" needs nothing here: finish() reports the end of the run.
}

void EventBridge::request_cancel() {
    _cancel.store(true);
    _stop_fn();
}

void EventBridge::finish(bool interrupted, bool history_cleared) {
    std::lock_guard<std::mutex> lock(_mutex);
    // A cancel wins: even if LLiMa ended by itself, the host asked to stop,
    // so it gets "cancelled". "length" = LLiMa said FULL; else "stop" (END).
    const std::string reason =
        (interrupted || _cancel.load()) ? "cancelled" : (_saw_full ? "length" : "stop");
    const double tps = _tps_seconds > 0.0 ? _tps_count / _tps_seconds : 0.0;
    _send_locked(kTagFinal, encode_final(_id, reason, _tokens, _ttft_s, tps, history_cleared));
}

void EventBridge::fail(const std::string& message, bool history_cleared) {
    std::lock_guard<std::mutex> lock(_mutex);
    _send_locked(kTagError, encode_error(_id, message, history_cleared));
}

void EventBridge::send_reply(const std::string& id, bool ok, const std::string& text) {
    std::lock_guard<std::mutex> lock(_mutex);
    _send_locked(kTagReply, encode_reply(id, ok, text));
}

void EventBridge::send_error(const std::string& id, const std::string& message) {
    std::lock_guard<std::mutex> lock(_mutex);
    _send_locked(kTagError, encode_error(id, message));
}

void EventBridge::_send_locked(const char* tag, const std::string& payload) {
    try {
        _out.notify(tag, payload);
    } catch (const std::exception& e) {
        // Never throw into LLiMa's streamer thread: an escaped exception there
        // would terminate the process. Report and go on. A lost event is
        // better than a dead backend; the next prompt still works.
        spdlog::warn("pcie-genai-backend: failed to send {}: {}", tag, e.what());
    }
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
