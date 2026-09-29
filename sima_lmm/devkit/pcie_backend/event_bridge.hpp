// EventBridge: turns LLiMa's streamer callbacks into genai.* notifications.
//
// Flow: host -> genai.prompt -> BackendLoop -> VlmGenerator/LLiMa ->
// EventBridge -> genai.token / genai.metrics / genai.final (or genai.error).
#ifndef _SIMA_LLIMA_PCIE_BACKEND_EVENT_BRIDGE_
#define _SIMA_LLIMA_PCIE_BACKEND_EVENT_BRIDGE_

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

#include "svc_client.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {

// Turns LLiMa's streamer callbacks into genai.* notifications for the host.
//
// on_text / on_info have the TextStreamer callback signatures and run on
// LLiMa's streamer thread; begin / finish / fail run on the generation worker;
// request_cancel and send_error may come from the receive thread. One mutex
// serializes all sends on the (send-only) svc handle.
class EventBridge {
    public:
        // `out` is the send-only svc client; `stop_fn` stops the running
        // generation (Generator::stop). Both must outlive the bridge.
        EventBridge(SvcClient& out, std::function<void()> stop_fn);

        void begin(const std::string& id);   // reset per-run state; clears a cancel
        void on_text(const std::string& text, bool stream_end, bool from_draft);
        void on_info(const std::string& metric, double value);
        void request_cancel();               // any thread
        // Sends genai.final; history_cleared = the Generator cleared its Chat.
        void finish(bool interrupted, bool history_cleared = false);
        // Sends genai.error for this run.
        void fail(const std::string& message, bool history_cleared = false);
        void send_error(const std::string& id, const std::string& message);
        void send_reply(const std::string& id, bool ok, const std::string& text);  // genai.reply
        // True if the host asked to cancel this run (set by request_cancel,
        // cleared by begin). Lets VlmGenerator clear the history when a cancel
        // raced the natural end of the answer.
        bool cancel_requested() const { return _cancel.load(); }

    private:
        void _send_locked(const char* tag, const std::string& payload);

        SvcClient& _out;
        std::function<void()> _stop_fn;
        std::mutex _mutex;
        std::atomic<bool> _cancel{false};
        std::string _id;
        double _ttft_s = 0.0;
        uint32_t _tokens = 0;
        uint32_t _tps_count = 0;
        double _tps_seconds = 0.0;
        bool _saw_full = false;
        std::uint64_t _token_seq = 0;   // per-run counter stamped on each token note
};

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
