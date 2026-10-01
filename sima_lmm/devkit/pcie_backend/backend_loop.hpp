// BackendLoop: the prompt loop of pcie-genai-backend.
//
// Flow: host -> genai.prompt -> BackendLoop -> Generator (VlmGenerator/LLiMa)
// -> EventBridge -> genai.token / genai.metrics / genai.final back to the host.
// It runs one prompt at a time; a second prompt gets a "busy" error.
// Also genai.chat commands (reset / print) on the Generator's kept Chat.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_BACKEND_LOOP_
#define _SIMA_LLIMA_PCIE_BACKEND_BACKEND_LOOP_

#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "event_bridge.hpp"
#include "genai_protocol.hpp"
#include "svc_client.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {

// What one run did. history_cleared = the Generator cleared its kept Chat
// (cancel or empty answer); it goes to the host in genai.final.
struct RunResult {
    bool completed = true;          // false = interrupted by stop()
    bool history_cleared = false;
};

// The answer to a genai.chat command, sent back as genai.reply.
struct ChatReply {
    bool ok = true;
    std::string text;               // print: the history JSON; not ok: why
};

// Thrown by Generator::run when a run fails; says whether the Chat was cleared.
class GenerationError : public std::runtime_error {
    public:
        GenerationError(const std::string& message, bool history_cleared)
          : std::runtime_error(message), history_cleared(history_cleared) {}
        bool history_cleared;
};

// What the loop runs for each prompt. The real one wraps VisionLanguageModel
// (VlmGenerator); tests use a fake.
class Generator {
    public:
        virtual ~Generator() = default;
        // Run one prompt to the end, feeding LLiMa's callbacks into `bridge`
        // (on_text / on_info). May throw (GenerationError says if the Chat was cleared).
        virtual RunResult run(const PromptRequest& request, EventBridge& bridge) = 0;
        // Ask the running generation to stop. Any thread; harmless when idle.
        virtual void stop() = 0;
        // genai.chat "reset": set the system prompt / thinking and clear the Chat.
        // Only called when no run is going.
        virtual ChatReply reset(const std::optional<std::string>& system_prompt,
                                bool enable_thinking) = 0;
        // genai.chat "print": the Chat as JSON. Only called when no run is going.
        virtual std::string history() = 0;
};

enum class LoopExit { Stopped, DaemonLost };

// Receives genai.prompt / genai.cancel / genai.chat on `in` (this thread only) and runs one
// prompt at a time on a worker thread, so a cancel can arrive mid-generation.
// Events go out through an EventBridge on `out`.
class BackendLoop {
    public:
        BackendLoop(SvcClient& in, SvcClient& out, Generator& generator);
        ~BackendLoop();   // stops and joins a running worker

        BackendLoop(const BackendLoop&) = delete;
        BackendLoop& operator=(const BackendLoop&) = delete;

        // Call before reporting READY: a notification with no listener is dropped.
        void subscribe();
        // Serve until `stop_requested` or until the daemon goes away.
        LoopExit run(const std::atomic<bool>& stop_requested, int recv_timeout_ms = 500);

    private:
        void _handle_prompt(const std::string& payload);
        void _handle_cancel();
        void _handle_chat(const std::string& payload);
        void _reap_worker(bool wait);

        SvcClient& _in;
        Generator& _generator;
        EventBridge _bridge;
        std::thread _worker;
        std::atomic<bool> _busy{false};
};

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
