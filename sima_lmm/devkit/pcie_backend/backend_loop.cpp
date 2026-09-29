// BackendLoop: receives genai.prompt / genai.cancel / genai.chat and runs each prompt on a
// worker thread. The receive thread stays free, so a cancel can arrive while
// LLiMa is still generating. Only one prompt runs at a time.
#include "backend_loop.hpp"

#include <exception>
#include <utility>

#include <spdlog/spdlog.h>

namespace simaai {
namespace llima {
namespace pcie_backend {

BackendLoop::BackendLoop(SvcClient& in, SvcClient& out, Generator& generator)
  : _in(in), _generator(generator), _bridge(out, [&generator] { generator.stop(); }) {}

BackendLoop::~BackendLoop() {
    if (_busy.load()) _bridge.request_cancel();
    _reap_worker(true);
}

void BackendLoop::subscribe() {
    _in.subscribe(kTagPrompt);
    _in.subscribe(kTagCancel);
    _in.subscribe(kTagChat);
}

LoopExit BackendLoop::run(const std::atomic<bool>& stop_requested, int recv_timeout_ms) {
    LoopExit exit = LoopExit::Stopped;
    while (!stop_requested.load()) {
        SvcNote note;
        const RecvStatus status = _in.recv(note, recv_timeout_ms);
        if (status == RecvStatus::Disconnected) {
            spdlog::error("pcie-genai-backend: lost the pep daemon");
            exit = LoopExit::DaemonLost;
            break;
        }
        // Join a worker whose run is over (_busy is false), so its thread is
        // freed. It does not wait for a generation that is still running.
        _reap_worker(false);
        if (status == RecvStatus::Timeout) continue;
        if (note.tag == kTagPrompt) {
            _handle_prompt(note.payload);
        } else if (note.tag == kTagCancel) {
            _handle_cancel();
        } else if (note.tag == kTagChat) {
            _handle_chat(note.payload);
        }
    }
    // Leaving the loop: stop a running generation and wait for it, so the
    // worker never outlives the model and the svc handles.
    if (_busy.load()) _bridge.request_cancel();
    _reap_worker(true);
    return exit;
}

void BackendLoop::_handle_prompt(const std::string& payload) {
    // One prompt at a time: LLiMa runs one generation per model, and the
    // tokens carry no id, so two runs would mix their text. Refuse the new
    // prompt with its own id; the running one goes on untouched.
    if (_busy.load()) {
        _bridge.send_error(try_read_id(payload), "busy: a generation is already running");
        return;
    }
    _reap_worker(true);
    PromptRequest request;
    try {
        request = parse_prompt(payload);
    } catch (const std::exception& e) {
        _bridge.send_error(try_read_id(payload), e.what());
        return;
    }
    // begin() on this thread, before the worker exists. begin() clears the
    // cancel flag. If the worker called it, a cancel that arrives right after
    // the prompt could be handled first and then wiped by the late begin().
    // The receive thread handles the prompt fully before it reads the cancel,
    // so here the order is always right.
    _bridge.begin(request.id);
    _busy.store(true);
    // _busy is cleared BEFORE finish()/fail() sends the final/error. The host
    // sends its next prompt as soon as it sees the final. If _busy were still
    // true then, that prompt would get a wrong "busy" error.
    // This is safe: _handle_prompt joins this worker (_reap_worker(true))
    // before it calls begin(), so the new run never starts while this thread
    // is still sending the old final.
    _worker = std::thread([this, request = std::move(request)] {
        try {
            const RunResult result = _generator.run(request, _bridge);
            _busy.store(false);
            _bridge.finish(!result.completed, result.history_cleared);
        } catch (const GenerationError& e) {
            spdlog::error("pcie-genai-backend: generation failed: {}", e.what());
            _busy.store(false);
            _bridge.fail(e.what(), e.history_cleared);
        } catch (const std::exception& e) {
            spdlog::error("pcie-genai-backend: generation failed: {}", e.what());
            _busy.store(false);
            _bridge.fail(e.what());
        } catch (...) {
            _busy.store(false);
            _bridge.fail("unknown error during generation");
        }
    });
}

void BackendLoop::_handle_cancel() {
    if (_busy.load()) _bridge.request_cancel();
}

// A chat command runs on this receive thread. It is refused while a
// generation runs, so the Chat is never touched by two threads at once.
void BackendLoop::_handle_chat(const std::string& payload) {
    if (_busy.load()) {
        _bridge.send_reply(try_read_id(payload), false, "busy: a generation is already running");
        return;
    }
    _reap_worker(true);
    ChatRequest request;
    try {
        request = parse_chat(payload);
    } catch (const std::exception& e) {
        _bridge.send_reply(try_read_id(payload), false, e.what());
        return;
    }
    ChatReply reply;
    try {
        if (request.op == ChatOp::Reset) {
            reply = _generator.reset(request.system_prompt, request.enable_thinking);
        } else {
            reply = ChatReply{true, _generator.history()};
        }
    } catch (const std::exception& e) {
        reply = ChatReply{false, e.what()};
    }
    _bridge.send_reply(request.id, reply.ok, reply.text);
}

void BackendLoop::_reap_worker(bool wait) {
    if (_worker.joinable() && (wait || !_busy.load())) _worker.join();
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
