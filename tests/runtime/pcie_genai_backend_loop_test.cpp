// Unit test for BackendLoop: the receive loop that turns genai.prompt and
// genai.cancel into generator runs, with a fake svc client and a fake
// generator that plays LLiMa's callback sequence. No daemon, no MLA.
#include <atomic>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include "backend_loop.hpp"
#include "genai_protocol.hpp"
#include "pcie_genai_fake_svc_client.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
using test::FakeSvcClient;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

class FakeGenerator final : public Generator {
    public:
        bool block_until_stop = false;
        bool throw_error = false;
        bool throw_generation_error = false;
        bool clear_history_on_run = false;
        bool throw_on_reset = false;
        std::string history_text = "[]";
        std::atomic<bool> started{false};
        std::atomic<int> stop_calls{0};
        std::atomic<int> reset_calls{0};
        std::string last_prompt;
        std::optional<std::string> last_reset_system;
        bool last_reset_thinking = false;

        RunResult run(const PromptRequest& request, EventBridge& bridge) override {
            last_prompt = request.prompt;
            started = true;
            if (throw_error) throw std::runtime_error("mla load failed");
            if (throw_generation_error) throw GenerationError("image pull failed", true);
            bridge.on_info("ttft", 0.5);
            bridge.on_text("Hel", false, false);
            if (block_until_stop) {
                while (stop_calls.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                bridge.on_info("END", 0.0);
                return RunResult{false, true};   // like run_model after stop_model(): no value
            }
            bridge.on_info("tps", 2.0);
            bridge.on_text("lo", false, false);
            bridge.on_info("END", 0.0);
            return RunResult{true, clear_history_on_run};
        }
        void stop() override { ++stop_calls; }
        ChatReply reset(const std::optional<std::string>& system_prompt, bool enable_thinking) override {
            if (throw_on_reset) throw std::runtime_error("reset failed");
            ++reset_calls;
            last_reset_system = system_prompt;
            last_reset_thinking = enable_thinking;
            return ChatReply{true, ""};
        }
        std::string history() override { return history_text; }
};

// Runs a BackendLoop on a thread for the length of a test.
struct Harness {
    FakeSvcClient in, out;
    FakeGenerator gen;
    std::atomic<bool> stop{false};
    BackendLoop loop{in, out, gen};
    std::thread thread;
    LoopExit exit = LoopExit::Stopped;

    void start() { thread = std::thread([this] { exit = loop.run(stop, 10); }); }
    void finish() { stop = true; if (thread.joinable()) thread.join(); }
    ~Harness() { finish(); }
};

std::string prompt(const std::string& id) {
    return R"({"id":")" + id + R"(","prompt":"Hi","enable_thinking":false})";
}

void test_subscribes_to_prompt_and_cancel() {
    Harness h;
    h.loop.subscribe();
    const auto subs = h.in.subscribed();
    expect(subs.size() == 3 && subs[0] == kTagPrompt && subs[1] == kTagCancel && subs[2] == kTagChat,
           "subscribe() listens on genai.prompt, genai.cancel and genai.chat");
}

void test_prompt_runs_to_final() {
    Harness h;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagFinal), "a prompt ends with genai.final");
    const auto sent = h.out.sent();
    expect(sent.size() == 4, "metric + 2 tokens + final");
    expect(sent.back().payload == encode_final("h1-1", "stop", 2, 0.5, 2.0), "final payload");
    expect(h.gen.last_prompt == "Hi", "the prompt text reached the generator");
}

void test_cancel_mid_run() {
    Harness h;
    h.gen.block_until_stop = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagToken), "the run started");
    h.in.push(kTagCancel, R"({"id":"h1-1"})");
    expect(h.out.wait_sent(kTagFinal), "a cancelled run still ends with genai.final");
    expect(h.out.sent().back().payload == encode_final("h1-1", "cancelled", 1, 0.5, 0.0, true),
           "finish_reason cancelled, history cleared");
    expect(h.gen.stop_calls.load() >= 1, "cancel stopped the generator");
}

void test_busy_prompt_is_refused() {
    Harness h;
    h.gen.block_until_stop = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagToken), "first run started");
    h.in.push(kTagPrompt, prompt("h1-2"));
    expect(h.out.wait_sent(kTagError), "second prompt is refused");
    bool busy_for_second = false;
    for (const auto& n : h.out.sent()) {
        if (n.tag == kTagError && n.payload.find(R"("id":"h1-2")") != std::string::npos &&
            n.payload.find("busy") != std::string::npos) busy_for_second = true;
    }
    expect(busy_for_second, "the busy error names the second request");
    h.in.push(kTagCancel, "");
    expect(h.out.wait_sent(kTagFinal), "the first run still ends normally");
}

void test_bad_prompt_gets_an_error() {
    Harness h;
    h.start();
    h.in.push(kTagPrompt, R"({"id":"h1-1","prompt":""})");
    expect(h.out.wait_sent(kTagError), "an empty prompt gets genai.error");
    expect(h.out.sent()[0].payload.find(R"("id":"h1-1")") != std::string::npos,
           "the error carries the prompt id when it can be read");
    expect(!h.gen.started.load(), "the generator never ran");
}

void test_generator_exception_is_an_error() {
    Harness h;
    h.gen.throw_error = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagError), "a failing run sends genai.error");
    expect(h.out.sent().back().payload == encode_error("h1-1", "mla load failed"), "error payload");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    for (const auto& n : h.out.sent()) expect(n.tag != kTagFinal, "no final after an error");
}

void test_cancel_while_idle_is_ignored() {
    Harness h;
    h.start();
    h.in.push(kTagCancel, "");
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    expect(h.out.sent().empty(), "an idle cancel sends nothing");
    expect(h.gen.stop_calls.load() == 0, "an idle cancel does not touch the generator");
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagFinal), "the next prompt still works");
    expect(h.out.sent().back().payload == encode_final("h1-1", "stop", 2, 0.5, 2.0),
           "and is not treated as cancelled");
}

void test_daemon_loss_ends_the_loop() {
    Harness h;
    h.start();
    h.in.disconnect();
    h.thread.join();
    expect(h.exit == LoopExit::DaemonLost, "a lost daemon ends the loop with DaemonLost");
}

void test_stop_during_a_run_ends_cleanly() {
    Harness h;
    h.gen.block_until_stop = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagToken), "run started");
    h.finish();   // like SIGTERM: stop flag set
    expect(h.gen.stop_calls.load() >= 1, "stopping the loop stops the generator");
    expect(h.out.sent().back().tag == kTagFinal, "the run still reports a final");
}

void test_back_to_back_prompts() {
    Harness h;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagFinal, 1), "the first prompt ends with genai.final");
    h.in.push(kTagPrompt, prompt("h1-2"));   // right after the final, like the host does
    expect(h.out.wait_sent(kTagFinal, 2), "the second prompt also ends with genai.final");
    for (const auto& n : h.out.sent()) expect(n.tag != kTagError, "no error (no busy refusal)");
    expect(h.out.sent().back().payload == encode_final("h1-2", "stop", 2, 0.5, 2.0),
           "the last final is for the second prompt");
}
void test_chat_reset_replies() {
    Harness h;
    h.start();
    h.in.push(kTagChat, R"({"id":"h1-1","op":"reset","system_prompt":"Be brief.","enable_thinking":true})");
    expect(h.out.wait_sent(kTagReply), "a chat command gets genai.reply");
    expect(h.out.sent().back().payload == encode_reply("h1-1", true, ""), "reset reply payload");
    expect(h.gen.reset_calls.load() == 1 && h.gen.last_reset_system == std::string("Be brief.") &&
           h.gen.last_reset_thinking, "reset reached the generator with its settings");
}

void test_chat_print_replies_history() {
    Harness h;
    h.gen.history_text = R"([{"role":"user","content":"Hi"}])";
    h.start();
    h.in.push(kTagChat, R"({"id":"h1-2","op":"print"})");
    expect(h.out.wait_sent(kTagReply), "print gets genai.reply");
    expect(h.out.sent().back().payload ==
           encode_reply("h1-2", true, R"([{"role":"user","content":"Hi"}])"), "print reply is the history");
}

void test_chat_refused_while_busy() {
    Harness h;
    h.gen.block_until_stop = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagToken), "run started");
    h.in.push(kTagChat, R"({"id":"h1-2","op":"reset","enable_thinking":false})");
    expect(h.out.wait_sent(kTagReply), "a chat command during a run is answered");
    expect(h.out.sent().back().payload ==
           encode_reply("h1-2", false, "busy: a generation is already running"), "busy reply");
    expect(h.gen.reset_calls.load() == 0, "the Chat is not touched during a run");
    h.in.push(kTagCancel, "");
    expect(h.out.wait_sent(kTagFinal), "the run still ends");
}

void test_bad_chat_gets_a_not_ok_reply() {
    Harness h;
    h.start();
    h.in.push(kTagChat, R"({"id":"h1-3","op":"nope"})");
    expect(h.out.wait_sent(kTagReply), "a bad chat command is answered");
    const std::string p = h.out.sent().back().payload;
    expect(p.find(R"("id":"h1-3")") != std::string::npos && p.find(R"("ok":false)") != std::string::npos,
           "not ok, with the command's id");
}

void test_reset_exception_is_a_not_ok_reply() {
    Harness h;
    h.gen.throw_on_reset = true;
    h.start();
    h.in.push(kTagChat, R"({"id":"h1-4","op":"reset","enable_thinking":false})");
    expect(h.out.wait_sent(kTagReply), "a failing reset is answered");
    expect(h.out.sent().back().payload == encode_reply("h1-4", false, "reset failed"), "reply text");
}

void test_history_cleared_reaches_the_final() {
    Harness h;
    h.gen.clear_history_on_run = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagFinal), "final sent");
    expect(h.out.sent().back().payload == encode_final("h1-1", "stop", 2, 0.5, 2.0, true),
           "history_cleared in the final");
}

void test_generation_error_carries_history_cleared() {
    Harness h;
    h.gen.throw_generation_error = true;
    h.start();
    h.in.push(kTagPrompt, prompt("h1-1"));
    expect(h.out.wait_sent(kTagError), "error sent");
    expect(h.out.sent().back().payload == encode_error("h1-1", "image pull failed", true),
           "history_cleared in the error");
}
}  // namespace

int main() {
    test_subscribes_to_prompt_and_cancel();
    test_prompt_runs_to_final();
    test_cancel_mid_run();
    test_busy_prompt_is_refused();
    test_bad_prompt_gets_an_error();
    test_generator_exception_is_an_error();
    test_cancel_while_idle_is_ignored();
    test_daemon_loss_ends_the_loop();
    test_stop_during_a_run_ends_cleanly();
    test_back_to_back_prompts();
    test_chat_reset_replies();
    test_chat_print_replies_history();
    test_chat_refused_while_busy();
    test_bad_chat_gets_a_not_ok_reply();
    test_reset_exception_is_a_not_ok_reply();
    test_history_cleared_reaches_the_final();
    test_generation_error_carries_history_cleared();
    if (failures == 0) std::cout << "pcie_genai_backend_loop_test passed\n";
    return failures == 0 ? 0 : 1;
}
