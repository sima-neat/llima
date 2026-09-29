// Unit test for EventBridge: LLiMa text/info callbacks become genai.* events.
// Drives the bridge exactly the way LLiMa's TextStreamer does (ttft, then one
// "tps" per token, then END or FULL), with a fake svc client.
#include <iostream>
#include <string>
#include <vector>

#include "event_bridge.hpp"
#include "genai_protocol.hpp"
#include "pcie_genai_fake_svc_client.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
using test::FakeSvcClient;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

// ttft + one tps event = 2 tokens; tps 2.0 means 0.5 s per token, so the
// average is exactly 2.0 (keeps the golden string exact).
void play_two_tokens(EventBridge& b) {
    b.on_info("ttft", 0.5);
    b.on_text("Hel", false, false);
    b.on_info("tps", 2.0);
    b.on_text("lo", false, false);
    b.on_text("", true, false);   // stream_end carries no text
}

void test_normal_run() {
    FakeSvcClient out;
    int stops = 0;
    EventBridge b(out, [&] { ++stops; });
    b.begin("h1-1");
    play_two_tokens(b);
    b.on_info("END", 0.0);
    b.finish(false);
    const auto sent = out.sent();
    expect(sent.size() == 4, "metric + 2 tokens + final");
    expect(sent[0].tag == kTagMetrics && sent[0].payload == R"({"type":"ttft","value":0.5})",
           "ttft metric is sent live");
    expect(sent[1].tag == kTagToken && sent[1].payload == encode_token(0, "Hel"),
           "first token, seq 0");
    expect(sent[2].tag == kTagToken && sent[2].payload == encode_token(1, "lo"),
           "second token, seq 1");
    expect(sent[3].tag == kTagFinal &&
               sent[3].payload == encode_final("h1-1", "stop", 2, 0.5, 2.0),
           "final: stop, 2 tokens, averaged tps");
    expect(stops == 0, "no stop on a normal run");
}

void test_cache_full_is_length() {
    FakeSvcClient out;
    EventBridge b(out, [] {});
    b.begin("h1-1");
    play_two_tokens(b);
    b.on_info("FULL", 0.0);
    b.finish(false);
    expect(out.sent().back().payload == encode_final("h1-1", "length", 2, 0.5, 2.0),
           "FULL maps to finish_reason length (as web.cpp does)");
}

void test_interrupted_is_cancelled() {
    FakeSvcClient out;
    EventBridge b(out, [] {});
    b.begin("h1-1");
    b.on_info("ttft", 0.5);
    b.on_text("Hel", false, false);
    b.finish(true);   // run_model returned no value
    expect(out.sent().back().payload == encode_final("h1-1", "cancelled", 1, 0.5, 0.0),
           "an interrupted run ends as cancelled");
}

void test_cancel_stops_and_silences() {
    FakeSvcClient out;
    int stops = 0;
    EventBridge b(out, [&] { ++stops; });
    b.begin("h1-1");
    b.on_info("ttft", 0.5);
    b.on_text("Hel", false, false);
    b.request_cancel();
    expect(stops == 1, "request_cancel calls stop");
    // Review focus 1: run_model() resets its running flag when it starts, so a
    // stop sent just before that is lost. Every later event repeats the stop.
    b.on_info("tps", 2.0);
    expect(stops == 2, "the stop is repeated on the next LLiMa event");
    b.on_text("late", false, false);
    b.finish(false);
    const auto sent = out.sent();
    for (const auto& n : sent) expect(n.payload != "late", "no tokens after a cancel");
    expect(sent.back().tag == kTagFinal &&
               sent.back().payload == encode_final("h1-1", "cancelled", 1, 0.5, 0.0),
           "a cancelled run ends as cancelled even if run_model completed");
}

void test_begin_resets_the_run() {
    FakeSvcClient out;
    int stops = 0;
    EventBridge b(out, [&] { ++stops; });
    b.begin("h1-1");
    play_two_tokens(b);
    b.request_cancel();
    b.finish(true);
    b.begin("h1-2");
    play_two_tokens(b);
    b.finish(false);
    expect(out.sent().back().payload == encode_final("h1-2", "stop", 2, 0.5, 2.0),
           "a new run starts from zero and is not cancelled");
}

void test_token_seq_resets_each_run() {
    FakeSvcClient out;
    EventBridge b(out, [] {});
    b.begin("h1-1");
    b.on_text("a", false, false);
    b.on_text("b", false, false);
    b.begin("h1-2");              // a new run
    b.on_text("c", false, false);
    const auto sent = out.sent();
    expect(sent.size() == 3, "three tokens sent");
    expect(sent[0].payload == encode_token(0, "a"), "run 1 token 0");
    expect(sent[1].payload == encode_token(1, "b"), "run 1 token 1");
    expect(sent[2].payload == encode_token(0, "c"), "run 2 starts back at seq 0");
}

void test_fail_and_send_error() {
    FakeSvcClient out;
    EventBridge b(out, [] {});
    b.begin("h1-1");
    b.fail("mla load failed");
    b.send_error("h1-9", "busy");
    const auto sent = out.sent();
    expect(sent.size() == 2, "two errors");
    expect(sent[0].tag == kTagError && sent[0].payload == encode_error("h1-1", "mla load failed"),
           "fail() reports the current run id");
    expect(sent[1].tag == kTagError && sent[1].payload == encode_error("h1-9", "busy"),
           "send_error() reports the given id");
}

void test_ttft_after_cancel_does_not_stop() {
    FakeSvcClient out;
    int stops = 0;
    EventBridge b(out, [&] { ++stops; });
    b.begin("h1-1");
    b.request_cancel();
    expect(stops == 1, "request_cancel calls stop");
    b.on_info("ttft", 0.5);
    expect(stops == 1, "ttft after cancel does not repeat the stop");
    expect(out.sent().size() == 0, "ttft after cancel sends nothing");
    b.on_info("tps", 2.0);
    expect(stops == 2, "tps after cancel repeats the stop");
    b.finish(true);
    expect(out.sent().back().payload == encode_final("h1-1", "cancelled", 0, 0.0, 0.0),
           "a run cancelled before any tokens ends with 0 tokens");
}
// VlmGenerator asks whether a cancel raced the end of the answer, and
// genai.chat replies go through the bridge.
void test_cancel_requested_and_reply() {
    FakeSvcClient out;
    EventBridge b(out, [] {});
    b.begin("h1-1");
    expect(!b.cancel_requested(), "no cancel after begin");
    b.request_cancel();
    expect(b.cancel_requested(), "cancel_requested after request_cancel");
    b.begin("h1-2");
    expect(!b.cancel_requested(), "begin clears it");
    b.send_reply("h1-3", true, "");
    expect(out.sent().back().tag == kTagReply &&
           out.sent().back().payload == encode_reply("h1-3", true, ""), "send_reply payload");
}
}  // namespace

int main() {
    test_cancel_requested_and_reply();
    test_normal_run();
    test_cache_full_is_length();
    test_interrupted_is_cancelled();
    test_cancel_stops_and_silences();
    test_begin_resets_the_run();
    test_token_seq_resets_each_run();
    test_fail_and_send_error();
    test_ttft_after_cancel_does_not_stop();
    if (failures == 0) std::cout << "pcie_genai_event_bridge_test passed\n";
    return failures == 0 ? 0 : 1;
}
