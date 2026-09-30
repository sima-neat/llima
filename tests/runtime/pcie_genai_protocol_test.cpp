// Unit test for the card side of the PCIe GenAI wire contract. The golden
// strings here are the same ones the host test uses
// (neat/core/pcie_host/tests/unit_pcie_host_genai_protocol_test.cpp), so the
// two repos cannot drift apart unnoticed.
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "genai_protocol.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

template <typename Fn>
bool refuses(Fn&& fn) {
    try { fn(); } catch (const std::invalid_argument&) { return true; }
    return false;
}

void test_parses_the_host_golden_prompt() {
    const PromptRequest r = parse_prompt(
        R"({"id":"h1-1","prompt":"Hi","system_prompt":"Be brief.","max_new_tokens":64,"enable_thinking":false})");
    expect(r.id == "h1-1", "id");
    expect(r.prompt == "Hi", "prompt");
    expect(r.system_prompt == "Be brief.", "system_prompt");
    expect(r.max_new_tokens == 64, "max_new_tokens");
    expect(!r.enable_thinking, "enable_thinking");
}

void test_optional_fields_default() {
    const PromptRequest r = parse_prompt(R"({"prompt":"Hi"})");
    expect(r.id.empty(), "id defaults to empty");
    expect(!r.system_prompt.has_value(), "no system prompt");
    expect(!r.max_new_tokens.has_value(), "no max_new_tokens = model default");
    const PromptRequest zero = parse_prompt(R"({"prompt":"Hi","max_new_tokens":0})");
    expect(!zero.max_new_tokens.has_value(), "0 means model default");
    const PromptRequest big = parse_prompt(R"({"prompt":"Hi","max_new_tokens":100000})");
    expect(big.max_new_tokens == 65535, "run_model takes uint16_t: clamp to 65535");
}

void test_refuses_bad_prompts() {
    expect(refuses([] { parse_prompt("not json"); }), "non-JSON refused");
    expect(refuses([] { parse_prompt("[1]"); }), "non-object refused");
    expect(refuses([] { parse_prompt(R"({"id":"x"})"); }), "missing prompt refused");
    expect(refuses([] { parse_prompt(R"({"prompt":""})"); }), "empty prompt refused");
    expect(refuses([] { parse_prompt(R"({"prompt":"Hi","max_new_tokens":-1})"); }),
           "negative max_new_tokens refused");
    // A wrong-type system_prompt must not be ignored (that would silently use the
    // model's default prompt). null means "model default", like genai.chat.
    expect(refuses([] { parse_prompt(R"({"prompt":"Hi","system_prompt":5})"); }),
           "non-string system_prompt refused");
    expect(refuses([] { parse_prompt(R"({"prompt":"Hi","system_prompt":["a"]})"); }),
           "list system_prompt refused");
    expect(!parse_prompt(R"({"prompt":"Hi","system_prompt":null})").system_prompt.has_value(),
           "null system_prompt = model default");
}

void test_reads_image_names() {
    const PromptRequest r = parse_prompt(
        R"({"id":"h1-1","prompt":"what","images":["pcie-genai/h1-1-0.jpg","pcie-genai/h1-1-1.png"]})");
    expect(r.images.size() == 2 && r.images[0] == "pcie-genai/h1-1-0.jpg" &&
           r.images[1] == "pcie-genai/h1-1-1.png", "image names read in order");
    expect(parse_prompt(R"({"prompt":"x"})").images.empty(), "no images key = no image");
    expect(parse_prompt(R"({"prompt":"x","images":[]})").images.empty(), "empty list = no image");
}

void test_refuses_unsafe_image_names() {
    // Each name must stay inside the serve root the card pulls from.
    expect(refuses([] { parse_prompt(R"({"prompt":"x","images":["/etc/passwd"]})"); }),
           "absolute image name refused");
    expect(refuses([] { parse_prompt(R"({"prompt":"x","images":["ok/a.jpg","../../secret"]})"); }),
           "one bad name refuses the whole prompt");
    expect(refuses([] { parse_prompt(R"({"prompt":"x","images":["a/../../b"]})"); }),
           "embedded .. image name refused");
    expect(refuses([] { parse_prompt(R"({"prompt":"x","images":[""]})"); }),
           "empty image name refused");
    expect(refuses([] { parse_prompt(R"({"prompt":"x","images":[5]})"); }),
           "non-string image refused");
    expect(refuses([] { parse_prompt(R"({"prompt":"x","images":"a.jpg"})"); }),
           "images must be a list");
    // An older host sends the old single "image" key. Refuse it with a clear
    // error instead of answering without the image.
    expect(refuses([] { parse_prompt(R"({"prompt":"x","image":"pcie-genai/h1-1.jpg"})"); }),
           "the old \"image\" key is refused (host too old)");
}

void test_parses_chat() {
    // Golden strings from the host test (unit_pcie_host_genai_protocol_test.cpp).
    const ChatRequest a = parse_chat(
        R"({"id":"h1-4","op":"reset","system_prompt":"Be brief.","enable_thinking":false})");
    expect(a.id == "h1-4" && a.op == ChatOp::Reset && a.system_prompt == "Be brief." &&
           !a.enable_thinking, "reset with system prompt");
    const ChatRequest b = parse_chat(R"({"id":"h1-5","op":"reset","enable_thinking":true})");
    expect(!b.system_prompt.has_value() && b.enable_thinking, "reset, model default prompt");
    const ChatRequest c =
        parse_chat(R"({"id":"h1-6","op":"reset","system_prompt":"","enable_thinking":false})");
    expect(c.system_prompt == std::string(""), "reset, no system prompt");
    const ChatRequest d = parse_chat(R"({"id":"h1-7","op":"print"})");
    expect(d.id == "h1-7" && d.op == ChatOp::Print, "print");
    expect(refuses([] { parse_chat(R"({"id":"x","op":"nope"})"); }), "unknown op refused");
    expect(refuses([] { parse_chat(R"({"id":"x"})"); }), "missing op refused");
    expect(refuses([] { parse_chat("not json"); }), "bad JSON refused");
    expect(refuses([] { parse_chat(R"({"op":"reset","system_prompt":5})"); }),
           "non-string system prompt refused");
}

void test_reply_and_history_cleared_goldens() {
    expect(encode_reply("h1-4", true, "") == R"({"id":"h1-4","ok":true,"text":""})",
           "reply golden");
    expect(encode_reply("h1-6", false, "Thinking is not supported for this model.") ==
           R"({"id":"h1-6","ok":false,"text":"Thinking is not supported for this model."})",
           "not-ok reply golden");
    expect(encode_final("h1-1", "cancelled", 1, 0.5, 0.0, true) ==
           R"({"id":"h1-1","finish_reason":"cancelled","generated_tokens":1,"ttft":0.5,"tps":0.0,"history_cleared":true})",
           "final with history_cleared");
    expect(encode_final("h1-1", "stop", 2, 0.5, 2.0, false) ==
           encode_final("h1-1", "stop", 2, 0.5, 2.0), "false writes no field (old golden)");
    expect(encode_error("h1-1", "pull failed", true) ==
           R"({"id":"h1-1","message":"pull failed","history_cleared":true})",
           "error with history_cleared");
}

void test_special_characters_round_trip() {
    nlohmann::ordered_json j;
    j["prompt"] = "He said \"hi\"\nnaïve 🙂";
    expect(parse_prompt(j.dump()).prompt == "He said \"hi\"\nnaïve 🙂", "special characters");
}

void test_try_read_id() {
    expect(try_read_id(R"({"id":"h1-2","prompt":""})") == "h1-2", "reads id from a bad prompt");
    expect(try_read_id("garbage").empty(), "empty id when unreadable");
}

void test_encoders_match_the_golden_strings() {
    expect(encode_metric("ttft", 0.5) == R"({"type":"ttft","value":0.5})", "metric golden");
    expect(encode_final("h1-1", "stop", 2, 0.5, 2.0) ==
               R"({"id":"h1-1","finish_reason":"stop","generated_tokens":2,"ttft":0.5,"tps":2.0})",
           "final golden");
    expect(encode_error("h1-1", "busy") == R"({"id":"h1-1","message":"busy"})", "error golden");
}

void test_encode_token_prefixes_the_sequence_number() {
    // Wire format shared with the host: "<seq>\n<raw text>".
    expect(encode_token(0, "Hel") == "0\nHel", "seq 0 then text");
    expect(encode_token(42, "lo") == "42\nlo", "multi-digit seq");
    // The text keeps every byte after the first newline, even more newlines.
    expect(encode_token(7, "a\nb") == "7\na\nb", "text may contain newlines");
    expect(encode_token(3, "") == "3\n", "empty text still carries its seq");
}

void test_bad_utf8_never_throws() {
    std::string out;
    bool threw = false;
    try { out = encode_error("h1-1", std::string("bad \xff byte")); } catch (...) { threw = true; }
    expect(!threw, "an invalid UTF-8 message must not throw");
    expect(nlohmann::json::accept(out), "the result is still valid JSON");
}
}  // namespace

int main() {
    test_parses_the_host_golden_prompt();
    test_optional_fields_default();
    test_refuses_bad_prompts();
    test_reads_image_names();
    test_refuses_unsafe_image_names();
    test_parses_chat();
    test_reply_and_history_cleared_goldens();
    test_special_characters_round_trip();
    test_try_read_id();
    test_encoders_match_the_golden_strings();
    test_encode_token_prefixes_the_sequence_number();
    test_bad_utf8_never_throws();
    if (failures == 0) std::cout << "pcie_genai_protocol_test passed\n";
    return failures == 0 ? 0 : 1;
}
