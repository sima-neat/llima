// Unit test for the pure chat rules VlmGenerator uses. No LLiMa,
// no MLA, no PCIe.
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "chat_policy.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

void test_effective_system_prompt() {
    expect(effective_system_prompt(std::nullopt, "Default.") == "Default.", "absent = model default");
    expect(effective_system_prompt(std::string(""), "Default.").empty(), "\"\" = no system prompt");
    expect(effective_system_prompt(std::string("Pirate."), "Default.") == "Pirate.", "text = text");
}

void test_after_run() {
    expect(after_run(false, "partial") == AfterRun::ClearInterrupted, "stopped run clears");
    expect(after_run(true, " \n\t") == AfterRun::ClearEmpty, "blank answer clears");
    expect(after_run(true, "") == AfterRun::ClearEmpty, "empty answer clears");
    expect(after_run(true, "A cat.") == AfterRun::KeepAnswer, "real answer is kept");
    expect(trim_answer("  A cat.\n") == "A cat.", "answers are trimmed like the devkit");
}

void test_kept_files() {
    int a = 0, b = 0;
    {
        KeptFiles files;
        files.add([&] { ++a; });
        files.add([&] { throw std::runtime_error("cannot delete"); });
        files.add([&] { ++b; });
        expect(files.size() == 3, "three kept");
        files.clear();
        expect(a == 1 && b == 1, "clear runs every remover, even after one throws");
        expect(files.size() == 0, "clear forgets them");
        files.clear();
        expect(a == 1 && b == 1, "a second clear runs nothing");
        files.add([&] { ++a; });
    }
    expect(a == 2, "the destructor clears what is left");
}
}  // namespace

void test_cap_max_new_tokens() {
    expect(!cap_max_new_tokens(std::nullopt, 2048).has_value(), "no value = model default");
    expect(cap_max_new_tokens(uint16_t{128}, 2048) == uint16_t{128}, "a normal value is kept");
    expect(cap_max_new_tokens(uint16_t{65535}, 2048) == uint16_t{2048},
           "a huge value is capped at the context (it used to wrap to an empty answer)");
    expect(cap_max_new_tokens(uint16_t{2048}, 2048) == uint16_t{2048}, "exactly the context is kept");
    // A context over half of uint16: context + cap must still fit.
    expect(cap_max_new_tokens(uint16_t{65535}, 40000) == uint16_t{25535},
           "context + cap never passes 65535");
}

// Feed chunks the way LLiMa streams them; collect what goes where.
StreamSplit::Out feed(StreamSplit& split, const std::vector<std::string>& chunks) {
    StreamSplit::Out all;
    for (std::size_t i = 0; i < chunks.size(); ++i) {
        const auto out = split.add(chunks[i], i + 1 == chunks.size(), false);
        all.to_host += out.to_host;
        all.to_answer += out.to_answer;
    }
    return all;
}

void test_stream_split() {
    using simaai::llima::ReasoningFormat;
    const std::vector<std::string> thought = {"<thi", "nk>secret plan", "</think>", "Hello", "!"};
    // LFM2 with thinking off still reasons; that must not reach the host.
    {
        StreamSplit split(ReasoningFormat::Lfm2, false);
        const auto out = feed(split, thought);
        expect(out.to_host.find("secret") == std::string::npos &&
                   out.to_host.find("<think>") == std::string::npos,
               "hidden LFM2 reasoning is not sent: " + out.to_host);
        expect(out.to_host == "Hello!", "the host gets the answer: " + out.to_host);
        expect(out.to_answer == "Hello!", "the history gets the answer");
    }
    // Thinking on: the host still sees the raw text (markers show the thinking);
    // the history keeps only the answer.
    {
        StreamSplit split(ReasoningFormat::Lfm2, true);
        const auto out = feed(split, thought);
        expect(out.to_host == "<think>secret plan</think>Hello!", "thinking on: raw to host");
        expect(out.to_answer == "Hello!", "thinking on: answer only in history");
    }
    // A model without reasoning: everything is the answer.
    {
        StreamSplit split(ReasoningFormat::None, false);
        const auto out = feed(split, {"Hel", "lo"});
        expect(out.to_host == "Hello" && out.to_answer == "Hello", "no reasoning: pass-through");
    }
}

int main() {
    test_effective_system_prompt();
    test_after_run();
    test_cap_max_new_tokens();
    test_kept_files();
    test_stream_split();
    if (failures == 0) std::cout << "pcie_genai_chat_policy_test passed\n";
    return failures == 0 ? 0 : 1;
}
