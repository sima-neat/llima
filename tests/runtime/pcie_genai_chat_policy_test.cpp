// Unit test for the pure chat rules VlmGenerator uses. No LLiMa,
// no MLA, no PCIe.
#include <iostream>
#include <stdexcept>
#include <string>

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

int main() {
    test_effective_system_prompt();
    test_after_run();
    test_cap_max_new_tokens();
    test_kept_files();
    if (failures == 0) std::cout << "pcie_genai_chat_policy_test passed\n";
    return failures == 0 ? 0 : 1;
}
