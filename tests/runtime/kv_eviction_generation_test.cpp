// DevKit test for runtime KV cache eviction: runs past the compiled context on a text model and
// checks the mechanics (not answer quality, which is evaluated separately).

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/common.h>

#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

namespace {

using simaai::llima::KvEvictionConfig;
using simaai::llima::VisionLanguageModel;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

constexpr const char* kSystemPrompt = "You are a concise assistant.";

std::string filler(size_t sentences) {
    static const char* subjects[] = {"The harbor", "The old mill", "The market", "The river",
                                     "The school", "The bakery", "The station", "The library"};
    static const char* verbs[] = {"was busy", "stayed quiet", "smelled of rain", "looked grey",
                                  "was repaired", "was crowded", "felt cold", "was painted"};
    std::string text;
    for (size_t i = 0; i < sentences; ++i) {
        text += std::string(subjects[(i * 7) % 8]) + " " + verbs[(i * 5 + i / 8) % 8] + " on day "
            + std::to_string(i % 31 + 1) + ". ";
    }
    return text;
}

// Runs the turns as one chat; returns the responses and fills `ttft` per turn. ("FULL" is also
// reported when a response reaches max_new_tokens, so it is not checked here.)
std::vector<std::string> ask(VisionLanguageModel& model, simaai::llima::Chat& chat,
                             const std::vector<std::string>& turns, std::vector<double>* ttft = nullptr) {
    // Called from the streamer thread, also by later runs: capture the pointer, not this frame.
    model.set_info_callback([ttft](const std::string& type, double value) {
        if (type == "ttft" && ttft) ttft->push_back(value);
    });
    std::vector<std::string> responses;
    for (const auto& turn : turns) {
        chat.add_query(turn);
        const auto response = model.run_model(chat, 16);
        require(response.has_value(), "generation was interrupted");
        responses.push_back(simaai::llima::test::trim(*response));
        chat.add_response(responses.back());
    }
    return responses;
}

// A new chat with the system prompt, or none if it is empty. A new chat inherits the last system
// prompt, and clearing it keeps the inherited system message, so the messages are reset too.
simaai::llima::Chat new_chat(VisionLanguageModel& model, const std::string& system_prompt) {
    auto chat = model.create_chat();
    chat.set_system_prompt(system_prompt);
    if (system_prompt.empty()) chat.set_messages(nlohmann::ordered_json::array());
    return chat;
}

std::vector<std::string> ask(VisionLanguageModel& model, const std::vector<std::string>& turns,
                             std::vector<double>* ttft = nullptr,
                             const std::string& system_prompt = kSystemPrompt) {
    auto chat = new_chat(model, system_prompt);
    return ask(model, chat, turns, ttft);
}

void test_fitting_chat_is_unchanged(VisionLanguageModel& model) {
    const std::vector<std::string> turns = {
        filler(40) + " The ferry leaves at 7:15. When does the ferry leave?", "Repeat only the time.",
    };
    model.set_kv_eviction({"off"});
    const auto reference = ask(model, turns);
    for (const auto* name : {"sink_window", "keydiff"}) {
        model.set_kv_eviction({name});
        require(ask(model, turns) == reference, std::string(name) + ": a chat that fits changed");
    }
}

void test_generation_continues_past_the_cache(VisionLanguageModel& model, uint16_t context) {
    // Arbitrary valid token ids and no stop tokens: generation only ends at the limit.
    std::vector<uint32_t> input_ids;
    for (uint32_t i = 0; i + 64 < context; ++i) input_ids.push_back(1000 + (i * 37) % 1000);
    const uint16_t limit = context + 192;
    model.set_kv_eviction({"off"});
    require(
        input_ids.size() + model.run_model(input_ids, limit, std::set<uint32_t>{}).size() <= context,
        "off: generation went past the compiled cache"
    );
    for (const auto* name : {"sink_window", "keydiff"}) {
        model.set_kv_eviction({name});
        require(
            input_ids.size() + model.run_model(input_ids, limit, std::set<uint32_t>{}).size() == limit,
            std::string(name) + ": generation stopped before the limit"
        );
    }
}

void test_long_chat(VisionLanguageModel& model, uint16_t context) {
    // The first turn is about twice the cache; the fact sits in the recent window. No system
    // prompt: some templates (Mistral v0.3) move it to the latest user message, which changes the
    // earlier tokens and re-processes the conversation by design.
    const std::string fact = " My flight number is LH 438. Reply with OK.";
    const std::vector<std::string> turns = {
        filler(static_cast<size_t>(context) * 2 / 9) + fact,
        "What is my flight number? Answer with the number only.",
        "Thanks. Reply with OK.",
    };
    // Recall is checked only if the model recalls the fact in a chat that fits.
    model.set_kv_eviction({"off"});
    const bool recalls = ask(model, {filler(context / 18) + fact, turns[1]}, nullptr, "")[1]
        .find("438") != std::string::npos;
    for (const auto* name : {"sink_window", "keydiff"}) {
        model.set_kv_eviction({name});
        std::vector<double> ttft;
        auto chat = new_chat(model, "");
        const auto responses = ask(model, chat, turns, &ttft);
        require(ttft.size() == turns.size(), "missing TTFT");
        std::cout << name << ": '" << responses[1] << "', ttft " << ttft[0] << "s, " << ttft[1]
                  << "s, " << ttft[2] << "s\n";
        require(
            !recalls || responses[1].find("438") != std::string::npos,
            std::string(name) + ": not recalled"
        );
        // Follow-up turns continue from the cache instead of re-processing the conversation.
        require(
            ttft[1] < 0.25 * ttft[0] && ttft[2] < 0.25 * ttft[0],
            std::string(name) + ": a follow-up turn re-processed the conversation"
        );

        // Editing an early message after evictions takes the full re-processing path.
        auto messages = chat.get_messages();
        messages[0]["content"] = "Note: " + messages[0]["content"].get<std::string>();
        messages.erase(messages.end() - 3, messages.end());  // ends with the second question
        chat.set_messages(messages);
        require(model.run_model(chat, 16).has_value(), "the edited chat was interrupted");

        // Deterministic: a fresh run gives the same responses.
        model.set_kv_eviction({name});
        require(
            ask(model, turns, nullptr, "") == responses,
            std::string(name) + ": a fresh run differs"
        );
    }
}

void test_system_prompt_must_fit_the_budget(VisionLanguageModel& model, uint16_t context) {
    model.set_kv_eviction({"keydiff"});
    try {
        ask(model, {filler(context / 4) + " Reply with OK."}, nullptr,
            std::string(kSystemPrompt) + " " + filler(context / 9));
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find("shorten the system prompt") != std::string::npos, message);
        return;
    }
    throw std::runtime_error("a system prompt longer than the budget was accepted");
}

void test_vlm_is_rejected() {
    VisionLanguageModel vlm(simaai::llima::test::resolve_model_dir(
        "SIMA_TEST_LLIMA_VLM_MODEL", simaai::llima::test::kDefaultVlmModelName, "LLiMa VLM",
        "devkit/vlm_config.json"
    ));
    try {
        vlm.set_kv_eviction({"keydiff"});
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find("vision-language") != std::string::npos, message);
        return;
    }
    throw std::runtime_error("enabling KV eviction on a VLM did not fail");
}

}  // namespace

int main() {
    bool connected = false;
    try {
        const auto model_dir = simaai::llima::test::resolve_model_dir(
            "SIMA_TEST_LLIMA_TEXT_MODEL", simaai::llima::test::kDefaultTextModelName, "LLiMa text",
            "devkit/vlm_config.json"
        );
        const uint16_t context = nlohmann::json::parse(std::ifstream(model_dir / "devkit/vlm_config.json"))
            .at("pipeline_cfg").at("max_num_tokens").get<uint16_t>();
        std::cout << "LLIMA_KV_EVICTION model_dir=" << model_dir << " context=" << context << '\n';

        simaai::llima::connect({}, "/tmp/sima_lmm_kv_eviction_generation_test.log", spdlog::level::info);
        connected = true;
        {
            VisionLanguageModel model(model_dir);
            model.set_text_callback([](const std::string&, bool, bool) {});
            test_fitting_chat_is_unchanged(model);
            test_generation_continues_past_the_cache(model, context);
            test_long_chat(model, context);
            test_system_prompt_must_fit_the_budget(model, context);
        }
        test_vlm_is_rejected();
        simaai::llima::disconnect();
        connected = false;
    } catch (const std::exception& error) {
        if (connected) {
            try { simaai::llima::disconnect(); } catch (...) {}
        }
        std::cerr << "KV eviction generation test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "KV eviction generation test passed\n";
    return 0;
}
