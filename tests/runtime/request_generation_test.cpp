#include <atomic>
#include <csignal>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include "language_model.hpp"
#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"
#include "vlm_helper.hpp"

using namespace simaai::llima;
using json = nlohmann::ordered_json;
namespace {
VisionLanguageModel* active_model = nullptr;
void interrupt(int) { if (active_model) active_model->stop_model(); }
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejected(F&& action, const char* message) {
    try { action(); } catch (const std::exception&) { return; }
    throw std::runtime_error(message);
}

void check_direct_pair(const std::filesystem::path& target_root, const std::filesystem::path& draft_root) {
    const auto cfg = json::parse(std::ifstream(target_root / "devkit/vlm_config.json")).get<VlmConfig>();
    VlmHelper helper(cfg, target_root / "devkit", std::string(""), std::nullopt);
    TextStreamer target_stream(helper.get_tokenizer()), draft_stream(helper.get_tokenizer());
    target_stream.disable();
    draft_stream.disable();
    LanguageModel target(target_root, helper.get_stop_token_ids(), helper.get_image_token_id(),
                         helper.get_pad_token_id(), target_stream);
    auto make_draft = [&] {
        return std::make_unique<LanguageModel>(draft_root, helper.get_stop_token_ids(), std::nullopt,
                                               helper.get_pad_token_id(), draft_stream);
    };
    auto draft = make_draft();
    auto prepare = [&](const std::string& query) {
        Chat chat(helper);
        chat.clear_system_prompt();
        chat.set_enable_thinking(false);
        chat.add_query(query);
        return helper.preprocess(chat).input_token_ids;
    };
    const auto a = prepare("Name one planet.");
    const auto b = prepare("Name one color.");
    const auto limit_for = [&](const auto& prompt) {
        return static_cast<uint16_t>(prompt.size() + cfg.lm_cfg.get_single_num_tokens() + 12);
    };
    auto generate = [&](const std::vector<uint32_t>& prompt, bool direct_mtp = false) {
        target.create_input_buffers(prompt);
        const auto limit = limit_for(prompt);
        auto result = direct_mtp
            ? target.run_model_gemma4_mtp(*draft, prompt, limit)
            : target.run_model_speculative_decoding(*draft, prompt, limit);
        require(result && !result->empty(), "Direct speculative generation did not return tokens");
        return *result;
    };

    const bool mtp = cfg.lm_cfg.is_gemma4_mtp_target();
    const auto gold_a = generate(a, mtp);
    const auto gold_b = generate(b);
    require(generate(a) == gold_a && generate(b, mtp) == gold_b,
            "Direct speculative calls disagreed with their fresh references");
    const std::vector<uint32_t> invalid{cfg.lm_cfg.token_cfg.vocab_size};
    rejected([&] { generate(invalid, mtp); }, "Direct speculative request accepted an invalid token");
    require(draft->get_kv_cache_len() == 0 && target.get_kv_cache_len() == 0,
            "A failed direct request retained target or draft cache state");
    require(generate(b, mtp) == gold_b, "Direct speculative recovery changed tokens");
    std::atomic<bool> cancelled_on_first_token = false;
    target_stream.set_info_callback([&](const std::string& key, double) {
        if (key == "ttft") {
            target.stop_model();
            cancelled_on_first_token = true;
        }
    });
    target_stream.enable();
    target.create_input_buffers(a);
    const auto interrupted = target.run_model_speculative_decoding(*draft, a, limit_for(a));
    target_stream.disable();
    target_stream.set_info_callback([](const std::string&, double) {});
    require(cancelled_on_first_token, "The direct speculative request did not emit a first-token event");
    require(!interrupted, "Direct speculative generation returned tokens after cancellation");
    require(draft->get_kv_cache_len() == 0, "Cancellation retained the draft cache");
    require(target.get_kv_cache_len() == 0, "Cancellation retained the target cache");
    require(generate(b, mtp) == gold_b, "Cancellation changed the next direct request");
    require(target.get_kv_cache_len() > 0, "Recovery did not populate the target cache");
    const auto previous_length = target.get_kv_cache_len();
    const auto previous_draft_length = draft->get_kv_cache_len();
    rejected([&] { target.run_model_speculative_decoding(target, a, limit_for(a)); },
             "Direct speculative generation accepted the target as its own draft");
    require(target.get_kv_cache_len() == previous_length && draft->get_kv_cache_len() == previous_draft_length,
            "Rejecting an incompatible draft changed a model cache");
    auto replacement = make_draft();
    target.attach_draft(*replacement);
    require(target.get_kv_cache_len() == 0,
            "Replacing a draft runner over the same package retained the old pair's cache");
    draft = std::move(replacement);
    require(generate(a, mtp) == gold_a, "Replacing the draft runner changed greedy output");
    std::cout << "Direct speculative pairing passed\n";
}

static_assert(std::is_aggregate_v<GenerationOptions>);

}

int main(int argc, char** argv) {
    bool connected = false;
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--direct-pair") {
            connect({}, "/tmp/llima-direct-pair.log", spdlog::level::warn);
            connected = true;
            check_direct_pair(argv[2], argv[3]);
            disconnect();
            return 0;
        }
        const auto root = argc > 1 ? std::filesystem::path(argv[1]) : test::resolve_model_dir(
            "SIMA_TEST_LLIMA_TEXT_MODEL", test::kDefaultTextModelName, "generation text", "devkit/vlm_config.json");
        const auto cfg = json::parse(std::ifstream(root / "devkit/vlm_config.json")).get<VlmConfig>();
        connect({}, "/tmp/llima-request-generation.log", spdlog::level::warn);
        connected = true;
        std::signal(SIGINT, interrupt);
        auto first = std::make_unique<VisionLanguageModel>(root, std::string(""));
        active_model = first.get();
        std::unique_ptr<VisionLanguageModel> draft;
        std::weak_ptr<FileProvider> draft_files;
        if (argc > 2 && std::string_view(argv[2]) != "-") {
            const std::filesystem::path draft_root(argv[2]);
            auto files = std::make_shared<DiskFileProvider>(draft_root);
            draft_files = files;
            draft = std::make_unique<VisionLanguageModel>(
                draft_root, std::nullopt, std::nullopt, std::move(files));
            first->set_draft_vlm(draft.get());
        }
        VlmHelper helper(cfg, root / "devkit", std::string(""), std::nullopt);
        std::string prefix;
        for (int i = 0; i < 24; ++i) prefix += "Reference note: keep each answer clear and accurate. ";
        auto prepare = [&](const std::string& query) {
            Chat chat(helper);
            chat.clear_system_prompt();
            chat.set_enable_thinking(false);
            chat.add_query(prefix + query);
            return helper.preprocess(chat).input_token_ids;
        };
        const auto a = prepare("List the first eight prime numbers, one per line.");
        const auto b = prepare("Write a short paragraph about the moon.");
        auto generate = [&](const auto& input) {
            return first->run_model(input, {.max_new_tokens = 64});
        };
        const auto gold_a = generate(a);
        const auto gold_b = generate(b);
        require(!gold_a.empty() && !gold_b.empty(), "Fresh references are empty");
        require(generate(a) == gold_a && generate(b) == gold_b, "Branched prompt reuse changed output");
        const auto workspace = first->workspace_memory_bytes();

        std::atomic<bool> busy_rejected = false;
        GenerationOptions cancel_options{.max_new_tokens = 512};
        cancel_options.callbacks.info = [&](const std::string& key, double) {
            if (key != "ttft") return;
            const auto busy = [](auto&& action) {
                try { action(); } catch (const std::logic_error&) { return true; }
                return false;
            };
            busy_rejected = busy([&] { first->unset_reloc(); })
                         && busy([&] { first->set_reloc("unused"); })
                         && first->workspace_memory_bytes() == workspace;
            first->stop_model();
        };
        auto cancel_chat = first->create_chat();
        cancel_chat.clear_system_prompt();
        cancel_chat.set_enable_thinking(false);
        cancel_chat.add_query("Write a long detailed story about a journey through space.");
        const auto cancelled = first->generate(cancel_chat, cancel_options);
        require(busy_rejected, "A busy workspace operation was not rejected from the callback");
        require(cancelled.finish == ChatResult::Finish::Cancelled, "Cancelled request returned a completed response");
        require(generate(a) == gold_a, "Cancellation recovery differs from fresh reference");
        const std::vector<uint32_t> invalid{cfg.lm_cfg.token_cfg.vocab_size};
        rejected([&] { first->run_model(invalid, {.max_new_tokens = 64}); }, "Out-of-vocabulary token was accepted");
        require(generate(b) == gold_b, "Exception recovery differs from fresh reference");
        std::cout << json{{"case", "cancel_exception_recovery"}, {"runtime_ok", true}}.dump() << '\n';

        if (draft && cfg.lm_cfg.is_dflash()) {
            GenerationOptions block_options{.max_new_tokens = 128, .stop_token_ids = std::set<uint32_t>{}};
            bool stopped_after_verification = false;
            bool signalled_full = false;
            block_options.callbacks.info = [&](const std::string& key, double) {
                if (key == "tps" && !stopped_after_verification) {
                    stopped_after_verification = true;
                    first->stop_model();
                }
                if (key == "FULL") signalled_full = true;
            };
            const auto interrupted = first->generate(cancel_chat, block_options);
            require(interrupted.finish == ChatResult::Finish::Cancelled && stopped_after_verification && !signalled_full,
                    "DFlash reported successful completion or a full cache after cancellation");
            require(generate(a) == gold_a, "DFlash block cancellation left stale state");
        }

        std::atomic<unsigned> request_chunks = 0;
        std::atomic<unsigned> request_events = 0;
        std::atomic<unsigned> request_completions = 0;
        GenerationOptions options{.max_new_tokens = 16};
        options.callbacks.text = [&](const std::string&, bool end, bool) {
            ++request_chunks;
            if (end) ++request_completions;
        };
        options.callbacks.info = [&](const std::string&, double) { ++request_events; };
        auto option_chat = first->create_chat();
        option_chat.add_query("Name three planets.");
        require(first->generate(option_chat, options).finish != ChatResult::Finish::Cancelled, "Request options failed");
        require(request_chunks > 0 && request_events > 0 && request_completions == 1,
                "Generation returned before its callbacks completed");
        const auto chunks = request_chunks.load();
        const auto events = request_events.load();
        std::atomic<unsigned> next_chunks = 0;
        first->generate(option_chat, {.max_new_tokens = 16,
            .callbacks = {.text = [&](const std::string&, bool, bool) { ++next_chunks; }}});
        require(next_chunks > 0 && request_chunks == chunks && request_events == events,
                "Request callbacks leaked into a later request");
        std::cout << json{{"case", "request_callback_isolation"}, {"runtime_ok", true}}.dump() << '\n';

        options.callbacks.info = [](const std::string& key, double) {
            if (key == "ttft") throw std::runtime_error("Intentional request callback failure");
        };
        rejected([&] { first->generate(option_chat, options); }, "Callback exception was swallowed");
        require(generate(a) == gold_a, "Callback exception left the next request corrupted");

        unsigned completions = 0;
        options.callbacks.info = [&](const std::string& key, double) {
            if (key == "END" || key == "FULL") {
                ++completions;
                first->stop_model();
            }
        };
        require(first->generate(option_chat, options).finish == ChatResult::Finish::Cancelled,
                "Cancellation from the terminal callback arrived after result assembly");
        require(completions == 1, "Streaming completion was delivered more than once");

        std::stop_source retired;
        options.stop = retired.get_token();
        options.max_new_tokens = 32;
        options.stop_token_ids = std::set<uint32_t>{};
        options.performance_summary = false;
        options.preserve_structural_tokens = false;
        options.callbacks.info.reset();
        const auto reference = first->generate(option_chat, options);
        require(reference.finish == ChatResult::Finish::Length && !reference.token_ids.empty(),
                "Expected the token budget to end the request");
        require(reference.text == helper.get_tokenizer()->decode(reference.token_ids, true)
                    && reference.performance.generated_tokens == reference.token_ids.size(),
                "ChatResult text or metrics disagree with its token IDs");

        std::stop_source current;
        options.stop = current.get_token();
        options.callbacks.info = [&](const std::string& key, double) {
            if (key == "ttft") retired.request_stop();
        };
        auto next = first->generate(option_chat, options);
        require(retired.stop_requested() && next.finish == reference.finish && next.token_ids == reference.token_ids,
                "Stopping a completed request interrupted the next request");

        options.stop = retired.get_token();
        options.callbacks.info = [](const std::string&, double) { throw std::runtime_error("Pre-cancel callback"); };
        const auto skipped = first->generate(option_chat, options);
        require(skipped.finish == ChatResult::Finish::Cancelled && skipped.token_ids.empty()
                    && skipped.text.empty() && skipped.performance.generated_tokens == 0,
                "An already-cancelled request generated output");
        require(first->run_model(a, options).empty(), "Pre-cancelled token generation returned tokens");
        rejected([&] { first->run_model_for_ttnt(a, options); }, "Pre-cancelled timing request succeeded");
        rejected([&] { first->run_model_for_logits(a, options); }, "Pre-cancelled logits request succeeded");
        rejected([&] { first->run_model_for_loglikelihood(a, 0, a, options); },
                 "Pre-cancelled log-likelihood request succeeded");
        options.stop = current.get_token();
        options.callbacks.info = [&](const std::string& key, double) {
            if (key == "END" || key == "FULL") current.request_stop();
        };
        next = first->generate(option_chat, options);
        require(next.finish == ChatResult::Finish::Cancelled && next.token_ids.empty() && next.text.empty(),
                "A terminal callback cancelled too late to affect ChatResult");

        const auto timing = first->run_model_for_ttnt(a);
        require(timing.generated_tokens > 1, "Nested prefill leaked its output limit into decode");
        require(generate(a) == gold_a, "Performance request left stale prompt state");
        if (!draft) {
            const std::set<uint32_t> no_stops;
            GenerationOptions limits{.max_new_tokens = 4, .stop_token_ids = no_stops};
            const auto four_tokens = first->run_model(a, limits);
            require(four_tokens.size() == 4, "Ordinary generation ignored the token budget");
            for (uint16_t budget : {0, 1, 4, 40}) {
                const auto metrics = first->run_model_for_ttnt(a,
                    {.max_new_tokens = budget, .stop_token_ids = no_stops});
                require(metrics.generated_tokens == budget && metrics.token_durations.size() == budget,
                        "Ordinary timing must report exactly one sample per emitted token");
            }
            limits.stop_token_ids = std::set{four_tokens.front()};
            const auto stopped = first->run_model_for_ttnt(a, limits);
            require(stopped.generated_tokens == 1 && stopped.token_durations.size() == 1,
                    "An immediate stop token must have exactly one timing sample");
        }
        if (cfg.pipeline_cfg.return_logits && !draft) {
            const auto suffix = std::span<const uint32_t>(b).last(2);
            first->run_model_for_loglikelihood(b, b.size() - 2, suffix);
            require(generate(a) == gold_a, "Scoring left stale generation history");
        }
        std::cout << json{{"case", "metrics_scoring_recovery"}, {"runtime_ok", true}}.dump() << '\n';

        auto independent_chat = first->create_chat();
        independent_chat.set_system_prompt("Only this conversation has a custom system prompt.");
        require(first->create_chat().get_messages().empty(), "Chat mutated shared system-prompt defaults");

        // Another wrapper leases the same programs/resources. Releasing the first must not unload it.
        auto survivor = std::make_unique<VisionLanguageModel>(root, std::string(""));
        if (draft) {
            survivor->set_draft_vlm(draft.get());
        }
        auto concurrent_a = std::async(std::launch::async, [&] { return generate(a); });
        auto concurrent_b = std::async(std::launch::async, [&] {
            return survivor->run_model(b, {.max_new_tokens = 64});
        });
        const auto result_a = concurrent_a.get();
        const auto result_b = concurrent_b.get();
        require(result_a == gold_a, "Concurrent first workspace differs from its fresh reference");
        require(result_b == gold_b, "Concurrent second workspace differs from its fresh reference");
        std::cout << json{{"case", "concurrent_workspaces"}, {"runtime_ok", true}}.dump() << '\n';
        if (draft) {
            require(survivor->run_model(a, {.max_new_tokens = 64}) == gold_a,
                    "Shared draft workspace changed the new target's output");
            require(generate(a) == gold_a,
                    "Another target corrupted this target's paired draft context");
        }
        active_model = nullptr;
        auto detached_chat = first->create_chat();
        if (draft) {
            active_model = survivor.get();
            GenerationOptions teardown_options{.max_new_tokens = 16};
            teardown_options.callbacks.info = [&](const std::string& key, double) {
                if (key == "END" || key == "FULL") first.reset();
            };
            auto teardown_chat = survivor->create_chat();
            teardown_chat.add_query("Name one planet.");
            require(survivor->generate(teardown_chat, teardown_options).finish != ChatResult::Finish::Cancelled,
                    "Callback teardown of another target sharing this draft failed");
            require(!first, "The callback did not release the other target");
        } else {
            GenerationOptions callback_options{.max_new_tokens = 16};
            callback_options.callbacks.info = [&](const std::string& key, double) {
                if (key == "END" || key == "FULL")
                    require(survivor->run_model(a, {.max_new_tokens = 64}) == gold_a,
                            "Cross-model callback generation differs from the reference");
            };
            require(first->generate(option_chat, callback_options).finish != ChatResult::Finish::Cancelled,
                    "Cross-model callback generation blocked completion");
            first.reset();
        }
        if (draft) {
            draft.reset();
            require(!draft_files.expired(), "Surviving target released its draft runtime assets");
        }
        // The target retains language execution after the draft facade is released.
        active_model = survivor.get();
        detached_chat.add_query("Name a planet.");
        require(detached_chat.get_last_query() == "Name a planet.", "Chat retained a released helper");
        detached_chat.update_last_query("Name one planet only.");
        require(survivor->generate(detached_chat, {.max_new_tokens = 16}).finish != ChatResult::Finish::Cancelled,
                "Detached chat cannot use the surviving model");
        require(survivor->run_model(a, {.max_new_tokens = 64}) == gold_a,
                "Releasing one wrapper unloaded or corrupted the surviving wrapper");
        active_model = nullptr;
        survivor.reset();
        require(draft_files.expired(), "Draft runtime assets leaked after the last target was released");
        std::cout << json{{"case", "shared_program_lifetime"}, {"runtime_ok", true}}.dump() << '\n';
        disconnect();
        connected = false;
    } catch (const std::exception& e) {
        if (connected) { try { disconnect(); } catch (...) {} }
        std::cerr << "Request generation test failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "Request generation test passed\n";
    return 0;
}
