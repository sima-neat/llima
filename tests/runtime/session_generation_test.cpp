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
    auto generate = [&](SessionId id, const std::vector<uint32_t>& prompt, bool direct_mtp = false) {
        target.select_session(id);
        target.create_input_buffers(prompt);
        const auto limit = limit_for(prompt);
        auto result = direct_mtp
            ? target.run_model_gemma4_mtp(*draft, prompt, limit)
            : target.run_model_speculative_decoding(*draft, prompt, limit);
        require(result && !result->empty(), "Direct speculative generation did not return tokens");
        return *result;
    };

    constexpr uint16_t sentinel = 7;
    draft->set_kv_cache_len(sentinel);
    const auto session = target.create_session();
    const bool mtp = cfg.lm_cfg.is_gemma4_mtp_target();
    const auto gold_a = generate(0, a, mtp);
    const auto gold_b = generate(session, b);
    require(draft->get_kv_cache_len() == sentinel, "Direct generation modified the public draft session");
    target.reset_session(session);
    require(generate(0, a) == gold_a && generate(session, b, mtp) == gold_b,
            "Direct speculative sessions disagreed with their fresh references");
    const std::vector<uint32_t> invalid{cfg.lm_cfg.token_cfg.vocab_size};
    rejected([&] { generate(session, invalid, mtp); }, "Direct speculative request accepted an invalid token");
    require(draft->get_kv_cache_len() == sentinel && target.get_kv_cache_len() == 0,
            "A failed direct request did not preserve the public draft or clear the target");
    require(generate(session, b, mtp) == gold_b, "Direct speculative recovery changed tokens");
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
    require(draft->get_kv_cache_len() == sentinel, "Cancellation changed the public draft cache");
    require(target.get_kv_cache_len() == 0, "Cancellation retained the target cache");
    require(generate(session, b, mtp) == gold_b, "Cancellation changed the next direct request");
    target.release_session(session);
    require(target.get_kv_cache_len() > 0, "Expected the original target session to retain its prefix");
    const auto previous_length = target.get_kv_cache_len();
    rejected([&] { target.run_model_speculative_decoding(target, a, limit_for(a)); },
             "Direct speculative generation accepted the target as its own draft");
    require(target.get_kv_cache_len() == previous_length && draft->get_kv_cache_len() == sentinel,
            "Rejecting an incompatible draft changed a session cache");
    auto replacement = make_draft();
    target.attach_draft(*replacement);
    require(target.get_kv_cache_len() == 0,
            "Replacing a draft runner over the same package retained the old pair's cache");
    draft = std::move(replacement);
    require(generate(0, a, mtp) == gold_a, "Replacing the draft runner changed greedy output");
    std::cout << "Direct speculative session pairing passed\n";
}

static_assert(std::is_const_v<std::remove_reference_t<
    decltype(std::declval<LanguageModel&>().get_model_map("pre"))>>);

}

int main(int argc, char** argv) {
    bool connected = false;
    try {
        if (argc == 4 && std::string_view(argv[1]) == "--direct-pair") {
            connect({}, "/tmp/llima-direct-session.log", spdlog::level::warn);
            connected = true;
            check_direct_pair(argv[2], argv[3]);
            disconnect();
            return 0;
        }
        const auto root = argc > 1 ? std::filesystem::path(argv[1]) : test::resolve_model_dir(
            "SIMA_TEST_LLIMA_TEXT_MODEL", test::kDefaultTextModelName, "session text", "devkit/vlm_config.json");
        const auto cfg = json::parse(std::ifstream(root / "devkit/vlm_config.json")).get<VlmConfig>();
        connect({}, "/tmp/llima-session-generation.log", spdlog::level::warn);
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
            const auto unused = draft->create_session();
            draft->release_session(unused);
            require(draft->session_memory_bytes(0) == 0,
                    "Draft setup or releasing an unused session allocated its public cache");
            first->set_draft_vlm(draft.get());
        }
        first->set_text_callback([](const std::string&, bool, bool) {});
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
        auto generate = [&](SessionId id, const auto& input) {
            return first->run_model(id, input, static_cast<uint16_t>(input.size() + 64));
        };
        first->reset_session(0);
        const auto gold_a = generate(0, a);
        if (draft) {
            require(draft->session_memory_bytes(0) == 0,
                    "Target generation wrote the draft's public session");
            draft->reset_session(0);
            require(draft->session_memory_bytes(0) == 0,
                    "Reset of an unused public draft session allocated its cache");
            require(generate(0, a) == gold_a,
                    "Resetting the draft's public session corrupted a paired target cache");
            const auto branch_after_reset = generate(0, b);
            first->reset_session(0);
            require(branch_after_reset == generate(0, b),
                    "Resetting the draft's public session corrupted a branched prompt");

            const auto limit = static_cast<uint16_t>(b.size() + 16);
            const auto reference_repeat = first->run_model_for_ttnt(0, b, limit);
            first->reset_session(0);
            require(generate(0, b) == branch_after_reset, "Fresh target output changed");
            draft->reset_session(0);
            const auto after_reset = first->run_model_for_ttnt(0, b, limit);
            require(reference_repeat.accepted_draft_tokens.has_value()
                        && after_reset.accepted_draft_tokens == reference_repeat.accepted_draft_tokens
                        && after_reset.generated_tokens == reference_repeat.generated_tokens,
                    "Resetting the public draft session changed speculative acceptance");
            std::cout << json{{"case", "paired_draft_isolation"}, {"runtime_ok", true},
                {"generated_tokens", after_reset.generated_tokens},
                {"accepted_draft_tokens", *after_reset.accepted_draft_tokens}}.dump() << '\n';
        }
        first->reset_session(0);
        const auto gold_b = generate(0, b);
        require(!gold_a.empty() && !gold_b.empty(), "Fresh references are empty");
        first->reset_session(0);
        require(generate(0, a) == gold_a, "Default session reset changed the reference");
        const auto shared = first->shared_resource_bytes();
        const auto workspace = first->workspace_memory_bytes();
        const SessionId sa = 0;
        const auto sb = first->create_session();
        require(first->session_memory_bytes(sb) == 0,
                "Unused session allocated device buffers or recurrent checkpoint snapshots");
        require(generate(sa, a) == gold_a, "Session A differs from fresh reference");
        require(generate(sb, b) == gold_b, "Session B differs from fresh reference");
        require(generate(sa, a) == gold_a, "Session A repeat uses another session's state");
        require(generate(sb, b) == gold_b, "Session B repeat uses another session's state");
        require(generate(sa, b) == gold_b, "Branched session reuse differs from fresh reference");
        first->reset_session(sa);
        require(generate(sa, a) == gold_a, "Reset session differs from fresh reference");
        require(first->shared_resource_bytes() == shared, "Creating sessions duplicated embeddings");
        require(first->workspace_memory_bytes() == workspace, "Creating sessions duplicated workspace");
        std::cout << json{{"case", "alternating_sessions"}, {"runtime_ok", true},
            {"shared_bytes", shared}, {"workspace_bytes", workspace},
            {"session_bytes", first->session_memory_bytes(sa)}, {"token_ids", gold_a}}.dump() << '\n';
        first->set_session_limit(2);
        rejected([&] { first->create_session(); }, "Session limit was ignored");
        rejected([&] { first->reset_session(999999); }, "Unknown session was accepted");
        first->release_session(sb);
        const auto replacement = first->create_session();
        require(replacement != sb, "Released session handle was reused");
        rejected([&] { generate(sb, b); }, "Released session was accepted");
        first->release_session(replacement);

        std::atomic<bool> busy_rejected = false;
        first->set_info_callback([&](const std::string& key, double) {
            if (key != "ttft") return;
            const auto busy = [](auto&& action) {
                try { action(); } catch (const std::logic_error&) { return true; }
                return false;
            };
            busy_rejected = busy([&] { first->reset_session(sa); })
                         && busy([&] { first->unset_reloc(sa); })
                         && busy([&] { first->set_reloc(sa, "unused"); })
                         && busy([&] { first->workspace_memory_bytes(); });
            first->stop_model();
        });
        auto cancel_chat = first->create_chat();
        cancel_chat.clear_system_prompt();
        cancel_chat.set_enable_thinking(false);
        cancel_chat.add_query("Write a long detailed story about a journey through space.");
        const auto cancelled = first->run_model(sa, cancel_chat, 512);
        first->wait_for_streamer_completion();
        require(busy_rejected, "A busy workspace operation was not rejected from the callback");
        require(!cancelled.has_value(), "Cancelled request returned a completed response");
        first->set_info_callback([](const std::string&, double) {});
        require(generate(sa, a) == gold_a, "Cancellation recovery differs from fresh reference");
        const std::vector<uint32_t> invalid{cfg.lm_cfg.token_cfg.vocab_size};
        rejected([&] { first->run_model(sa, invalid, 64); }, "Out-of-vocabulary token was accepted");
        first->wait_for_streamer_completion();
        require(generate(sa, b) == gold_b, "Exception recovery differs from fresh reference");
        std::cout << json{{"case", "cancel_exception_recovery"}, {"runtime_ok", true}}.dump() << '\n';

        if (draft && cfg.lm_cfg.is_dflash()) {
            GenerationOptions block_options;
            block_options.max_new_tokens = 128;
            block_options.stop_token_ids = std::set<uint32_t>{};
            bool stopped_after_verification = false;
            bool signalled_full = false;
            block_options.info_callback = [&](const std::string& key, double) {
                if (key == "tps" && !stopped_after_verification) {
                    stopped_after_verification = true;
                    first->stop_model();
                }
                if (key == "FULL") signalled_full = true;
            };
            const auto interrupted = first->run_model(sa, cancel_chat, block_options);
            require(!interrupted && stopped_after_verification && !signalled_full,
                    "DFlash reported successful completion or a full cache after cancellation");
            require(generate(sa, a) == gold_a, "DFlash block cancellation left stale state");
        }

        std::atomic<unsigned> default_chunks = 0;
        std::atomic<unsigned> default_events = 0;
        std::atomic<unsigned> request_chunks = 0;
        first->set_text_callback([&](const std::string&, bool, bool) { ++default_chunks; });
        first->set_info_callback([&](const std::string&, double) { ++default_events; });
        GenerationOptions options;
        options.max_new_tokens = 16;
        options.text_callback = [&](const std::string&, bool, bool) { ++request_chunks; };
        auto option_chat = first->create_chat();
        option_chat.add_query("Name three planets.");
        require(first->run_model(sa, option_chat, options).has_value(), "Request options failed");
        require(request_chunks > 0 && default_chunks == 0, "Request callback did not replace the default");
        require(default_events > 0, "Overriding text suppressed the default info callback");
        first->run_model(sa, option_chat, 16);
        require(default_chunks > 0, "Request callback leaked into a later request");
        first->set_text_callback([](const std::string&, bool, bool) {});
        first->set_info_callback([](const std::string&, double) {});
        std::cout << json{{"case", "request_callback_isolation"}, {"runtime_ok", true}}.dump() << '\n';

        options.info_callback = [](const std::string& key, double) {
            if (key == "ttft") throw std::runtime_error("Intentional request callback failure");
        };
        rejected([&] { first->run_model(sa, option_chat, options); }, "Callback exception was swallowed");
        require(generate(sa, a) == gold_a, "Callback exception left the next request corrupted");

        unsigned completions = 0;
        options.info_callback = [&](const std::string& key, double) {
            if (key == "END" || key == "FULL") {
                ++completions;
                first->stop_model();
            }
        };
        require(!first->run_model(sa, option_chat, options),
                "Cancellation from the terminal callback arrived after result assembly");
        require(completions == 1, "Streaming completion was delivered more than once");

        const auto timing = first->run_model_for_ttnt(sa, a);
        require(timing.generated_tokens > 1, "Nested prefill leaked its output limit into decode");
        require(generate(sa, a) == gold_a, "Performance request left stale prompt state");
        if (!draft) {
            const std::set<uint32_t> no_stops;
            const auto limit = static_cast<uint16_t>(a.size() + 4);
            const auto four_tokens = first->run_model(sa, a, limit, no_stops);
            require(four_tokens.size() == 4, "Ordinary generation ignored the token budget");
            for (uint16_t budget : {0, 1, 4, 40}) {
                const auto metrics = first->run_model_for_ttnt(sa, a, a.size() + budget, no_stops);
                require(metrics.generated_tokens == budget && metrics.token_durations.size() == budget,
                        "Ordinary timing must report exactly one sample per emitted token");
            }
            const auto stopped = first->run_model_for_ttnt(sa, a, limit, std::set{four_tokens.front()});
            require(stopped.generated_tokens == 1 && stopped.token_durations.size() == 1,
                    "An immediate stop token must have exactly one timing sample");
        }
        if (cfg.pipeline_cfg.return_logits && !draft) {
            const auto suffix = std::span<const uint32_t>(b).last(2);
            first->run_model_for_loglikelihood(sa, b, b.size() - 2, suffix);
            require(generate(sa, a) == gold_a, "Scoring left stale generation history");
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
        survivor->set_text_callback([](const std::string&, bool, bool) {});
        first->reset_session(sa);
        survivor->reset_session(0);
        auto concurrent_a = std::async(std::launch::async, [&] { return generate(sa, a); });
        auto concurrent_b = std::async(std::launch::async, [&] {
            return survivor->run_model(b, static_cast<uint16_t>(b.size() + 64));
        });
        const auto result_a = concurrent_a.get();
        const auto result_b = concurrent_b.get();
        require(result_a == gold_a, "Concurrent first workspace differs from its fresh reference");
        require(result_b == gold_b, "Concurrent second workspace differs from its fresh reference");
        std::cout << json{{"case", "concurrent_workspaces"}, {"runtime_ok", true}}.dump() << '\n';
        if (draft) {
            require(survivor->run_model(a, static_cast<uint16_t>(a.size() + 64)) == gold_a,
                    "Shared draft workspace changed the new target's output");
            require(generate(sa, a) == gold_a,
                    "Another target corrupted this target's paired draft context");
        }
        active_model = nullptr;
        auto detached_chat = first->create_chat();
        if (draft) first->create_session();
        if (draft) {
            active_model = survivor.get();
            survivor->set_info_callback([&](const std::string& key, double) {
                if (key == "END" || key == "FULL") first.reset();
            });
            auto teardown_chat = survivor->create_chat();
            teardown_chat.add_query("Name one planet.");
            require(survivor->run_model(teardown_chat, 16).has_value(),
                    "Callback teardown of another target sharing this draft failed");
            require(!first, "The callback did not release the other target");
            survivor->set_info_callback([](const std::string&, double) {});
        } else {
            first->set_info_callback([&](const std::string& key, double) {
                if (key == "END" || key == "FULL")
                    require(survivor->run_model(a, static_cast<uint16_t>(a.size() + 64)) == gold_a,
                            "Cross-model callback generation differs from the reference");
            });
            require(first->run_model(option_chat, 16).has_value(),
                    "Cross-model callback generation blocked completion");
            first.reset();
        }
        if (draft) {
            const auto released_slot = draft->create_session();
            draft->release_session(released_slot);
            draft.reset();
            require(!draft_files.expired(), "Surviving target released its draft runtime assets");
        }
        // The target retains language execution after the draft facade is released.
        active_model = survivor.get();
        detached_chat.add_query("Name a planet.");
        require(detached_chat.get_last_query() == "Name a planet.", "Chat retained a released helper");
        detached_chat.update_last_query("Name one planet only.");
        require(survivor->run_model(detached_chat, 16).has_value(), "Detached chat cannot use the surviving model");
        require(survivor->run_model(a, static_cast<uint16_t>(a.size() + 64)) == gold_a,
                "Releasing one wrapper unloaded or corrupted the surviving wrapper");
        active_model = nullptr;
        survivor.reset();
        require(draft_files.expired(), "Draft runtime assets leaked after the last target was released");
        std::cout << json{{"case", "shared_program_lifetime"}, {"runtime_ok", true}}.dump() << '\n';
        disconnect();
        connected = false;
    } catch (const std::exception& e) {
        if (connected) { try { disconnect(); } catch (...) {} }
        std::cerr << "Session generation test failed: " << e.what() << '\n';
        return 1;
    }
    std::cout << "Session generation test passed\n";
    return 0;
}
