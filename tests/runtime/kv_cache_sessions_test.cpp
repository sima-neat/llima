#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

#include <iostream>
#include <optional>
#include <stdexcept>
#include <stop_token>

using namespace simaai::llima;

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void exercise(const std::filesystem::path& path) {
    VisionLanguageModel model(path, ModelOptions{2});
    require(model.kv_cache_count() == 0, "construction retained a warmup context");
    require(model.kv_cache_bytes_per_slot() > 0, "missing per-slot memory accounting");
    auto chat = model.create_chat();
    chat.set_system_prompt("You are concise.");
    chat.add_query("What is the capital of Germany?");
    auto other = model.create_chat();
    other.set_system_prompt("Answer briefly.");
    other.add_query("What is the capital of France?");
    auto run = [&](const Chat& prompt, std::optional<std::string> id, bool created,
                   bool warm) {
        GenerationOptions options;
        options.max_new_tokens = 24;
        options.performance_summary = false;
        options.cache_id = id;
        std::optional<bool> callback_created;
        std::optional<uint32_t> callback_cached;
        options.callbacks.info = [&](const std::string& name, double value) {
            if (name == "cache_created") callback_created = value != 0;
            if (name == "cached_prompt_tokens") callback_cached = static_cast<uint32_t>(value);
        };
        options.callbacks.text = [](const std::string&, bool, bool) {};
        auto result = model.generate(prompt, options);
        require(result.finish != ChatResult::Finish::Cancelled, "unexpected cancellation");
        require(!test::trim(result.text).empty(), "missing generated text");
        require(result.performance.cache_created == created, "wrong cache creation metric");
        require((result.performance.cached_prompt_tokens > 0) == warm, "wrong cached-token metric");
        require(callback_created == created, "cache creation callback disagrees with result");
        require(callback_cached == result.performance.cached_prompt_tokens,
                "cached-token callback disagrees with result");
        return result.text;
    };
    const auto a = run(chat, "a", true, false);
    const auto b = run(other, "b", true, false);
    for (int i = 0; i < 3; ++i) {
        require(run(chat, "a", false, true) == a, "switching contexts changed A output");
        require(run(other, "b", false, true) == b, "switching contexts changed B output");
    }
    for (const auto id : {std::optional<std::string>{"c"}, std::optional<std::string>{}}) {
        bool capacity = false;
        try { run(chat, id, true, false); }
        catch (const KVCacheCapacityError&) { capacity = true; }
        require(capacity, "new context bypassed capacity limit");
    }
    require(model.kv_cache_count() == 2, "capacity failure changed assignments");
    require(!model.remove_kv_cache("missing"), "missing cache was removed");
    require(model.remove_kv_cache("a"), "failed to remove A");
    require(run(chat, std::nullopt, true, false) == a, "legacy context changed output");
    require(run(other, "b", false, true) == b, "removing A invalidated B");
    model.clear_kv_caches();
    require(model.kv_cache_count() == 0, "clear did not release assignments");
    require(run(chat, "a", true, false) == a, "recreated A changed output");
    require(run(other, "b", true, false) == b, "recreated B changed output");

    // Failed/cancelled requests invalidate only their selected context.
    std::stop_source stop;
    stop.request_stop();
    GenerationOptions cancelled;
    cancelled.cache_id = "a";
    cancelled.stop = stop.get_token();
    cancelled.max_new_tokens = 24;
    require(model.generate(chat, cancelled).finish == ChatResult::Finish::Cancelled,
            "pre-cancelled request completed");
    require(run(other, "b", false, true) == b, "cancelling A invalidated B");
    require(run(chat, "a", false, true) == a, "pre-cancelled A lost its cache");
    GenerationOptions active_cancel;
    active_cancel.max_new_tokens = 512;
    active_cancel.cache_id = "a";
    bool busy_rejected = false;
    active_cancel.callbacks.info = [&](const std::string& key, double) {
        if (key != "ttft") return;
        try { model.clear_kv_caches(); }
        catch (const std::logic_error&) { busy_rejected = true; }
        model.stop_model();
    };
    auto story = model.create_chat();
    story.add_query("Write a long detailed story about a journey through space.");
    require(model.generate(story, active_cancel).finish == ChatResult::Finish::Cancelled,
            "active cancellation failed");
    require(busy_rejected, "cache control from a callback did not reject the busy model");
    require(run(other, "b", false, true) == b, "active cancellation of A invalidated B");
    require(run(chat, "a", false, false) == a, "cancelled A was not safely rebuilt");
    bool invalid = false;
    try { run(chat, "", false, false); }
    catch (const std::invalid_argument&) { invalid = true; }
    require(invalid, "empty cache ID accepted");
    model.clear_kv_caches();
}
} // namespace

int main() {
    try {
        const auto path = test::resolve_model_dir("SIMA_TEST_LLIMA_TEXT_MODEL",
            test::kDefaultTextModelName, "text", "devkit/vlm_config.json");
        connect({}, "/tmp/sima_lmm_cache_sessions_test.log", spdlog::level::info);
        try { exercise(path); }
        catch (...) { disconnect(); throw; }
        disconnect();
        std::cout << "KV_CACHE_SESSIONS result=PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "KV_CACHE_SESSIONS result=FAIL error=" << error.what() << '\n';
        return 1;
    }
}
