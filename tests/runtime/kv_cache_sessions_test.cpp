#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include <spdlog/common.h>

#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";
constexpr const char* kExpectedGermany = "The capital of Germany is Berlin.";

struct CacheMetrics {
    std::optional<uint32_t> cached_prompt_tokens;
    std::optional<bool> cache_created;
};

std::string run_session(
    simaai::llima::VisionLanguageModel& model,
    const simaai::llima::Chat& chat,
    const std::string& cache_id,
    CacheMetrics& metrics,
    std::mutex& metrics_mutex
) {
    {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        metrics = {};
    }
    auto result = model.run_model(chat, 24, cache_id);
    if (!result.has_value()) {
        throw std::runtime_error("Text generation was interrupted");
    }
    {
        std::lock_guard<std::mutex> lock(metrics_mutex);
        if (!metrics.cached_prompt_tokens.has_value() || !metrics.cache_created.has_value()) {
            throw std::runtime_error("KV cache observability metrics were not reported");
        }
    }
    return simaai::llima::test::trim(std::move(*result));
}

}  // namespace

int main() {
    bool connected = false;
    try {
        const std::filesystem::path model_dir =
            simaai::llima::test::resolve_model_dir(
                kModelEnv,
                simaai::llima::test::kDefaultTextModelName,
                "LLiMa text",
                "devkit/vlm_config.json"
            );
        std::cout << "LLIMA_KV_SESSIONS model_dir=" << model_dir << '\n';

        simaai::llima::connect(
            {},
            "/tmp/sima_lmm_kv_cache_sessions_test.log",
            spdlog::level::info
        );
        connected = true;

        CacheMetrics metrics;
        std::mutex metrics_mutex;
        {
            simaai::llima::VisionLanguageModel model(
                model_dir, std::nullopt, std::nullopt, 2
            );
            model.set_info_callback(
                [&](const std::string& metric, double value) {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    if (metric == "cached_prompt_tokens") {
                        metrics.cached_prompt_tokens = static_cast<uint32_t>(value);
                    } else if (metric == "cache_created") {
                        metrics.cache_created = value != 0.0;
                    }
                }
            );

            auto chat_a = model.create_chat();
            chat_a.set_system_prompt("You are concise.");
            chat_a.add_query("What is the capital of Germany?");
            auto chat_b = model.create_chat();
            chat_b.set_system_prompt("Answer in one short sentence.");
            chat_b.add_query("What is the capital of France?");

            const auto a1 = run_session(
                model, chat_a, "session-a", metrics, metrics_mutex
            );
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (*metrics.cached_prompt_tokens != 0 || !*metrics.cache_created) {
                    throw std::runtime_error("A1 should create a cold cache");
                }
            }
            const auto b1 = run_session(
                model, chat_b, "session-b", metrics, metrics_mutex
            );
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (*metrics.cached_prompt_tokens != 0 || !*metrics.cache_created) {
                    throw std::runtime_error("B1 should create a separate cold cache");
                }
            }
            const auto a2 = run_session(
                model, chat_a, "session-a", metrics, metrics_mutex
            );
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (*metrics.cached_prompt_tokens == 0 || *metrics.cache_created) {
                    throw std::runtime_error("A2 should reuse session-a's cached prefix");
                }
            }
            const auto b2 = run_session(
                model, chat_b, "session-b", metrics, metrics_mutex
            );
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (*metrics.cached_prompt_tokens == 0 || *metrics.cache_created) {
                    throw std::runtime_error("B2 should reuse session-b's cached prefix");
                }
            }

            if (a1 != kExpectedGermany || a2 != a1 || b2 != b1 || b1.empty()) {
                throw std::runtime_error("Alternating cache sessions changed generated output");
            }
            if (model.kv_cache_count() != 2) {
                throw std::runtime_error("Expected two assigned KV cache slots");
            }

            bool capacity_rejected = false;
            try {
                (void)model.run_model(chat_a, 24, "session-c");
            } catch (const simaai::llima::KVCacheCapacityError&) {
                capacity_rejected = true;
            }
            if (!capacity_rejected) {
                throw std::runtime_error("A third cache ID should exhaust capacity");
            }
            if (!model.remove_kv_cache("session-a")) {
                throw std::runtime_error("Expected session-a cache removal to succeed");
            }
            const auto c1 = run_session(
                model, chat_a, "session-c", metrics, metrics_mutex
            );
            if (c1 != a1 || model.kv_cache_count() != 2) {
                throw std::runtime_error("Released cache slot was not reused correctly");
            }
            model.clear_kv_caches();
            if (model.kv_cache_count() != 0) {
                throw std::runtime_error("Expected clear_kv_caches to release all associations");
            }
        }

        simaai::llima::disconnect();
        connected = false;
    } catch (const std::exception& error) {
        if (connected) {
            try {
                simaai::llima::disconnect();
            } catch (...) {
            }
        }
        std::cerr << "KV cache sessions test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "KV cache sessions test passed\n";
    return 0;
}
