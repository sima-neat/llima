#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/sinks/base_sink.h>

#include "language_model.hpp"
#include "setup.hpp"

namespace {

using namespace simaai::llima;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// Observe the actual grouped/pointwise prefill calls without adding a runtime
// API just for tests. MTP verification uses its own target-batch entry point.
class PrefillTrace : public spdlog::sinks::base_sink<std::mutex> {
public:
    void reset(LanguageModel* cancel_on_prefill = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        positions_.clear();
        cancel_on_prefill_ = cancel_on_prefill;
    }

    std::vector<size_t> positions() {
        std::lock_guard<std::mutex> lock(mutex_);
        return positions_;
    }

private:
    void sink_it_(const spdlog::details::log_msg& message) override {
        const std::string text(message.payload.data(), message.payload.size());
        constexpr std::string_view prefix = "Processing token no. ";
        if (!text.starts_with(prefix)) return;
        positions_.push_back(std::stoul(text.substr(prefix.size())));
        if (cancel_on_prefill_ != nullptr) {
            cancel_on_prefill_->stop_model();
            cancel_on_prefill_ = nullptr;
        }
    }

    void flush_() override {}

    std::vector<size_t> positions_;
    LanguageModel* cancel_on_prefill_ = nullptr;
};

struct Metrics {
    std::optional<uint32_t> cached_tokens;
    std::optional<double> ttft;
};

struct Result {
    std::optional<std::vector<uint32_t>> tokens;
    Metrics metrics;
    std::vector<size_t> prefill_positions;
};

} // namespace

// Deliberately manual until a compiled MTP pair is available in the CI fixture
// manifest. Both arguments are individual deployed target/draft directories.
int main(int argc, char** argv) {
    using namespace simaai::llima;
    bool connected = false;
    try {
        require(argc == 3, "Usage: sima_lmm_gemma4_mtp_cache_test TARGET_DIR DRAFT_DIR");
        const std::filesystem::path target_dir(argv[1]);
        const auto cfg = nlohmann::json::parse(
            std::ifstream(target_dir / "devkit/vlm_config.json")
        ).get<VlmConfig>();
        require(cfg.lm_cfg.is_gemma4_mtp_target(), "Expected a Gemma4 MTP target");
        const size_t group = cfg.pipeline_cfg.input_token_group_size;
        const auto& offsets = cfg.pipeline_cfg.input_token_group_offsets;
        require(group > 17 && offsets.has_value(), "Test requires grouped prefill");
        for (size_t offset : {size_t{0}, group, 2 * group}) {
            require(std::find(offsets->begin(), offsets->end(), offset) != offsets->end(),
                "Test requires the first three prefill group offsets");
        }
        require(3 * group + 64 < cfg.pipeline_cfg.max_num_tokens,
            "Test prompts exceed the model's context capacity");
        const auto tokenizer = Tokenizer::from_hf_json(target_dir / "devkit/tokenizer.json");
        const auto seed = tokenizer->encode("The capital of Germany is Berlin. ", true);
        const auto other = tokenizer->encode("Different", false);
        require(!seed.empty() && !other.empty(), "Tokenizer produced no test tokens");
        const uint32_t replacement = other.front();

        // Gemma byte-fallback encodes the ideographic space as three token
        // IDs. A draft/target boundary inside it must not flush an incomplete
        // UTF-8 prefix as replacement characters.
        const std::string ideographic_space = "\xE3\x80\x80";
        const auto byte_tokens = tokenizer->encode(ideographic_space, false);
        require(byte_tokens.size() > 1, "Test requires a split UTF-8 token sequence");
        std::string streamed_space;
        {
            TextStreamer utf8_streamer(tokenizer.get());
            utf8_streamer.set_text_callback(
                [&](const std::string& text, bool, bool) { streamed_space += text; }
            );
            for (size_t i = 0; i < byte_tokens.size(); ++i) {
                utf8_streamer.put(byte_tokens[i], i == 0);
            }
            utf8_streamer.end();
        }
        require(streamed_space == ideographic_space,
            "Draft provenance boundary corrupted a split UTF-8 character");

        connect({}, "/tmp/llima-gemma4-mtp-cache.log", spdlog::level::info);
        connected = true;
        auto trace = std::make_shared<PrefillTrace>();
        // Model loggers clone this synchronous logger so trace assertions and
        // cancellation do not race the asynchronous file-logging worker.
        auto old_logger = spdlog::get("llima");
        auto logger = std::make_shared<spdlog::logger>(
            "llima", old_logger->sinks().begin(), old_logger->sinks().end()
        );
        logger->sinks().push_back(trace);
        logger->set_level(spdlog::level::info);
        spdlog::drop("llima");
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);

        {
            Metrics metrics;
            std::mutex metrics_mutex;
            TextStreamer streamer(tokenizer.get());
            streamer.set_text_callback([](const std::string&, bool, bool) {});
            streamer.set_info_callback([&](const std::string& name, double value) {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (name == "cached_prompt_tokens") metrics.cached_tokens = static_cast<uint32_t>(value);
                else if (name == "ttft") metrics.ttft = value;
            });
            LanguageModel target(target_dir, {}, std::nullopt, std::nullopt, streamer);
            LanguageModel draft(argv[2], {}, std::nullopt, std::nullopt, streamer);

            auto run = [&](const std::vector<uint32_t>& prompt,
                           uint16_t generated = 16, bool cancel = false) {
                target.create_input_buffers(prompt);
                {
                    std::lock_guard<std::mutex> lock(metrics_mutex);
                    metrics = {};
                }
                trace->reset(cancel ? &target : nullptr);
                auto tokens = target.run_model_gemma4_mtp(
                    draft, prompt, static_cast<uint16_t>(prompt.size() + generated),
                    std::nullopt, nullptr
                );
                std::lock_guard<std::mutex> lock(metrics_mutex);
                require(metrics.cached_tokens.has_value(), "Missing cache reuse metric");
                if (!cancel) {
                    require(
                        tokens.has_value() && !tokens->empty()
                            && tokens->size() <= generated,
                        "MTP produced no tokens or exceeded the requested count"
                    );
                    require(metrics.ttft.has_value(), "Missing TTFT metric");
                    std::cout << "prompt=" << prompt.size()
                              << " cached=" << *metrics.cached_tokens
                              << " prefill_calls=" << trace->positions().size()
                              << " ttft=" << *metrics.ttft << '\n';
                }
                return Result{std::move(tokens), metrics, trace->positions()};
            };
            auto reset_cache = [&] {
                target.clear_cached_token_ids();
                draft.clear_cached_token_ids();
                target.set_kv_cache_len(0);
                draft.set_kv_cache_len(0);
            };
            auto cold_reference = [&](const std::vector<uint32_t>& prompt) {
                reset_cache();
                auto result = run(prompt);
                require(*result.metrics.cached_tokens == 0,
                    "Reference must start from a cold cache");
                return result;
            };
            auto make_prompt = [&](size_t size) {
                std::vector<uint32_t> prompt(size);
                for (size_t i = 0; i < size; ++i) prompt[i] = seed[i % seed.size()];
                return prompt;
            };

            // Verify group boundaries, including exact matches with no decode
            // round (one output token), where a saved next token can be stale.
            for (size_t length : {size_t{1}, group - 1, group, group + 1, 2 * group + 1}) {
                reset_cache();
                const auto prompt = make_prompt(length);
                const auto cold = run(prompt, 1);
                const auto warm = run(prompt, 1);
                require(cold.tokens == warm.tokens, "Repeated prompt changed its next token");
                require(*cold.metrics.cached_tokens == 0,
                    "Initial prompt should use a cold cache");
                require(*warm.metrics.cached_tokens == length,
                    "Repeated prompt should match its committed prefix");
                require(warm.prefill_positions == std::vector<size_t>{(length - 1) / group * group},
                    "Full match must run only the final prefill group");
            }

            reset_cache();
            const auto prompt_a = make_prompt(2 * group + 1);
            const auto a1 = run(prompt_a);

            // Include all returned tokens, including the final unprocessed
            // bonus. Reuse must stop at the committed KV length, even if the
            // physical buffer still contains extra verification rows.
            auto continuation = prompt_a;
            continuation.insert(continuation.end(), a1.tokens->begin(), a1.tokens->end());
            const auto committed = target.get_kv_cache_len();
            const auto continuation_warm = run(continuation);
            const auto continuation_cold = cold_reference(continuation);
            require(continuation_warm.tokens == continuation_cold.tokens,
                "Continuation differs between warm and cold prefill");
            require(*continuation_warm.metrics.cached_tokens == committed,
                "Continuation reused uncommitted speculative KV");

            auto partial = prompt_a;
            const size_t mismatch = group + 17;
            require(partial[mismatch] != replacement, "Test needs a distinct suffix token");
            partial[mismatch] = replacement;
            reset_cache();
            run(prompt_a);
            const auto partial_warm = run(partial);
            const auto partial_cold = cold_reference(partial);
            require(partial_warm.tokens == partial_cold.tokens,
                "Partial-prefix reuse changed generated tokens");
            require(*partial_warm.metrics.cached_tokens == mismatch,
                "Partial-prefix match has the wrong length");
            require(partial_warm.prefill_positions == std::vector<size_t>{group, 2 * group},
                "Partial-prefix reuse did not skip the first prefill group");

            // A logical KV truncation must win over a longer token history.
            target.set_kv_cache_len(static_cast<uint16_t>(group));
            const auto truncated = run(partial);
            require(*truncated.metrics.cached_tokens == group && truncated.tokens == partial_cold.tokens,
                "Reuse exceeded the valid KV length after truncation");

            // Stop synchronously at the first prefill dispatch, then verify the
            // single cache is invalidated and can recover cleanly.
            const auto reference = cold_reference(prompt_a);
            reset_cache();
            const auto interrupted = run(prompt_a, 16, true);
            require(!interrupted.tokens.has_value(), "Prefill ignored cancellation");
            require(target.get_kv_cache_len() == 0, "Interrupted target KV remains valid");
            const auto recovered = run(prompt_a);
            require(*recovered.metrics.cached_tokens == 0,
                "Recovery must recompute from an empty cache");
            require(recovered.tokens == reference.tokens, "Recovery changed generated tokens");

            // Exercise reuse across sliding-window and long-context mask
            // buckets, rather than testing only short system prompts.
            reset_cache();
            const size_t long_group = std::min<size_t>(4096, cfg.pipeline_cfg.max_num_tokens / 2)
                / group * group;
            require(std::find(offsets->begin(), offsets->end(), long_group) != offsets->end(),
                "Long-prefix test requires its final prefill group offset");
            auto long_prompt = make_prompt(long_group + 1);
            const auto long_cold = run(long_prompt);
            const auto long_warm = run(long_prompt);
            require(long_warm.tokens == long_cold.tokens,
                "Long-prefix reuse changed generated tokens");
            require(*long_warm.metrics.cached_tokens == long_prompt.size()
                    && long_warm.prefill_positions == std::vector<size_t>{long_group},
                "Long-prefix reuse did not skip all earlier prefill groups");
        }
        disconnect();
        connected = false;
        std::cout << "Gemma4 MTP prefix-cache tests passed\n";
    } catch (const std::exception& error) {
        if (connected) { try { disconnect(); } catch (...) {} }
        std::cerr << "Gemma4 MTP prefix-cache test failed: " << error.what() << '\n';
        return 1;
    }
}
