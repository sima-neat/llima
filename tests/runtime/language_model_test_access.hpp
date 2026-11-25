#pragma once

#include "language/language_model.hpp"
#include "language/language_model_state.hpp"
#include <spdlog/sinks/base_sink.h>

namespace simaai::llima {

struct LanguageModelTestAccess {
    static uint16_t kv_cache_len(const LanguageModel& model) { return model._cache->kv_cache_len; }
    static void reset_cache(LanguageModel& model) { model.invalidate_prompt_reuse(); }
    static void truncate_kv(LanguageModel& model, uint16_t length) { model.set_kv_cache_len(length); }
};

// Observe the actual grouped/pointwise prefill calls without adding a runtime
// API just for tests. MTP verification uses its own target-batch entry point.
class PrefillTrace : public spdlog::sinks::base_sink<std::mutex> {
public:
    static std::shared_ptr<PrefillTrace> install() {
        auto trace = std::make_shared<PrefillTrace>();
        auto previous = spdlog::get("llima");
        auto logger = std::make_shared<spdlog::logger>(
            "llima", previous->sinks().begin(), previous->sinks().end());
        logger->sinks().push_back(trace);
        logger->set_level(spdlog::level::info);
        spdlog::drop("llima");
        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);
        return trace;
    }

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

} // namespace simaai::llima
