#pragma once

#include "language_model_state.hpp"

namespace simaai::llima {

// Owns cancellation, streaming, and cleanup for one language operation.
// The caller keeps the target and optional draft alive and exclusively available.
class LanguageRequest {
  public:
    LanguageRequest(LanguageModel& target, LanguageModel* draft = nullptr,
                    TextStreamer::RequestOptions stream = {}, std::stop_token stop = {},
                    std::optional<std::set<uint32_t>> stop_ids = std::nullopt);
    ~LanguageRequest();
    LanguageRequest(const LanguageRequest&) = delete;
    LanguageRequest& operator=(const LanguageRequest&) = delete;

    // Wait for callbacks before recording the final cancellation state.
    void finish();
    bool cancelled() const { return _completed ? _cancelled_at_finish : _request.cancelled(); }
    bool length_limit_reached() const { return _request.length_limit_reached; }
    LanguageModel* draft() const { return _draft; }

  private:
    LanguageModel& _target;
    LanguageModel* _draft;
    LanguageRequestState _request;
    bool _completed = false;
    bool _cancelled_at_finish = false;
};

} // namespace simaai::llima
