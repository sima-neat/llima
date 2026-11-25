#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace simaai::llima {

// Timing/status events and finalized text chunks for one request.
// Text callbacks receive (text, stream_end, from_draft).
using InfoCallback = std::function<void(const std::string&, double)>;
using TextCallback = std::function<void(const std::string&, bool, bool)>;

// Text and status callbacks selected for one generation or transcription request.
struct StreamCallbacks {
    std::optional<InfoCallback> info;
    std::optional<TextCallback> text;
};

using PreservedTokens = std::vector<std::pair<uint32_t, std::string>>;

// Continuation log-probability and whether every scored token matches greedy decoding.
struct LogLikelihoodResult {
    double logprob;
    bool is_greedy;
};

// Timing and token counts collected for one generation, including draft acceptance.
struct GenerationPerformanceResult {
    std::vector<double> token_durations;
    uint32_t generated_tokens = 0;
    std::optional<uint32_t> accepted_draft_tokens;
};

// Completed generation output: token IDs, decoded text, finish reason and performance.
struct ChatResult {
    enum class Finish { Stop, Length, Cancelled };
    std::vector<uint32_t> token_ids;
    std::string text;
    Finish finish = Finish::Stop;
    GenerationPerformanceResult performance;
};

} // namespace simaai::llima
