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

struct StreamCallbacks {
    std::optional<InfoCallback> info;
    std::optional<TextCallback> text;
};

using PreservedTokens = std::vector<std::pair<uint32_t, std::string>>;

struct LogLikelihoodResult {
    double logprob;
    bool is_greedy;
};

struct GenerationPerformanceResult {
    std::vector<double> token_durations;
    uint32_t generated_tokens = 0;
    std::optional<uint32_t> accepted_draft_tokens;
};

struct ChatResult {
    enum class Finish { Stop, Length, Cancelled };
    std::vector<uint32_t> token_ids;
    std::string text;
    Finish finish = Finish::Stop;
    GenerationPerformanceResult performance;
};

} // namespace simaai::llima
