#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::llima {

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
