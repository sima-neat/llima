#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace simaai::llima {

using SessionId = uint64_t;

struct LogLikelihoodResult {
    double logprob;
    bool is_greedy;
};

struct GenerationPerformanceResult {
    std::vector<double> token_durations;
    uint32_t generated_tokens = 0;
    std::optional<uint32_t> accepted_draft_tokens;
};

} // namespace simaai::llima
