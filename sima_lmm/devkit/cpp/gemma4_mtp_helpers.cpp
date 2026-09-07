#include "gemma4_mtp_helpers.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace simaai {
namespace llima {
namespace gemma4_mtp_helpers {

uint16_t draft_query_position(uint16_t shared_kv_len, size_t max_num_tokens) {
    if (shared_kv_len == 0) {
        throw std::runtime_error(
            "Gemma4 MTP draft requires at least one target KV row"
        );
    }
    const uint16_t position_id = static_cast<uint16_t>(shared_kv_len - 1);
    if (position_id >= max_num_tokens) {
        throw std::runtime_error("Gemma4 MTP draft position exceeds cache capacity");
    }
    return position_id;
}

std::vector<uint32_t> select_candidate_tokens(
    std::span<const Eigen::bfloat16> centroid_logits,
    std::span<const uint32_t> token_ordering,
    uint32_t top_k_centroids
) {
    if (
        centroid_logits.empty() || token_ordering.empty()
        || token_ordering.size() % centroid_logits.size() != 0
    ) {
        throw std::runtime_error(
            "Gemma4 masked lm_head vocabulary must divide evenly across centroids"
        );
    }
    if (top_k_centroids == 0 || top_k_centroids > centroid_logits.size()) {
        throw std::runtime_error("Invalid Gemma4 masked lm_head centroid top-k");
    }

    std::vector<uint32_t> centroid_indices(centroid_logits.size());
    std::iota(centroid_indices.begin(), centroid_indices.end(), 0);
    std::partial_sort(
        centroid_indices.begin(),
        centroid_indices.begin() + top_k_centroids,
        centroid_indices.end(),
        [&](uint32_t left, uint32_t right) {
            const float left_value = static_cast<float>(centroid_logits[left]);
            const float right_value = static_cast<float>(centroid_logits[right]);
            return left_value != right_value
                ? left_value > right_value
                : left < right;
        }
    );

    const size_t tokens_per_centroid = (
        token_ordering.size() / centroid_logits.size()
    );
    std::vector<uint32_t> candidate_tokens;
    candidate_tokens.reserve(
        static_cast<size_t>(top_k_centroids) * tokens_per_centroid
    );
    for (uint32_t rank = 0; rank < top_k_centroids; ++rank) {
        const size_t ordering_begin = (
            static_cast<size_t>(centroid_indices[rank]) * tokens_per_centroid
        );
        for (size_t offset = 0; offset < tokens_per_centroid; ++offset) {
            const uint32_t token = token_ordering[ordering_begin + offset];
            if (token >= token_ordering.size()) {
                throw std::runtime_error(
                    "Gemma4 token_ordering contains an out-of-range token ID"
                );
            }
            candidate_tokens.emplace_back(token);
        }
    }
    return candidate_tokens;
}

uint32_t select_masked_token(
    std::span<const Eigen::bfloat16> token_logits,
    std::span<const Eigen::bfloat16> centroid_logits,
    std::span<const uint32_t> token_ordering,
    uint32_t top_k_centroids
) {
    if (token_logits.empty() || token_logits.size() != token_ordering.size()) {
        throw std::runtime_error(
            "Gemma4 masked lm_head requires one token-ordering entry per logit"
        );
    }

    const auto candidate_tokens = select_candidate_tokens(
        centroid_logits, token_ordering, top_k_centroids
    );
    uint32_t best_token = std::numeric_limits<uint32_t>::max();
    float best_value = -std::numeric_limits<float>::infinity();
    for (const uint32_t token : candidate_tokens) {
        const float value = static_cast<float>(token_logits[token]);
        if (
            value > best_value
            || (value == best_value && token < best_token)
        ) {
            best_value = value;
            best_token = token;
        }
    }
    return best_token;
}

} // namespace gemma4_mtp_helpers
} // namespace llima
} // namespace simaai
