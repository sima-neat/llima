#include "gemma4_mtp_helpers.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace simaai {
namespace llima {
namespace gemma4_mtp_helpers {

uint16_t draft_query_position(size_t input_length, size_t max_num_tokens) {
    if (input_length == 0) {
        throw std::runtime_error("Gemma4 MTP draft requires at least one input token");
    }
    const size_t position_id = input_length - 1;
    if (
        position_id >= max_num_tokens
        || position_id > std::numeric_limits<uint16_t>::max()
    ) {
        throw std::runtime_error("Gemma4 MTP draft position exceeds cache capacity");
    }
    return static_cast<uint16_t>(position_id);
}

uint16_t draft_visible_shared_kv_len(
    size_t input_length,
    size_t available_shared_kv_len,
    size_t max_num_tokens
) {
    if (input_length == 0 || available_shared_kv_len == 0) {
        throw std::runtime_error(
            "Gemma4 MTP draft requires at least one target KV row"
        );
    }
    if (
        input_length > max_num_tokens
        || available_shared_kv_len > max_num_tokens
        || max_num_tokens > std::numeric_limits<uint16_t>::max()
    ) {
        throw std::runtime_error("Gemma4 MTP shared KV length exceeds cache capacity");
    }
    // The target may have computed speculative rows beyond the accepted prefix.
    // Transformers exposes those rows only through the current input length. A
    // partial rejection therefore makes the visible KV length one greater than
    // the assistant query position, while initial/full-acceptance rounds remain
    // bounded by the number of rows the target actually produced.
    return static_cast<uint16_t>(std::min(input_length, available_shared_kv_len));
}

std::vector<std::pair<uint32_t, bool>> resolve_draft_tokens(
    std::span<const uint32_t> draft_token_ids,
    std::span<const uint32_t> target_next_token_ids
) {
    if (
        draft_token_ids.empty()
        || target_next_token_ids.size() != draft_token_ids.size() + 1
    ) {
        throw std::runtime_error(
            "Gemma4 MTP verification requires one target result beyond the draft"
        );
    }

    std::vector<std::pair<uint32_t, bool>> emitted_tokens;
    emitted_tokens.reserve(draft_token_ids.size() + 1);
    for (size_t depth = 0; depth < draft_token_ids.size(); ++depth) {
        if (draft_token_ids[depth] != target_next_token_ids[depth]) {
            emitted_tokens.emplace_back(target_next_token_ids[depth], false);
            return emitted_tokens;
        }
        emitted_tokens.emplace_back(draft_token_ids[depth], true);
    }

    // When every proposal matches, the last verification row supplies the
    // target's bonus token. It remains unprocessed until the next round.
    emitted_tokens.emplace_back(target_next_token_ids.back(), false);
    return emitted_tokens;
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
