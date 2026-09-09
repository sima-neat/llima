#ifndef _SIMA_LLIMA_GEMMA4_MTP_HELPERS_
#define _SIMA_LLIMA_GEMMA4_MTP_HELPERS_

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace simaai {
namespace llima {
namespace gemma4_mtp_helpers {

uint16_t draft_query_position(size_t input_length, size_t max_num_tokens);

uint16_t draft_visible_shared_kv_len(
    size_t input_length,
    size_t available_shared_kv_len,
    size_t max_num_tokens
);

// Pack rows at the compiled cache graph's context width, with columns
// relative to the bound KV slice rather than the full model-context buffer.
std::vector<Eigen::bfloat16> build_causal_mask(
    uint16_t num_tokens,
    uint16_t first_visible_token_count,
    uint16_t valid_tokens,
    uint16_t cache_token_idx_begin,
    uint16_t context_length
);

std::vector<std::pair<uint32_t, bool>> resolve_draft_tokens(
    std::span<const uint32_t> draft_token_ids,
    std::span<const uint32_t> target_next_token_ids
);

std::vector<uint32_t> select_candidate_tokens(
    std::span<const Eigen::bfloat16> centroid_logits,
    std::span<const uint32_t> token_ordering,
    uint32_t top_k_centroids
);

uint32_t select_masked_token(
    std::span<const Eigen::bfloat16> token_logits,
    std::span<const Eigen::bfloat16> centroid_logits,
    std::span<const uint32_t> token_ordering,
    uint32_t top_k_centroids
);

} // namespace gemma4_mtp_helpers
} // namespace llima
} // namespace simaai

#endif
