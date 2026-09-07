#ifndef _SIMA_LLIMA_GEMMA4_MTP_HELPERS_
#define _SIMA_LLIMA_GEMMA4_MTP_HELPERS_

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include <Eigen/Core>

namespace simaai {
namespace llima {
namespace gemma4_mtp_helpers {

uint16_t draft_query_position(uint16_t shared_kv_len, size_t max_num_tokens);

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
