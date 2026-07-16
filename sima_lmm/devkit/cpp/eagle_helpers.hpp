//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2026 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#ifndef _SIMA_LLIMA_EAGLE_HELPERS_
#define _SIMA_LLIMA_EAGLE_HELPERS_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <omp.h>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

namespace simaai {
namespace llima {
namespace eagle_helpers {

// EAGLE3 tree attention mask: 4D shape (dim0, dim1, rows, cols).
// dim0 and dim1 are typically 1; the meaningful axes are rows (queries) x cols (keys)
struct EagleTreeMask {
    std::vector<std::vector<std::vector<std::vector<float>>>> data;
};


// Build a causal attention mask of shape (tgt_len, past_kv_len + tgt_len) in bfloat16.
// Past KV columns are all 0 (visible). Among the tgt_len new positions, position i can
// attend to positions [0, i] within the new block (lower triangular); rest is -inf.
inline std::vector<Eigen::bfloat16> make_causal_mask(
    size_t tgt_len, size_t past_kv_len
) {
    const size_t total_cols = past_kv_len + tgt_len;
    std::vector<Eigen::bfloat16> mask(tgt_len * total_cols);
    const Eigen::bfloat16 zero{0.0f};
    const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};

    for (size_t i = 0; i < tgt_len; ++i) {
        Eigen::bfloat16* row = mask.data() + i * total_cols;
        const size_t allowed_end = past_kv_len + i + 1;  // exclusive upper bound
        for (size_t j = 0; j < allowed_end; ++j) row[j] = zero;
        for (size_t j = allowed_end; j < total_cols; ++j) row[j] = neg_inf;
    }
    return mask;
}

// Overlay a tree mask onto the bottom-right (tree_rows × tree_cols) region of
// `mask`. Wherever tree[r,c] == 0, the corresponding mask slot is set to -inf.
inline void apply_tree_mask(
    std::vector<Eigen::bfloat16>& mask,
    size_t mask_rows, size_t mask_cols,
    const uint8_t* tree, size_t tree_rows, size_t tree_cols
) {
    const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
    const size_t row_offset = mask_rows - tree_rows;
    const size_t col_offset = mask_cols - tree_cols;
    for (size_t r = 0; r < tree_rows; ++r) {
        Eigen::bfloat16* mask_row = mask.data() + (row_offset + r) * mask_cols;
        const uint8_t* tree_row = tree + r * tree_cols;
        for (size_t c = 0; c < tree_cols; ++c) {
            if (tree_row[c] == 0) {
                mask_row[col_offset + c] = neg_inf;
            }
        }
    }
}

// Build the combined attention mask in row-major bf16. Steps:
//   1. Causal mask when seq_length > 1; otherwise all-zero (visible) mask.
//   2. Apply explicit attention_mask: cols with mask[c]==0 → -inf in every row.
//   3. Overlay tree_mask at the bottom-right (reads from [0][0][r][c]).
// Does NOT pad to (num_tokens, ...); caller pads per-stage.
inline std::vector<Eigen::bfloat16> prepare_attention_mask(
    const std::optional<std::vector<uint8_t>>& attention_mask,
    size_t seq_length,
    size_t past_key_values_length,
    const std::shared_ptr<EagleTreeMask>& tree_mask
) {
    const size_t seq_len_with_past = past_key_values_length + seq_length;

    std::vector<Eigen::bfloat16> mask;
    if (seq_length > 1) {
        mask = make_causal_mask(seq_length, past_key_values_length);
    } else {
        mask.assign(seq_length * seq_len_with_past, Eigen::bfloat16{0.0f});
    }

    if (attention_mask.has_value()) {
        const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
        const auto& am = attention_mask.value();
        const size_t src_len = am.size();  // expected == seq_len_with_past
        for (size_t r = 0; r < seq_length; ++r) {
            for (size_t c = 0; c < src_len; ++c) {
                if (am[c] == 0) {
                    mask[r * seq_len_with_past + c] = neg_inf;
                }
            }
        }
    }

    if (tree_mask && !tree_mask->data.empty()
        && !tree_mask->data[0].empty()
        && !tree_mask->data[0][0].empty()) {
        const auto& tm2d = tree_mask->data[0][0];  // (rows, cols)
        const size_t tree_rows = tm2d.size();
        const size_t tree_cols = tm2d[0].size();
        const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
        const size_t row_offset = seq_length - tree_rows;
        const size_t col_offset = seq_len_with_past - tree_cols;
        for (size_t r = 0; r < tree_rows; ++r) {
            for (size_t c = 0; c < tree_cols; ++c) {
                if (tm2d[r][c] == 0) {
                    mask[(row_offset + r) * seq_len_with_past + (col_offset + c)] = neg_inf;
                }
            }
        }
    }

    return mask;
}


// Greedy accept/reject for spec decoding. Per path, finds the longest prefix
// where target's argmax[i] == candidate[i+1]. Best path wins; sample_p is the
// target's logits row at the rejection (or last) position — caller takes its
// argmax for the bonus token.
//   logits:     (num_paths × candidate_len × vocab_size) bf16
//   candidates: (num_paths × candidate_len) int32; -1 marks path padding
struct EvaluatePosteriorResult {
    size_t best_candidate;
    int32_t accept_length;
    std::vector<Eigen::bfloat16> sample_p;  // bonus-position logits, length = vocab_size
};

// Argmax is OMP-parallel across rows; inline bf16→fp32 (shl 16) keeps the inner
// loop vectorizable. sample_p is the target's logits row at the rejection (or
// last) position — caller takes its argmax for the bonus token.
inline EvaluatePosteriorResult evaluate_posterior(
    const Eigen::bfloat16* logits,
    const int32_t* candidates,
    size_t num_paths,
    size_t candidate_len,
    size_t vocab_size
) {
    const size_t total_rows = num_paths * candidate_len;
    std::vector<int32_t> logits_argmax(total_rows);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_rows; ++i) {
        const Eigen::bfloat16* row = logits + i * vocab_size;
        float best_val = -std::numeric_limits<float>::infinity();
        int32_t best_idx = 0;
        for (size_t j = 0; j < vocab_size; ++j) {
            const float v = static_cast<float>(row[j]);
            if (v > best_val) {
                best_val = v;
                best_idx = static_cast<int32_t>(j);
            }
        }
        logits_argmax[i] = best_idx;
    }

    std::vector<int32_t> accept_lengths(num_paths, 0);
    for (size_t c = 0; c < num_paths; ++c) {
        for (size_t i = 0; i + 1 < candidate_len; ++i) {
            const int32_t arg = logits_argmax[c * candidate_len + i];
            const int32_t cand_next = candidates[c * candidate_len + (i + 1)];
            // -1 in candidates means "this path ends here" — treat as mismatch.
            if (cand_next < 0 || arg != cand_next) break;
            accept_lengths[c]++;
        }
    }

    int32_t max_accept = 0;
    size_t best_c = 0;
    for (size_t c = 0; c < num_paths; ++c) {
        if (accept_lengths[c] > max_accept) {
            max_accept = accept_lengths[c];
            best_c = c;
        }
    }

    EvaluatePosteriorResult result;
    result.best_candidate = best_c;
    result.accept_length = max_accept;
    const Eigen::bfloat16* sample_row =
        logits + (best_c * candidate_len + max_accept) * vocab_size;
    result.sample_p.assign(sample_row, sample_row + vocab_size);
    return result;
}

// For each query, returns the insertion index that would keep `sorted_arr`
// sorted. `right=false` uses std::lower_bound semantics; `right=true` uses
// std::upper_bound.
template <typename T>
std::vector<size_t> searchsorted(
    const std::vector<T>& sorted_arr,
    const std::vector<T>& queries,
    bool right = false
) {
    std::vector<size_t> result(queries.size());
    for (size_t i = 0; i < queries.size(); ++i) {
        auto it = right
            ? std::upper_bound(sorted_arr.begin(), sorted_arr.end(), queries[i])
            : std::lower_bound(sorted_arr.begin(), sorted_arr.end(), queries[i]);
        result[i] = static_cast<size_t>(std::distance(sorted_arr.begin(), it));
    }
    return result;
}

}  // namespace eagle_helpers
}  // namespace llima
}  // namespace simaai

#endif
