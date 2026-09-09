#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma4_mtp_helpers.hpp"

namespace {

using simaai::llima::gemma4_mtp_helpers::build_causal_mask;
using simaai::llima::gemma4_mtp_helpers::draft_query_position;
using simaai::llima::gemma4_mtp_helpers::draft_visible_shared_kv_len;
using simaai::llima::gemma4_mtp_helpers::resolve_draft_tokens;
using simaai::llima::gemma4_mtp_helpers::select_masked_token;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

template <typename Callable>
void expect_runtime_error(Callable&& callable, const std::string& message) {
    try {
        callable();
        expect(false, message);
    } catch (const std::runtime_error&) {
    }
}

void expect_mask_row(
    const std::vector<Eigen::bfloat16>& mask, size_t columns,
    size_t row, size_t visible_columns
) {
    for (size_t column = 0; column < columns; ++column) {
        const float value = static_cast<float>(mask[row * columns + column]);
        expect(
            column < visible_columns ? value == 0.0f : std::isinf(value) && value < 0.0f,
            "mask visibility must match the causal prefix at the compiled row stride"
        );
    }
}

void test_causal_mask_uses_compiled_context_stride() {
    // The allocation may cover 4096 positions, but this cache ELF consumes
    // seven compact 128-column rows. Full-context packing misreads rows 1..6.
    const auto mask = build_causal_mask(7, 40, 7, 0, 128);
    expect(mask.size() == 7 * 128, "short-context mask must use the compiled width");
    const size_t visible[] = {40, 41, 42, 43, 44, 45, 46};
    for (size_t row = 0; row < 7; ++row) {
        expect_mask_row(mask, 128, row, visible[row]);
    }
}

void test_causal_mask_spans_a_context_bucket_boundary() {
    const auto mask = build_causal_mask(7, 127, 7, 0, 256);
    expect(mask.size() == 7 * 256, "verification crossing 128 must use the next bucket");
    expect_mask_row(mask, 256, 0, 127);
    expect_mask_row(mask, 256, 6, 133);
}

void test_causal_mask_uses_relative_sliding_columns() {
    const auto mask = build_causal_mask(7, 1201, 7, 183, 1024);
    expect_mask_row(mask, 1024, 0, 1018);
    expect_mask_row(mask, 1024, 6, 1024);

    // The pointwise assistant can expose a complete shared-KV window.
    const auto draft_mask = build_causal_mask(1, 1201, 1, 177, 1024);
    expect_mask_row(draft_mask, 1024, 0, 1024);
}

void test_causal_mask_keeps_padding_masked_and_rejects_invalid_ranges() {
    const auto mask = build_causal_mask(7, 40, 1, 0, 128);
    expect_mask_row(mask, 128, 0, 40);
    for (size_t row = 1; row < 7; ++row) {
        expect_mask_row(mask, 128, row, 0);
    }
    expect_runtime_error([] { build_causal_mask(7, 40, 0, 0, 128); }, "empty batch");
    expect_runtime_error([] { build_causal_mask(7, 40, 8, 0, 128); }, "too many rows");
    expect_runtime_error([] { build_causal_mask(7, 40, 7, 40, 128); }, "empty visible prefix");
    expect_runtime_error([] { build_causal_mask(7, 127, 7, 0, 128); }, "insufficient compiled width");
}

void test_selects_only_from_active_centroids() {
    const std::vector<Eigen::bfloat16> token_logits = {
        Eigen::bfloat16{100.0f},
        Eigen::bfloat16{1.0f},
        Eigen::bfloat16{7.0f},
        Eigen::bfloat16{3.0f},
        Eigen::bfloat16{4.0f},
        Eigen::bfloat16{5.0f},
        Eigen::bfloat16{6.0f},
        Eigen::bfloat16{9.0f},
    };
    const std::vector<Eigen::bfloat16> centroid_logits = {
        Eigen::bfloat16{1.0f},
        Eigen::bfloat16{10.0f},
        Eigen::bfloat16{2.0f},
        Eigen::bfloat16{9.0f},
    };
    const std::vector<uint32_t> token_ordering = {0, 1, 4, 5, 2, 3, 6, 7};

    expect(
        select_masked_token(
            token_logits, centroid_logits, token_ordering, 2
        ) == 7,
        "the global maximum must be ignored when its centroid is inactive"
    );
}

void test_uses_canonical_token_ids_and_deterministic_ties() {
    const std::vector<Eigen::bfloat16> token_logits = {
        Eigen::bfloat16{0.0f},
        Eigen::bfloat16{0.0f},
        Eigen::bfloat16{0.0f},
        Eigen::bfloat16{0.0f},
        Eigen::bfloat16{9.0f},
        Eigen::bfloat16{0.0f},
        Eigen::bfloat16{0.0f},
        Eigen::bfloat16{9.0f},
    };
    const std::vector<Eigen::bfloat16> centroid_logits = {
        Eigen::bfloat16{1.0f},
        Eigen::bfloat16{10.0f},
        Eigen::bfloat16{2.0f},
        Eigen::bfloat16{9.0f},
    };
    const std::vector<uint32_t> token_ordering = {0, 1, 7, 5, 2, 3, 6, 4};

    expect(
        select_masked_token(
            token_logits, centroid_logits, token_ordering, 2
        ) == 4,
        "equal candidate logits must choose the lower canonical token ID"
    );
}

void test_rejects_invalid_metadata() {
    const std::vector<Eigen::bfloat16> token_logits(8, Eigen::bfloat16{0.0f});
    const std::vector<Eigen::bfloat16> centroid_logits(4, Eigen::bfloat16{0.0f});
    const std::vector<uint32_t> ordering = {0, 1, 2, 3, 4, 5, 6, 7};

    expect_runtime_error(
        [&] { select_masked_token(token_logits, centroid_logits, ordering, 0); },
        "zero centroid top-k must be rejected"
    );
    auto out_of_range = ordering;
    out_of_range[0] = 8;
    expect_runtime_error(
        [&] {
            select_masked_token(
                token_logits, centroid_logits, out_of_range, centroid_logits.size()
            );
        },
        "out-of-range canonical token IDs must be rejected"
    );
}

void test_draft_query_position_follows_the_input_length() {
    expect(
        draft_query_position(124, 2048) == 123,
        "the draft query must use input_length minus one"
    );
    expect_runtime_error(
        [] { draft_query_position(0, 2048); },
        "an empty input must be rejected"
    );
    expect_runtime_error(
        [] { draft_query_position(2049, 2048); },
        "a draft query outside cache capacity must be rejected"
    );
}

void test_partial_rejection_exposes_one_more_shared_kv_row_than_query_position() {
    const uint16_t query_position = draft_query_position(124, 2048);
    const uint16_t visible_shared_kv_len = draft_visible_shared_kv_len(
        124, 130, 2048
    );
    expect(query_position == 123, "partial rejection keeps the query on the last input");
    expect(
        visible_shared_kv_len == 124,
        "partial rejection exposes target KV through the current input length"
    );
    expect(
        visible_shared_kv_len == query_position + 1,
        "query position and visible shared KV length must remain independent"
    );
}

void test_shared_kv_visibility_never_exceeds_computed_target_rows() {
    expect(
        draft_visible_shared_kv_len(124, 123, 2048) == 123,
        "the initial or full-match round must not expose an uncomputed bonus row"
    );
    expect_runtime_error(
        [] { draft_visible_shared_kv_len(124, 0, 2048); },
        "an empty target KV cache must be rejected"
    );
    expect_runtime_error(
        [] { draft_visible_shared_kv_len(124, 2049, 2048); },
        "target KV availability outside cache capacity must be rejected"
    );
}

void test_verification_emits_the_first_target_mismatch() {
    const std::vector<uint32_t> drafts = {10, 11, 12, 13, 14, 15};
    const std::vector<uint32_t> target = {10, 21, 22, 23, 24, 25, 26};
    const auto emitted = resolve_draft_tokens(drafts, target);

    expect(emitted.size() == 2, "verification must stop at the first mismatch");
    expect(
        emitted[0] == std::pair<uint32_t, bool>{10, true},
        "the matching prefix is accepted"
    );
    expect(
        emitted[1] == std::pair<uint32_t, bool>{21, false},
        "the mismatch uses the target token"
    );
}

void test_verification_emits_bonus_after_a_full_match() {
    const std::vector<uint32_t> drafts = {10, 11, 12, 13, 14, 15};
    const std::vector<uint32_t> target = {10, 11, 12, 13, 14, 15, 16};
    const auto emitted = resolve_draft_tokens(drafts, target);

    expect(emitted.size() == 7, "a full match must also emit the target bonus token");
    for (size_t index = 0; index < drafts.size(); ++index) {
        expect(
            emitted[index] == std::pair<uint32_t, bool>{drafts[index], true},
            "every matching draft token must be marked accepted"
        );
    }
    expect(
        emitted.back() == std::pair<uint32_t, bool>{16, false},
        "the bonus token comes from target"
    );
}

void test_verification_rejects_an_incomplete_target_result() {
    const std::vector<uint32_t> drafts = {10, 11};
    const std::vector<uint32_t> target = {10, 11};
    expect_runtime_error(
        [&] { resolve_draft_tokens(drafts, target); },
        "verification requires one more target result than draft tokens"
    );
}

} // namespace

int main() {
    test_causal_mask_uses_compiled_context_stride();
    test_causal_mask_spans_a_context_bucket_boundary();
    test_causal_mask_uses_relative_sliding_columns();
    test_causal_mask_keeps_padding_masked_and_rejects_invalid_ranges();
    test_selects_only_from_active_centroids();
    test_uses_canonical_token_ids_and_deterministic_ties();
    test_rejects_invalid_metadata();
    test_draft_query_position_follows_the_input_length();
    test_partial_rejection_exposes_one_more_shared_kv_row_than_query_position();
    test_shared_kv_visibility_never_exceeds_computed_target_rows();
    test_verification_emits_the_first_target_mismatch();
    test_verification_emits_bonus_after_a_full_match();
    test_verification_rejects_an_incomplete_target_result();
    return failures == 0 ? 0 : 1;
}
