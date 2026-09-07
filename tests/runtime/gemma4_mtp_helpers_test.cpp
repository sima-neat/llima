#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma4_mtp_helpers.hpp"

namespace {

using simaai::llima::gemma4_mtp_helpers::draft_query_position;
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

void test_draft_query_position_stays_on_the_last_target_row() {
    expect(
        draft_query_position(123, 2048) == 122,
        "the draft query position must be derived only from the shared target KV length"
    );
    expect_runtime_error(
        [] { draft_query_position(0, 2048); },
        "an empty shared target KV cache must be rejected"
    );
    expect_runtime_error(
        [] { draft_query_position(2049, 2048); },
        "a draft query outside cache capacity must be rejected"
    );
}

} // namespace

int main() {
    test_selects_only_from_active_centroids();
    test_uses_canonical_token_ids_and_deterministic_ties();
    test_rejects_invalid_metadata();
    test_draft_query_position_stays_on_the_last_target_row();
    return failures == 0 ? 0 : 1;
}
