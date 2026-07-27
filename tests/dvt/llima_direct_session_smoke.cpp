#include "mla_buffer.hpp"
#include "setup.hpp"
#include "whisper_config.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

/*
 * Minimal DVT-only architecture smoke for the direct-kernel LLiMa session.
 *
 * This is intentionally not an LLM accuracy test.  It catches two foundational
 * integration failures before running an expensive model-family corpus:
 *
 *  1. padded tensor slices must use the physical MLA row stride; and
 *  2. connect/disconnect must open and tear down the Background /dev/mla
 *     session without Dispatcher, MLASHM, or MLA-RT.
 *
 * Inspect the resulting executable with `file` and run the AArch64 binary on
 * an authorized DVT; never execute it in the x86 build container.
 */
int main() {
    /* Older Whisper packages omit both language lookup arrays; current develop
     * emits both. A half-upgraded pair must fail while loading metadata rather
     * than indexing one table with an iterator derived from the other. */
    simaai::llima::WhisperConfig whisper_config{};
    whisper_config.language_token_ids = {50259};
    whisper_config.language_codes = {"en"};
    nlohmann::json malformed_whisper = whisper_config;
    malformed_whisper.erase("language_codes");
    bool rejected_mismatched_language_tables = false;
    try {
        (void)malformed_whisper.get<simaai::llima::WhisperConfig>();
    } catch (const nlohmann::json::exception&) {
        rejected_mismatched_language_tables = true;
    }
    if (!rejected_mismatched_language_tables) {
        std::cerr << "mismatched Whisper language tables were accepted\n";
        return 2;
    }

    simaai::llima::MLABuffer padded(
        "layout", {2, 3}, "bfloat16", true
    );
    const std::uint64_t second_row = padded.get_buf_addr_offset(
        std::vector<std::uint32_t>{1, 0}
    );

    // Three BF16 values occupy six logical bytes, but the second physical row
    // starts at the next 16-byte MLA row.  The old logical-stride calculation
    // incorrectly returned six.
    if (second_row != 16) {
        std::cerr << "padded offset mismatch: " << second_row << "\n";
        return 3;
    }

    /*
     * Exercise the checked padded upload/download path as well.  Padding must
     * never leak into the logical host tensor, and a partial padded upload is
     * rejected instead of silently addressing the wrong physical bytes.
     */
    padded.allocate();
    const std::array<std::uint16_t, 6> source = {
        1, 2, 3, 4, 5, 6
    };
    std::array<std::uint16_t, 6> round_trip{};
    padded.upload(source.data());
    padded.download(round_trip.data());
    if (source != round_trip) {
        std::cerr << "padded upload/download mismatch\n";
        return 4;
    }
    bool rejected_partial = false;
    try {
        padded.upload(source.data(), 1, 1);
    } catch (const std::invalid_argument&) {
        rejected_partial = true;
    }
    if (!rejected_partial) {
        std::cerr << "partial padded upload was not rejected\n";
        return 5;
    }
    padded.free();

    bool rejected_after_free = false;
    try {
        padded.clear();
    } catch (const std::logic_error&) {
        rejected_after_free = true;
    }
    if (!rejected_after_free) {
        std::cerr << "use-after-free buffer access was not rejected\n";
        return 6;
    }

    simaai::llima::connect(
        {}, "/tmp/llima-direct-smoke.log", spdlog::level::warn
    );
    simaai::llima::disconnect();

    std::cout
        << "LLIMA_DIRECT_SMOKE_PASS padded_row_offset="
        << second_row
        << " padded_round_trip=BYTE_EXACT checked_rejections=3\n";
    return 0;
}
