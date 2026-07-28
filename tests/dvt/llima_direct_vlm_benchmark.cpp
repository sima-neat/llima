#include "setup.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * DVT-only product-metric benchmark for a compiled vision-language model.
 *
 * Usage:
 *   llima_direct_vlm_benchmark <compiled-model-directory> <image>
 *                              [measured-decode-intervals [trials [prompt
 *                              [warmup-decode-intervals]]]]
 *
 * The published LFM2.5-VL-450M MLA-RT result reports response TPS after an
 * image plus a seven-token text prompt. Keep the same product boundary here:
 * TTFT includes image preprocessing/encoding and language prefill, while TPS
 * is the harmonic response rate of exactly N post-warmup decode intervals.
 * The harness disables semantic stop tokens for the measured call, produces
 * one first token plus the declared warmup and N measured intervals, and
 * rejects any shorter/longer result.  This fixes the old benchmark bug where
 * `max_new_tokens=256` silently stopped at 191 tokens and mixed a cold first
 * interval into the reported mean.  Do not use total process or
 * model-construction time as a substitute for this metric.
 *
 * VisionLanguageModel performs its normal two-token warmup in the constructor.
 * Callbacks are installed only afterward, so warmup cannot contaminate any
 * sample. The production direct-kernel session, package residency, checked
 * dma-buf views, and queue-ahead executor are otherwise unchanged.
 *
 * Inspect this executable with `file` and run the AArch64 artifact only on an
 * authorized Modalix DVT. It must never be executed in the x86 build container.
 */
namespace {

constexpr std::size_t kDefaultWarmupDecodeIntervals = 32;

std::uint32_t parse_bounded(
    const char* text, const char* name, std::uint32_t low, std::uint32_t high
) {
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < low || value > high) {
        throw std::invalid_argument(
            std::string(name) + " must be in [" + std::to_string(low) +
            "," + std::to_string(high) + "]"
        );
    }
    return static_cast<std::uint32_t>(value);
}

struct TrialMetrics {
    double ttft_s = -1.0;
    std::vector<double> decode_intervals_s;
    std::vector<std::uint32_t> output_token_ids;
    bool ended = false;
};

struct TrialResult {
    double ttft_s = 0.0;
    double response_tps = 0.0;
    double median_tps = 0.0;
    double wall_s = 0.0;
    std::size_t generated_tokens = 0;
    std::size_t response_bytes = 0;
};

TrialResult finalize_metrics(
    TrialMetrics metrics,
    std::size_t warmup_decode_intervals,
    std::size_t measured_decode_intervals,
    double wall_s,
    std::size_t response_bytes
) {
    if (!metrics.ended || metrics.ttft_s < 0.0 ||
        metrics.decode_intervals_s.empty()) {
        throw std::runtime_error(
            "generation did not produce a terminal event, TTFT, and at least "
            "one post-first-token interval"
        );
    }
    for (const double interval : metrics.decode_intervals_s) {
        if (!(interval > 0.0) || !std::isfinite(interval)) {
            throw std::runtime_error("invalid decode interval");
        }
    }

    const std::size_t expected_intervals =
        warmup_decode_intervals + measured_decode_intervals;
    if (metrics.decode_intervals_s.size() != expected_intervals) {
        throw std::runtime_error(
            "fixed-length benchmark expected " +
            std::to_string(expected_intervals) +
            " post-first-token intervals but observed " +
            std::to_string(metrics.decode_intervals_s.size())
        );
    }

    /*
     * Erase no observations after the measurement window begins.  The first
     * warmup decode intervals are a declared contract, not statistical
     * outlier removal; every one of the following N intervals contributes to
     * the per-trial harmonic rate.
     */
    metrics.decode_intervals_s.erase(
        metrics.decode_intervals_s.begin(),
        metrics.decode_intervals_s.begin() + warmup_decode_intervals
    );
    const double decode_s = std::accumulate(
        metrics.decode_intervals_s.begin(),
        metrics.decode_intervals_s.end(), 0.0
    );
    const double response_tps =
        static_cast<double>(metrics.decode_intervals_s.size()) / decode_s;

    const std::size_t middle = metrics.decode_intervals_s.size() / 2;
    std::nth_element(
        metrics.decode_intervals_s.begin(),
        metrics.decode_intervals_s.begin() + middle,
        metrics.decode_intervals_s.end()
    );

    return {
        .ttft_s = metrics.ttft_s,
        .response_tps = response_tps,
        .median_tps = 1.0 / metrics.decode_intervals_s[middle],
        .wall_s = wall_s,
        .generated_tokens = expected_intervals + 1,
        .response_bytes = response_bytes,
    };
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

double sample_standard_deviation(
    const std::vector<double>& values, double sample_mean
) {
    if (values.size() < 2) {
        return 0.0;
    }
    const double squared_error = std::accumulate(
        values.begin(), values.end(), 0.0,
        [sample_mean](double sum, double value) {
            const double error = value - sample_mean;
            return sum + error * error;
        }
    );
    return std::sqrt(squared_error / static_cast<double>(values.size() - 1));
}

/*
 * One-sided 95% Student-t critical values for df=1..19.  The benchmark caps
 * trials at 20, making a small audited table preferable to adding a target
 * dependency on a statistics package.  Index zero is unused.
 */
constexpr double kOneSidedT95[] = {
    0.0,
    6.313752, 2.919986, 2.353363, 2.131847, 2.015048,
    1.943180, 1.894579, 1.859548, 1.833113, 1.812461,
    1.795885, 1.782288, 1.770933, 1.761310, 1.753050,
    1.745884, 1.739607, 1.734064, 1.729133,
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 7) {
        std::cerr
            << "usage: " << argv[0]
            << " <compiled-model-directory> <image>"
               " [measured-decode-intervals [trials [prompt"
               " [warmup-decode-intervals]]]]\n";
        return 64;
    }

    const std::filesystem::path model_path(argv[1]);
    const std::filesystem::path image_path(argv[2]);
    if (!std::filesystem::is_directory(model_path / "devkit") ||
        !std::filesystem::is_directory(model_path / "elf_files")) {
        std::cerr << "model package lacks devkit/ or elf_files/: "
                  << model_path << "\n";
        return 65;
    }
    if (!std::filesystem::is_regular_file(image_path)) {
        std::cerr << "image does not exist: " << image_path << "\n";
        return 66;
    }

    std::uint32_t measured_decode_intervals = 256;
    std::uint32_t trials = 3;
    std::uint32_t warmup_decode_intervals =
        kDefaultWarmupDecodeIntervals;
    std::string prompt = "Describe what's in the image";
    try {
        if (argc >= 4) {
            measured_decode_intervals = parse_bounded(
                argv[3], "measured-decode-intervals", 2, 992
            );
        }
        if (argc >= 5) {
            trials = parse_bounded(argv[4], "trials", 1, 20);
        }
        if (argc >= 6) {
            prompt = argv[5];
            if (prompt.empty()) {
                throw std::invalid_argument("prompt must not be empty");
            }
        }
        if (argc == 7) {
            warmup_decode_intervals = parse_bounded(
                argv[6], "warmup-decode-intervals", 0, 992
            );
        }
        if (static_cast<std::uint64_t>(warmup_decode_intervals) +
                measured_decode_intervals >
            std::numeric_limits<std::uint16_t>::max()) {
            throw std::invalid_argument(
                "warmup plus measured intervals exceeds uint16 range"
            );
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 64;
    }

    simaai::llima::connect(
        {}, "/tmp/llima-direct-vlm-benchmark.log", spdlog::level::warn
    );
    try {
        /*
         * Make constructor warmup use the same physical image as the measured
         * run. Develop/MLA-RT also warmed the model before reporting response
         * rate; measuring first-use package admission would not be comparable.
         */
        simaai::llima::set_sample_image_file_name(image_path);
        simaai::llima::VisionLanguageModel model(model_path);
        if (!model.support_image()) {
            throw std::runtime_error("model does not advertise image support");
        }

        std::mutex metrics_mutex;
        TrialMetrics current;
        model.set_text_callback(
            [](const std::string&, bool, bool) {}
        );
        model.set_decode_callback(
            [&](const simaai::llima::DecodeCallbackData& data) {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (data.type ==
                    simaai::llima::DecodeCallbackType::TTFT) {
                    current.ttft_s = data.duration;
                    current.output_token_ids.emplace_back(data.token_id);
                } else if (
                    data.type ==
                    simaai::llima::DecodeCallbackType::TPS) {
                    if (!(data.duration > 0.0) ||
                        !std::isfinite(data.duration)) {
                        throw std::runtime_error(
                            "invalid decode interval callback"
                        );
                    }
                    current.decode_intervals_s.emplace_back(data.duration);
                    current.output_token_ids.emplace_back(data.token_id);
                }
            }
        );
        model.set_info_callback(
            [&](const std::string& type, double) {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (type == "END" || type == "FULL") {
                    current.ended = true;
                }
            }
        );

        std::vector<TrialResult> results;
        results.reserve(trials);
        /*
         * VisionLanguageModel's max-new-token boundary is expressed as the
         * number of decode transitions after the prefill seed.  The streamer
         * therefore observes requested_new_tokens + 1 generated tokens: one
         * TTFT token followed by that many TPS intervals.
         */
        const auto requested_new_tokens = static_cast<std::uint16_t>(
            warmup_decode_intervals + measured_decode_intervals
        );
        for (std::uint32_t trial = 0; trial < trials; ++trial) {
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                current = {};
            }

            auto chat = model.create_chat();
            chat.add_image(image_path);
            chat.add_query(prompt);

            const auto start = std::chrono::steady_clock::now();
            /*
             * Empty stop-token set is intentional for this benchmark only:
             * semantic EOS would make trial length prompt/content dependent.
             * The scoped LanguageModel override restores normal stop tokens
             * before returning, so subsequent product calls are unaffected.
             */
            const auto response = model.run_model(
                chat, requested_new_tokens, std::set<std::uint32_t>{}
            );
            model.wait_for_streamer_completion();
            const double wall_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start
            ).count();

            TrialMetrics snapshot;
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                snapshot = current;
            }
            if (snapshot.output_token_ids.size() !=
                snapshot.decode_intervals_s.size() + 1) {
                throw std::runtime_error(
                    "token observer did not preserve one token per interval"
                );
            }
            /*
             * Emit every unrounded observation before aggregation.  The
             * analyzer treats these lines as the source of truth; the PASS
             * line is only a convenient independently-computed summary.
             */
            std::cout
                << std::fixed << std::setprecision(9)
                << "LLIMA_DIRECT_VLM_FIRST"
                << " trial=" << (trial + 1)
                << " output_token_id=" << snapshot.output_token_ids.front()
                << " ttft_us=" << (snapshot.ttft_s * 1.0e6)
                << "\n";
            for (std::size_t interval = 0;
                 interval < snapshot.decode_intervals_s.size();
                 ++interval) {
                const bool warmup =
                    interval < warmup_decode_intervals;
                std::cout
                    << std::fixed << std::setprecision(9)
                    << "LLIMA_DIRECT_VLM_INTERVAL"
                    << " trial=" << (trial + 1)
                    << " interval=" << interval
                    << " phase=" << (warmup ? "warmup" : "measured")
                    << " output_token_id="
                    << snapshot.output_token_ids[interval + 1]
                    << " period_us="
                    << (snapshot.decode_intervals_s[interval] * 1.0e6)
                    << "\n";
            }
            auto result = finalize_metrics(
                std::move(snapshot), warmup_decode_intervals,
                measured_decode_intervals, wall_s,
                response.has_value() ? response->size() : 0U
            );
            results.emplace_back(result);
            std::cout
                << std::fixed << std::setprecision(3)
                << "LLIMA_DIRECT_VLM_TRIAL trial=" << (trial + 1)
                << " generated_tokens=" << result.generated_tokens
                << " warmup_intervals=" << warmup_decode_intervals
                << " measured_intervals=" << measured_decode_intervals
                << " response_tps=" << result.response_tps
                << " median_tps=" << result.median_tps
                << " ttft_ms=" << (result.ttft_s * 1000.0)
                << " wall_s=" << result.wall_s
                << " response_bytes=" << result.response_bytes << "\n";
        }

        std::vector<double> response_tps;
        std::vector<double> ttft_ms;
        for (const auto& result : results) {
            response_tps.emplace_back(result.response_tps);
            ttft_ms.emplace_back(result.ttft_s * 1000.0);
        }
        /*
         * Preserve every trial in `results`; sorted copies are used only for
         * descriptive medians.  Acceptance uses the untrimmed sample mean and
         * a one-sided 95% Student-t lower confidence bound.
         */
        const double response_tps_mean = mean(response_tps);
        const double response_tps_stddev =
            sample_standard_deviation(response_tps, response_tps_mean);
        const double response_tps_lcb95 = trials > 1
            ? response_tps_mean -
                kOneSidedT95[trials - 1] * response_tps_stddev /
                    std::sqrt(static_cast<double>(trials))
            : std::numeric_limits<double>::quiet_NaN();
        std::sort(response_tps.begin(), response_tps.end());
        std::sort(ttft_ms.begin(), ttft_ms.end());

        std::cout
            << std::fixed << std::setprecision(3)
            << "LLIMA_DIRECT_VLM_PASS"
            << " model=" << model_path.filename()
            << " trials=" << trials
            << " requested_new_tokens=" << requested_new_tokens
            << " warmup_intervals=" << warmup_decode_intervals
            << " measured_intervals=" << measured_decode_intervals
            << " response_tps_mean=" << response_tps_mean
            << " response_tps_stddev=" << response_tps_stddev
            << " response_tps_lcb95=" << response_tps_lcb95
            << " response_tps_median="
            << response_tps[response_tps.size() / 2]
            << " ttft_ms_mean=" << mean(ttft_ms)
            << " ttft_ms_median=" << ttft_ms[ttft_ms.size() / 2]
            << " prompt=\"" << prompt << "\"\n";
    } catch (const std::exception& error) {
        simaai::llima::disconnect();
        std::cerr << "VLM benchmark failed: " << error.what() << "\n";
        return 1;
    }
    simaai::llima::disconnect();
    return 0;
}
