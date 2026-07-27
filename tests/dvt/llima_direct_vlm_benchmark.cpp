#include "setup.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

/*
 * DVT-only product-metric benchmark for a compiled vision-language model.
 *
 * Usage:
 *   llima_direct_vlm_benchmark <compiled-model-directory> <image>
 *                              [max-new-tokens [trials [prompt]]]
 *
 * The published LFM2.5-VL-450M MLA-RT result reports response TPS after an
 * image plus a seven-token text prompt. Keep the same product boundary here:
 * TTFT includes image preprocessing/encoding and language prefill, while TPS
 * is the harmonic response rate of only the post-first-token decode intervals.
 * Do not use total process or model-construction time as a substitute for
 * either metric.
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
    TrialMetrics metrics, double wall_s, std::size_t response_bytes
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
        .generated_tokens = metrics.decode_intervals_s.size() + 1,
        .response_bytes = response_bytes,
    };
}

double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr
            << "usage: " << argv[0]
            << " <compiled-model-directory> <image>"
               " [max-new-tokens [trials [prompt]]]\n";
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

    std::uint32_t max_new_tokens = 64;
    std::uint32_t trials = 3;
    std::string prompt = "Describe what's in the image";
    try {
        if (argc >= 4) {
            max_new_tokens = parse_bounded(
                argv[3], "max-new-tokens", 2, 1024
            );
        }
        if (argc >= 5) {
            trials = parse_bounded(argv[4], "trials", 1, 20);
        }
        if (argc == 6) {
            prompt = argv[5];
            if (prompt.empty()) {
                throw std::invalid_argument("prompt must not be empty");
            }
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
        model.set_info_callback(
            [&](const std::string& type, double value) {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                if (type == "ttft") {
                    current.ttft_s = value;
                } else if (type == "tps") {
                    if (!(value > 0.0) || !std::isfinite(value)) {
                        throw std::runtime_error(
                            "invalid instantaneous TPS callback"
                        );
                    }
                    current.decode_intervals_s.emplace_back(1.0 / value);
                } else if (type == "END" || type == "FULL") {
                    current.ended = true;
                }
            }
        );

        std::vector<TrialResult> results;
        results.reserve(trials);
        for (std::uint32_t trial = 0; trial < trials; ++trial) {
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                current = {};
            }

            auto chat = model.create_chat();
            chat.add_image(image_path);
            chat.add_query(prompt);

            const auto start = std::chrono::steady_clock::now();
            const auto response = model.run_model(chat, max_new_tokens);
            model.wait_for_streamer_completion();
            const double wall_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start
            ).count();

            TrialMetrics snapshot;
            {
                std::lock_guard<std::mutex> lock(metrics_mutex);
                snapshot = current;
            }
            auto result = finalize_metrics(
                std::move(snapshot), wall_s,
                response.has_value() ? response->size() : 0U
            );
            results.emplace_back(result);
            std::cout
                << std::fixed << std::setprecision(3)
                << "LLIMA_DIRECT_VLM_TRIAL trial=" << (trial + 1)
                << " generated_tokens=" << result.generated_tokens
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
        std::sort(response_tps.begin(), response_tps.end());
        std::sort(ttft_ms.begin(), ttft_ms.end());

        std::cout
            << std::fixed << std::setprecision(3)
            << "LLIMA_DIRECT_VLM_PASS"
            << " model=" << model_path.filename()
            << " trials=" << trials
            << " max_new_tokens=" << max_new_tokens
            << " response_tps_mean=" << mean(response_tps)
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
