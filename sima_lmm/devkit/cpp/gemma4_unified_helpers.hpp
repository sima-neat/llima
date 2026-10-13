#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "vlm_config.hpp"
#include "audio_preprocessing.hpp"

namespace simaai::llima {

using ImageSpan = std::pair<uint16_t, uint16_t>;

// Split raw PCM into fixed frames; the padded final frame is a valid token.
inline std::vector<Eigen::bfloat16> gemma4_audio_frames(
    const ArrayXf& samples, const AudioModelConfig& cfg
) {
    if (cfg.samples_per_token == 0 || cfg.sampling_rate != 16000) {
        throw std::invalid_argument("Unified audio requires a positive frame size and 16-kHz samples");
    }
    const size_t padded = (samples.size() + cfg.samples_per_token - 1) / cfg.samples_per_token * cfg.samples_per_token;
    std::vector<Eigen::bfloat16> frames(padded, Eigen::bfloat16{0.0f});
    for (Eigen::Index i = 0; i < samples.size(); ++i) frames[i] = Eigen::bfloat16{samples[i]};
    return frames;
}

inline std::vector<ImageSpan> gemma4_image_windows(std::span<const ImageSpan> spans, uint16_t group) {
    std::vector<ImageSpan> windows;
    for (const auto& [begin, end] : spans) {
        const uint16_t first = begin / group * group;
        const uint16_t last = (end + group - 1) / group * group;
        if (!windows.empty() && first < windows.back().second) {
            windows.back().second = std::max(last, windows.back().second);
        } else {
            windows.emplace_back(first, last);
        }
    }
    return windows;
}

inline std::pair<uint16_t, uint16_t> gemma4_cache_view(
    uint16_t query_begin, uint16_t rows, uint16_t context, uint16_t window,
    std::span<const ImageSpan> spans
) {
    const uint16_t begin = std::max(0, static_cast<int>(query_begin) + 1 - window);
    uint16_t end = query_begin + rows;
    for (const auto& [image_begin, image_end] : spans) {
        if (image_begin < query_begin + rows && image_end > query_begin) end = std::max(end, image_end);
    }
    const uint16_t capacity = round_up_to(end - begin, rows);
    return {std::min<uint16_t>(begin, context - capacity), capacity};
}

inline std::vector<Eigen::bfloat16> gemma4_attention_mask(
    uint16_t query_begin, uint16_t rows, uint16_t key_begin, uint16_t capacity,
    uint16_t valid_end, std::span<const ImageSpan> spans, uint16_t window
) {
    std::vector<Eigen::bfloat16> mask(static_cast<size_t>(rows) * capacity, Eigen::bfloat16{-3.3895313892515355e38f});
    for (uint32_t row = 0; row < rows; ++row) {
        const int q = query_begin + row;
        if (q >= valid_end) {
            mask[row * capacity] = Eigen::bfloat16{0.0f};
            continue;
        }
        for (uint32_t col = 0; col < capacity; ++col) {
            const int k = key_begin + col;
            bool allowed = k <= q && (window == 0 || k > q - window);
            if (window) {
                for (const auto& [begin, end] : spans) {
                    allowed |= q >= begin && q < end && k >= begin && k < end;
                }
            }
            if (allowed && k < valid_end) mask[row * capacity + col] = Eigen::bfloat16{0.0f};
        }
    }
    return mask;
}

}  // namespace simaai::llima
