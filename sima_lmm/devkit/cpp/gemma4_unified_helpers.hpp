#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "media_span.hpp"
#include "vlm_config.hpp"
#include "audio_preprocessing.hpp"

namespace simaai::llima {

using ImageWindow = std::pair<uint16_t, uint16_t>;

// Resolve placeholders once, in prompt order, before narrowing to the runtime token type.
inline std::vector<MediaSpan> gemma4_media_spans(
    std::span<const uint32_t> ids, const VlmConfig& cfg, std::optional<uint32_t> image_token
) {
    if (ids.size() > cfg.pipeline_cfg.max_num_tokens)
        throw std::invalid_argument("Prompt exceeds the compiled context");
    std::vector<MediaSpan> spans;
    const auto audio_token = cfg.support_audio() ? std::optional(cfg.audio_cfg->token_id) : std::nullopt;
    size_t images = 0;
    size_t audios = 0;
    for (size_t begin = 0; begin < ids.size();) {
        const auto token = ids[begin];
        const bool image = image_token == token;
        if (!image && audio_token != token) {
            ++begin;
            continue;
        }
        size_t end = begin + 1;
        while (end < ids.size() && ids[end] == token) ++end;
        if (image && end - begin > cfg.vm_cfg->max_soft_tokens)
            throw std::invalid_argument("Image exceeds the compiled token budget");
        if (!image && end - begin > cfg.audio_cfg->max_tokens)
            throw std::invalid_argument("Audio exceeds the compiled token budget");
        spans.push_back({image ? MediaKind::Image : MediaKind::Audio, image ? images++ : audios++,
                         static_cast<uint16_t>(begin), static_cast<uint16_t>(end)});
        begin = end;
    }
    return spans;
}

struct FinalPostRows {
    uint32_t first_input;
    uint32_t result;
};

// Place the last valid prompt row in a fixed-width post graph, including short final chunks.
inline FinalPostRows final_post_rows(uint32_t last_valid, uint32_t width) {
    const uint32_t result = std::min(last_valid, width - 1);
    return {last_valid - result, result};
}

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

inline std::vector<ImageWindow> gemma4_image_windows(std::span<const MediaSpan> spans, uint16_t group) {
    std::vector<ImageWindow> windows;
    for (const auto& span : spans) {
        if (span.kind != MediaKind::Image) continue;
        const uint16_t first = span.begin / group * group;
        const uint16_t last = (span.end + group - 1) / group * group;
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
    std::span<const MediaSpan> spans
) {
    const uint16_t begin = std::max(0, static_cast<int>(query_begin) + 1 - window);
    uint16_t end = query_begin + rows;
    for (const auto& span : spans) {
        if (span.kind == MediaKind::Image && span.begin < query_begin + rows && span.end > query_begin)
            end = std::max(end, span.end);
    }
    const uint16_t capacity = round_up_to(end - begin, rows);
    return {std::min<uint16_t>(begin, context - capacity), capacity};
}

inline std::vector<Eigen::bfloat16> gemma4_attention_mask(
    uint16_t query_begin, uint16_t rows, uint16_t key_begin, uint16_t capacity,
    uint16_t valid_end, std::span<const MediaSpan> spans, uint16_t window
) {
    std::vector<Eigen::bfloat16> mask(static_cast<size_t>(rows) * capacity, Eigen::bfloat16{-3.3895313892515355e38f});
    for (uint32_t row = 0; row < rows; ++row) {
        const int q = query_begin + row;
        if (q >= valid_end) {
            mask[row * capacity] = Eigen::bfloat16{0.0f};
            continue;
        }
        const auto image = window ? std::find_if(spans.begin(), spans.end(), [&](const MediaSpan& span) {
            return span.kind == MediaKind::Image && span.begin <= q && q < span.end;
        }) : spans.end();
        for (uint32_t col = 0; col < capacity; ++col) {
            const int k = key_begin + col;
            bool allowed = k <= q && (window == 0 || k > q - window);
            if (image != spans.end() && image->begin <= k && k < image->end) allowed = true;
            if (allowed && k < valid_end) mask[row * capacity + col] = Eigen::bfloat16{0.0f};
        }
    }
    return mask;
}

}  // namespace simaai::llima
