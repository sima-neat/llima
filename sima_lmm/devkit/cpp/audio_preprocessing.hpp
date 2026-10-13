#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include "eigen_types.hpp"

namespace simaai::llima {

// Decode/resample to mono 16-kHz PCM. Return only real samples; padding belongs
// to the model frontend. Whisper opts into truncation, while Gemma rejects it.
ArrayXf load_audio(const std::filesystem::path& path, size_t max_samples, bool truncate = false);

// Decode uploaded audio through the same frontend; its temporary file is
// closed and removed before returning, including on decoding errors.
ArrayXf load_audio(std::span<const uint8_t> data, size_t max_samples);

}
