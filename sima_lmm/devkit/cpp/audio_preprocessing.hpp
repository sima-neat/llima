#pragma once

#include <algorithm>
#include <cstdint>
#include <complex>
#include <filesystem>
#include <span>
#include <vector>
#include <fftw3.h>
#include "eigen_types.hpp"
#include "file_provider.hpp"
#include "vlm_config.hpp"

namespace simaai::llima {

// Decode/resample to mono 16-kHz PCM. Return only real samples; padding belongs
// to the model frontend. Whisper opts into truncation, while Gemma rejects it.
ArrayXf load_audio(const std::filesystem::path& path, size_t max_samples, bool truncate = false);

// Decode uploaded audio through the same frontend; its temporary file is
// closed and removed before returning, including on decoding errors.
ArrayXf load_audio(std::span<const uint8_t> data, size_t max_samples);

struct AudioInput {
    std::vector<Eigen::bfloat16> features;
    std::vector<Eigen::bfloat16> mask;
    uint32_t num_tokens;
};

class Gemma4AudioPreprocessor {
    public:
        Gemma4AudioPreprocessor(const AudioModelConfig& config, FileProvider& files);
        ~Gemma4AudioPreprocessor();
        Gemma4AudioPreprocessor(const Gemma4AudioPreprocessor&) = delete;
        Gemma4AudioPreprocessor& operator=(const Gemma4AudioPreprocessor&) = delete;
        AudioInput preprocess(const ArrayXf& samples);
        size_t max_samples() const {
            return std::min<uint32_t>(30 * _sampling_rate, _cfg.num_frames() * _hop_length);
        }
    private:
        const AudioModelConfig& _cfg;
        uint32_t _sampling_rate, _num_mel_bins, _frame_length, _hop_length, _fft_length;
        ArrayXf _window;
        ArrayXXd _mel_filters;
        ArrayXd _mean, _stddev;
        double _mel_floor;
        std::vector<double> _fft_input;
        std::vector<std::complex<double>> _fft_output;
        fftw_plan _fft_plan;
};

}
