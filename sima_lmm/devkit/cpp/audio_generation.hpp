#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace simaai::llima {

struct AudioSamplingOptions {
    std::optional<float> temperature;
    std::optional<uint32_t> top_k;
};

uint32_t sample_audio_model_token(
    std::span<const float> logits, const AudioSamplingOptions& options, std::mt19937_64& random
);

struct AudioGeneratedToken {
    bool is_audio = false;
    uint32_t text = 0;
    std::array<uint32_t, 8> codes{};
};

// Private execution seam: implementations own persistent temporal/depth caches.
class AudioGenerationBackend {
public:
    virtual ~AudioGenerationBackend() = default;
    virtual size_t hidden_size() const = 0;
    virtual void reset() = 0;
    virtual std::vector<float> forward(std::span<const float> embeddings) = 0;
    virtual std::vector<float> text_logits(std::span<const float> hidden) = 0;
    virtual std::array<uint32_t, 8> audio_codes(
        std::span<const float> hidden, const AudioSamplingOptions&, std::mt19937_64&
    ) = 0;
    virtual std::vector<float> text_embedding(uint32_t token) = 0;
    virtual std::vector<float> audio_embedding(const std::array<uint32_t, 8>& codes) = 0;
};

class AudioGenerationEngine {
public:
    enum class Mode { Sequential, Interleaved };
    using Callback = std::function<bool(const AudioGeneratedToken&)>;

    AudioGenerationEngine(
        AudioGenerationBackend& backend, uint32_t text_run = 6, uint32_t audio_run = 12,
        uint64_t seed = 0
    );
    void generate(
        Mode mode, std::span<const float> prompt_embeddings, uint32_t max_new_tokens,
        const AudioSamplingOptions& text_options, const AudioSamplingOptions& audio_options,
        const Callback& callback
    );
    void stop() noexcept { _stopped.store(true); }

private:
    AudioGenerationBackend& _backend;
    uint32_t _text_run;
    uint32_t _audio_run;
    std::mt19937_64 _random;
    std::atomic<bool> _stopped{false};
    std::mutex _generation_mutex;
};

// Streaming form of the reference decoder's "same"-padded inverse STFT.
// FFT implementation details stay private; the saved checkpoint window is used.
class AudioWaveformSynthesizer {
public:
    explicit AudioWaveformSynthesizer(std::span<const float> window);
    ~AudioWaveformSynthesizer();
    AudioWaveformSynthesizer(const AudioWaveformSynthesizer&) = delete;
    AudioWaveformSynthesizer& operator=(const AudioWaveformSynthesizer&) = delete;
    std::vector<float> push(std::span<const float> spectrum);
    std::vector<float> finish();
    void reset();

    static constexpr size_t FFT_SIZE = 1280;
    static constexpr size_t HOP = 320;
    static constexpr size_t BINS = FFT_SIZE / 2 + 1;

private:
    struct Impl;
    Impl* _impl;
};

struct AudioMelFeatures {
    std::vector<float> values; // Feature-major [128, frames].
    size_t frames = 0;
};

class AudioFeatureExtractor {
public:
    AudioFeatureExtractor(std::span<const float> window, std::span<const float> filters);
    ~AudioFeatureExtractor();
    AudioFeatureExtractor(const AudioFeatureExtractor&) = delete;
    AudioFeatureExtractor& operator=(const AudioFeatureExtractor&) = delete;
    AudioMelFeatures process(std::span<const float> pcm);
private:
    struct Impl;
    Impl* _impl;
};

} // namespace simaai::llima
