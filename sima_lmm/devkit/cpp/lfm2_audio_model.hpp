#pragma once

#include "audio_generation.hpp"
#include <filesystem>
#include <memory>

namespace simaai::llima {

struct AudioPromptMessage {
    std::string role;
    std::string text;
    // Optional encoder output, appended after text within this message.
    std::vector<float> audio_embeddings;
};

struct AudioDecodeTimings {
    double network_ms = 0.;
    double waveform_cpu_ms = 0.;
};

// Private native audio runtime. It consumes the explicit audio_runtime.json
// bundle, not the VisionLanguage/ASR GenAI task interface.
class Lfm2AudioModel final : public AudioGenerationBackend {
public:
    explicit Lfm2AudioModel(const std::filesystem::path& runtime_directory);
    ~Lfm2AudioModel() override;
    Lfm2AudioModel(const Lfm2AudioModel&) = delete;
    Lfm2AudioModel& operator=(const Lfm2AudioModel&) = delete;
    size_t hidden_size() const override;
    void reset() override;
    std::vector<float> forward(std::span<const float> embeddings) override;
    std::vector<float> text_logits(std::span<const float> hidden) override;
    std::array<uint32_t, 8> audio_codes(
        std::span<const float> hidden, const AudioSamplingOptions&, std::mt19937_64&
    ) override;
    std::vector<float> text_embedding(uint32_t token) override;
    std::vector<float> audio_embedding(const std::array<uint32_t, 8>& codes) override;
    std::vector<float> make_prompt(std::span<const AudioPromptMessage> messages);
    std::string decode_text(const std::vector<uint32_t>& tokens);

    // Feature-major log-mel data must already use the saved preprocessing
    // contract. physical_frames is the original padded-batch maximum length.
    std::vector<float> encode_mel(
        std::span<const float> features, size_t frames, size_t physical_frames
    );
    std::vector<float> encode_pcm(std::span<const float> mono_16khz_pcm);
    std::vector<float> decode_codes(
        std::span<const std::array<uint32_t, 8>> codes, bool finish = false
    );
    AudioDecodeTimings decode_timings() const;

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace simaai::llima
