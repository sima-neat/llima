#include "audio_generation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <string>

using namespace simaai::llima;

namespace {
class Backend final : public AudioGenerationBackend {
public:
    std::vector<uint32_t> text;
    std::vector<uint32_t> semantic;
    std::vector<char> trace;
    std::vector<std::array<uint32_t, 8>> embeddings;
    size_t text_index = 0;
    size_t audio_index = 0;
    size_t hidden_size() const override { return 2; }
    void reset() override { text_index = audio_index = 0; trace.clear(); embeddings.clear(); }
    std::vector<float> forward(std::span<const float>) override { return {1.f, 2.f}; }
    std::vector<float> text_logits(std::span<const float>) override {
        trace.push_back('T');
        std::vector<float> logits(131, -10.f);
        logits.at(text.at(text_index++)) = 10.f;
        return logits;
    }
    std::array<uint32_t, 8> audio_codes(
        std::span<const float>, const AudioSamplingOptions&, std::mt19937_64&
    ) override {
        trace.push_back('A');
        std::array<uint32_t, 8> codes{0, 1, 2, 3, 4, 5, 6, 7};
        codes[0] = audio_index < semantic.size() ? semantic[audio_index] : 17;
        ++audio_index;
        return codes;
    }
    std::vector<float> text_embedding(uint32_t) override { return {1.f, 2.f}; }
    std::vector<float> audio_embedding(const std::array<uint32_t, 8>& codes) override {
        embeddings.push_back(codes);
        return {1.f, 2.f};
    }
};

void generation_modes() {
    Backend backend;
    AudioGenerationEngine engine(backend, 2, 3);
    const std::array prompt{1.f, 2.f};
    std::vector<AudioGeneratedToken> output;
    auto collect = [&](const auto& token) { output.push_back(token); return true; };
    backend.text = {128, 7};
    backend.semantic = {42, 2048};
    engine.generate(AudioGenerationEngine::Mode::Sequential, prompt, 20, {}, {}, collect);
    assert(std::string(backend.trace.begin(), backend.trace.end()) == "TAAT");
    assert(output.size() == 4 && output.back().text == 7);
    assert(output[2].is_audio && std::all_of(output[2].codes.begin(), output[2].codes.end(), [](auto x) { return x == 2048; }));
    assert(backend.embeddings.size() == 2 && backend.embeddings.back()[7] == 2048);

    output.clear();
    backend.text = {12, 13, 7};
    backend.semantic.clear();
    engine.generate(AudioGenerationEngine::Mode::Interleaved, prompt, 20, {}, {}, collect);
    assert(std::string(backend.trace.begin(), backend.trace.end()) == "TTAAAT");
    assert(output.size() == 5); // Interleaved EOS is not yielded.

    output.clear();
    backend.text = {130, 7};
    backend.semantic = {4, 5, 6, 7, 2048};
    engine.generate(AudioGenerationEngine::Mode::Interleaved, prompt, 20, {}, {}, collect);
    assert(std::string(backend.trace.begin(), backend.trace.end()) == "TAAAAAT");
    assert(output.size() == 6); // text_end keeps audio active past its ordinary run.

    backend.text = {12, 13};
    engine.generate(AudioGenerationEngine::Mode::Interleaved, prompt, 20, {}, {}, [](const auto&) { return false; });
    assert(backend.trace.size() == 1);
}

void sampling() {
    std::mt19937_64 random(5);
    const std::array logits{1.f, 3.f, 3.f, -1.f};
    assert(sample_audio_model_token(logits, {}, random) == 1);
    std::array<size_t, 4> counts{};
    for (size_t index = 0; index < 1000; ++index) {
        ++counts[sample_audio_model_token(logits, {.temperature = 0.8f, .top_k = 2}, random)];
    }
    assert(counts[0] == 0 && counts[3] == 0 && counts[1] > 300 && counts[2] > 300);
}

void waveform() {
    constexpr auto size = AudioWaveformSynthesizer::FFT_SIZE;
    constexpr auto bins = AudioWaveformSynthesizer::BINS;
    constexpr float pi = 3.14159265358979323846f;
    std::vector<float> window(size);
    for (size_t index = 0; index < size; ++index) {
        window[index] = 0.5f * (1.f - std::cos(2.f * pi * static_cast<float>(index) / size));
    }
    std::vector<float> spectrum(2 * bins, 0.f);
    for (size_t index = 0; index < bins; ++index) spectrum[bins + index] = index % 2 ? pi : 0.f;
    AudioWaveformSynthesizer one(window);
    auto actual = one.push(spectrum);
    auto tail = one.finish();
    actual.insert(actual.end(), tail.begin(), tail.end());
    assert(actual.size() == 320);
    for (size_t index = 0; index < actual.size(); ++index) {
        assert(std::abs(actual[index] - (index == 160 ? 1.f : 0.f)) < 1e-5f);
    }
    one.reset();
    std::vector<float> batch;
    for (size_t index = 0; index < 17; ++index) batch.insert(batch.end(), spectrum.begin(), spectrum.end());
    auto whole = one.push(batch);
    tail = one.finish();
    whole.insert(whole.end(), tail.begin(), tail.end());
    one.reset();
    std::vector<float> chunks;
    for (size_t index = 0; index < 17; ++index) {
        auto chunk = one.push(spectrum);
        chunks.insert(chunks.end(), chunk.begin(), chunk.end());
    }
    tail = one.finish();
    chunks.insert(chunks.end(), tail.begin(), tail.end());
    assert(whole.size() == 17 * 320 && whole == chunks);
}
void frontend() {
    std::vector<float> window(400, 1.f), filters(128 * 257, 0.f), pcm(511);
    for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = (static_cast<int>(i % 29) - 14) / 16.f;
    filters[257] = 1.f;
    filters[2 * 257 + 256] = 1.f;
    filters[3 * 257] = 1.f;
    filters[3 * 257 + 37] = .5f;
    filters[3 * 257 + 256] = 2.f;
    const std::array<float, 5> triangle{.2f, .4f, 1.f, .4f, .2f};
    std::copy(triangle.begin(), triangle.end(), filters.begin() + 4 * 257 + 91);
    for (size_t i = 0; i < 257; ++i) filters[5 * 257 + i] = (i % 11) / 16.f;
    AudioFeatureExtractor extractor(window, filters);
    const auto output = extractor.process(pcm);
    // Golden values from the original dense frontend: empty, first/last-bin,
    // gapped, triangular and dense filters, including the final zero frame.
    const std::array<float, 24> expected{
        0.f, 0.f, 0.f, 0.f,
        0x1.a550b6p-3f, -0x1.163ba8p+0f, 0x1.c3232p-1f, 0.f,
        -0x1.1e642cp-1f, -0x1.30b696p-1f, 0x1.278d62p+0f, 0.f,
        0x1.29cf6ap-1f, -0x1.2797eap+0f, 0x1.25607ep-1f, 0.f,
        -0x1.357888p-1f, -0x1.197d8ep-1f, 0x1.277b0cp+0f, 0.f,
        -0x1.26bb32p+0f, 0x1.ff33eap-2f, 0x1.4ddc98p-1f, 0.f,
    };
    assert(output.frames == 4 && output.values.size() == 128 * 4);
    for (size_t i = 0; i < expected.size(); ++i) assert(std::abs(output.values[i] - expected[i]) < 3e-6f);
    for (size_t i = expected.size(); i < output.values.size(); ++i) assert(output.values[i] == 0.f);
    // Zero coefficients must still participate when finite PCM overflows the
    // power calculation, retaining the dense frontend's 0*Inf -> NaN behavior.
    std::fill(pcm.begin(), pcm.end(), 1e38f);
    assert(std::isnan(extractor.process(pcm).values[0]));
}
} // namespace

std::vector<float> read_floats(const char* path) {
    const auto bytes = std::filesystem::file_size(path);
    if (bytes % sizeof(float)) throw std::invalid_argument("Invalid float fixture");
    std::vector<float> values(bytes / sizeof(float));
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file) throw std::runtime_error("Cannot read float fixture");
    return values;
}

int main(int argc, char** argv) {
    if (argc == 6 && std::string(argv[1]) == "--frontend") {
        AudioFeatureExtractor extractor(read_floats(argv[3]), read_floats(argv[4]));
        auto output = extractor.process(read_floats(argv[2]));
        std::ofstream file(argv[5], std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.values.data()), output.values.size() * sizeof(float));
        std::cout << "frames=" << output.frames << '\n';
        return file ? 0 : 1;
    }
    if (argc == 5 && std::string(argv[1]) == "--synthesize") {
        AudioWaveformSynthesizer synthesizer(read_floats(argv[3]));
        auto output = synthesizer.push(read_floats(argv[2]));
        auto tail = synthesizer.finish();
        output.insert(output.end(), tail.begin(), tail.end());
        std::ofstream file(argv[4], std::ios::binary);
        file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
        return file ? 0 : 1;
    }
    if (argc != 1) throw std::invalid_argument("Unknown audio test invocation");
    generation_modes();
    sampling();
    waveform();
    frontend();
    std::cout << "Audio generation and waveform checks passed\n";
}
