#include "audio_generation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <fftw3.h>

namespace simaai::llima {

uint32_t sample_audio_model_token(
    std::span<const float> logits, const AudioSamplingOptions& options, std::mt19937_64& random
) {
    if (logits.empty()) throw std::invalid_argument("Cannot sample an empty vocabulary");
    if (std::any_of(logits.begin(), logits.end(), [](float value) {
        return std::isnan(value) || value == std::numeric_limits<float>::infinity();
    })) throw std::invalid_argument("Invalid audio-model logits");
    if (!options.temperature || *options.temperature <= 0 || options.top_k == 1) {
        return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    if (!std::isfinite(*options.temperature)) throw std::invalid_argument("Invalid sampling temperature");
    if (options.top_k && (*options.top_k == 0 || *options.top_k > logits.size())) {
        throw std::invalid_argument("top_k is outside the vocabulary");
    }
    std::vector<float> scores(logits.begin(), logits.end());
    for (auto& score : scores) score /= *options.temperature;
    float threshold = -std::numeric_limits<float>::infinity();
    if (options.top_k) {
        auto sorted = scores;
        auto nth = sorted.begin() + *options.top_k - 1;
        std::nth_element(sorted.begin(), nth, sorted.end(), std::greater<float>());
        threshold = *nth;
    }
    const auto maximum = *std::max_element(scores.begin(), scores.end());
    if (!std::isfinite(maximum)) throw std::invalid_argument("No finite sampling candidate");
    std::vector<float> weights(scores.size());
    for (size_t index = 0; index < scores.size(); ++index) {
        weights[index] = scores[index] < threshold ? 0.f : std::exp(scores[index] - maximum);
    }
    // Keep all ties at the kth score, matching the reference threshold mask.
    std::discrete_distribution<uint32_t> distribution(weights.begin(), weights.end());
    return distribution(random);
}

AudioGenerationEngine::AudioGenerationEngine(
    AudioGenerationBackend& backend, uint32_t text_run, uint32_t audio_run, uint64_t seed
) : _backend(backend), _text_run(text_run), _audio_run(audio_run), _random(seed) {
    if (!text_run || !audio_run) throw std::invalid_argument("Audio generation runs must be positive");
}

void AudioGenerationEngine::generate(
    Mode mode, std::span<const float> prompt_embeddings, uint32_t max_new_tokens,
    const AudioSamplingOptions& text_options, const AudioSamplingOptions& audio_options,
    const Callback& callback
) {
    std::lock_guard lock(_generation_mutex);
    const auto width = _backend.hidden_size();
    if (!width || prompt_embeddings.empty() || prompt_embeddings.size() % width || !callback) {
        throw std::invalid_argument("Invalid audio generation prompt or callback");
    }
    _stopped.store(false);
    _backend.reset();
    std::vector<float> input(prompt_embeddings.begin(), prompt_embeddings.end());
    bool audio = false;
    bool text_done = false;
    int64_t left = _text_run;
    for (uint32_t index = 0; index < max_new_tokens && !_stopped.load(); ++index) {
        if (mode == Mode::Interleaved) --left;
        auto hidden = _backend.forward(input);
        if (hidden.size() != width) throw std::runtime_error("Audio backbone returned an invalid hidden state");
        AudioGeneratedToken token;
        token.is_audio = audio;
        if (!audio) {
            token.text = sample_audio_model_token(_backend.text_logits(hidden), text_options, _random);
            // Sequential generation yields EOS. Interleaved generation stops before yielding it.
            if (mode == Mode::Interleaved && token.text == 7) return;
            if (!callback(token) || _stopped.load()) return;
            if (mode == Mode::Sequential) {
                if (token.text == 7) return;
                if (token.text == 128) audio = true;
            } else {
                if (token.text == 130) text_done = true;
                if (left == 0 || text_done) {
                    audio = true;
                    left = _audio_run;
                }
            }
            input = _backend.text_embedding(token.text);
        } else {
            // The backend executes every depth step, including steps following an
            // end-of-audio semantic token. Overwrite the frame only after sampling.
            token.codes = _backend.audio_codes(hidden, audio_options, _random);
            if (mode == Mode::Interleaved && left == 0 && !text_done) {
                audio = false;
                left = _text_run;
            }
            if (token.codes[0] == 2048) {
                token.codes.fill(2048);
                audio = false;
            }
            if (!callback(token) || _stopped.load()) return;
            input = _backend.audio_embedding(token.codes);
        }
        if (input.size() != width) throw std::runtime_error("Audio embedding has the wrong width");
    }
}

struct AudioWaveformSynthesizer::Impl {
    std::vector<float> window;
    std::vector<float> sum;
    std::vector<float> envelope;
    fftw_complex* frequency = nullptr;
    double* time = nullptr;
    fftw_plan plan = nullptr;
    size_t skip = (FFT_SIZE - HOP) / 2;
    bool seen_frame = false;
    bool closed = false;

    explicit Impl(std::span<const float> source) : window(source.begin(), source.end()) {
        if (window.size() != FFT_SIZE || std::any_of(window.begin(), window.end(), [](float x) {
            return !std::isfinite(x);
        })) throw std::invalid_argument("Invalid audio synthesis window");
        frequency = fftw_alloc_complex(BINS);
        time = fftw_alloc_real(FFT_SIZE);
        if (!frequency || !time) {
            fftw_free(frequency);
            fftw_free(time);
            throw std::bad_alloc();
        }
        plan = fftw_plan_dft_c2r_1d(FFT_SIZE, frequency, time, FFTW_ESTIMATE);
        if (!plan) {
            fftw_free(frequency);
            fftw_free(time);
            throw std::runtime_error("Cannot create audio inverse FFT plan");
        }
    }
    ~Impl() {
        if (plan) fftw_destroy_plan(plan);
        fftw_free(frequency);
        fftw_free(time);
    }
    std::vector<float> take(size_t count) {
        const size_t drop = std::min(skip, count);
        skip -= drop;
        std::vector<float> output;
        output.reserve(count - drop);
        for (size_t index = drop; index < count; ++index) {
            if (!(envelope[index] > 1e-11f)) throw std::runtime_error("Invalid audio overlap envelope");
            output.push_back(sum[index] / envelope[index]);
        }
        sum.erase(sum.begin(), sum.begin() + count);
        envelope.erase(envelope.begin(), envelope.begin() + count);
        return output;
    }
};

AudioWaveformSynthesizer::AudioWaveformSynthesizer(std::span<const float> window)
    : _impl(new Impl(window)) {}
AudioWaveformSynthesizer::~AudioWaveformSynthesizer() { delete _impl; }

std::vector<float> AudioWaveformSynthesizer::push(std::span<const float> spectrum) {
    auto& state = *_impl;
    if (state.closed) throw std::logic_error("Audio synthesis is finished; reset before another stream");
    if (spectrum.size() % (2 * BINS)) throw std::invalid_argument("Invalid audio spectrum shape");
    if (std::any_of(spectrum.begin(), spectrum.end(), [](float value) {
        return !std::isfinite(value);
    })) throw std::invalid_argument("Non-finite audio spectrum");
    const auto frames = spectrum.size() / (2 * BINS);
    if (!frames) return {};
    state.sum.resize(frames * HOP + FFT_SIZE - HOP, 0.f);
    state.envelope.resize(state.sum.size(), 0.f);
    for (size_t frame = 0; frame < frames; ++frame) {
        const auto data = spectrum.subspan(frame * 2 * BINS, 2 * BINS);
        for (size_t bin = 0; bin < BINS; ++bin) {
            const auto magnitude = std::exp(data[bin]);
            if (!std::isfinite(magnitude)) throw std::invalid_argument("Audio spectrum magnitude overflow");
            state.frequency[bin][0] = magnitude * std::cos(data[BINS + bin]);
            state.frequency[bin][1] = magnitude * std::sin(data[BINS + bin]);
        }
        fftw_execute(state.plan);
        for (size_t index = 0; index < FFT_SIZE; ++index) {
            const float window = state.window[index];
            const float value = static_cast<float>(state.time[index] / FFT_SIZE) * window;
            state.sum[frame * HOP + index] += value;
            state.envelope[frame * HOP + index] += window * window;
        }
    }
    state.seen_frame = true;
    return state.take(frames * HOP);
}

std::vector<float> AudioWaveformSynthesizer::finish() {
    auto& state = *_impl;
    if (state.closed) throw std::logic_error("Audio synthesis was already finished");
    state.closed = true;
    if (!state.seen_frame) return {};
    auto output = state.take((FFT_SIZE - HOP) / 2);
    state.sum.clear();
    state.envelope.clear();
    return output;
}

void AudioWaveformSynthesizer::reset() {
    auto& state = *_impl;
    state.sum.clear();
    state.envelope.clear();
    state.skip = (FFT_SIZE - HOP) / 2;
    state.seen_frame = false;
    state.closed = false;
}

struct AudioFeatureExtractor::Impl {
    std::vector<float> window, filters;
    std::array<size_t, 128> filter_begin{}, filter_end{};
    double* time = nullptr;
    fftw_complex* frequency = nullptr;
    fftw_plan plan = nullptr;
    Impl(std::span<const float> source_window, std::span<const float> source_filters)
        : window(source_window.begin(), source_window.end()), filters(source_filters.begin(), source_filters.end()) {
        if (window.size() != 400 || filters.size() != 128 * 257) throw std::invalid_argument("Invalid log-mel assets");
        for (size_t feature = 0; feature < filter_begin.size(); ++feature) {
            size_t begin = 0, end = 257;
            while (begin < end && filters[feature * 257 + begin] == 0.f) ++begin;
            while (end > begin && filters[feature * 257 + end - 1] == 0.f) --end;
            // Retain complete four-float groups so the Modalix compiler uses
            // the same multiply/add rounding as the original dense loop.
            if (begin < end) {
                begin = begin / 4 * 4;
                end = std::min(size_t{257}, (end + 3) / 4 * 4);
            }
            filter_begin[feature] = begin;
            filter_end[feature] = end;
        }
        time = fftw_alloc_real(512);
        frequency = fftw_alloc_complex(257);
        if (!time || !frequency) {
            fftw_free(time); fftw_free(frequency); throw std::bad_alloc();
        }
        plan = fftw_plan_dft_r2c_1d(512, time, frequency, FFTW_ESTIMATE);
        if (!plan) { fftw_free(time); fftw_free(frequency); throw std::runtime_error("Cannot create audio analysis FFT plan"); }
    }
    ~Impl() {
        if (plan) fftw_destroy_plan(plan);
        fftw_free(time); fftw_free(frequency);
    }
};

AudioFeatureExtractor::AudioFeatureExtractor(std::span<const float> window, std::span<const float> filters)
    : _impl(new Impl(window, filters)) {}
AudioFeatureExtractor::~AudioFeatureExtractor() { delete _impl; }

AudioMelFeatures AudioFeatureExtractor::process(std::span<const float> pcm) {
    if (pcm.empty() || std::any_of(pcm.begin(), pcm.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Expected nonempty finite mono PCM at 16 kHz");
    }
    auto& state = *_impl;
    const size_t valid = pcm.size() / 160;
    AudioMelFeatures result{std::vector<float>(128 * (valid + 1), 0.f), valid + 1};
    if (!valid) return result;
    std::vector<float> emphasized(pcm.size());
    emphasized[0] = pcm[0];
    for (size_t index = 1; index < pcm.size(); ++index) emphasized[index] = pcm[index] - .97f * pcm[index - 1];
    std::array<float, 257> power{};
    for (size_t frame = 0; frame < valid; ++frame) {
        std::fill_n(state.time, 512, 0.);
        for (size_t index = 0; index < 400; ++index) {
            const int64_t sample = static_cast<int64_t>(frame * 160 + index) - 200;
            if (sample >= 0 && static_cast<size_t>(sample) < emphasized.size()) {
                state.time[56 + index] = emphasized[sample] * state.window[index];
            }
        }
        fftw_execute(state.plan);
        for (size_t bin = 0; bin < 257; ++bin) {
            const float real = static_cast<float>(state.frequency[bin][0]);
            const float imaginary = static_cast<float>(state.frequency[bin][1]);
            const float magnitude = std::sqrt(real * real + imaginary * imaginary);
            power[bin] = magnitude * magnitude;
        }
        const bool finite_power = std::all_of(power.begin(), power.end(), [](float value) {
            return std::isfinite(value);
        });
        for (size_t feature = 0; feature < 128; ++feature) {
            float energy = 0.f;
            // Omit only leading/trailing zero coefficients, keeping nonzero
            // additions in their original order. Preserve 0*Inf/NaN behavior
            // when finite PCM overflows the float power calculation.
            const size_t begin = finite_power ? state.filter_begin[feature] : 0;
            const size_t end = finite_power ? state.filter_end[feature] : power.size();
            for (size_t bin = begin; bin < end; ++bin) energy += state.filters[feature * 257 + bin] * power[bin];
            result.values[feature * result.frames + frame] = std::log(energy + 0x1.p-24f);
        }
    }
    for (size_t feature = 0; feature < 128; ++feature) {
        auto* values = result.values.data() + feature * result.frames;
        double sum = 0.;
        for (size_t time = 0; time < valid; ++time) sum += values[time];
        const float mean = static_cast<float>(sum / valid);
        double variance = 0.;
        for (size_t time = 0; time < valid; ++time) {
            const float delta = values[time] - mean;
            variance += delta * delta;
        }
        const float deviation = (valid > 1 ? std::sqrt(static_cast<float>(variance / (valid - 1))) : 0.f) + 1e-5f;
        for (size_t time = 0; time < valid; ++time) values[time] = (values[time] - mean) / deviation;
        // The processor returns the centered STFT's extra frame as zero. Its
        // ChatState uses the physical frame count, including this final frame.
        values[valid] = 0.f;
    }
    return result;
}

} // namespace simaai::llima
