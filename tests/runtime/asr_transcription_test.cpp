#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <spdlog/common.h>

#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "whisper_model.hpp"

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_ASR_MODEL";

std::string normalize_transcript(const std::string& value) {
    std::string normalized;
    bool pending_space = false;
    for (const unsigned char character : value) {
        if (std::isalnum(character)) {
            if (pending_space && !normalized.empty()) {
                normalized.push_back(' ');
            }
            normalized.push_back(static_cast<char>(std::tolower(character)));
            pending_space = false;
        } else if (std::isspace(character) || std::ispunct(character)) {
            pending_space = true;
        }
    }
    return normalized;
}

size_t count_occurrences(const std::string& text, const std::string& phrase) {
    size_t count = 0;
    for (auto pos = text.find(phrase); pos != std::string::npos; pos = text.find(phrase, pos + 1))
        ++count;
    return count;
}

}  // namespace

int main() {
    bool connected = false;
    try {
        const std::filesystem::path model_dir =
            simaai::llima::test::resolve_model_dir(
                kModelEnv,
                simaai::llima::test::kDefaultAsrModelName,
                "LLiMa ASR",
                "devkit/whisper_config.json"
            );
        const std::filesystem::path audio =
            simaai::llima::test::resolve_asset("why_is_the_sky_blue.wav");
        std::cout << "LLIMA_ASR model_dir=" << model_dir << '\n';
        std::cout << "LLIMA_ASR audio=" << audio << '\n';

        simaai::llima::connect(
            {},
            "/tmp/sima_lmm_asr_transcription_test.log",
            spdlog::level::info
        );
        connected = true;

        using TranscriptionOptions = simaai::llima::WhisperModel::TranscriptionOptions;
        simaai::llima::WhisperModel::TranscriptionResult result;
        std::string prompted_text;
        std::string long_prompt_text;
        std::string prefixed_long_prompt_text;
        std::string long_audio_text;
        std::string unconditioned_long_audio_text;
        std::string repeated_text;
        bool rejected_special_tokens = false;
        {
            simaai::llima::WhisperModel model(model_dir);
            result = model.run_model(audio, "auto");
            prompted_text = model.run_model(
                audio, TranscriptionOptions{.prompt = "Rayleigh scattering of sunlight."}
            ).text;
            // Both prompts exceed the token limit and share their last tokens, so truncation
            // must give identical decoder inputs. Long repetitive prompts make Whisper repeat
            // them, so only the equality is checked, not the transcript.
            std::string long_prompt;
            for (int i = 0; i < 100; ++i)
                long_prompt += "Rayleigh scattering. ";
            std::string prefixed_long_prompt;
            for (int i = 0; i < 40; ++i)
                prefixed_long_prompt += "Agenda item. ";
            prefixed_long_prompt += long_prompt;
            long_prompt_text = model.run_model(
                audio, TranscriptionOptions{.prompt = long_prompt}
            ).text;
            prefixed_long_prompt_text = model.run_model(
                audio, TranscriptionOptions{.prompt = prefixed_long_prompt}
            ).text;
            // The sample at 0 s and again at 31 s spans two 30 s windows.
            constexpr uint32_t sample_rate = simaai::llima::WhisperPreprocessor::SAMPLE_RATE;
            const auto sample =
                simaai::llima::WhisperPreprocessor(model_dir / "devkit").load_audio(audio);
            std::vector<float> long_audio(31 * sample_rate + sample.size(), 0.0F);
            std::copy_n(sample.data(), sample.size(), long_audio.begin());
            std::copy_n(sample.data(), sample.size(), long_audio.begin() + 31 * sample_rate);
            long_audio_text = model.run_model_from_pcm(
                long_audio, sample_rate, TranscriptionOptions{}
            ).text;
            unconditioned_long_audio_text = model.run_model_from_pcm(
                long_audio, sample_rate, TranscriptionOptions{.condition_on_previous_text = false}
            ).text;
            try {
                model.run_model(audio, TranscriptionOptions{.prompt = "<|en|> sky"});
            } catch (const std::invalid_argument&) {
                rejected_special_tokens = true;
            }
            repeated_text = model.run_model(audio, "auto").text;
        }

        simaai::llima::disconnect();
        connected = false;

        const std::string transcript = normalize_transcript(result.text);
        std::cout << "LLIMA_ASR text=" << result.text << '\n';
        std::cout << "LLIMA_ASR language=" << result.language << '\n';
        if (transcript.find("why is the sky blue") == std::string::npos) {
            throw std::runtime_error("Unexpected ASR transcript: " + result.text);
        }
        if (result.language != "en") {
            throw std::runtime_error("Unexpected ASR language: " + result.language);
        }
        if (
            !std::isfinite(result.no_speech_prob)
            || result.no_speech_prob < 0.0F
            || result.no_speech_prob > 1.0F
        ) {
            throw std::runtime_error("ASR no_speech_prob is invalid");
        }
        if (!result.avg_logprob.has_value() || !std::isfinite(*result.avg_logprob)) {
            throw std::runtime_error("ASR avg_logprob is invalid");
        }
        std::cout << "LLIMA_ASR prompted_text=" << prompted_text << '\n';
        if (normalize_transcript(prompted_text).find("why is the sky blue") == std::string::npos) {
            throw std::runtime_error("Unexpected prompted ASR transcript: " + prompted_text);
        }
        if (prefixed_long_prompt_text != long_prompt_text) {
            throw std::runtime_error("ASR prompt truncation did not keep the last prompt tokens");
        }
        std::cout << "LLIMA_ASR long_audio_text=" << long_audio_text << '\n';
        for (const auto& text : {long_audio_text, unconditioned_long_audio_text}) {
            if (count_occurrences(normalize_transcript(text), "why is the sky blue") != 2) {
                throw std::runtime_error("Unexpected long-form ASR transcript: " + text);
            }
        }
        if (!rejected_special_tokens) {
            throw std::runtime_error("ASR prompt with special tokens was not rejected");
        }
        if (repeated_text != result.text) {
            throw std::runtime_error("Unprompted ASR transcript changed after prompted runs");
        }
    } catch (const std::exception& error) {
        if (connected) {
            try {
                simaai::llima::disconnect();
            } catch (...) {
            }
        }
        std::cerr << "ASR transcription test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "ASR transcription test passed\n";
    return 0;
}
