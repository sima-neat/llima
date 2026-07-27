#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

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

        simaai::llima::WhisperModel::TranscriptionResult result;
        {
            simaai::llima::WhisperModel model(model_dir);
            result = model.run_model(audio, "auto");
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
