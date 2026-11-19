#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>
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
            auto first = std::make_unique<simaai::llima::WhisperModel>(model_dir);
            result = first->run_model(audio, "auto");
            try {
                first->run_model(audio, "invalid_language");
                throw std::logic_error("Invalid ASR language was accepted");
            } catch (const std::runtime_error&) {}
            first->wait_for_streamer_completion();
            if (first->run_model(audio, "auto").text != result.text)
                throw std::runtime_error("ASR exception recovery changed transcription");
            unsigned default_callbacks = 0, request_callbacks = 0;
            first->set_text_callback([&](const std::string&, bool, bool) { ++default_callbacks; });
            simaai::llima::TranscriptionOptions options;
            options.language = "auto";
            options.text_callback = [&](const std::string&, bool, bool) { ++request_callbacks; };
            if (first->run_model(audio, options).text != result.text || !request_callbacks || default_callbacks)
                throw std::runtime_error("ASR request callback isolation failed");
            first->run_model(audio, "auto");
            if (!default_callbacks) throw std::runtime_error("ASR request callback leaked into defaults");
            options.info_callback = [](const std::string& key, double) {
                if (key == "ttft") throw std::runtime_error("Intentional ASR callback failure");
            };
            bool callback_failed = false;
            try { first->run_model(audio, options); }
            catch (const std::runtime_error&) { callback_failed = true; }
            if (!callback_failed || first->run_model(audio, "auto").text != result.text)
                throw std::runtime_error("ASR callback failure/recovery failed");
            first->set_text_callback([](const std::string&, bool, bool) {});
            std::vector<float> silence(16000);
            const auto pcm = first->run_model_from_pcm(silence, 16000, "en");
            auto survivor = std::make_unique<simaai::llima::WhisperModel>(model_dir);
            first.reset();
            if (survivor->run_model_from_pcm(silence, 16000, "en").text != pcm.text)
                throw std::runtime_error("ASR PCM or shared-program lifetime changed transcription");
            if (survivor->run_model(audio, "auto").text != result.text)
                throw std::runtime_error("Releasing one ASR wrapper corrupted the survivor");
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
