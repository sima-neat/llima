#include "setup.hpp"
#include "whisper_model.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

/*
 * DVT-only, end-to-end Whisper qualification for the direct kernel path.
 *
 * Usage:
 *   llima_direct_whisper <compiled-whisper-directory> <audio-file>
 *
 * The test deliberately calls the production file/preprocessor path rather
 * than a single QMLA ELF.  It covers encoder residency, decoder init,
 * layer/cache/post segments, CPU/MLA ownership boundaries and tokenizer
 * decode.  Two independent transcriptions must be exactly repeatable.  The
 * existing model/application test remains the accuracy oracle; this test
 * proves that removing MLA-RT and Dispatcher did not make the execution
 * lifecycle nondeterministic.
 *
 * Inspect this executable with `file` and run the AArch64 artifact only on an
 * authorized Modalix DVT.  It must never be executed in the x86 build
 * container.
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << " <compiled-whisper-directory> <audio-file>\n";
        return 64;
    }
    const std::filesystem::path model_path(argv[1]);
    const std::filesystem::path audio_path(argv[2]);
    if (!std::filesystem::is_directory(model_path / "devkit") ||
        !std::filesystem::is_directory(model_path / "elf_files") ||
        !std::filesystem::is_regular_file(audio_path)) {
        std::cerr << "invalid model or audio path\n";
        return 65;
    }

    simaai::llima::connect(
        {}, "/tmp/llima-direct-whisper.log", spdlog::level::warn
    );
    simaai::llima::WhisperModel::TranscriptionResult first;
    simaai::llima::WhisperModel::TranscriptionResult second;
    try {
        /* Destroy model/buffer owners before disconnecting their session. */
        simaai::llima::WhisperModel model(model_path);
        first = model.run_model(audio_path, "en", "transcribe");
        second = model.run_model(audio_path, "en", "transcribe");
    } catch (const std::exception& error) {
        simaai::llima::disconnect();
        std::cerr << "Whisper failed: " << error.what() << "\n";
        return 1;
    }
    simaai::llima::disconnect();

    const bool probabilities_valid =
        std::isfinite(first.no_speech_prob) &&
        first.no_speech_prob >= 0.0F && first.no_speech_prob <= 1.0F &&
        (!first.avg_logprob.has_value() ||
         std::isfinite(first.avg_logprob.value()));
    if (first.text.empty() || first.text != second.text ||
        first.language != "en" || first.task != "transcribe" ||
        second.language != first.language || second.task != first.task ||
        !probabilities_valid) {
        std::cerr << "Whisper repeatability/metadata mismatch: text_bytes="
                  << first.text.size() << " language=" << first.language
                  << " task=" << first.task << "\n";
        return 2;
    }

    std::cout << "LLIMA_DIRECT_WHISPER_PASS text_repeat=EXACT"
              << " language=" << first.language
              << " task=" << first.task
              << " text_bytes=" << first.text.size()
              << " transcript=" << first.text << "\n";
    return 0;
}
