#include "lfm2_audio_model.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>

using namespace simaai::llima;

// Explicit experimental runner; keep the installed GenAI task ABI unchanged.
// Arguments: bundle text output-f32 [max-steps] [input-mono-16k-f32|-] [task] [repeats]
int main(int argc, char** argv) {
    try {
        using Clock = std::chrono::steady_clock;
        auto milliseconds = [](auto begin, auto end) {
            return std::chrono::duration<double, std::milli>(end - begin).count();
        };
        if (argc < 4 || argc > 8) throw std::invalid_argument(
            "Usage: sima_lmm_lfm2_audio_smoke BUNDLE TEXT OUTPUT_F32 [MAX_STEPS] [INPUT_16K_F32|-] [interleaved|asr|tts] [REPEATS]");
        const std::string task = argc >= 7 ? argv[6] : "interleaved";
        if (task != "interleaved" && task != "asr" && task != "tts") throw std::invalid_argument("Unknown audio task");
        const size_t repeats = argc == 8 ? std::stoul(argv[7]) : 1;
        if (!repeats || repeats > 100) throw std::invalid_argument("Repeat count must be between 1 and 100");
        const bool has_audio = argc >= 6 && std::string(argv[5]) != "-";
        if (task == "asr" && !has_audio) throw std::invalid_argument("ASR requires input mono PCM at 16 kHz");
        const auto system = task == "asr" ? "Perform ASR." : task == "tts"
            ? "Perform TTS. Use the UK male voice." : "Respond with interleaved text and audio.";
        const auto construction_started = Clock::now();
        Lfm2AudioModel model(argv[1]);
        const auto construction_finished = Clock::now();
        std::vector<float> pcm;
        if (has_audio) {
            const auto size = std::filesystem::file_size(argv[5]);
            if (!size || size % sizeof(float)) throw std::invalid_argument("Input must be raw mono float32 PCM at 16 kHz");
            pcm.resize(size / sizeof(float));
            std::ifstream input(argv[5], std::ios::binary);
            input.read(reinterpret_cast<char*>(pcm.data()), size);
            if (!input) throw std::runtime_error("Cannot read input PCM");
        }
        const auto maximum = argc >= 5 ? std::stoull(argv[4]) : 32;
        const auto mode = task == "interleaved" ? AudioGenerationEngine::Mode::Interleaved : AudioGenerationEngine::Mode::Sequential;
        const AudioSamplingOptions audio_options = task == "tts"
            ? AudioSamplingOptions{.temperature = .8f, .top_k = 64} : AudioSamplingOptions{};
        for (size_t trial = 0; trial < repeats; ++trial) {
            const auto request_started = Clock::now();
            std::vector<AudioPromptMessage> messages{
                {"system", system, {}}, {"user", argv[2], {}}
            };
            if (has_audio) messages.back().audio_embeddings = model.encode_pcm(pcm);
            const auto encoding_finished = Clock::now();
            const auto prompt = model.make_prompt(messages);
            const auto generation_started = Clock::now();
            // Keep weights resident, but reset all session state and the sampling
            // seed for each independent request through a fresh generation engine.
            AudioGenerationEngine engine(model);
            std::vector<uint32_t> text;
            std::vector<std::array<uint32_t, 8>> codes;
            std::optional<double> first_event, first_text, first_audio, first_pcm;
            engine.generate(mode, prompt, maximum, {}, audio_options, [&](const auto& token) {
                const auto elapsed = milliseconds(request_started, Clock::now());
                if (!first_event) first_event = elapsed;
                if (token.is_audio) {
                    if (std::none_of(token.codes.begin(),token.codes.end(),[](auto code) {return code == 2048;})) {
                        if (!first_audio) first_audio = elapsed;
                        codes.push_back(token.codes);
                    }
                } else {
                    text.push_back(token.text);
                    if (!first_text && model.decode_text(text).find_first_not_of(" \t\r\n") != std::string::npos) first_text = elapsed;
                }
                return true;
            });
            const auto generation_finished = Clock::now();
            // This runner buffers the complete code sequence before decoding.
            // first_audio_codes_ms is not playable-audio latency; first_pcm_ms
            // includes the complete generation plus this waveform decoder call.
            auto waveform = model.decode_codes(codes, true);
            const auto decoding_finished = Clock::now();
            if (!waveform.empty()) first_pcm = milliseconds(request_started, decoding_finished);
            std::ofstream output(argv[3], std::ios::binary);
            output.write(reinterpret_cast<const char*>(waveform.data()), waveform.size() * sizeof(float));
            output.close();
            if (!output) throw std::runtime_error("Cannot write output PCM");
            const auto request_finished = Clock::now();
            const auto elapsed = milliseconds(generation_started, decoding_finished) / 1000.;
            std::cout << "text=" << model.decode_text(text) << '\n'
                      << "text_tokens=" << text.size() << " audio_frames=" << codes.size()
                      << " samples=" << waveform.size() << " sample_rate=24000 elapsed_seconds=" << elapsed << '\n';
            nlohmann::json metrics{
                {"trial", trial}, {"state", trial ? "model_resident" : "first_request"}, {"task", task},
                {"constructor_ms", milliseconds(construction_started, construction_finished)},
                {"audio_encode_ms", has_audio ? milliseconds(request_started, encoding_finished) : 0.},
                {"prompt_prepare_ms", milliseconds(encoding_finished, generation_started)},
                {"generation_ms", milliseconds(generation_started, generation_finished)},
                {"waveform_decode_ms", milliseconds(generation_finished, decoding_finished)},
                {"request_ms", milliseconds(request_started, request_finished)},
                {"prompt_tokens", prompt.size() / model.hidden_size()}, {"max_steps", maximum},
                {"text_tokens", text.size()}, {"audio_frames", codes.size()}, {"samples", waveform.size()},
                {"input_audio_seconds", pcm.size() / 16000.}, {"output_audio_seconds", waveform.size() / 24000.},
                {"text", model.decode_text(text)}, {"waveform_delivery", "buffered_after_generation"},
            };
            auto add_optional = [&](const char* name, const auto& value) {
                metrics[name] = value ? nlohmann::json(*value) : nlohmann::json(nullptr);
            };
            add_optional("first_event_ms", first_event);
            add_optional("ttft_visible_ms", first_text);
            add_optional("first_audio_codes_ms", first_audio);
            add_optional("first_pcm_ms", first_pcm);
            std::cout << "metrics_json=" << metrics.dump() << '\n' << std::flush;
            if (text.empty() && codes.empty()) return 1;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "LFM2 Audio smoke failed: " << error.what() << '\n';
        return 1;
    }
}
