// Experimental single-session audio server. Compiler Python is not required.
#include "lfm2_audio_model.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>

using namespace simaai::llima;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }

std::string pcm_base64(std::span<const float> samples) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::vector<uint8_t> bytes;
    bytes.reserve(samples.size() * 2);
    for (float sample : samples) {
        if (!std::isfinite(sample)) throw std::runtime_error("Decoder produced nonfinite PCM");
        const auto value = static_cast<uint16_t>(static_cast<int16_t>(
            std::lrint(std::clamp(sample, -1.f, 1.f) * 32767.f)));
        bytes.push_back(value & 255); bytes.push_back(value >> 8);
    }
    std::string output;
    output.reserve((bytes.size() + 2) / 3 * 4);
    for (size_t i = 0; i < bytes.size(); i += 3) {
        const uint32_t bits = (uint32_t(bytes[i]) << 16)
            | (i + 1 < bytes.size() ? uint32_t(bytes[i + 1]) << 8 : 0)
            | (i + 2 < bytes.size() ? bytes[i + 2] : 0);
        output += alphabet[(bits >> 18) & 63]; output += alphabet[(bits >> 12) & 63];
        output += i + 1 < bytes.size() ? alphabet[(bits >> 6) & 63] : '=';
        output += i + 2 < bytes.size() ? alphabet[bits & 63] : '=';
    }
    return output;
}

struct Request {
    std::string id, task = "interleaved", text;
    uint32_t max_steps = 320;
    std::vector<float> pcm;
    std::atomic<bool> cancelled{false};
    AudioGenerationEngine engine;
    explicit Request(Lfm2AudioModel& model) : engine(model) {}
    void stop() { cancelled.store(true); engine.stop(); }
};

void error(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(Json{{"error", message}}.dump(), "application/json");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3 || argc > 5) throw std::invalid_argument(
            "Usage: sima_lmm_lfm2_audio_server BUNDLE WEB_DIRECTORY [PORT=8096] [HOST=127.0.0.1]");
        const auto root = std::filesystem::path(argv[1]);
        Json config;
        std::ifstream(root / "audio_runtime.json") >> config;
        const size_t max_samples = config.at("options").at("audio_frames").get<size_t>() * 160 - 1;
        const size_t context = config.at("options").at("context_length");
        const int port = argc >= 4 ? std::stoi(argv[3]) : 8096;
        if (port < 1 || port > 65535) throw std::invalid_argument("Invalid TCP port");
        const std::string host = argc >= 5 ? argv[4] : "127.0.0.1";
        Lfm2AudioModel model(root);
        httplib::Server server;
        server.new_task_queue = [] { return new httplib::ThreadPool(4); };
        server.set_payload_max_length(std::max(size_t(65536), max_samples * sizeof(float)));
        server.set_read_timeout(10, 0);
        server.set_write_timeout(5, 0);
        if (!server.set_mount_point("/", argv[2])) throw std::invalid_argument("Web directory does not exist");
        std::mutex state_mutex;
        std::shared_ptr<Request> active;
        server.Get("/api/health", [&](const auto&, auto& response) {
            std::lock_guard lock(state_mutex);
            response.set_header("Cache-Control", "no-store");
            response.set_content(Json{{"model", "LFM2.5-Audio-1.5B"}, {"experimental", true},
                {"busy", bool(active)}, {"input_sample_rate", 16000}, {"output_sample_rate", 24000},
                {"max_input_samples", max_samples}, {"max_input_seconds", max_samples / 16000.},
                {"session", "independent requests; one generation at a time"}}.dump(), "application/json");
        });
        auto same_origin = [](const httplib::Request& req, httplib::Response& response) {
            if (req.has_header("Origin") && req.get_header_value("Origin") != "http://" + req.get_header_value("Host")) {
                error(response, 403, "Cross-origin requests are disabled"); return false;
            }
            return true;
        };
        server.Post("/api/stop", [&](const auto& req, auto& response) {
            if (!same_origin(req, response)) return;
            std::lock_guard lock(state_mutex);
            const bool matched = active && active->id == req.get_param_value("id");
            if (matched) active->stop();
            response.set_content(Json{{"stopping", matched}}.dump(), "application/json");
        });
        auto generate = [&](const httplib::Request& req, httplib::Response& response) {
            if (!same_origin(req, response)) return;
            auto job = std::make_shared<Request>(model);
            try {
                job->id = req.get_param_value("id");
                if (job->id.empty() || job->id.size() > 64) throw std::invalid_argument("Supply a request id of 1 to 64 characters");
                if (req.path == "/api/speech") {
                    if (req.get_header_value("Content-Type") != "application/octet-stream") {
                        error(response, 415, "Expected application/octet-stream: mono little-endian float32 PCM at 16 kHz"); return;
                    }
                    if (req.body.empty() || req.body.size() % sizeof(float) || req.body.size() / sizeof(float) > max_samples) {
                        throw std::invalid_argument("Input exceeds the compiled duration or is not mono float32 PCM");
                    }
                    job->pcm.resize(req.body.size() / sizeof(float));
                    std::memcpy(job->pcm.data(), req.body.data(), req.body.size());
                    if (std::any_of(job->pcm.begin(), job->pcm.end(), [](float x) { return !std::isfinite(x) || std::abs(x) > 1.f; })) {
                        throw std::invalid_argument("PCM samples must be finite and between -1 and 1");
                    }
                    if (req.has_param("task")) job->task = req.get_param_value("task");
                    if (job->task != "interleaved" && job->task != "asr") throw std::invalid_argument("Speech task must be interleaved or asr");
                } else {
                    if (req.get_header_value("Content-Type") != "application/json") {
                        error(response, 415, "Expected application/json"); return;
                    }
                    const auto body = Json::parse(req.body);
                    job->text = body.at("text").get<std::string>();
                    job->task = body.value("task", std::string("interleaved"));
                    if (body.contains("max_steps") && !body.at("max_steps").is_number_integer()) {
                        throw std::invalid_argument("max_steps must be an integer");
                    }
                    job->max_steps = body.value("max_steps", 320u);
                    if (job->text.empty() || job->text.size() > 2048) throw std::invalid_argument("Text must contain 1 to 2048 bytes");
                    if (job->task != "interleaved" && job->task != "tts") throw std::invalid_argument("Text task must be interleaved or tts");
                    if (!job->max_steps || job->max_steps > 512) throw std::invalid_argument("max_steps must be between 1 and 512");
                }
            } catch (const std::exception& exception) { error(response, 400, exception.what()); return; }
            {
                std::lock_guard lock(state_mutex);
                if (active) { error(response, 409, "Model is busy; stop or wait for the active request"); return; }
                active = job;
            }
            response.set_header("Cache-Control", "no-store");
            response.set_header("X-Accel-Buffering", "no");
            response.set_chunked_content_provider("application/x-ndjson",
                [&, job](size_t, httplib::DataSink& sink) {
                    const auto start = Clock::now();
                    auto elapsed = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - start).count(); };
                    auto send = [&](const Json& event) {
                        const auto line = event.dump(-1, ' ', false, Json::error_handler_t::replace) + '\n';
                        return sink.write(line.data(), line.size());
                    };
                    try {
                        if (!send({{"type", "start"}, {"id", job->id}})) return false;
                        const auto system = job->task == "asr" ? "Perform ASR." : job->task == "tts"
                            ? "Perform TTS. Use the UK male voice." : "Respond with interleaved text and audio.";
                        std::vector<AudioPromptMessage> messages{{"system", system, {}}, {"user", job->text, {}}};
                        if (!job->pcm.empty()) messages.back().audio_embeddings = model.encode_pcm(job->pcm);
                        auto prompt = model.make_prompt(messages);
                        if (prompt.size() / model.hidden_size() + job->max_steps > context) {
                            throw std::invalid_argument("Prompt and generation limit exceed the compiled context");
                        }
                        std::vector<uint32_t> text;
                        size_t frames = 0, samples = 0;
                        std::optional<double> first_text, first_codes, first_pcm;
                        bool connected = true;
                        auto emit_pcm = [&](const std::vector<float>& pcm) {
                            if (pcm.empty()) return true;
                            if (!first_pcm) first_pcm = elapsed();
                            samples += pcm.size();
                            return send({{"type", "audio"}, {"sample_rate", 24000}, {"pcm_s16le", pcm_base64(pcm)}});
                        };
                        if (!job->cancelled.load()) job->engine.generate(
                            job->task == "interleaved" ? AudioGenerationEngine::Mode::Interleaved : AudioGenerationEngine::Mode::Sequential,
                            prompt, job->max_steps, {}, job->task == "tts" ? AudioSamplingOptions{.temperature=.8f, .top_k=64} : AudioSamplingOptions{},
                            [&](const AudioGeneratedToken& token) {
                                if (job->cancelled.load() || !sink.is_writable()) return false;
                                if (token.is_audio) {
                                    if (std::all_of(token.codes.begin(), token.codes.end(), [](auto c) { return c < 2048; })) {
                                        ++frames;
                                        if (!first_codes) first_codes = elapsed();
                                        connected = emit_pcm(model.decode_codes(std::span(&token.codes, 1)));
                                    }
                                } else {
                                    text.push_back(token.text);
                                    const auto decoded = model.decode_text(text);
                                    if (!first_text && decoded.find_first_not_of(" \t\r\n") != std::string::npos) first_text = elapsed();
                                    connected = send({{"type", "text"}, {"text", decoded}});
                                }
                                return connected && !job->cancelled.load();
                            });
                        if (connected && !job->cancelled.load()) connected = emit_pcm(model.decode_codes({}, true));
                        Json metrics{{"type", "done"}, {"id", job->id}, {"cancelled", job->cancelled.load()},
                            {"request_ms", elapsed()}, {"audio_frames", frames}, {"samples", samples},
                            {"output_audio_seconds", samples / 24000.}, {"text", model.decode_text(text)},
                            {"input_audio_seconds", job->pcm.size() / 16000.}, {"waveform_delivery", "streamed"}};
                        metrics["ttft_visible_ms"] = first_text ? Json(*first_text) : Json(nullptr);
                        metrics["first_audio_codes_ms"] = first_codes ? Json(*first_codes) : Json(nullptr);
                        metrics["first_pcm_ms"] = first_pcm ? Json(*first_pcm) : Json(nullptr);
                        std::cout << "metrics_json=" << metrics.dump() << std::endl;
                        if (!connected || !send(metrics)) return false;
                    } catch (const std::exception& exception) {
                        if (!send({{"type", "error"}, {"error", exception.what()}})) return false;
                    }
                    sink.done(); return true;
                }, [&, job](bool) {
                    std::lock_guard lock(state_mutex);
                    if (active == job) active.reset();
                });
        };
        server.Post("/api/speech", generate);
        server.Post("/api/text", generate);
        std::signal(SIGINT, interrupt); std::signal(SIGTERM, interrupt); std::signal(SIGHUP, interrupt);
        std::jthread monitor([&](std::stop_token stop) {
            while (!stop.stop_requested()) {
                if (interrupted) {
                    { std::lock_guard lock(state_mutex); if (active) active->stop(); }
                    server.stop(); return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
        std::cout << "LFM2 audio server http://" << host << ':' << port << " (single session)" << std::endl;
        const bool listened = server.listen(host, port);
        monitor.request_stop();
        return listened ? 0 : 1;
    } catch (const std::exception& exception) {
        std::cerr << "Audio server: " << exception.what() << '\n'; return 1;
    }
}
