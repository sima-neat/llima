#include <fstream>
#include <variant>
#include <vector>

#include "web.hpp"

#include <fmt/chrono.h>
#include <nlohmann/json.hpp>


#include "tool_call_parser.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

static std::string get_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    return fmt::format("{:%FT%T%z}", now);
}

WEB::WEB(
    std::filesystem::path vlm_model_path,
    std::optional<std::filesystem::path> whisper_model_path,
    std::optional<std::filesystem::path> draft_model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    bool do_parallel_load
) : _vision_language_model_ptr(
        std::make_unique<VisionLanguageModel>(
            vlm_model_path, system_prompt, chat_template,
            do_parallel_load
        )
    )
{
    if (_singleton_ptr)
        throw std::runtime_error("Only one WEB instance can be created");
    _singleton_ptr = this;

    if (whisper_model_path.has_value()) {
        _whisper_model_ptr = std::make_unique<WhisperModel>(
            whisper_model_path.value(), do_parallel_load
        );
    }

    if (draft_model_path.has_value()) {
        _vision_language_draft_model_ptr = std::make_unique<VisionLanguageModel>(
            draft_model_path.value(), system_prompt, chat_template, do_parallel_load
        );
        _vision_language_model_ptr->set_draft_vlm(_vision_language_draft_model_ptr.get());
    }

    auto llima_logger = spdlog::get("llima");
    _logger = llima_logger? llima_logger->clone("WEB") : spdlog::default_logger();

    struct sigaction new_sigint_action;
    new_sigint_action.sa_handler = [](int sig) {
        assert(sig == SIGINT);
        if (_singleton_ptr) {
            _singleton_ptr->stop();
        }
    };
    sigemptyset(&new_sigint_action.sa_mask);
    new_sigint_action.sa_flags = 0;
    sigaction(SIGINT, &new_sigint_action, &_old_sigint_action);
}


WEB::~WEB() {
    stop();
    if (std::filesystem::is_regular_file(_AUDIO_FILE_NAME)) {
        std::filesystem::remove(_AUDIO_FILE_NAME);
    }
    sigaction(SIGINT, &_old_sigint_action, nullptr);
}


void WEB::run() {
    _http_server.Post(
        "/stop",
        [this](const httplib::Request& req, httplib::Response& res) {
            this->_handle_stop(req, res);
        }
    );

    _http_server.Post(
        "/set_lora",
        [this](const httplib::Request& req, httplib::Response& res) {
            this->_handle_set_lora(req, res);
        }
    );

    _http_server.Post(
        "/unset_lora",
        [this](const httplib::Request& req, httplib::Response& res) {
            this->_handle_unset_lora(req, res);
        }
    );

    // OpenAI protocol
    auto openai_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_handle_chat_completions(req, res, true);
    };
    _http_server.Post("/v1/chat/completions", openai_handler);
    _http_server.Post("/v1/completions", openai_handler);

    // Ollama protocol
    auto ollama_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_handle_chat_completions(req, res, false);
    };
    _http_server.Post("/api/chat", ollama_handler);
    _http_server.Post("/api/generate", ollama_handler);

    // Audio transcription endpoints (OpenAI compatible)
    auto audio_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_handle_audio_transcriptions(req, res);
    };
    _http_server.Post("/v1/audio/transcriptions", audio_handler);
    _http_server.Post("/audio/transcriptions", audio_handler);  // Alternative route without /v1

    auto cors_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_set_cors_headers(res);
        res.status = 200;
    };

    _http_server.Options("/v1/chat/completions", cors_handler);
    _http_server.Options("/api/chat", cors_handler);
    _http_server.Options("/api/generate", cors_handler);
    _http_server.Options("/v1/chat", cors_handler);
    _http_server.Options("/v1/audio/transcriptions", cors_handler);
    _http_server.Options("/audio/transcriptions", cors_handler);
    _http_server.Options("/stop", cors_handler);

    auto msg = fmt::format("Starting the HTTP server and listening on port {}", _SERVER_PORT);
    _logger->info(msg);
    std::cout << msg << std::endl << std::flush;
    _http_server.listen("0.0.0.0", _SERVER_PORT);
}


void WEB::stop() {
    _logger->info("User interrupt received. Stopping the HTTP server.");
    _http_server.stop();
}


void WEB::_stop_and_detach_vlm_thread() {
    if (_vlm_thread.joinable()) {
        _logger->info("Received stop from remote.");
        _vision_language_model_ptr->stop_model();

        // Wait for VLM to send STOP token and queue to drain
        _vision_language_model_ptr->wait_for_streamer_completion();

        _vlm_thread.detach();
        _logger->info("VLM stopped.");
    }
}

void WEB::_handle_stop(const httplib::Request& req, httplib::Response& res) {
    _set_cors_headers(res);

    _stop_and_detach_vlm_thread();

    // Send response immediately
    res.status = 200;
    _logger->info("VLM completed.");
}

void WEB::_set_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

nlohmann::ordered_json WEB::_parse_endpoint_messages(const nlohmann::json& messages) {
    nlohmann::ordered_json internal_messages = nlohmann::ordered_json::array();

    for (const auto& msg : messages) {
        nlohmann::ordered_json internal_msg;
        internal_msg["role"] = msg.value("role", "user");

        // Handle Ollama's images array format first
        bool has_ollama_images = msg.contains("images") && msg["images"].is_array();

        // Prepare content array (may be from content field or constructed from text + images)
        nlohmann::ordered_json internal_content = nlohmann::ordered_json::array();

        if (msg.contains("content")) {
            if (msg["content"].is_string()) {
                // Simple text content
                if (has_ollama_images) {
                    // Need to convert to array format to include images
                    internal_content.push_back({{"type", "text"}, {"text", msg["content"]}});
                } else {
                    // Just text, no images. Set content and fall through so that
                    // tool_calls / tool_call_id preservation below still runs.
                    internal_msg["content"] = msg["content"];
                }
            } else if (msg["content"].is_array()) {
                // OpenAI array format:
                // [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
                for (const auto& item : msg["content"]) {
                    std::string type = item.value("type", "");

                    if (type == "image_url") {
                        // OpenAI standard format:
                        // {"type": "image_url", "image_url": {"url": "data:..."}}
                        std::string url = item["image_url"].value("url", "");
                            internal_content.push_back({{"type", "image"}, {"image", url}});
                    } else if (type == "image" && item.contains("image")) {
                        // Internal format: {"type": "image", "image": "data:..."}
                        internal_content.push_back(nlohmann::ordered_json(item));
                    } else {
                        // Other types (text, etc.) - pass through as-is
                        internal_content.push_back(nlohmann::ordered_json(item));
                    }
                }
            }
        }

        // Handle Ollama's images array: convert base64 to data URI
        if (has_ollama_images) {
            for (const auto& image_data : msg["images"]) {
                if (image_data.is_string()) {
                    std::string base64_data = image_data.get<std::string>();
                    // Ollama sends raw base64, convert to data URI
                    // Default to JPEG, though we could try to detect format
                    std::string data_uri = "data:image/jpeg;base64," + base64_data;
                    internal_content.push_back({{"type", "image"}, {"image", data_uri}});
                }
            }
        }

        // Set the content (either original string or constructed array)
        if (internal_content.is_array() && !internal_content.empty()) {
            // Check if content array contains ONLY text and no images
            bool has_images = false;
            std::string combined_text = "";

            for (const auto& item : internal_content) {
                std::string item_type = item.value("type", "");
                if (item_type == "image") {
                    has_images = true;
                    break;
                } else if (item_type == "text") {
                    if (!combined_text.empty()) combined_text += " ";
                    combined_text += item.value("text", "");
                }
            }

            // If only text (no images), collapse to string for tokenizer compatibility
            if (!has_images && !combined_text.empty()) {
                internal_msg["content"] = combined_text;
            } else {
                internal_msg["content"] = internal_content;
            }
        } else if (!msg.contains("content") || msg["content"].is_null()) {
            // Assistant messages carrying only tool_calls have null/missing content
            // but must be preserved so the chat template can render the prior tool
            // dispatch.
            const bool has_valid_tool_calls =
                msg.contains("tool_calls") && msg["tool_calls"].is_array() &&
                !msg["tool_calls"].empty();
            if (!has_valid_tool_calls) {
                continue;
            }
            internal_msg["content"] = "";
        }

        if (msg.contains("tool_calls") && msg["tool_calls"].is_array()) {
            internal_msg["tool_calls"] = msg["tool_calls"];
        }
        if (msg.contains("tool_call_id") && msg["tool_call_id"].is_string()) {
            internal_msg["tool_call_id"] = msg["tool_call_id"];
        }

        internal_messages.push_back(internal_msg);
    }
    return internal_messages;
}

std::string WEB::_format_openai_sse_chunk(
    const std::string& content,
    const std::string& model,
    bool finished,
    std::optional<std::string> finish_reason,
    std::optional<double> ttft,
    std::optional<double> tps,
    bool from_draft
) {
    nlohmann::json chunk;
    chunk["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
    chunk["object"] = "chat.completion.chunk";
    chunk["created"] = std::time(nullptr);
    chunk["model"] = model;
    chunk["system_fingerprint"] = "fp_sima_vlm";

    // Add performance metrics (custom extension)
    if (ttft.has_value()) {
        chunk["ttft"] = ttft.value();  // in seconds
    }
    if (tps.has_value()) {
        chunk["tps"] = tps.value();  // tokens per second
    }

    nlohmann::json choice;
    choice["index"] = 0;

    if (finished) {
        choice["delta"] = nlohmann::json::object();
        choice["finish_reason"] = finish_reason.value_or("stop");
    } else {
        // Custom extension: mark chunks whose tokens were accepted from the
        // draft model during speculative decoding so the client can render
        // them differently.
        choice["delta"] = {{"content", content}, {"from_draft", from_draft}};
        choice["finish_reason"] = nullptr;
    }

    chunk["choices"] = nlohmann::json::array({choice});
    return "data: " + chunk.dump() + "\n\n";
}

std::string WEB::_format_ollama_ndjson_chunk(
    const std::string& content,
    const std::string& model,
    bool finished,
    std::optional<std::string> finish_reason,
    std::optional<double> ttft,
    std::optional<double> tps,
    bool from_draft
) {
    nlohmann::json chunk;
    chunk["model"] = model;
    chunk["created_at"] = get_iso_timestamp();
    chunk["done"] = finished;

    // Add performance metrics (custom extension)
    if (ttft.has_value()) {
        chunk["ttft"] = ttft.value();
    }
    if (tps.has_value()) {
        chunk["tps"] = tps.value();
    }

    if (finished && finish_reason.has_value()) {
        chunk["finish_reason"] = finish_reason.value();
    }

    if (!finished) {
        chunk["message"] = {{"role", "assistant"}, {"content", content}};
        chunk["response"] = content;
        // Custom extension: flag draft-accepted chunks for client rendering.
        chunk["from_draft"] = from_draft;
    }

    return chunk.dump() + "\n";
}

std::string WEB::_format_audio_sse_chunk(
    const std::string& text,
    bool finished,
    std::optional<std::string> finish_reason,
    std::optional<double> ttft,
    std::optional<double> tps
) {
    nlohmann::json chunk;
    chunk["object"] = finished ? "audio.transcription.done" : "audio.transcription.chunk";
    chunk["text"] = text;
    if (finished) {
        chunk["finish_reason"] = finish_reason.value_or("stop");
    }
    if (ttft.has_value()) {
        chunk["ttft"] = ttft.value();
    }
    if (tps.has_value()) {
        chunk["tps"] = tps.value();
    }
    return "data: " + chunk.dump() + "\n\n";
}


void WEB::_handle_chat_completions(
    const httplib::Request& req,
    httplib::Response& res,
    bool is_openai
) {
    _set_cors_headers(res);
    try {
        std::string model;
        bool stream = false;

        // 1. Prepare
        std::optional<Chat> chat_opt = _prepare_chat_context(req, res, model, stream);
        if (!chat_opt.has_value()) {
            return;
        }
        Chat chat = std::move(chat_opt.value());

        // 2. Barge-in check
        _stop_and_detach_vlm_thread();

        // 3. Execution
        if (stream) {
            _execute_streaming_chat(res, chat, model, is_openai);
        } else {
            _execute_normal_chat(res, chat, model, is_openai);
        }
    } catch (const std::exception& e) {
        _logger->error("Error in chat handler: {}", e.what());
        res.status = 500;
        nlohmann::json error_body = {{"error", e.what()}};
        if (is_openai) {
             error_body = {
                {"error", {{"message", e.what()}, {"type", "internal_error"}}}
             };
        }
        res.set_content(error_body.dump(), "application/json");
    }
}

void WEB::_handle_audio_transcriptions(const httplib::Request& req, httplib::Response& res) {
    _set_cors_headers(res);
    try {
        if (!req.has_file("file")) {
            res.status = 400;
            res.set_content(R"({"error": "No 'file' part"})", "application/json");
            return;
        }

        // Barge-in check
        _stop_and_detach_vlm_thread();

        const auto file = req.get_file_value("file");

        std::ofstream output_file(_AUDIO_FILE_NAME, std::ios::out | std::ios::binary);
        output_file.write(file.content.c_str(), file.content.size());
        output_file.close();

        // Get language from form data
        std::string language = "en";
        // Older cpp-httplib stores plain multipart fields in params.
        if (req.has_param("language")) {
            language = req.get_param_value("language");
        } else if (req.has_file("language")) {
            const auto lang_file = req.get_file_value("language");
            if (!lang_file.content.empty()) {
                language = lang_file.content;
            }
        }

        if (!_whisper_model_ptr) {
            res.status = 503;
            res.set_content(R"({"error": "Whisper model not loaded"})", "application/json");
            return;
        }

        bool stream = false;
        if (req.has_param("stream")) {
            auto value = req.get_param_value("stream");
            stream = value == "true" || value == "1" || value == "True";
        } else if (req.has_file("stream")) {
            auto value = req.get_file_value("stream").content;
            stream = value == "true" || value == "1" || value == "True";
        }

        if (stream) {
            _execute_streaming_audio_transcription(res, language);
            return;
        }

        auto text = _whisper_model_ptr->run_model(_AUDIO_FILE_NAME, language);
        nlohmann::json response = {{"text", text}};
        res.set_content(response.dump(), "application/json");

    } catch (const std::exception& e) {
        _logger->error("Error in transcription: {}", e.what());
        res.status = 500;
        res.set_content(nlohmann::json({{"error", e.what()}}).dump(), "application/json");
    }
}

void WEB::_execute_streaming_audio_transcription(
    httplib::Response& res,
    const std::string& language
) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, language](size_t offset, httplib::DataSink &sink) {
            (void)offset;
            std::optional<double> ttft_value;
            std::optional<double> tps_value;

            struct WhisperCallbackGuard {
                WhisperModel* model;
                ~WhisperCallbackGuard() {
                    model->set_info_callback([](const std::string&, double) {});
                    model->set_text_callback([](const std::string&, bool, bool) {});
                }
            } callback_guard{_whisper_model_ptr.get()};

            auto info_callback = [&](const std::string& metric_type, double metric_value) {
                if (metric_type == "ttft") {
                    ttft_value = metric_value;
                    return;
                }
                if (metric_type == "tps") {
                    tps_value = metric_value;
                    return;
                }
                if (metric_type == "END" || metric_type == "FULL") {
                    const std::string finish_reason = metric_type == "FULL" ? "length" : "stop";
                    std::string chunk = _format_audio_sse_chunk(
                        "", true, finish_reason, ttft_value, tps_value
                    ) + "data: [DONE]\n\n";
                    sink.write(chunk.data(), chunk.size());
                }
            };

            auto text_callback = [&](const std::string& text, bool stream_end, bool) {
                if (text.empty())
                    return;
                std::string chunk = _format_audio_sse_chunk(text, false);
                sink.write(chunk.data(), chunk.size());
            };

            try {
                _whisper_model_ptr->set_info_callback(info_callback);
                _whisper_model_ptr->set_text_callback(text_callback);
                _whisper_model_ptr->run_model(_AUDIO_FILE_NAME, language);
            } catch (const std::exception& e) {
                nlohmann::json error;
                error["object"] = "audio.transcription.error";
                error["error"] = e.what();
                std::string chunk = "data: " + error.dump() + "\n\ndata: [DONE]\n\n";
                sink.write(chunk.data(), chunk.size());
            }

            sink.done();
            return true;
        }
    );
}



std::optional<Chat> WEB::_prepare_chat_context(
    const httplib::Request& req,
    httplib::Response& res,
    std::string& model,
    bool& stream
) {
    auto json_data = nlohmann::json::parse(req.body);
    model = json_data.value("model", "default-model");
    stream = json_data.value("stream", false);

    Chat chat = _vision_language_model_ptr->create_chat();
    if (json_data.contains("tools") && json_data["tools"].is_array()) {
        chat.set_tools(nlohmann::ordered_json(json_data["tools"]));
    }

    if (json_data.contains("messages")) {
        auto internal_messages = _parse_endpoint_messages(json_data["messages"]);
        chat.set_messages(internal_messages);
    } else if (json_data.contains("prompt")) {
        chat.add_query(json_data["prompt"]);
    } else {
        res.status = 400;
        res.set_content(R"({"error": "Missing messages or prompt"})", "application/json");
        return std::nullopt;
    }
    return chat;
}

static nlohmann::json openai_tool_calls_to_ollama(const nlohmann::json& openai_tool_calls);

void WEB::_execute_streaming_chat(
    httplib::Response& res,
    Chat& chat,
    const std::string& model,
    bool is_openai
) {
    res.set_header("Content-Type", is_openai ? "text/event-stream" : "application/x-ndjson");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider(
        is_openai ? "text/event-stream" : "application/x-ndjson",
        [this, chat, model, is_openai](size_t offset, httplib::DataSink &sink) {
            const bool has_tools = chat.has_tools();
            bool sent_initial_chunk = false;
            bool ttft_sent = false;
            std::optional<double> ttft_value;
            std::optional<double> tps_value;
            ToolCallStreamParser tool_parser;
            nlohmann::json pending_ollama_tool_calls = nullptr;

            auto send_openai_initial = [&]() {
                if (sent_initial_chunk) return;
                nlohmann::json initial_chunk;
                initial_chunk["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
                initial_chunk["object"] = "chat.completion.chunk";
                initial_chunk["created"] = std::time(nullptr);
                initial_chunk["model"] = model;
                initial_chunk["system_fingerprint"] = "fp_sima_vlm";
                initial_chunk["choices"] = nlohmann::json::array({{
                    {"index", 0},
                    {"delta", {{"role", "assistant"}, {"content", nullptr}}},
                    {"finish_reason", nullptr}
                }});
                std::string initial_output = "data: " + initial_chunk.dump() + "\n\n";
                sink.write(initial_output.data(), initial_output.size());
                sent_initial_chunk = true;
            };

            auto take_ttft_once = [&]() -> std::optional<double> {
                if (!ttft_sent && ttft_value.has_value()) {
                    ttft_sent = true;
                    return ttft_value;
                }
                return std::nullopt;
            };

            auto send_content = [&](const std::string& text) {
                if (text.empty()) return;
                if (is_openai) {
                    send_openai_initial();
                    auto chunk = _format_openai_sse_chunk(
                        text, model, false, std::nullopt, take_ttft_once(), tps_value
                    );
                    sink.write(chunk.data(), chunk.size());
                } else {
                    auto chunk = _format_ollama_ndjson_chunk(
                        text, model, false, std::nullopt, take_ttft_once(), tps_value
                    );
                    sink.write(chunk.data(), chunk.size());
                }
            };

            auto send_openai_tool_calls = [&](const nlohmann::json& parsed_tool_calls) {
                send_openai_initial();
                nlohmann::json delta_tool_calls = nlohmann::json::array();
                for (size_t i = 0; i < parsed_tool_calls.size(); ++i) {
                    nlohmann::json tool_call = parsed_tool_calls[i];
                    tool_call["index"] = static_cast<int>(i);
                    delta_tool_calls.push_back(tool_call);
                }

                nlohmann::json tool_chunk;
                tool_chunk["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
                tool_chunk["object"] = "chat.completion.chunk";
                tool_chunk["created"] = std::time(nullptr);
                tool_chunk["model"] = model;
                tool_chunk["system_fingerprint"] = "fp_sima_vlm";
                if (auto ttft_for_chunk = take_ttft_once(); ttft_for_chunk.has_value())
                    tool_chunk["ttft"] = ttft_for_chunk.value();
                if (tps_value.has_value())
                    tool_chunk["tps"] = tps_value.value();
                tool_chunk["choices"] = nlohmann::json::array({{
                    {"index", 0},
                    {"delta", {{"tool_calls", delta_tool_calls}}},
                    {"finish_reason", nullptr}
                }});
                std::string tool_output = "data: " + tool_chunk.dump() + "\n\n";
                sink.write(tool_output.data(), tool_output.size());
            };

            bool saw_tool_calls = false;
            auto handle_tool_parser_events = [&](std::vector<ToolCallStreamParser::Event> events) {
                for (auto& event : events) {
                    if (std::holds_alternative<ToolCallStreamParser::Content>(event)) {
                        send_content(std::get<ToolCallStreamParser::Content>(event).text);
                    } else {
                        const auto& calls = std::get<ToolCallStreamParser::ToolCalls>(event).calls;
                        saw_tool_calls = true;
                        if (is_openai) {
                            send_openai_tool_calls(calls);
                        } else {
                            pending_ollama_tool_calls = calls;
                        }
                    }
                }
            };

            // Info callback handles timing/status information
            auto info_callback = [&](const std::string& metric_type, double metric_value) {
                // Store TTFT metric
                if (metric_type == "ttft") {
                    ttft_value = metric_value;
                    return;
                }

                // Store TPS metric
                if (metric_type == "tps") {
                    tps_value = metric_value;
                    return;
                }

                // Handle END and FULL signals (metric_value is 0.0 for these)
                if (metric_type == "END" || metric_type == "FULL") {
                    const std::string default_finish_reason =
                        (metric_type == "FULL") ? "length" : "stop";

                    if (has_tools) {
                        handle_tool_parser_events(tool_parser.add("", true));

                        if (is_openai) {
                            send_openai_initial();
                            std::string finish_reason =
                                saw_tool_calls ? "tool_calls" : default_finish_reason;

                            std::string final_chunk =
                                _format_openai_sse_chunk("", model, true, finish_reason)
                                + "data: [DONE]\n\n";
                            sink.write(final_chunk.data(), final_chunk.size());
                        } else {
                            nlohmann::json final_obj;
                            final_obj["model"] = model;
                            final_obj["created_at"] = get_iso_timestamp();
                            final_obj["done"] = true;
                            final_obj["finish_reason"] = default_finish_reason;
                            if (ttft_value.has_value())
                                final_obj["ttft"] = ttft_value.value();
                            if (tps_value.has_value())
                                final_obj["tps"] = tps_value.value();

                            nlohmann::json message = {{"role", "assistant"}};
                            if (!pending_ollama_tool_calls.is_null()) {
                                message["content"] = "";
                                message["tool_calls"] =
                                    openai_tool_calls_to_ollama(pending_ollama_tool_calls);
                                final_obj["finish_reason"] = "tool_calls";
                            } else {
                                message["content"] = "";
                            }
                            final_obj["message"] = message;

                            std::string final_line = final_obj.dump() + "\n";
                            sink.write(final_line.data(), final_line.size());
                        }
                        return;
                    }

                    std::string finish_reason = default_finish_reason;
                    std::string formatted_chunk;

                    if (is_openai) {
                        formatted_chunk = _format_openai_sse_chunk(
                            "", model, true, finish_reason
                        ) + "data: [DONE]\n\n";
                    } else {
                        formatted_chunk = _format_ollama_ndjson_chunk(
                            "", model, true, finish_reason
                        );
                    }
                    sink.write(formatted_chunk.data(), formatted_chunk.size());
                }
            };

            // Text callback handles generated text chunks. from_draft is
            // attached to each streamed chunk as a JSON field so the client
            // can render draft-accepted text differently from target-only.
            auto text_callback = [&](
                const std::string& text, bool stream_end, bool from_draft
            ) {
                if (has_tools) {
                    handle_tool_parser_events(tool_parser.add(text, stream_end));
                    return;
                }

                // Send initial role chunk for OpenAI (first time only)
                if (is_openai && !sent_initial_chunk) {
                    nlohmann::json initial_chunk;
                    initial_chunk["id"] = "chatcmpl-" + std::to_string(std::time(nullptr));
                    initial_chunk["object"] = "chat.completion.chunk";
                    initial_chunk["created"] = std::time(nullptr);
                    initial_chunk["model"] = model;
                    initial_chunk["system_fingerprint"] = "fp_sima_vlm";

                    nlohmann::json initial_choice;
                    initial_choice["index"] = 0;
                    initial_choice["delta"] = {
                        {"role", "assistant"}, {"content", nullptr}
                    };
                    initial_choice["finish_reason"] = nullptr;
                    initial_chunk["choices"] = nlohmann::json::array({initial_choice});

                    std::string initial_output = "data: " + initial_chunk.dump() + "\n\n";
                    sink.write(initial_output.data(), initial_output.size());
                    sent_initial_chunk = true;
                }

                // Regular text chunk - include metrics
                std::string formatted_chunk;
                if (is_openai) {
                    std::optional<double> ttft_for_chunk = std::nullopt;
                    if (!ttft_sent && ttft_value.has_value()) {
                        ttft_for_chunk = ttft_value;
                        ttft_sent = true;
                    }
                    formatted_chunk = _format_openai_sse_chunk(
                        text, model, false, std::nullopt,
                        ttft_for_chunk,
                        tps_value,
                        from_draft
                    );
                } else {
                    std::optional<double> ttft_for_chunk = std::nullopt;
                    if (!ttft_sent && ttft_value.has_value()) {
                        ttft_for_chunk = ttft_value;
                        ttft_sent = true;
                    }
                    formatted_chunk = _format_ollama_ndjson_chunk(
                        text, model, false, std::nullopt,
                        ttft_for_chunk,
                        tps_value,
                        from_draft
                    );
                }

                sink.write(formatted_chunk.data(), formatted_chunk.size());
            };

            _vision_language_model_ptr->set_info_callback(info_callback);
            _vision_language_model_ptr->set_text_callback(text_callback);

            // Create thread and track it for /stop endpoint
            _vlm_thread = std::jthread([this, chat]() {
                _vision_language_model_ptr->run_model(chat);
            });

            // Wait for completion
            _vlm_thread.join();

            // No need to reset callbacks - they'll be overwritten on next request

            sink.done();
            return true;
        }
    );
}

static nlohmann::json openai_tool_calls_to_ollama(const nlohmann::json& openai_tool_calls) {
    nlohmann::json out = nlohmann::json::array();
    if (!openai_tool_calls.is_array()) {
        return out;
    }
    for (const auto& tool_call : openai_tool_calls) {
        if (!tool_call.contains("function") || !tool_call["function"].is_object()) {
            continue;
        }
        const auto& function = tool_call["function"];
        nlohmann::json args = nlohmann::json::object();
        if (function.contains("arguments")) {
            if (function["arguments"].is_string()) {
                try {
                    args = nlohmann::json::parse(function["arguments"].get<std::string>());
                } catch (...) {
                    args = nlohmann::json::object();
                }
            } else if (function["arguments"].is_object()) {
                args = function["arguments"];
            }
        }
        out.push_back({
            {"function", {
                {"name", function.value("name", std::string{})},
                {"arguments", args},
            }},
        });
    }
    return out;
}


void WEB::_execute_normal_chat(
    httplib::Response& res,
    Chat& chat,
    const std::string& model,
    bool is_openai
) {
    std::string full_response = "";

    // Info callback: ignore timing/status info in non-streaming mode
    auto info_callback = [](const std::string& metric_type, double metric_value) {
        // Do nothing - we don't need timing info in non-streaming mode
    };

    // Text callback: accumulate all text. from_draft is ignored in
    // non-streaming mode -- the whole response is returned as a single string.
    auto text_callback = [&](const std::string& text, bool stream_end, bool) {
        full_response += text;
    };

    _vision_language_model_ptr->set_info_callback(info_callback);
    _vision_language_model_ptr->set_text_callback(text_callback);

    // Run in thread for consistency with barge-in logic
    _vlm_thread = std::jthread([this, chat]() {
        _vision_language_model_ptr->run_model(chat);
    });
    _vlm_thread.join();

    // No need to reset callbacks - they'll be overwritten on next request

    nlohmann::json response;
    if (is_openai) {
        auto tool_calls = try_parse_tool_calls(full_response);
        nlohmann::json message = {{"role", "assistant"}};
        std::string finish_reason;
        if (!tool_calls.is_null()) {
            message["content"] = nullptr;
            message["tool_calls"] = tool_calls;
            finish_reason = "tool_calls";
        } else {
            message["content"] = full_response;
            finish_reason = "stop";
        }
        response = {
            {"id", "chatcmpl-" + std::to_string(std::time(nullptr))},
            {"object", "chat.completion"},
            {"created", std::time(nullptr)},
            {"model", model},
            {"choices", {{
                {"index", 0},
                {"message", message},
                {"finish_reason", finish_reason}
            }}}
        };
    } else {
        auto tool_calls = try_parse_tool_calls(full_response);
        nlohmann::json message = {{"role", "assistant"}};
        if (!tool_calls.is_null()) {
            message["content"] = "";
            message["tool_calls"] = openai_tool_calls_to_ollama(tool_calls);
        } else {
            message["content"] = full_response;
        }
        response = {
            {"model", model},
            {"created_at", get_iso_timestamp()},
            {"response", tool_calls.is_null() ? full_response : std::string("")},
            {"message", message},
            {"done", true}
        };
    }
    res.set_content(response.dump(), "application/json");
}


void WEB::_handle_set_lora(const httplib::Request& req, httplib::Response& res) {
    _set_cors_headers(res);

    auto json_data = nlohmann::json::parse(req.body);
    if (!json_data.contains("name")) {
        res.status = 400;
        res.set_content(R"({"error": "Missing name"})", "application/json");
        return;
    }

    _stop_and_detach_vlm_thread();

    try {
        _vision_language_model_ptr->set_reloc(json_data["name"]);
        res.status = 200;
        _logger->info("Set LoRA weights completed.");
    } catch (const std::exception& e) {
        _logger->error("Failed to set LoRA weights");
        res.status = 500;
        nlohmann::json error_body = {{"error", e.what()}};
        res.set_content(error_body.dump(), "application/json");
    }
}


void WEB::_handle_unset_lora(const httplib::Request& req, httplib::Response& res) {
    _set_cors_headers(res);

    _stop_and_detach_vlm_thread();
    _vision_language_model_ptr->unset_reloc();

    // Send response immediately
    res.status = 200;
    _logger->info("Set LoRA weights completed.");
}



}
}
