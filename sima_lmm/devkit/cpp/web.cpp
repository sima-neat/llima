#include <fstream>
#include <variant>
#include <vector>

#include "web.hpp"

#include <fmt/chrono.h>
#include <nlohmann/json.hpp>


#include "reasoning_parser.hpp"
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
    bool enable_thinking
) : _vision_language_model_ptr(
        std::make_unique<VisionLanguageModel>(vlm_model_path, system_prompt, chat_template)
    ),
    _enable_thinking(enable_thinking)
{
    if (_singleton_ptr)
        throw std::runtime_error("Only one WEB instance can be created");
    _singleton_ptr = this;

    if (whisper_model_path.has_value()) {
        _whisper_model_ptr = std::make_unique<WhisperModel>(whisper_model_path.value());
    }

    if (draft_model_path.has_value()) {
        _vision_language_draft_model_ptr = std::make_unique<VisionLanguageModel>(
            draft_model_path.value(), system_prompt, chat_template
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
        this->_handle_chat_completions(req, res, ChatProtocol::OpenAI);
    };
    _http_server.Post("/v1/chat/completions", openai_handler);
    _http_server.Post("/v1/completions", openai_handler);

    // Ollama protocol
    auto ollama_chat_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_handle_chat_completions(req, res, ChatProtocol::OllamaChat);
    };
    auto ollama_generate_handler = [this](
        const httplib::Request& req, httplib::Response& res
    ) {
        this->_handle_chat_completions(req, res, ChatProtocol::OllamaGenerate);
    };
    _http_server.Post("/api/chat", ollama_chat_handler);
    _http_server.Post("/api/generate", ollama_generate_handler);

    // Audio transcription endpoints (OpenAI compatible)
    auto audio_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_handle_audio_transcriptions(req, res, "transcribe");
    };
    auto audio_translation_handler = [this](const httplib::Request& req, httplib::Response& res) {
        this->_handle_audio_transcriptions(req, res, "translate");
    };
    _http_server.Post("/v1/audio/transcriptions", audio_handler);
    _http_server.Post("/audio/transcriptions", audio_handler);  // Alternative route without /v1
    _http_server.Post("/v1/audio/translations", audio_translation_handler);
    _http_server.Post("/audio/translations", audio_translation_handler);

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
    _http_server.Options("/v1/audio/translations", cors_handler);
    _http_server.Options("/audio/translations", cors_handler);
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
    const std::string& completion_id,
    std::time_t created,
    bool finished,
    std::optional<std::string> finish_reason,
    std::optional<double> ttft,
    std::optional<double> tps,
    bool from_draft,
    bool reasoning
) {
    nlohmann::json chunk;
    chunk["id"] = completion_id;
    chunk["object"] = "chat.completion.chunk";
    chunk["created"] = created;
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
        choice["delta"] = {
            {reasoning ? "reasoning_content" : "content", content},
            {"from_draft", from_draft}
        };
        choice["finish_reason"] = nullptr;
    }

    chunk["choices"] = nlohmann::json::array({choice});
    return "data: " + chunk.dump() + "\n\n";
}

std::string WEB::_format_ollama_ndjson_chunk(
    const std::string& content,
    const std::string& model,
    ChatProtocol protocol,
    bool finished,
    std::optional<std::string> finish_reason,
    std::optional<double> ttft,
    std::optional<double> tps,
    bool from_draft,
    bool reasoning
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

    if (protocol == ChatProtocol::OllamaChat) {
        nlohmann::json message = {{"role", "assistant"}, {"content", ""}};
        if (!finished) {
            if (reasoning) {
                message["thinking"] = content;
            } else {
                message["content"] = content;
            }
        }
        chunk["message"] = std::move(message);
    } else {
        chunk["response"] = reasoning || finished ? std::string("") : content;
        if (!finished && reasoning) chunk["thinking"] = content;
    }

    if (!finished) {
        // Custom extension: flag draft-accepted chunks for client rendering.
        chunk["from_draft"] = from_draft;
    }

    return chunk.dump() + "\n";
}

std::string WEB::_format_audio_sse_chunk(
    const std::string& text,
    const std::string& event_task,
    bool finished,
    std::optional<std::string> finish_reason,
    std::optional<double> ttft,
    std::optional<double> tps,
    std::optional<std::string> language,
    std::optional<std::string> task,
    std::optional<float> no_speech_prob,
    std::optional<float> avg_logprob
) {
    nlohmann::json chunk;
    const auto object_prefix = event_task == "translate"
        ? "audio.translation"
        : "audio.transcription";
    chunk["object"] = object_prefix + std::string(finished ? ".done" : ".chunk");
    chunk["text"] = text;
    if (finished) {
        chunk["finish_reason"] = finish_reason.value_or("stop");
        if (language.has_value()) {
            chunk["language"] = language.value();
        }
        if (task.has_value()) {
            chunk["task"] = task.value();
        }
        if (no_speech_prob.has_value()) {
            chunk["no_speech_prob"] = no_speech_prob.value();
        }
        if (avg_logprob.has_value()) {
            chunk["avg_logprob"] = avg_logprob.value();
        }
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
    ChatProtocol protocol
) {
    const bool is_openai = protocol == ChatProtocol::OpenAI;
    _set_cors_headers(res);
    try {
        std::string model;
        bool stream = false;

        // 1. Prepare
        std::optional<Chat> chat_opt = _prepare_chat_context(
            req, res, model, stream, protocol
        );
        if (!chat_opt.has_value()) {
            return;
        }
        Chat chat = std::move(chat_opt.value());

        // 2. Barge-in check
        _stop_and_detach_vlm_thread();

        // 3. Execution
        if (stream) {
            _execute_streaming_chat(res, chat, model, protocol);
        } else {
            _execute_normal_chat(res, chat, model, protocol);
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

void WEB::_handle_audio_transcriptions(
    const httplib::Request& req,
    httplib::Response& res,
    const std::string& task
) {
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
        std::string language = "auto";
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
            _execute_streaming_audio_transcription(res, language, task);
            return;
        }

        auto result = _whisper_model_ptr->run_model(_AUDIO_FILE_NAME, language, task);
        nlohmann::json response = {
            {"text", result.text},
            {"language", result.language},
            {"task", result.task},
            {"no_speech_prob", result.no_speech_prob}
        };
        if (result.avg_logprob.has_value())
            response["avg_logprob"] = result.avg_logprob.value();
        res.set_content(response.dump(), "application/json");

    } catch (const std::exception& e) {
        _logger->error("Error in transcription: {}", e.what());
        res.status = 500;
        res.set_content(nlohmann::json({{"error", e.what()}}).dump(), "application/json");
    }
}

void WEB::_execute_streaming_audio_transcription(
    httplib::Response& res,
    const std::string& language,
    const std::string& task
) {
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider(
        "text/event-stream",
        [this, language, task](size_t offset, httplib::DataSink &sink) {
            (void)offset;
            std::optional<double> ttft_value;
            std::optional<double> tps_value;
            std::optional<std::string> finish_reason;

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
                    finish_reason = metric_type == "FULL" ? "length" : "stop";
                }
            };

            auto text_callback = [&](const std::string& text, bool stream_end, bool) {
                if (text.empty())
                    return;
                std::string chunk = _format_audio_sse_chunk(text, task, false);
                sink.write(chunk.data(), chunk.size());
            };

            try {
                _whisper_model_ptr->set_info_callback(info_callback);
                _whisper_model_ptr->set_text_callback(text_callback);
                auto result = _whisper_model_ptr->run_model(_AUDIO_FILE_NAME, language, task);
                std::string chunk = _format_audio_sse_chunk(
                    "", task, true, finish_reason.value_or("stop"), ttft_value, tps_value,
                    result.language, result.task, result.no_speech_prob, result.avg_logprob
                ) + "data: [DONE]\n\n";
                sink.write(chunk.data(), chunk.size());
            } catch (const std::exception& e) {
                nlohmann::json error;
                error["object"] = task == "translate"
                    ? "audio.translation.error"
                    : "audio.transcription.error";
                error["error"] = e.what();
                std::string chunk = "data: " + error.dump() + "\n\ndata: [DONE]\n\n";
                sink.write(chunk.data(), chunk.size());
            }

            sink.done();
            return true;
        }
    );
}


namespace {

bool json_bool_or_default(
    const nlohmann::json& json_data, const char* key, bool default_value
) {
    if (!json_data.contains(key))
        return default_value;
    const auto& value = json_data.at(key);
    if (value.is_boolean())
        return value.get<bool>();
    if (value.is_string()) {
        auto text = value.get<std::string>();
        return text == "true" || text == "1" || text == "True";
    }
    return default_value;
}

bool request_enable_thinking(
    const nlohmann::json& json_data, bool default_value, ChatProtocol protocol
) {
    if (protocol != ChatProtocol::OpenAI)
        return json_bool_or_default(json_data, "think", default_value);

    bool enable_thinking = json_bool_or_default(
        json_data, "enable_thinking", default_value
    );
    if (
        json_data.contains("chat_template_kwargs")
        && json_data.at("chat_template_kwargs").is_object()
    ) {
        enable_thinking = json_bool_or_default(
            json_data.at("chat_template_kwargs"), "enable_thinking", enable_thinking
        );
    }
    return enable_thinking;
}

}

std::optional<Chat> WEB::_prepare_chat_context(
    const httplib::Request& req,
    httplib::Response& res,
    std::string& model,
    bool& stream,
    ChatProtocol protocol
) {
    auto json_data = nlohmann::json::parse(req.body);
    model = json_data.value("model", "default-model");
    stream = json_data.value("stream", false);

    Chat chat = _vision_language_model_ptr->create_chat();
    chat.set_enable_thinking(
        request_enable_thinking(json_data, _enable_thinking, protocol)
    );
    bool tools_enabled = true;
    if (json_data.contains("tool_choice") && !json_data["tool_choice"].is_null()) {
        if (!json_data["tool_choice"].is_string()) {
            res.status = 400;
            res.set_content(
                R"({"error": "Only tool_choice 'auto' or 'none' is supported"})",
                "application/json"
            );
            return std::nullopt;
        }
        const auto tool_choice = json_data["tool_choice"].get<std::string>();
        if (tool_choice == "none") {
            tools_enabled = false;
        } else if (tool_choice != "auto") {
            res.status = 400;
            res.set_content(
                R"({"error": "Only tool_choice 'auto' or 'none' is supported"})",
                "application/json"
            );
            return std::nullopt;
        }
    }
    if (json_data.contains("tools") && !json_data["tools"].is_array()) {
        res.status = 400;
        res.set_content(R"({"error": "tools must be an array"})", "application/json");
        return std::nullopt;
    }
    if (protocol == ChatProtocol::OllamaGenerate &&
        json_data.contains("tools") && !json_data["tools"].empty()) {
        res.status = 400;
        res.set_content(
            R"({"error": "tools are not supported by /api/generate"})",
            "application/json"
        );
        return std::nullopt;
    }
    if (tools_enabled && json_data.contains("tools")) {
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

static std::vector<std::string> tool_names_from_definitions(
    const nlohmann::ordered_json& tools
) {
    std::vector<std::string> names;
    if (!tools.is_array()) return names;

    for (const auto& tool : tools) {
        if (tool.is_object() && tool.contains("function") && tool["function"].is_object() &&
            tool["function"].contains("name") && tool["function"]["name"].is_string()) {
            names.push_back(tool["function"]["name"].get<std::string>());
        }
    }
    return names;
}

void WEB::_execute_streaming_chat(
    httplib::Response& res,
    Chat& chat,
    const std::string& model,
    ChatProtocol protocol
) {
    const bool is_openai = protocol == ChatProtocol::OpenAI;
    res.set_header("Content-Type", is_openai ? "text/event-stream" : "application/x-ndjson");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    res.set_chunked_content_provider(
        is_openai ? "text/event-stream" : "application/x-ndjson",
        [this, chat, model, protocol](size_t offset, httplib::DataSink &sink) {
            (void)offset;
            const bool is_openai = protocol == ChatProtocol::OpenAI;
            const auto created = std::time(nullptr);
            const auto completion_id = "chatcmpl-" + std::to_string(created);
            const bool has_tools = chat.has_tools();
            bool sent_initial_chunk = false;
            bool ttft_sent = false;
            std::optional<double> ttft_value;
            std::optional<double> tps_value;
            ToolCallStreamParser tool_parser(
                _vision_language_model_ptr->tool_call_format(),
                tool_names_from_definitions(chat.get_tools()));
            const auto reasoning_format = reasoning_format_for_model(
                _vision_language_model_ptr->model_type()
            );
            const auto& messages = chat.get_messages();
            const bool prompt_opens_reasoning = reasoning_format == ReasoningFormat::Gemma4 &&
                !messages.empty() && messages.back().value("role", std::string{}) == "tool";
            ReasoningStreamParser reasoning_parser(
                reasoning_format, chat.get_enable_thinking(), prompt_opens_reasoning
            );
            nlohmann::json pending_ollama_tool_calls = nullptr;

            auto send_openai_initial = [&]() {
                if (sent_initial_chunk) return;
                nlohmann::json initial_chunk;
                initial_chunk["id"] = completion_id;
                initial_chunk["object"] = "chat.completion.chunk";
                initial_chunk["created"] = created;
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

            auto send_text = [&](const std::string& text, bool from_draft, bool reasoning) {
                if (text.empty()) return;
                if (is_openai) {
                    send_openai_initial();
                    auto chunk = _format_openai_sse_chunk(
                        text, model, completion_id, created, false, std::nullopt,
                        take_ttft_once(), tps_value, from_draft, reasoning
                    );
                    sink.write(chunk.data(), chunk.size());
                } else {
                    auto chunk = _format_ollama_ndjson_chunk(
                        text, model, protocol, false, std::nullopt, take_ttft_once(),
                        tps_value, from_draft, reasoning
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
                tool_chunk["id"] = completion_id;
                tool_chunk["object"] = "chat.completion.chunk";
                tool_chunk["created"] = created;
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
                        const auto& content =
                            std::get<ToolCallStreamParser::Content>(event);
                        send_text(content.text, content.from_draft, false);
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

                    if (has_tools) handle_tool_parser_events(tool_parser.add("", true));
                    const std::string finish_reason =
                        saw_tool_calls ? "tool_calls" : default_finish_reason;
                    if (is_openai) {
                        send_openai_initial();
                        auto formatted_chunk = _format_openai_sse_chunk(
                            "", model, completion_id, created, true, finish_reason
                        ) + "data: [DONE]\n\n";
                        sink.write(formatted_chunk.data(), formatted_chunk.size());
                    } else {
                        nlohmann::json final_obj = {
                            {"model", model},
                            {"created_at", get_iso_timestamp()},
                            {"done", true},
                            {"finish_reason", finish_reason},
                        };
                        if (ttft_value.has_value()) final_obj["ttft"] = *ttft_value;
                        if (tps_value.has_value()) final_obj["tps"] = *tps_value;
                        if (protocol == ChatProtocol::OllamaChat) {
                            nlohmann::json message = {
                                {"role", "assistant"}, {"content", ""}
                            };
                            if (!pending_ollama_tool_calls.is_null()) {
                                message["tool_calls"] =
                                    openai_tool_calls_to_ollama(pending_ollama_tool_calls);
                            }
                            final_obj["message"] = std::move(message);
                        } else {
                            final_obj["response"] = "";
                        }
                        auto formatted_chunk = final_obj.dump() + "\n";
                        sink.write(formatted_chunk.data(), formatted_chunk.size());
                    }
                }
            };

            auto text_callback = [&](
                const std::string& text, bool stream_end, bool from_draft
            ) {
                for (auto& event : reasoning_parser.add(text, stream_end, from_draft)) {
                    if (event.reasoning) {
                        send_text(event.text, event.from_draft, true);
                        continue;
                    }

                    if (has_tools) {
                        handle_tool_parser_events(
                            tool_parser.add(event.text, false, event.from_draft)
                        );
                    } else {
                        send_text(event.text, event.from_draft, false);
                    }
                }
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
    ChatProtocol protocol
) {
    std::string reasoning_response;
    std::string full_response;
    std::string finish_reason = "stop";
    const auto reasoning_format = reasoning_format_for_model(
        _vision_language_model_ptr->model_type()
    );
    const auto& messages = chat.get_messages();
    const bool prompt_opens_reasoning = reasoning_format == ReasoningFormat::Gemma4 &&
        !messages.empty() && messages.back().value("role", std::string{}) == "tool";
    ReasoningStreamParser reasoning_parser(
        reasoning_format, chat.get_enable_thinking(), prompt_opens_reasoning
    );

    auto info_callback = [](const std::string&, double) {};

    auto text_callback = [&](const std::string& text, bool stream_end, bool from_draft) {
        for (auto& event : reasoning_parser.add(text, stream_end, from_draft)) {
            if (event.reasoning) {
                reasoning_response += event.text;
            } else {
                full_response += event.text;
            }
        }
    };

    _vision_language_model_ptr->set_info_callback(info_callback);
    _vision_language_model_ptr->set_text_callback(text_callback);

    // Run in thread for consistency with barge-in logic
    _vlm_thread = std::jthread([this, chat]() {
        _vision_language_model_ptr->run_model(chat);
    });
    _vlm_thread.join();

    // No need to reset callbacks - they'll be overwritten on next request

    nlohmann::json tool_calls = nullptr;
    if (chat.has_tools()) {
        tool_calls = try_parse_tool_calls(
            _vision_language_model_ptr->tool_call_format(), full_response,
            tool_names_from_definitions(chat.get_tools()));
        if (!tool_calls.is_null()) {
            full_response.clear();
            finish_reason = "tool_calls";
        }
    }

    nlohmann::json response;
    if (protocol == ChatProtocol::OpenAI) {
        nlohmann::json message = {{"role", "assistant"}};
        if (!tool_calls.is_null()) {
            message["content"] = nullptr;
            message["tool_calls"] = tool_calls;
        } else {
            message["content"] = full_response;
        }
        if (!reasoning_response.empty()) {
            message["reasoning_content"] = reasoning_response;
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
    } else if (protocol == ChatProtocol::OllamaChat) {
        nlohmann::json message = {{"role", "assistant"}};
        if (!tool_calls.is_null()) {
            message["content"] = "";
            message["tool_calls"] = openai_tool_calls_to_ollama(tool_calls);
        } else {
            message["content"] = full_response;
        }
        if (!reasoning_response.empty()) message["thinking"] = reasoning_response;
        response = {
            {"model", model},
            {"created_at", get_iso_timestamp()},
            {"message", message},
            {"done", true},
            {"finish_reason", finish_reason}
        };
    } else {
        response = {
            {"model", model},
            {"created_at", get_iso_timestamp()},
            {"response", tool_calls.is_null() ? full_response : std::string("")},
            {"done", true},
            {"finish_reason", finish_reason}
        };
        if (!reasoning_response.empty()) response["thinking"] = reasoning_response;
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
