#include <array>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <dlfcn.h>
#include <openssl/evp.h>

#include <fmt/ranges.h>

#include "zmq_server.hpp"
#include "reasoning_parser.hpp"

namespace simaai {
namespace llima {

namespace {
std::string sha256(std::istream& stream) {
    auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
        EVP_MD_CTX_new(), EVP_MD_CTX_free
    );
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("Cannot initialize artifact hashing");
    std::array<char, 1024 * 1024> buffer;
    while (stream.read(buffer.data(), buffer.size()) || stream.gcount()) {
        if (EVP_DigestUpdate(context.get(), buffer.data(), stream.gcount()) != 1)
            throw std::runtime_error("Cannot hash artifact");
    }
    if (stream.bad())
        throw std::runtime_error("Cannot read artifact for hashing");
    std::array<unsigned char, EVP_MAX_MD_SIZE> digest;
    unsigned int size = 0;
    if (EVP_DigestFinal_ex(context.get(), digest.data(), &size) != 1)
        throw std::runtime_error("Cannot finalize artifact hash");
    std::string result;
    for (size_t i = 0; i < size; ++i)
        result += fmt::format("{:02x}", digest[i]);
    return result;
}

std::string hash_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open artifact for hashing: " + path.string());
    return sha256(file);
}
}  // namespace


std::string ZMQServer::_multimodal_info() {
    if (_multimodal_identity.has_value())
        return _multimodal_identity.value();
    auto& model = *_vision_language_model_ptr;
    if (!model.support_image() || _vision_language_draft_model_ptr != nullptr)
        throw std::invalid_argument("Image accuracy requires a non-speculative VLM");
    nlohmann::ordered_json info = {
        {"protocol_version", 1}, {"model_type", model._cfg.model_type},
        {"max_num_tokens", model._cfg.pipeline_cfg.max_num_tokens},
        {"decoding", "greedy"}, {"image_transport", "lossless-png"},
        {"multi_image", true}, {"files", nlohmann::ordered_json::object()}
    };
    std::vector<std::filesystem::path> files;
    for (const auto& dir : {"devkit", "elf_files"}) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(_model_path / dir)) {
            if (entry.is_regular_file())
                files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    for (const auto& file : files) {
        const auto name = file.lexically_relative(_model_path).generic_string();
        info["files"][name] = {
            {"sha256", hash_file(file)}, {"bytes", std::filesystem::file_size(file)}
        };
    }
    std::istringstream artifact_manifest(info["files"].dump());
    info["artifact_sha256"] = sha256(artifact_manifest);
    for (const auto& name : {"vlm_config.json", "preprocessor_config.json", "processor_config.json"}) {
        const auto file = _model_path / "devkit" / name;
        if (std::filesystem::is_regular_file(file)) {
            std::ifstream input(file);
            info[name] = nlohmann::ordered_json::parse(input);
        }
    }
    Dl_info library{};
    if (dladdr(reinterpret_cast<void*>(&hash_file), &library) == 0 || !library.dli_fname)
        throw std::runtime_error("Cannot identify the loaded LLiMa runtime");
    info["runtime"] = {{"library_sha256", hash_file(library.dli_fname)}};
    std::ifstream buildinfo("/etc/buildinfo");
    if (buildinfo)
        info["runtime"]["platform_buildinfo"] = std::string(
            std::istreambuf_iterator<char>(buildinfo), std::istreambuf_iterator<char>()
        );
    _multimodal_identity = info.dump();
    return _multimodal_identity.value();
}


std::string ZMQServer::_generate_multimodal(
    const ZMQRequestMetadata& metadata, const std::string& payload,
    ZMQResponseMetadata& response
) {
    _multimodal_info();  // Validate capabilities before handling any images.
    if (!metadata.max_num_tokens || !metadata.max_new_tokens
        || metadata.max_new_tokens.value() == 0 || metadata.stop_token_ids)
        throw std::invalid_argument("Image accuracy requires context/output limits and native stop tokens");
    const auto request = nlohmann::ordered_json::parse(payload);
    if (!request.is_object() || request.size() != 1 || !request.contains("messages"))
        throw std::invalid_argument("Expected a messages object");
    const auto& messages = request.at("messages");
    if (!messages.is_array() || messages.empty())
        throw std::invalid_argument("Expected nonempty ordered messages");
    size_t image_count = 0;
    for (const auto& message : messages) {
        if (!message.is_object() || message.size() != 2 || !message.contains("content")
            || !message.contains("role") || !message["role"].is_string())
            throw std::invalid_argument("Messages require role and content");
        const auto role = message.at("role").get<std::string>();
        if (role != "user" && role != "assistant" && role != "system")
            throw std::invalid_argument("Unsupported message role");
        const auto& content = message.at("content");
        if (content.is_string())
            continue;
        if (!content.is_array())
            throw std::invalid_argument("Expected text or ordered image/text content");
        for (const auto& item : content) {
            const auto type = item.at("type").get<std::string>();
            if (type == "image") {
                const auto& data = item.at("image").get_ref<const std::string&>();
                if (item.size() != 2 || (!data.starts_with("data:image/png;base64,")
                    && !data.starts_with("data:image/jpeg;base64,")))
                    throw std::invalid_argument("Images must be inline PNG/JPEG data, not server paths");
                ++image_count;
            } else if (type != "text" || item.size() != 2 || !item.at("text").is_string()) {
                throw std::invalid_argument("Only image and text content are supported");
            }
        }
    }
    if (!image_count)
        throw std::invalid_argument("Image accuracy requires at least one image");
    auto& model = *_vision_language_model_ptr;
    auto chat = model.create_chat();
    auto effective_messages = messages;
    const auto& defaults = chat.get_messages();
    if (messages.front().at("role") != "system" && defaults.is_array() && !defaults.empty()
        && defaults.front().at("role") == "system") {
        effective_messages.insert(effective_messages.begin(), defaults.front());
    }
    chat.set_messages(std::move(effective_messages));
    chat.set_enable_thinking(metadata.enable_thinking);
    PreprocessedChat prepared;
    struct GenerationState {
        uint32_t tokens = 0;
        std::string finish_reason;
        std::string raw;
        std::string answer;
        ReasoningStreamParser parser;
        GenerationState(ReasoningFormat format, bool thinking) : parser(format, thinking) {}
    };
    // Streamer callbacks run on its worker thread and may outlive an exception.
    auto state = std::make_shared<GenerationState>(
        reasoning_format_for_model(model._cfg.model_type), metadata.enable_thinking
    );
    model.set_info_callback([state](const std::string& event, double) {
        if (event == "ttft" || event == "tps")
            ++state->tokens;
        else if (event == "END")
            state->finish_reason = "stop";
        else if (event == "FULL")
            state->finish_reason = "length";
    });
    model.set_text_callback([state](const std::string& text, bool end, bool from_draft) {
        state->raw += text;
        for (const auto& event : state->parser.add(text, end, from_draft)) {
            if (!event.reasoning)
                state->answer += event.text;
        }
    });
    auto tokens = model._run_chat_model(
        chat, metadata.max_new_tokens, metadata.max_num_tokens, &prepared
    );
    if (!tokens)
        throw std::runtime_error("Image generation was interrupted");
    response.input_tokens = prepared.input_token_ids.size();
    response.generated_tokens = state->tokens;
    response.finish_reason = state->finish_reason;
    response.raw_answer = state->raw;
    std::istringstream prompt(prepared.formatted_prompt);
    response.prompt_sha256 = sha256(prompt);
    return state->answer;
}


ZMQServer::ZMQServer(
    const std::filesystem::path& model_path,
    uint32_t port,
    std::optional<std::filesystem::path> draft_model_path
) : _model_path(model_path), _port(port), _zmq_ctx(1),
    _zmq_socket(_zmq_ctx, zmq::socket_type::rep), _is_running(false) {
    if (_singleton_ptr)
        throw std::runtime_error("Only one ZMQServer instance can be created");
    _singleton_ptr = this;

    // Create local logger.
    auto llima_logger = spdlog::get("llima");
    _logger = llima_logger? llima_logger->clone("ZMQ") : spdlog::default_logger();

    // Create the vlm model.
    _vision_language_model_ptr = std::make_unique<VisionLanguageModel>(model_path);
    if (draft_model_path.has_value()) {
        _vision_language_draft_model_ptr = std::make_unique<VisionLanguageModel>(
            draft_model_path.value()
        );
        _vision_language_model_ptr->set_draft_vlm(
            _vision_language_draft_model_ptr.get()
        );
    }

    // Override the signal handler for ctrl-c.
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

    // Start the zmq server.
    _zmq_socket.set(zmq::sockopt::curve_secretkey, "+ZyM3eTG#JY(P)U75QLd<jK&f>ZL0/4XSO.M(+]8");
    _zmq_socket.set(zmq::sockopt::curve_server, 1);
}


ZMQServer::~ZMQServer() {
    _zmq_socket.close();
    _zmq_ctx.close();
    sigaction(SIGINT, &_old_sigint_action, nullptr);
    _logger->info("ZMQ server desctructed");
}


void ZMQServer::run() {
    auto uri = fmt::format("tcp://*:{}", _port);
    _zmq_socket.bind(uri);
    _logger->info("ZMQ server listening on port {} to receive requests", uri);
    _is_running = true;
    try {
        while (_is_running) {
            // Receive the request messages.
            std::vector<zmq::message_t> request_messages;
            const auto ret = zmq::recv_multipart(_zmq_socket, std::back_inserter(request_messages));
            if (!ret.has_value()) {
                // The server is terminated.
                break;
            } else if (ret.value() == 1 && request_messages[0].to_string() == "stop") {
                // Client requests to stop the server.
                _logger->info("ZMQ client requested stop");
                break;
            }
            _logger->info("Received a request");

            ZMQResponseMetadata response_metadata;
            std::array<zmq::message_t, 2> response_messages;
            ChronoTimer inference_timer(true);
            try {
                if (request_messages.size() != 2)
                    throw std::invalid_argument("Expected exactly two benchmark request frames");
                // Unpack the metadata and check the values.
                msgpack::object_handle handle = msgpack::unpack(
                    reinterpret_cast<char*>(request_messages[0].data()), request_messages[0].size()
                );
                auto request_metadata = handle.get().as<ZMQRequestMetadata>();
                if (request_metadata.type == "multimodal_info"
                    || request_metadata.type == "generate_multimodal") {
                    auto result = request_metadata.type == "multimodal_info"
                        ? _multimodal_info()
                        : _generate_multimodal(
                            request_metadata, request_messages[1].to_string(), response_metadata
                        );
                    response_metadata.tensor_dtype = "utf8";
                    response_metadata.tensor_shape = {result.size()};
                    response_metadata.result_type = request_metadata.type;
                    response_messages[1] = zmq::message_t(result.data(), result.size());
                } else {
                    if (
                        request_metadata.tensor_shape.size() != 2
                        || request_metadata.tensor_shape[0] != 1
                        || request_metadata.tensor_shape[1] <= 0
                    )
                        throw std::runtime_error(
                            fmt::format("Invalid shape: {}", request_metadata.tensor_shape)
                        );
                    if (request_metadata.tensor_dtype != "uint32")
                        throw std::runtime_error(
                            fmt::format("Invalid dtype: {}", request_metadata.tensor_dtype)
                        );
                    if (request_messages[1].size() % sizeof(uint32_t) != 0
                        || request_metadata.tensor_shape[1] != request_messages[1].size() / sizeof(uint32_t))
                        throw std::invalid_argument("Token frame size does not match tensor shape");
                    _logger->info(
                        "type: {}, dtype: {}, shape: {}, max_num_tokens: {}, stop_token_ids: {}, "
                        "continuation_start: {}, continuation_tokens: {}, use_group_prefill: {}",
                        request_metadata.type,
                        request_metadata.tensor_dtype,
                        request_metadata.tensor_shape,
                        request_metadata.max_num_tokens.value_or(0),
                        request_metadata.stop_token_ids.value_or(std::set<uint32_t>()),
                        request_metadata.continuation_start.value_or(0),
                        request_metadata.continuation_token_ids.value_or(std::vector<uint32_t>()),
                        request_metadata.use_group_prefill
                    );

                    if (
                        _vision_language_draft_model_ptr != nullptr
                        && (
                            request_metadata.type == "generate"
                            || request_metadata.type == "model_call"
                        )
                    ) {
                        throw std::runtime_error(
                            "Accuracy benchmarking is not supported for "
                            "speculative-decoding models"
                        );
                    }

                    // Construct input token ids.
                    std::span<const uint32_t> input_token_ids(
                        reinterpret_cast<const uint32_t*>(request_messages[1].data()),
                        request_metadata.tensor_shape[1]
                    );

                    // Service the request and populate the response messages.
                    if (request_metadata.type == "generate") {
                        // Given a list of input token ids, return the list of generated token ids.
                        auto result = _vision_language_model_ptr->run_model(
                            input_token_ids, request_metadata.max_num_tokens,
                            request_metadata.stop_token_ids
                        );
                        response_metadata.tensor_dtype = "uint32";
                        response_metadata.tensor_shape = {1, result.size()};
                        response_messages[1] = {result.data(), result.size() * sizeof(uint32_t)};
                    } else if (request_metadata.type == "model_call") {
                        if (request_metadata.continuation_token_ids.has_value()) {
                            const auto& continuation_token_ids = (
                                request_metadata.continuation_token_ids.value()
                            );
                            if (
                                !request_metadata.continuation_start.has_value()
                                && continuation_token_ids.size() > input_token_ids.size()
                            ) {
                                throw std::runtime_error(
                                    fmt::format(
                                        "Continuation length {} exceeds input length {}",
                                        continuation_token_ids.size(),
                                        input_token_ids.size()
                                    )
                                );
                            }
                            const auto continuation_start = request_metadata.continuation_start.value_or(
                                input_token_ids.size() - continuation_token_ids.size()
                            );
                            auto result = _vision_language_model_ptr->run_model_for_loglikelihood(
                                input_token_ids, continuation_start, continuation_token_ids,
                                request_metadata.use_group_prefill
                            );
                            std::array<double, 2> response_data{
                                result.logprob, result.is_greedy? 1.0 : 0.0
                            };
                            response_metadata.tensor_dtype = "float64";
                            response_metadata.tensor_shape = {1, response_data.size()};
                            response_metadata.result_type = "loglikelihood";
                            response_messages[1] = {
                                response_data.data(), response_data.size() * sizeof(double)
                            };
                        } else {
                            // Given a list of input token ids, return the list of computed logits.
                            auto result = _vision_language_model_ptr->run_model_for_logits(
                                input_token_ids
                            );
                            response_metadata.tensor_dtype = "bfloat16";
                            response_metadata.tensor_shape = {
                                1, input_token_ids.size(), result.size() / input_token_ids.size()
                            };
                            response_messages[1] = {
                                result.data(), result.size() * sizeof(Eigen::bfloat16)
                            };
                        }
                    } else if (request_metadata.type == "generate_for_perf") {
                        // Given a list of input token ids, return the list of time to generate the next
                        // token. The first item of the list is the TTFT.
                        auto result = _vision_language_model_ptr->run_model_for_ttnt(
                            input_token_ids, request_metadata.max_num_tokens,
                            request_metadata.stop_token_ids
                        );
                        response_metadata.tensor_dtype = "float64";
                        response_metadata.tensor_shape = {1, result.token_durations.size()};
                        response_metadata.generated_tokens = result.generated_tokens;
                        response_metadata.accepted_draft_tokens = result.accepted_draft_tokens;
                        response_messages[1] = {
                            result.token_durations.data(),
                            result.token_durations.size() * sizeof(double)
                        };
                    } else {
                        throw std::runtime_error(
                            fmt::format("Unknown request type: {}", request_metadata.type)
                        );
                    }
                }
            } catch (const std::exception& e) {
                response_metadata.tensor_dtype = "";
                response_metadata.tensor_shape = {};
                response_metadata.error = e.what();
                response_metadata.error_type = dynamic_cast<const std::invalid_argument*>(&e)
                    ? "unsupported" : "runtime";
                response_messages[1] = zmq::message_t();
                _logger->error("Request failed: {}", e.what());
            }
            response_metadata.infer_time_ns = inference_timer.stop<std::nano>();

            // Pack the metadata and send the response messages.
            msgpack::sbuffer response_metadata_sbuffer;
            msgpack::pack(response_metadata_sbuffer, response_metadata);
            response_messages[0] = {
                response_metadata_sbuffer.data(), response_metadata_sbuffer.size()
            };
            zmq::send_multipart(_zmq_socket, std::move(response_messages));
            _logger->info("Sent response");
        }
    } catch (const zmq::error_t& e) {
        // When zmq recv or send function is in action and ctrl-c is triggered, cppzmq throws an
        // zmq::error exception. Nothing to be done.
    }
    _is_running = false;
    _logger->info("ZMQ server stopped");
}


void ZMQServer::stop() {
    if (_is_running.exchange(false)) {
        _logger->info("ZMQ server shutdown requested");
        _zmq_ctx.shutdown();
    }
}


}
}
