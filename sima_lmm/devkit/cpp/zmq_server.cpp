#include <fmt/ranges.h>

#include "zmq_server.hpp"

namespace simaai {
namespace llima {


ZMQServer::ZMQServer(
    const std::filesystem::path& model_path, uint32_t port
) : _port(port), _zmq_ctx(1), _zmq_socket(_zmq_ctx, zmq::socket_type::rep), _is_running(false) {
    if (_singleton_ptr)
        throw std::runtime_error("Only one ZMQServer instance can be created");
    _singleton_ptr = this;

    // Create local logger.
    auto llima_logger = spdlog::get("llima");
    _logger = llima_logger? llima_logger->clone("ZMQ") : spdlog::default_logger();

    // Create the vlm model.
    _vision_language_model_ptr = std::make_unique<VisionLanguageModel>(model_path);

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
            } else if (ret.value() != 2) {
                throw std::runtime_error("Failed to receive zmq messages");
            }
            _logger->info("Received a request");

            // Unpack the metadata and check the values.
            msgpack::object_handle handle = msgpack::unpack(
                reinterpret_cast<char*>(request_messages[0].data()), request_messages[0].size()
            );
            auto request_metadata = handle.get().as<ZMQRequestMetadata>();
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
            _logger->info(
                "type: {}, dtype: {}, shape: {}, max_num_tokens: {}, stop_token_ids: {}",
                request_metadata.type,
                request_metadata.tensor_dtype,
                request_metadata.tensor_shape,
                request_metadata.max_num_tokens.value_or(0),
                request_metadata.stop_token_ids.value_or(std::set<uint32_t>())
            );

            // Construct input token ids.
            std::span<const uint32_t> input_token_ids(
                reinterpret_cast<const uint32_t*>(request_messages[1].data()),
                request_metadata.tensor_shape[1]
            );

            // Service the request and populate the response messages.
            ZMQResponseMetadata response_metadata;
            std::array<zmq::message_t, 2> response_messages;
            ChronoTimer inference_timer(true);
            if (request_metadata.type == "generate") {
                // Given a list of input token ids, return the list of generated token ids.
                auto result = _vision_language_model_ptr->run_model(
                    input_token_ids, request_metadata.max_num_tokens,
                    request_metadata.stop_token_ids
                );
                response_metadata.tensor_dtype = "uint32";
                response_metadata.tensor_shape = {1, result.size()};
                response_messages[1] = {result.data(), result.size() * 4};
            } else if (request_metadata.type == "model_call") {
                // Given a list of input token ids, return the list of computed logits.
                auto result = _vision_language_model_ptr->run_model_for_logits(input_token_ids);
                response_metadata.tensor_dtype = "bfloat16";
                response_metadata.tensor_shape = {
                    1, input_token_ids.size(), result.size() / input_token_ids.size()
                };
                response_messages[1] = {result.data(), result.size() * 2};
            } else if (request_metadata.type == "generate_for_perf") {
                // Given a list of input token ids, return the list of time to generate the next
                // token. The first item of the list is the TTFT.
                auto result = _vision_language_model_ptr->run_model_for_ttnt(
                    input_token_ids, request_metadata.max_num_tokens,
                    request_metadata.stop_token_ids
                );
                response_metadata.tensor_dtype = "float64";
                response_metadata.tensor_shape = {1, result.size()};
                response_messages[1] = {result.data(), result.size() * 8};
            } else {
                throw std::runtime_error(
                    fmt::format("Unknown request type: {}", request_metadata.type)
                );
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
