#ifndef _SIMA_LLIMA_ZMQ_SERVER_
#define _SIMA_LLIMA_ZMQ_SERVER_

#include <atomic>
#include <csignal>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <msgpack.hpp>
#include <spdlog/spdlog.h>
#include <zmq_addon.hpp>

#include "utils.hpp"
#include "vision_language_model.hpp"

namespace simaai {
namespace llima {


struct ZMQRequestMetadata {
    std::string type;
    std::string tensor_dtype;
    std::vector<size_t> tensor_shape;
    std::optional<uint16_t> max_num_tokens;
    std::optional<std::set<uint32_t>> stop_token_ids;
    std::optional<size_t> continuation_start;
    std::optional<std::vector<uint32_t>> continuation_token_ids;
    bool use_group_prefill = true;

    MSGPACK_DEFINE_MAP(
        type, tensor_dtype, tensor_shape, max_num_tokens, stop_token_ids, continuation_start,
        continuation_token_ids, use_group_prefill
    );
};


struct ZMQResponseMetadata {
    std::string tensor_dtype;
    std::vector<size_t> tensor_shape;
    size_t infer_time_ns = 0;
    std::optional<std::string> result_type;
    std::optional<std::string> error;

    MSGPACK_DEFINE_MAP(tensor_dtype, tensor_shape, infer_time_ns, result_type, error);
};


class EXPORT ZMQServer {
    public:
        ZMQServer(const std::filesystem::path& model_path, uint32_t port);
        ~ZMQServer();

        void run();
        void stop();

    private:
        uint32_t _port;
        zmq::context_t _zmq_ctx;
        zmq::socket_t _zmq_socket;

        std::unique_ptr<VisionLanguageModel> _vision_language_model_ptr;

        std::shared_ptr<spdlog::logger> _logger;
        std::atomic<bool> _is_running;

        inline static ZMQServer* _singleton_ptr = nullptr;
        inline static struct sigaction _old_sigint_action = {};
};

}
}


#endif
