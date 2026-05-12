//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

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

    MSGPACK_DEFINE_MAP(type, tensor_dtype, tensor_shape, max_num_tokens, stop_token_ids);
};


struct ZMQResponseMetadata {
    std::string tensor_dtype;
    std::vector<size_t> tensor_shape;
    size_t infer_time_ns;

    MSGPACK_DEFINE_MAP(tensor_dtype, tensor_shape, infer_time_ns);
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
