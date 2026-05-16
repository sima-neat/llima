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

#include <atomic>
#include <memory>

#include <llama.h>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "eigen_types.hpp"
#include "image_processor.hpp"
#include "mla_model.hpp"
#include "setup.hpp"

namespace simaai {
namespace llima {

static std::shared_ptr<spdlog::logger> llama_logger = nullptr;

static void setup_llama_cpp_logger() {
   auto llima_logger = spdlog::get("llima");
   llama_logger = llima_logger? llima_logger->clone("llama") : spdlog::default_logger();

   llama_log_set(
        [](enum ggml_log_level level, const char* text, void* user_data) {
            if (!text || !llama_logger || !llama_logger->should_log(spdlog::level::debug)) return;

            spdlog::level::level_enum msg_level;
            switch (level) {
                case GGML_LOG_LEVEL_ERROR: msg_level = spdlog::level::err;   break;
                case GGML_LOG_LEVEL_WARN:  msg_level = spdlog::level::warn;  break;
                case GGML_LOG_LEVEL_INFO:  msg_level = spdlog::level::debug; break;
                case GGML_LOG_LEVEL_DEBUG: msg_level = spdlog::level::debug; break;
                default:                   msg_level = spdlog::level::trace; break;
            }

            std::string msg(text);
            if (!msg.empty() && msg.back() == '\n') msg.pop_back();
            llama_logger->log(msg_level, "{}", msg);
        },
        nullptr
    );
}

void set_log_level(spdlog::level::level_enum log_level) {
    spdlog::set_level(log_level);
    if (auto default_logger = spdlog::default_logger()) {
        default_logger->set_level(log_level);
    }
    for (const char* logger_name : {"llima", "VLM", "STREAM", "llama", "Whisper"}) {
        if (auto logger = spdlog::get(logger_name)) {
            logger->set_level(log_level);
        }
    }
}

void connect(
    const std::vector<std::string>& mla_rt_args,
    const std::filesystem::path& log_file_name,
    const spdlog::level::level_enum log_level,
    const std::optional<std::filesystem::path>& sample_image_file_name,
    const std::optional<std::filesystem::path>& sample_audio_file_name
) {
    // Setup logger. Disable the logger if no file name provided.
    auto logger = spdlog::basic_logger_mt<spdlog::async_factory>("llima", log_file_name, true);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");
    spdlog::set_default_logger(logger);
    set_log_level(log_level);
    spdlog::flush_every(std::chrono::seconds(1));

    // Configure the llama.cpp log level.
    setup_llama_cpp_logger();

    // Connect to the MLA dispatcher.
    connect_mla_rt(mla_rt_args);

    // Read environment variables.
    MLAModelWithBuffer::read_env_vars();
    ImageProcessor::read_env_vars();

    // Setup eigen device.
    get_eigen_device();

    // Set the sample file names.
    if (sample_image_file_name.has_value())
        set_sample_image_file_name(sample_image_file_name.value());
    if (sample_audio_file_name.has_value())
        set_sample_audio_file_name(sample_audio_file_name.value());
}


void disconnect() {
    disconnect_mla_rt();
    spdlog::shutdown();
}

}
}
