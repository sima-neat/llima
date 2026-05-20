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

void initialize_default_sample_files() {
    const std::filesystem::path assets_dir = SIMA_LMM_DEFAULT_ASSETS_DIR;
    set_sample_image_file_name(assets_dir / "sjc.jpg");
    set_sample_audio_file_name(assets_dir / "why_is_the_sky_blue.wav");
}

void connect(
    const std::vector<std::string>& mla_rt_args,
    const std::filesystem::path& log_file_name,
    const spdlog::level::level_enum log_level
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

    // Set default warm-up sample files.
    initialize_default_sample_files();
}


void disconnect() {
    disconnect_mla_rt();
    spdlog::shutdown();
}

}
}
