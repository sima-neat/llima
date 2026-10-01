#ifndef _SIMA_LLIMA_CLI_
#define _SIMA_LLIMA_CLI_

#include <csignal>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "chat.hpp"
#include "file_provider.hpp"
#include "readline_helper.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"
#include "whisper_model.hpp"


namespace simaai {
namespace llima {

class EXPORT CLI {
    public:
        // Reads the model files from disk. Same signature as before the
        // FileProvider seam: libsima_lmm_runtime exported it, so code built
        // against the older library still links.
        CLI(
            std::filesystem::path vlm_model_path,
            std::optional<std::filesystem::path> whisper_model_path,
            std::optional<std::filesystem::path> draft_model_path,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template
        );
        // file_provider: nullptr = read from disk. No default, so no call
        // can match both constructors.
        CLI(
            std::filesystem::path vlm_model_path,
            std::optional<std::filesystem::path> whisper_model_path,
            std::optional<std::filesystem::path> draft_model_path,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template,
            std::shared_ptr<FileProvider> file_provider
        );
        ~CLI();

        void run();
        void stop();

    private:
        std::unique_ptr<VisionLanguageModel> _vision_language_model_ptr;
        std::unique_ptr<WhisperModel> _whisper_model_ptr;
        std::unique_ptr<VisionLanguageModel> _vision_language_draft_model_ptr;

        // Logging.
        std::shared_ptr<spdlog::logger> _logger;

        static const std::string _COMMANDS;

        inline static CLI* _singleton_ptr = nullptr;
        inline static struct sigaction _old_sigint_action = {};
};


}
}


#endif
