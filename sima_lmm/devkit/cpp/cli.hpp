#ifndef _SIMA_LLIMA_CLI_
#define _SIMA_LLIMA_CLI_

#include <csignal>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include "chat.hpp"
#include "readline_helper.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"
#include "whisper_model.hpp"

namespace simaai {
namespace llima {

class EXPORT CLI {
public:
  CLI(std::filesystem::path vlm_model_path, std::optional<std::filesystem::path> whisper_model_path,
      std::optional<std::filesystem::path> draft_model_path,
      std::optional<std::string> system_prompt, std::optional<std::string> chat_template);
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

} // namespace llima
} // namespace simaai

#endif
