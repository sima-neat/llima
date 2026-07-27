
#ifndef _SIMA_LLIMA_WEB_
#define _SIMA_LLIMA_WEB_

#include <csignal>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "chat.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"
#include "whisper_model.hpp"

#include <httplib.h>

namespace simaai {
namespace llima {

class EXPORT WEB {
public:
  WEB(std::filesystem::path vlm_model_path, std::optional<std::filesystem::path> whisper_model_path,
      std::optional<std::filesystem::path> draft_model_path,
      std::optional<std::string> system_prompt, std::optional<std::string> chat_template);
  ~WEB();

  void run();
  void stop();

private:
  void _stop_and_detach_vlm_thread(); // Helper to stop and detach active VLM thread
  void _handle_stop(const httplib::Request& req, httplib::Response& res);
  void _handle_set_lora(const httplib::Request& req, httplib::Response& res);
  void _handle_unset_lora(const httplib::Request& req, httplib::Response& res);

  // Endpoints Handlers
  void _handle_chat_completions(const httplib::Request& req, httplib::Response& res,
                                bool is_openai);
  void _handle_audio_transcriptions(const httplib::Request& req, httplib::Response& res,
                                    const std::string& task);

  // Helpers
  void _set_cors_headers(httplib::Response& res);
  nlohmann::ordered_json _parse_endpoint_messages(const nlohmann::json& messages);
  std::string _format_openai_sse_chunk(const std::string& content, const std::string& model,
                                       bool finished,
                                       std::optional<std::string> finish_reason = std::nullopt,
                                       std::optional<double> ttft = std::nullopt,
                                       std::optional<double> tps = std::nullopt,
                                       bool from_draft = false);
  std::string _format_ollama_ndjson_chunk(const std::string& content, const std::string& model,
                                          bool finished,
                                          std::optional<std::string> finish_reason = std::nullopt,
                                          std::optional<double> ttft = std::nullopt,
                                          std::optional<double> tps = std::nullopt,
                                          bool from_draft = false);
  std::string _format_audio_sse_chunk(const std::string& text, const std::string& event_task,
                                      bool finished,
                                      std::optional<std::string> finish_reason = std::nullopt,
                                      std::optional<double> ttft = std::nullopt,
                                      std::optional<double> tps = std::nullopt,
                                      std::optional<std::string> language = std::nullopt,
                                      std::optional<std::string> task = std::nullopt,
                                      std::optional<float> no_speech_prob = std::nullopt,
                                      std::optional<float> avg_logprob = std::nullopt);

  // Helpers for chat completion
  std::optional<Chat> _prepare_chat_context(const httplib::Request& req, httplib::Response& res,
                                            std::string& model, bool& stream);

  void _execute_streaming_chat(httplib::Response& res, Chat& chat, const std::string& model,
                               bool is_openai);

  void _execute_normal_chat(httplib::Response& res, Chat& chat, const std::string& model,
                            bool is_openai);
  void _execute_streaming_audio_transcription(httplib::Response& res, const std::string& language,
                                              const std::string& task);
  std::unique_ptr<VisionLanguageModel> _vision_language_model_ptr;
  std::unique_ptr<VisionLanguageModel> _vision_language_draft_model_ptr;
  std::unique_ptr<WhisperModel> _whisper_model_ptr;
  std::jthread _vlm_thread;

  // HTTP server. For HTTPS, use SSLServer and define CPPHTTPLIB_OPENSSL_SUPPORT before
  // including the httplib.h header.
  httplib::Server _http_server;

  // Logging.
  std::shared_ptr<spdlog::logger> _logger;

  inline static const uint32_t _SERVER_PORT = 9998;
  inline static const std::string _AUDIO_FILE_NAME = "/tmp/audio.webm";
  inline static WEB* _singleton_ptr = nullptr;
  inline static struct sigaction _old_sigint_action = {};
};

} // namespace llima
} // namespace simaai

#endif
