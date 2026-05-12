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


#ifndef _SIMA_LLIMA_WEB_
#define _SIMA_LLIMA_WEB_

#include <csignal>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include <httplib.h>

#include "chat.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"
#include "whisper_model.hpp"


namespace simaai {
namespace llima {


class EXPORT WEB {
    public:
        WEB(
            std::filesystem::path vlm_model_path,
            std::optional<std::filesystem::path> whisper_model_path,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template,
            bool do_parallel_load
        );
        ~WEB();

        void run();
        void stop();

    private:
        void _stop_and_detach_vlm_thread();  // Helper to stop and detach active VLM thread
        void _handle_stop(const httplib::Request& req, httplib::Response& res);
        void _handle_set_lora(const httplib::Request& req, httplib::Response& res);
        void _handle_unset_lora(const httplib::Request& req, httplib::Response& res);

        // Endpoints Handlers
        void _handle_chat_completions(
            const httplib::Request& req,
            httplib::Response& res,
            bool is_openai
        );
        void _handle_audio_transcriptions(const httplib::Request& req, httplib::Response& res);

        // Helpers
        void _set_cors_headers(httplib::Response& res);
        nlohmann::ordered_json _parse_endpoint_messages(const nlohmann::json& messages);
        std::string _format_openai_sse_chunk(
            const std::string& content,
            const std::string& model,
            bool finished,
            std::optional<std::string> finish_reason = std::nullopt,
            std::optional<double> ttft = std::nullopt,
            std::optional<double> tps = std::nullopt
        );
        std::string _format_ollama_ndjson_chunk(
            const std::string& content,
            const std::string& model,
            bool finished,
            std::optional<std::string> finish_reason = std::nullopt,
            std::optional<double> ttft = std::nullopt,
            std::optional<double> tps = std::nullopt
        );

        // Helpers for chat completion
        std::optional<Chat> _prepare_chat_context(
            const httplib::Request& req, 
            httplib::Response& res,
            std::string& model, 
            bool& stream
        );

        void _execute_streaming_chat(
            httplib::Response& res, 
            Chat& chat, 
            const std::string& model, 
            bool is_openai
        );

        void _execute_normal_chat(
            httplib::Response& res, 
            Chat& chat, 
            const std::string& model, 
            bool is_openai
        );

        std::unique_ptr<VisionLanguageModel> _vision_language_model_ptr;
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


}
}

#endif
