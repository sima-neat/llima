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
        CLI(
            std::filesystem::path vlm_model_path,
            std::optional<std::filesystem::path> whisper_model_path,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template,
            bool do_parallel_load
        );
        ~CLI();

        void run();
        void stop();

    private:
        std::unique_ptr<VisionLanguageModel> _vision_language_model_ptr;
        std::unique_ptr<WhisperModel> _whisper_model_ptr;

        // Logging.
        std::shared_ptr<spdlog::logger> _logger;

        static const std::string _COMMANDS;

        inline static CLI* _singleton_ptr = nullptr;
        inline static struct sigaction _old_sigint_action = {};
};


}
}


#endif
