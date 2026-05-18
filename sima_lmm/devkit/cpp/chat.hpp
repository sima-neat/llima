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

#ifndef _SIMA_LLIMA_CHAT_
#define _SIMA_LLIMA_CHAT_

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "vlm_helper.hpp"

namespace simaai {
namespace llima {

class VlmHelper;

class Chat {
    public:
        Chat(VlmHelper& vlm_helper);
        ~Chat() {}

        void set_system_prompt(std::string system_prompt);
        void clear_system_prompt();

        void add_query(std::string query);
        std::string get_last_query() const;
        void update_last_query(std::string query);

        void add_response(std::string response);

        void set_messages(nlohmann::ordered_json messages);
        const nlohmann::ordered_json& get_messages() const { return _messages; }
        void clear_messages();

        void set_tools(nlohmann::ordered_json tools);
        const nlohmann::ordered_json& get_tools() const { return _tools; }
        bool has_tools() const { return _tools.is_array() && !_tools.empty(); }

        void add_image(std::filesystem::path image_path);
        void clear_images();
        const std::vector<std::filesystem::path>& get_images() const { return _images; }

        void clear_history() { clear_messages(); clear_images(); }
        void print_history() const { std::cout << _messages << std::endl << std::flush; }

    protected:
        VlmHelper& _vlm_helper;
        nlohmann::ordered_json _messages;
        nlohmann::ordered_json _tools;
        std::vector<std::filesystem::path> _images;
};


}
}

#endif
