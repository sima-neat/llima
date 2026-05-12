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

#include <fmt/ranges.h>

#include "chat.hpp"

namespace simaai {
namespace llima {


Chat::Chat(VlmHelper& vlm_helper) : _vlm_helper(vlm_helper) {
    auto system_prompt = vlm_helper.get_system_prompt().value_or("");
    if (!system_prompt.empty())
        set_system_prompt(system_prompt);
}


void Chat::set_system_prompt(std::string system_prompt) {
    _vlm_helper.set_system_prompt(system_prompt);

    if (system_prompt.empty()) {
        clear_system_prompt();
        return;
    }

    // Start a new message with the system prompt.
    nlohmann::ordered_json system_message;
    system_message["role"] = "system";
    if (_vlm_helper.is_multimodal()) {
        system_message["content"].push_back({{"type", "text"}, {"text", std::move(system_prompt)}});
    } else {
        system_message["content"] = std::move(system_message);
    }
    _messages = nlohmann::ordered_json::array({system_message});
}


void Chat::clear_system_prompt() {
    _vlm_helper.set_system_prompt(std::nullopt);
    clear_history();
}


void Chat::add_query(std::string query) {
    if (!_vlm_helper.is_multimodal()) {
        _messages.push_back({{"role", "user"}, {"content", std::move(query)}});
        return;
    }

    // Check how many images need to be added to this message in addition to the query.
    uint32_t num_images_in_messages = 0;
    for (const auto& message: _messages) {
        for (const auto& item: message["content"]) {
            if (item.at("type") == "image")
                ++num_images_in_messages;
        }
    }

    nlohmann::ordered_json content;
    if (!_images.empty()) {
        for (uint32_t i = num_images_in_messages; i < _images.size(); ++i) {
            content.push_back({{"type", "image"}, {"image", _images[i].string()}});
        }
    }
    content.push_back({{"type", "text"}, {"text", std::move(query)}});
    _messages.push_back({{"role", "user"}, {"content", std::move(content)}});
}


std::string Chat::get_last_query() const {
    if (!_messages.empty() && _messages.back()["role"] == "user") {
        if (_vlm_helper.is_multimodal()) {
            for (const auto& item: _messages.back()["content"]) {
                if (item["type"] == "text")
                    return item["text"];
            }
        } else {
            return _messages.back()["content"];
        }
    }
    throw std::runtime_error("Unable to find the last query");
}


void Chat::update_last_query(std::string query) {
    if (!_messages.empty() && _messages.back()["role"] == "user") {
        if (_vlm_helper.is_multimodal()) {
            for (auto& item: _messages.back()["content"]) {
                if (item["type"] == "text") {
                    item["text"] = std::move(query);
                    return;
                }
            }
        } else {
            _messages.back()["content"] = std::move(query);
            return;
        }
    }
    throw std::runtime_error("Unable to find the last query");
}


void Chat::add_response(std::string response) {
    nlohmann::ordered_json message;
    message["role"] = "assistant";
    if (_vlm_helper.is_multimodal()) {
        message["content"].push_back({{"type", "text"}, {"text", std::move(response)}});
    } else {
        message["content"] = std::move(response);
    }
    _messages.push_back(message);
}


void Chat::set_messages(nlohmann::ordered_json messages) {
    if (messages.is_string()) {
        clear_history();
        add_query(messages);
    } else {
        assert(_images.size() == 0);
        _messages = std::move(messages);
    }
}


void Chat::clear_messages() {
    if (_messages.empty() || _messages[0]["role"] != "system")
        _messages.clear();
    else
        _messages.erase(_messages.begin() + 1, _messages.end());
}


void Chat::set_tools(nlohmann::ordered_json tools) {
    _tools = std::move(tools);
}


void Chat::add_image(std::filesystem::path image_path) {
    _images.emplace_back(std::move(image_path));
}


void Chat::clear_images() {
    _images.clear();
}


}
}
