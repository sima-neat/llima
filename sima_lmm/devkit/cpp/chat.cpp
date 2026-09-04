#include <fmt/ranges.h>

#include "chat.hpp"

namespace simaai {
namespace llima {

namespace {

uint32_t count_images_in_messages(const nlohmann::ordered_json& messages) {
    uint32_t num_images = 0;
    for (const auto& message: messages) {
        if (!message.contains("content"))
            continue;
        const auto& content = message["content"];
        if (!content.is_array())
            continue;
        for (const auto& item: content) {
            if (item.is_object() && item.value("type", "") == "image")
                ++num_images;
        }
    }
    return num_images;
}

}


Chat::Chat(VlmHelper& vlm_helper) : _vlm_helper(vlm_helper),
    _enable_thinking(vlm_helper.get_enable_thinking()) {
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

    // System/developer content is consumed as a plain string by Gemma4 and other chat templates.
    nlohmann::ordered_json system_message;
    system_message["role"] = "system";
    system_message["content"] = std::move(system_prompt);
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
    auto num_images_in_messages = count_images_in_messages(_messages);

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


void Chat::clear_queued_images() {
    // Images already submitted with a query are part of chat history. Only discard images queued
    // since the last query; clear_history() removes both submitted and queued images.
    auto num_submitted_images = count_images_in_messages(_messages);
    if (_images.size() > num_submitted_images)
        _images.resize(num_submitted_images);
}


}
}
