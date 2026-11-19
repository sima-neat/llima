#include <algorithm>
#include <fmt/ranges.h>

#include "chat.hpp"

namespace simaai {
namespace llima {


Chat::Chat(VlmHelper& vlm_helper) : _vlm_helper(vlm_helper),
    _enable_thinking(vlm_helper.get_enable_thinking()),
    _reasoning_effort(vlm_helper.get_reasoning_effort()) {
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
    clear_messages();
    _messages = nlohmann::ordered_json::array({system_message});
}


void Chat::clear_system_prompt() {
    _vlm_helper.set_system_prompt(std::nullopt);
    if (!_messages.empty() && _messages[0]["role"] == "system")
        _messages.erase(_messages.begin());
    clear_history();
}


void Chat::add_query(std::string query) {
    if (!_vlm_helper.is_multimodal()) {
        _messages.push_back({{"role", "user"}, {"content", std::move(query)}});
        return;
    }

    auto content = std::move(_pending_media);
    _pending_media = nlohmann::ordered_json::array();
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
        assert(_images.empty() && _pending_media.empty());
        _messages = std::move(messages);
    }
}


void Chat::clear_messages() {
    // Keep attachments for replay, including any not yet submitted.
    auto media = nlohmann::ordered_json::array();
    for (const auto& message : _messages) {
        const auto& content = message.at("content");
        if (!content.is_array()) continue;
        for (const auto& item : content) {
            if (item.at("type") == "image" || item.at("type") == "audio") media.push_back(item);
        }
    }
    for (const auto& item : _pending_media) media.push_back(item);
    _pending_media = std::move(media);
    _reset_messages();
}


void Chat::_reset_messages() {
    if (_messages.empty() || _messages[0]["role"] != "system")
        _messages.clear();
    else
        _messages.erase(_messages.begin() + 1, _messages.end());
}


void Chat::clear_history() {
    _reset_messages();
    _images.clear();
    _pending_media.clear();
}


void Chat::set_tools(nlohmann::ordered_json tools) {
    _tools = std::move(tools);
}


void Chat::add_image(std::filesystem::path image_path) {
    _pending_media.push_back({{"type", "image"}, {"image", image_path.string()}});
    _images.emplace_back(std::move(image_path));
}

void Chat::add_audio(std::filesystem::path audio_path) {
    if (!_vlm_helper.support_audio()) throw std::invalid_argument("This model has no compiled audio frontend");
    _pending_media.push_back({{"type", "audio"}, {"audio", audio_path.string()}});
}


void Chat::clear_images() {
    _images.clear();
    auto remove_images = [](auto& content) {
        if (content.is_array()) {
            content.erase(std::remove_if(content.begin(), content.end(),
                [](const auto& item) { return item.at("type") == "image"; }), content.end());
        }
    };
    remove_images(_pending_media);
    for (auto& message : _messages) remove_images(message["content"]);
}


}
}
