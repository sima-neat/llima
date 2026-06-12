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
