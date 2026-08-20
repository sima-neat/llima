#ifndef _SIMA_LLIMA_VLM_HELPER_
#define _SIMA_LLIMA_VLM_HELPER_

#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <minja/chat-template.hpp>

#include "chat.hpp"
#include "image_processor.hpp"
#include "tokenizer.hpp"
#include "vlm_config.hpp"

namespace simaai {
namespace llima {


struct PreprocessedChat {
    std::string formatted_prompt;
    std::vector<uint32_t> input_token_ids;
    std::vector<std::vector<Eigen::bfloat16>> image_tensors;
};

class Chat;
class VlmHelper {
    public:
        VlmHelper(
            const VlmConfig& vlm_cfg,
            const std::filesystem::path& devkit_dir,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template,
            bool enable_thinking = false
        );
        ~VlmHelper() {};

        void set_system_prompt(const std::optional<std::string>& system_prompt);
        std::optional<std::string> get_system_prompt() const { return _system_prompt; }
        void set_enable_thinking(bool enable_thinking) { _enable_thinking = enable_thinking; }
        bool get_enable_thinking() const { return _enable_thinking; }
        PreprocessedChat preprocess(const Chat& chat);

        bool is_multimodal() const { return _vlm_cfg.is_multimodal(); }
        bool support_image() const { return _vlm_cfg.support_image(); }
        Tokenizer* get_tokenizer() { return _tokenizer_ptr.get(); }
        std::set<uint32_t> get_stop_token_ids() const { return _stop_token_ids; }
        std::optional<uint32_t> get_image_token_id() const { return _image_token_id; }
        std::optional<uint32_t> get_pad_token_id() const { return _pad_token_id; }

    private:
        void _init_chat_template(
            const std::filesystem::path& devkit_dir,
            const nlohmann::json& tokenizer_config_json,
            std::optional<std::string> override_chat_template
        );
        void _init_stop_token_ids(
            const std::filesystem::path& devkit_dir,
            const nlohmann::json* tokenizer_config_json = nullptr
        );
        void _init_image_token_id(const nlohmann::json& tokenizer_config_json);
        void _init_pad_token_id(const nlohmann::json& tokenizer_config_json);
        void _init_image_processor(const std::filesystem::path& devkit_dir);

        const VlmConfig& _vlm_cfg;
        std::unique_ptr<Tokenizer> _tokenizer_ptr;
        std::unique_ptr<minja::chat_template> _chat_template_ptr;
        std::unique_ptr<ImageProcessor> _image_processor_ptr;
        std::optional<std::string> _system_prompt;
        bool _enable_thinking;
        std::string _bos_token;
        std::set<uint32_t> _stop_token_ids;
        std::optional<uint32_t> _image_token_id;
        std::optional<uint32_t> _pad_token_id;
};

}
}
#endif
