#include <fstream>
#include <string_view>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "utils.hpp"
#include "vlm_helper.hpp"

namespace simaai {
namespace llima {


uint32_t find_and_replace_all(
    std::string& text, const std::string& search, const std::string& replace
) {
    uint32_t num_replaced = 0;
    size_t pos = 0;
    while ((pos = text.find(search, pos)) != std::string::npos) {
        text.replace(pos, search.size(), replace);
        pos += replace.size();
        ++num_replaced;
    }
    return num_replaced;
}


VlmHelper::VlmHelper(
    const VlmConfig& vlm_cfg,
    const std::filesystem::path& devkit_dir,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    bool enable_thinking
) : _vlm_cfg(vlm_cfg), _enable_thinking(enable_thinking) {
    if (system_prompt.has_value())
        _system_prompt = system_prompt;
    else
        _system_prompt = _vlm_cfg.pipeline_cfg.system_prompt;

    // Spec drafts ship no tokenization files and reuse the target's tokenizer;
    // skip init entirely (the draft's vlm_helper is never used for preprocess).
    const bool is_spec_decode_draft = _vlm_cfg.lm_cfg.is_spec_decode()
        && _vlm_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
    if (is_spec_decode_draft) {
        return;
    }

    if (_vlm_cfg.gguf_file_name.empty()) {
        _tokenizer_ptr = Tokenizer::from_hf_json(devkit_dir / "tokenizer.json");
        // Huggingface format.
        auto tokenizer_config_json = nlohmann::json::parse(
            std::ifstream(devkit_dir / "tokenizer_config.json")
        );
        _init_chat_template(devkit_dir, tokenizer_config_json, chat_template);
        _init_stop_token_ids(devkit_dir);
        if (_vlm_cfg.is_multimodal()) {
            _init_image_token_id(tokenizer_config_json);
            _init_pad_token_id(tokenizer_config_json);
            _init_image_processor(devkit_dir);
        }
    } else {
        // GGUf format.
        assert(!_vlm_cfg.is_multimodal());
        _tokenizer_ptr = Tokenizer::from_gguf(devkit_dir / _vlm_cfg.gguf_file_name);
        auto chat_template_str = _tokenizer_ptr->get_chat_template();
        _bos_token = _tokenizer_ptr->get_bos_token();
        auto eos_token = _tokenizer_ptr->get_eos_token();
        _chat_template_ptr = std::make_unique<minja::chat_template>(
            chat_template_str, _bos_token, eos_token
        );
        _init_stop_token_ids(devkit_dir);
    }
}


void VlmHelper::set_system_prompt(const std::optional<std::string>& system_prompt) {
    _system_prompt = std::move(system_prompt);
}


PreprocessedChat VlmHelper::preprocess(const Chat& chat) {
    // Apply chat template.
    minja::chat_template_inputs inputs;
    inputs.messages = chat.get_messages();
    inputs.tools = chat.get_tools();
    inputs.extra_context["enable_thinking"] = chat.get_enable_thinking();
    auto formatted_prompt = _chat_template_ptr->apply(inputs);

    // Construct the actual prompt with full image tokens. Manually process the prompt because there
    // is no known automatic way.
    std::vector<std::vector<Eigen::bfloat16>> image_tensors;
    if (_vlm_cfg.support_image()) {
        uint32_t num_images;
        if (_vlm_cfg.model_type == "vlm-llava") {
            std::string search = "<image>";
            std::string replace = "";
            for (uint32_t i = 0; i < _vlm_cfg.mm_cfg.value().mm_tokens_per_image; ++i) {
                replace += "<image>";
            }
            num_images = find_and_replace_all(formatted_prompt, search, replace);
        } else if (_vlm_cfg.model_type == "vlm-gemma3") {
            std::string search = "<start_of_image>";
            std::string replace = "\n\n<start_of_image>";
            for (uint32_t i = 0; i < _vlm_cfg.mm_cfg.value().mm_tokens_per_image; ++i) {
                replace += "<image_soft_token>";
            }
            replace += "<end_of_image>\n\n";
            num_images = find_and_replace_all(formatted_prompt, search, replace);
        } else if (_vlm_cfg.model_type.starts_with("vlm-lfm2")) {
            std::string search = "<image>";
            std::string replace = "<|image_start|>";
            for (uint32_t i = 0; i < _vlm_cfg.mm_cfg.value().mm_tokens_per_image; ++i) {
                replace += "<image>";
            }
            replace += "<|image_end|>";
            num_images = find_and_replace_all(formatted_prompt, search, replace);
        } else if (_vlm_cfg.model_type.starts_with("vlm-qwen")) {
            std::string search = "<|image_pad|>";
            std::string replace = "";
            for (uint32_t i = 0; i < _vlm_cfg.mm_cfg.value().mm_tokens_per_image; ++i) {
                // Use <|image_pad|> instead of <|placeholder|> so that the language model can find
                // the image_token locations.
                replace += "<|image_pad|>";
            }
            num_images = find_and_replace_all(formatted_prompt, search, replace);
        } else if (_vlm_cfg.model_type == "vlm-gemma4") {
            std::string search = "<|image|>";
            std::string replace = "";
            for (uint32_t i = 0; i < _vlm_cfg.mm_cfg.value().mm_tokens_per_image; ++i) {
                replace += "<|image|>";
            }
            num_images = find_and_replace_all(formatted_prompt, search, replace);
        } else {
            throw std::runtime_error(
                fmt::format("Image support for {} is not implemented", _vlm_cfg.model_type)
            );
        }

        // Preprocess images.
        if (chat.get_images().size() == 0 && num_images > 0) {
            // Images are embedded in the messages.
            for (const auto& message: chat.get_messages()) {
                const auto& content = message.at("content");
                // OpenAI-compatible history may mix plain-string text messages with array content.
                if (!content.is_array()) {
                    continue;
                }
                for (const auto& item: content) {
                    if (item.at("type") != "image")
                        continue;
                    std::string_view image_str = item.at("image").get_ref<const std::string&>();
                    const std::string_view base64_prefix = "data:image/jpeg;base64,";
                    if (image_str.starts_with(base64_prefix)) {
                        auto image_bytes = base64_decode(image_str.substr(base64_prefix.size()));
                        image_tensors.emplace_back(_image_processor_ptr->preprocess(image_bytes));
                    } else {
                        // Assume regular file.
                        assert(std::filesystem::is_regular_file(image_str));
                        image_tensors.emplace_back(_image_processor_ptr->preprocess(image_str));
                    }
                }
            }
        } else {
            assert(chat.get_images().size() == num_images);
            for (const auto& image_file_name: chat.get_images()) {
                image_tensors.emplace_back(_image_processor_ptr->preprocess(image_file_name));
            }
        }
    }

    // Encode the formatted prompt to token ids.
    auto add_special_tokens = !formatted_prompt.starts_with(_bos_token);
    auto input_token_ids = _tokenizer_ptr->encode(formatted_prompt, add_special_tokens);
    return {
        std::move(formatted_prompt),
        std::move(input_token_ids),
        std::move(image_tensors)
    };
}


void VlmHelper::_init_chat_template(
    const std::filesystem::path& devkit_dir,
    const nlohmann::json& tokenizer_config_json,
    std::optional<std::string> override_chat_template
) {
    // Determine the chat template.
    auto chat_template_jinja_file_name = devkit_dir / "chat_template.jinja";
    auto chat_template_json_file_name = devkit_dir / "chat_template.json";
    std::string chat_template_str;
    if (override_chat_template.has_value()) {
        // Use custom chat template if specified by user.
        chat_template_str = override_chat_template.value();
    } else if (_vlm_cfg.pipeline_cfg.chat_template.has_value()) {
        // Use custom chat template if specified by user.
        chat_template_str = _vlm_cfg.pipeline_cfg.chat_template.value();
    } else if (std::filesystem::is_regular_file(chat_template_jinja_file_name)) {
        // Try the chat_template.jinja.
        auto size = std::filesystem::file_size(chat_template_jinja_file_name);
        std::ifstream file(chat_template_jinja_file_name, std::ios::binary);
        chat_template_str.assign(size, '\0');
        file.read(chat_template_str.data(), size);
    } else if (std::filesystem::is_regular_file(chat_template_json_file_name)) {
        // Try the chat_template.json.
        auto chat_template_json = nlohmann::json::parse(
            std::ifstream(chat_template_json_file_name)
        );
        chat_template_str = chat_template_json.at("chat_template");
    } else if (
        tokenizer_config_json.contains("chat_template")
        && !tokenizer_config_json.at("chat_template").is_null()
    ) {
        // Last location to find is tokenizer_config.json.
        chat_template_str = tokenizer_config_json.at("chat_template");
    } else {
        throw std::runtime_error(
            fmt::format(
                "Model '{}' is missing a chat template. This runtime requires one of: "
                "--chat_template, --chat_template_file, devkit/chat_template.jinja, "
                "devkit/chat_template.json, or tokenizer_config.json['chat_template']. "
                "Use an instruct/chat model or provide a template.",
                _vlm_cfg.model_name
            )
        );
    }

    auto bos_token_json = tokenizer_config_json.at("bos_token");
    auto eos_token_json = tokenizer_config_json.at("eos_token");
    std::string eos_token;
    if (bos_token_json.is_null())
        _bos_token = "";
    else if (bos_token_json.is_string())
        _bos_token = bos_token_json.get<std::string>();
    else if (bos_token_json.contains("content"))
        _bos_token = bos_token_json["content"].get<std::string>();
    else
        throw std::runtime_error("Failed to find bos token");
    if (eos_token_json.is_null())
        eos_token = "";
    else if (eos_token_json.is_string())
        eos_token = eos_token_json.get<std::string>();
    else if (eos_token_json.contains("content"))
        eos_token = eos_token_json["content"].get<std::string>();
    else
        throw std::runtime_error("Failed to find eos token");
    _chat_template_ptr = std::make_unique<minja::chat_template>(
        chat_template_str, _bos_token, eos_token
    );
}


void VlmHelper::_init_stop_token_ids(const std::filesystem::path& devkit_dir) {
    // Draft models use the target's tokenization scheme; stop tokens come from
    // the target model, not the draft.
    if (_vlm_cfg.lm_cfg.speculative_decoding_cfg.has_value()
        && _vlm_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft) {
        return;
    }

    auto generation_config_file_name = devkit_dir / "generation_config.json";
    if (std::filesystem::is_regular_file(generation_config_file_name)) {
        auto json = nlohmann::json::parse(std::ifstream(generation_config_file_name));
        auto eos_token_id = json.at("eos_token_id");
        if (eos_token_id.is_number_unsigned()) {
            _stop_token_ids.emplace(eos_token_id.get<uint32_t>());
        } else {
            eos_token_id.get_to(_stop_token_ids);
        }
    } else if (!_vlm_cfg.gguf_file_name.empty()) {
        auto eos_token_id = _tokenizer_ptr->get_eos_token_id();
        _stop_token_ids.emplace(eos_token_id);
    } else {
        throw std::runtime_error("Failed to determine the stop token ids");
    }
}


void VlmHelper::_init_image_token_id(const nlohmann::json& tokenizer_config_json) {
    std::string image_token;
    if (tokenizer_config_json.contains("image_token")) {
        image_token = tokenizer_config_json["image_token"];
    } else if (_vlm_cfg.model_type.find("qwen") != std::string::npos) {
        // Qwen sets default image token as <|image_pad|>.
        image_token = "<|image_pad|>";
    } else {
        throw std::runtime_error("Failed to find image token");
    }
    _image_token_id = _tokenizer_ptr->token_to_id(image_token);
}


void VlmHelper::_init_pad_token_id(const nlohmann::json& tokenizer_config_json) {
    if (!tokenizer_config_json.contains("pad_token") || tokenizer_config_json["pad_token"].is_null())
        return;
    _pad_token_id = _tokenizer_ptr->token_to_id(tokenizer_config_json["pad_token"].get<std::string>());
}



void VlmHelper::_init_image_processor(const std::filesystem::path& devkit_dir) {
    auto p1 = devkit_dir / "preprocessor_config.json";
    auto p2 = devkit_dir / "processor_config.json";
    auto json_root = nlohmann::json::parse(std::ifstream(std::filesystem::exists(p1) ? p1 : p2));
    const auto& json = json_root.contains("image_processor") ? json_root["image_processor"] : json_root;

    bool do_pad_to_square;
    if (json.contains("do_pad") && json["do_pad"].is_boolean()) {
        do_pad_to_square = json["do_pad"];
    } else if (
        _vlm_cfg.model_type.starts_with("vlm-lfm2")
        || _vlm_cfg.model_type.starts_with("vlm-qwen")
    ) {
        do_pad_to_square = true;
    } else {
        do_pad_to_square = false;
    }

    bool do_center_crop;
    if (json.contains("do_center_crop") && json["do_center_crop"].is_boolean()) {
        do_center_crop = json["do_center_crop"];
    } else {
        do_center_crop = false;
    }

    // If resample is not set in the config file, then use INTER_CUBIC by default.
    int resample = json.value("resample", 3);
    cv::InterpolationFlags interpolation;
    switch (resample) {
        case 2:
            interpolation = cv::INTER_LINEAR;
            break;
        case 3:
            interpolation = cv::INTER_CUBIC;
            break;
        default:
            throw std::runtime_error(fmt::format("Unsupported resample type: {}", resample));
    }

    double rescale_factor = json.value("rescale_factor", 1.0 / 255.0);
    std::vector<double> image_mean = json["image_mean"];
    std::vector<double> image_std = json["image_std"];

    _image_processor_ptr = std::make_unique<ImageProcessor>(
        _vlm_cfg,
        do_pad_to_square,
        do_center_crop,
        interpolation,
        rescale_factor,
        std::move(image_mean),
        std::move(image_std)
    );
}


}
}
