#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <string_view>
#include <utility>

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


namespace {
// The legacy VlmHelper constructor gets the devkit folder itself, while the
// provider API names files from the model root ("devkit/tokenizer.json").
// Drop that prefix and read from the given folder, whatever it is called.
class DevkitDirProvider final : public FileProvider {
    public:
        explicit DevkitDirProvider(std::filesystem::path devkit_dir)
            : _disk(std::move(devkit_dir)) {}
        std::filesystem::path get_path(std::string_view name) override {
            return _disk.get_path(_strip(name));
        }
        std::unique_ptr<std::istream> open_stream(std::string_view name) override {
            return _disk.open_stream(_strip(name));
        }
    private:
        static std::string_view _strip(std::string_view name) {
            constexpr std::string_view prefix = "devkit/";
            return name.substr(0, prefix.size()) == prefix ? name.substr(prefix.size()) : name;
        }
        DiskFileProvider _disk;
};
}  // namespace


// The provider only lives for the delegated constructor: VlmHelper reads all
// its files while it is built and keeps no reference to the provider.
VlmHelper::VlmHelper(
    const VlmConfig& vlm_cfg,
    const std::filesystem::path& devkit_dir,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    bool enable_thinking,
    std::string reasoning_effort
) : VlmHelper(
        vlm_cfg, *std::make_unique<DevkitDirProvider>(devkit_dir), std::move(system_prompt),
        std::move(chat_template), enable_thinking, std::move(reasoning_effort)
    ) {}


VlmHelper::VlmHelper(
    const VlmConfig& vlm_cfg,
    FileProvider& files,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    bool enable_thinking,
    std::string reasoning_effort
) : _vlm_cfg(vlm_cfg), _enable_thinking(enable_thinking),
    _reasoning_effort(std::move(reasoning_effort)) {
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
        _tokenizer_ptr = Tokenizer::from_hf_json(*files.open_stream("devkit/tokenizer.json"));
        // Huggingface format.
        auto tokenizer_config_json = nlohmann::json::parse(
            *files.open_stream("devkit/tokenizer_config.json")
        );
        _init_chat_template(files, tokenizer_config_json, chat_template);
        _init_stop_token_ids(files, &tokenizer_config_json);
        if (_vlm_cfg.is_multimodal()) {
            _init_image_token_id(tokenizer_config_json);
            _init_pad_token_id(tokenizer_config_json);
            _init_image_processor(files);
        }
        if (_vlm_cfg.support_audio())
            _audio_processor_ptr = std::make_unique<Gemma4AudioPreprocessor>(*_vlm_cfg.audio_cfg, files);
    } else {
        // GGUf format.
        assert(!_vlm_cfg.is_multimodal());
        const std::string gguf_name = "devkit/" + _vlm_cfg.gguf_file_name;
        _tokenizer_ptr = Tokenizer::from_gguf(files.get_path(gguf_name));
        // The tokenizer loads only the vocabulary (vocab_only: no tensors, no
        // mmap) and keeps no hold on the file. For a model compiled from GGUF
        // this file is the whole source model (GBs), so let a pulling provider
        // (PCIe) delete it now instead of keeping it for the whole run. The
        // disk provider keeps it (release() does nothing there).
        files.release(gguf_name);
        auto chat_template_str = _tokenizer_ptr->get_chat_template();
        _bos_token = _tokenizer_ptr->get_bos_token();
        auto eos_token = _tokenizer_ptr->get_eos_token();
        _chat_template_ptr = std::make_unique<minja::chat_template>(
            chat_template_str, _bos_token, eos_token
        );
        _init_stop_token_ids(files);
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
    // gpt-oss (harmony) renders this into the system block as "Reasoning: <effort>".
    // Leave it unset when empty so the template keeps its own default.
    if (!chat.get_reasoning_effort().empty())
        inputs.extra_context["reasoning_effort"] = chat.get_reasoning_effort();
    auto formatted_prompt = _chat_template_ptr->apply(inputs);

    // Construct the actual prompt with full image tokens. Manually process the prompt because there
    // is no known automatic way.
    std::vector<std::vector<Eigen::bfloat16>> image_tensors;
    if (_vlm_cfg.support_image()) {
        auto expand_images = [&](const std::string& placeholder, std::string_view token,
                                 std::string_view prefix = {}, std::string_view suffix = {}) {
            std::string expanded(prefix);
            for (uint32_t i = 0; i < _vlm_cfg.mm_cfg.value().mm_tokens_per_image; ++i) expanded += token;
            expanded += suffix;
            return find_and_replace_all(formatted_prompt, placeholder, expanded);
        };
        uint32_t num_images;
        if (_vlm_cfg.model_type == "vlm-llava")
            num_images = expand_images("<image>", "<image>");
        else if (_vlm_cfg.model_type == "vlm-gemma3")
            num_images = expand_images("<start_of_image>", "<image_soft_token>",
                                       "\n\n<start_of_image>", "<end_of_image>\n\n");
        else if (_vlm_cfg.model_type.starts_with("vlm-lfm2"))
            num_images = expand_images("<image>", "<image>", "<|image_start|>", "<|image_end|>");
        else if (_vlm_cfg.model_type.starts_with("vlm-qwen"))
            num_images = expand_images("<|image_pad|>", "<|image_pad|>");
        else if (_vlm_cfg.model_type == "vlm-smolvlm")
            num_images = expand_images("<image>", "<image>",
                                       "<fake_token_around_image><global-img>", "<fake_token_around_image>");
        else if (_vlm_cfg.model_type == "vlm-gemma4")
            num_images = expand_images("<|image|>", "<|image|>");
        else
            throw std::runtime_error(fmt::format("Image support for {} is not implemented", _vlm_cfg.model_type));

        // CLI and HTTP attachments are both stored in message content.
        for (const auto& message : chat.get_messages()) {
            const auto& content = message.at("content");
            if (!content.is_array()) continue;
            for (const auto& item : content) {
                if (item.at("type") != "image") continue;
                std::string_view image = item.at("image").get_ref<const std::string&>();
                constexpr std::string_view base64_prefix = "data:image/jpeg;base64,";
                if (image.starts_with(base64_prefix)) {
                    auto bytes = base64_decode(image.substr(base64_prefix.size()));
                    image_tensors.emplace_back(_image_processor_ptr->preprocess(bytes));
                } else {
                    assert(std::filesystem::is_regular_file(image));
                    image_tensors.emplace_back(_image_processor_ptr->preprocess(image));
                }
            }
        }
        assert(image_tensors.size() == num_images);
    }

    std::vector<AudioInput> audio_tensors;
    size_t audio_cursor = 0;
    for (const auto& message : chat.get_messages()) {
        const auto& content = message.at("content");
        if (!content.is_array()) continue;
        for (const auto& item : content) {
            if (item.at("type") != "audio") continue;
            if (!_vlm_cfg.support_audio()) throw std::invalid_argument("This model has no compiled audio frontend");
            const auto& audio = *_vlm_cfg.audio_cfg;
            const auto& source = item.at("audio");
            const size_t budget = _audio_processor_ptr->max_samples();
            ArrayXf samples;
            if (source.is_string()) {
                const auto path = source.get<std::string>();
                if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("Audio file does not exist");
                samples = load_audio(path, budget);
            } else if (source.is_object() && source.contains("data") && source["data"].is_string()
                       && source.contains("format") && (source["format"] == "wav" || source["format"] == "mp3")) {
                const auto encoded = source["data"].get<std::string>();
                if (encoded.empty() || encoded.size() % 4
                    || encoded.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=") != std::string::npos)
                    throw std::invalid_argument("input_audio.data must contain valid base64 audio");
                const auto bytes = base64_decode(encoded);
                samples = load_audio(std::span<const uint8_t>(bytes), budget);
            } else {
                throw std::invalid_argument("Audio input requires a path or base64 data with wav/mp3 format");
            }
            auto features = _audio_processor_ptr->preprocess(samples);
            const auto at = formatted_prompt.find("<|audio|>", audio_cursor);
            if (at == std::string::npos) throw std::invalid_argument("Audio attachment has no prompt placeholder");
            std::string expanded = _tokenizer_ptr->id_to_token(audio.begin_token_id);
            for (uint32_t i = 0; i < features.num_tokens; ++i) expanded += _tokenizer_ptr->id_to_token(audio.token_id);
            expanded += _tokenizer_ptr->id_to_token(audio.end_token_id);
            formatted_prompt.replace(at, std::string_view("<|audio|>").size(), expanded);
            audio_cursor = at + expanded.size();
            audio_tensors.push_back(std::move(features));
        }
    }
    if (_vlm_cfg.support_audio()
        && formatted_prompt.find("<|audio|>", audio_cursor) != std::string::npos)
        throw std::invalid_argument("Audio placeholder has no matching attachment");

    // Encode the formatted prompt to token ids.
    auto add_special_tokens = !formatted_prompt.starts_with(_bos_token);
    auto input_token_ids = _tokenizer_ptr->encode(formatted_prompt, add_special_tokens);

    uint16_t stable_prefix_token_count = 0;
    const auto& messages = chat.get_messages();
    const auto& layer_types = _vlm_cfg.lm_cfg.layer_types;
    const auto& group_offsets = _vlm_cfg.pipeline_cfg.input_token_group_offsets;
    const bool uses_state_checkpoints = group_offsets.has_value() && !group_offsets->empty()
        && std::any_of(
            layer_types.begin(), layer_types.end(),
            [](const auto& type) { return type == "conv" || type == "linear_attention"; }
        );
    if (
        uses_state_checkpoints
        && !messages.empty()
        && (messages.front().value("role", "") == "system" || !inputs.tools.empty())
    ) {
        const auto first_user = std::find_if(
            messages.begin(), messages.end(),
            [](const auto& message) { return message.value("role", "") == "user"; }
        );
        if (first_user != messages.end()) {
            auto stable_inputs = inputs;
            stable_inputs.messages = nlohmann::ordered_json::array();
            if (messages.front().value("role", "") == "system")
                stable_inputs.messages.push_back(messages.front());
            stable_inputs.messages.push_back(*first_user);
            stable_inputs.messages.back()["content"] = "__llima_stable_prefix_end__";
            auto stable_prompt = _chat_template_ptr->apply(stable_inputs);
            auto stable_token_ids = _tokenizer_ptr->encode(
                stable_prompt, !stable_prompt.starts_with(_bos_token)
            );
            const auto mismatch = std::mismatch(
                stable_token_ids.begin(), stable_token_ids.end(),
                input_token_ids.begin(), input_token_ids.end()
            );
            stable_prefix_token_count = static_cast<uint16_t>(
                std::distance(stable_token_ids.begin(), mismatch.first)
            );
        }
    }
    return {
        std::move(formatted_prompt),
        std::move(input_token_ids),
        stable_prefix_token_count,
        std::move(image_tensors),
        std::move(audio_tensors)
    };
}


void VlmHelper::_init_chat_template(
    FileProvider& files,
    const nlohmann::json& tokenizer_config_json,
    std::optional<std::string> override_chat_template
) {
    // Determine the chat template. The devkit/chat_template.{jinja,json} files
    // are OPTIONAL (most models carry the template in tokenizer_config.json),
    // so probe with exists() — get_path() would treat their absence as a fatal
    // error. exists() also avoids pulling them over PCIe unless they are used.
    std::string chat_template_str;
    if (override_chat_template.has_value()) {
        // Use custom chat template if specified by user.
        chat_template_str = override_chat_template.value();
    } else if (_vlm_cfg.pipeline_cfg.chat_template.has_value()) {
        // Use custom chat template if specified by user.
        chat_template_str = _vlm_cfg.pipeline_cfg.chat_template.value();
    } else if (files.exists("devkit/chat_template.jinja")) {
        // Try the chat_template.jinja.
        auto stream = files.open_stream("devkit/chat_template.jinja");
        chat_template_str.assign(
            std::istreambuf_iterator<char>(*stream), std::istreambuf_iterator<char>()
        );
    } else if (files.exists("devkit/chat_template.json")) {
        // Try the chat_template.json.
        auto chat_template_json = nlohmann::json::parse(
            *files.open_stream("devkit/chat_template.json")
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


void VlmHelper::_init_stop_token_ids(
    FileProvider& files,
    const nlohmann::json* tokenizer_config_json
) {
    // Draft models use the target's tokenization scheme; stop tokens come from
    // the target model, not the draft.
    if (_vlm_cfg.lm_cfg.speculative_decoding_cfg.has_value()
        && _vlm_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft) {
        return;
    }

    // generation_config.json is OPTIONAL — fall back to tokenizer_config when
    // absent, so probe with exists() rather than get_path() (which throws on a
    // missing file) and skip the pull entirely when it is not present.
    if (files.exists("devkit/generation_config.json")) {
        auto json = nlohmann::json::parse(*files.open_stream("devkit/generation_config.json"));
        auto eos_token_id = json.at("eos_token_id");
        if (eos_token_id.is_number_integer() || eos_token_id.is_number_unsigned()) {
            _stop_token_ids.emplace(eos_token_id.get<uint32_t>());
        } else {
            eos_token_id.get_to(_stop_token_ids);
        }
    } else if (tokenizer_config_json != nullptr) {
        auto eos_token_json = tokenizer_config_json->at("eos_token");
        std::string eos_token;
        if (eos_token_json.is_string()) {
            eos_token = eos_token_json.get<std::string>();
        } else if (eos_token_json.contains("content")) {
            eos_token = eos_token_json["content"].get<std::string>();
        } else {
            throw std::runtime_error("Failed to determine the eos token from tokenizer_config.json");
        }
        _stop_token_ids.emplace(_tokenizer_ptr->token_to_id(eos_token));
    } else if (!_vlm_cfg.gguf_file_name.empty()) {
        auto eos_token_id = _tokenizer_ptr->get_eos_token_id();
        _stop_token_ids.emplace(eos_token_id);
    } else {
        throw std::runtime_error("Failed to determine the stop token ids");
    }

    if (_vlm_cfg.model_type == "vlm-qwen3_5" && tokenizer_config_json != nullptr) {
        _stop_token_ids.emplace(_tokenizer_ptr->token_to_id(
            tokenizer_config_json->at("eos_token").get<std::string>()
        ));
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



void VlmHelper::_init_image_processor(FileProvider& files) {
    // preprocessor_config.json is OPTIONAL — models without it use
    // processor_config.json instead. Probe with exists() (get_path() would
    // throw on the missing file) so the choice does not pull the wrong file.
    const bool use_preprocessor = files.exists("devkit/preprocessor_config.json");
    auto json_root = nlohmann::json::parse(*files.open_stream(
        use_preprocessor ? "devkit/preprocessor_config.json" : "devkit/processor_config.json"
    ));
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
        case 1:
            interpolation = cv::INTER_LANCZOS4;
            break;
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
