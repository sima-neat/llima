#include <spdlog/spdlog.h>

#include "utils.hpp"
#include "vision_language_model.hpp"

namespace simaai {
namespace llima {

VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template
) : BaseModel(model_path),
    _vlm_helper(_cfg, _devkit_dir, system_prompt, chat_template),
    _text_streamer(_vlm_helper.get_tokenizer(), std::nullopt, std::nullopt)
{
    _tool_call_format = tool_call_format_for_model(_cfg.model_type);
    _text_streamer.set_preserved_token_ids(
        resolve_tool_call_special_tokens(_tool_call_format, *_vlm_helper.get_tokenizer())
    );
    if (_cfg.support_image()) {
        _vision_model_ptr = std::make_unique<VisionModel>(model_path);
    }
    _language_model_ptr = std::make_unique<LanguageModel>(
        model_path,
        _vlm_helper.get_stop_token_ids(),
        _vlm_helper.get_image_token_id(),
        _vlm_helper.get_pad_token_id(),
        _text_streamer
    );

    // Dummy-query warmup. Skipped in spec mode: drafts have no standalone
    // run_model path.
    if (!_cfg.lm_cfg.is_spec_decode()) {
        Chat chat = create_chat();
        if (_cfg.support_image() && sample_image_file_name.has_value()) {
            chat.add_image(sample_image_file_name.value());
            chat.add_query("Describe what's in the image");
        } else {
            chat.add_query("Hi");
        }
        _text_streamer.disable();
        run_model(chat, 2);
        _text_streamer.enable();
    }

    // Pre-warm OMP thread pool to avoid libgomp first-call cost.
    warmup_omp();
}


std::optional<std::string> VisionLanguageModel::run_model(
    const Chat& chat, std::optional<uint16_t> max_new_tokens
) {
    // Acquire lock to ensure only one inference runs at a time
    std::lock_guard<std::mutex> lock(_run_mutex);
    _text_streamer.set_tool_call_enabled(chat.has_tools());
    
    ChronoTimer timer_ttft(true);

    // Preprocess the chat.
    ChronoTimer timer_preprocess(true);
    _logger->info("Chat:\n{}", chat.get_messages().dump(4));
    auto preprocessed_data = _vlm_helper.preprocess(chat);
    _logger->info("Formatted prompt: {}", preprocessed_data.formatted_prompt);
    _logger->info("Preprocess time: {:.2f}s", timer_preprocess.stop());

    // Create buffers for the input embeds and deepstack features (for qwen3) and ofm maps for
    // vision models.
    auto vision_ofm_maps = _language_model_ptr->create_input_buffers(
        preprocessed_data.input_token_ids
    );

    // Encode the image tensors to embeddings. The image embeddings are directly written to the
    // input embeddings for the language models.
    ChronoTimer timer_image_encode(true);
    for (size_t i = 0; i < vision_ofm_maps.size(); ++i) {
        _vision_model_ptr->run_model(preprocessed_data.image_tensors[i], &vision_ofm_maps[i]);
    }
    _logger->info("Image encode time: {:.2f}s", timer_image_encode.stop());

    // Decode the text and image embeddings. When a draft VLM is configured
    // (set_draft_vlm), dispatch to speculative decoding; otherwise run the
    // normal language-model decode loop.
    std::optional<uint16_t> max_num_tokens{};
    if (max_new_tokens.has_value())
        max_num_tokens = preprocessed_data.input_token_ids.size() + max_new_tokens.value();

    std::optional<std::vector<uint32_t>> output_token_ids;
    if (_draft_vlm_ptr != nullptr) {
        output_token_ids = _language_model_ptr->run_model_speculative_decoding(
            *_draft_vlm_ptr->_language_model_ptr,
            preprocessed_data.input_token_ids,
            max_num_tokens,
            timer_ttft
        );
    } else {
        output_token_ids = _language_model_ptr->run_model(
            preprocessed_data.input_token_ids, timer_ttft, max_num_tokens
        );
    }

    if (output_token_ids.has_value())
        return _vlm_helper.get_tokenizer()->decode(output_token_ids.value(), true);
    return std::nullopt;
}


std::vector<uint32_t> VisionLanguageModel::run_model(
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids
) {
    // Given a list of input token ids, return a list of generated token ids. The text streamer is
    // disabled for this mode.
    _text_streamer.set_tool_call_enabled(false);
    _text_streamer.disable();

    _language_model_ptr->create_input_buffers(input_token_ids);
    auto output_token_ids = _language_model_ptr->run_model(
        input_token_ids, std::nullopt, override_max_num_tokens, override_stop_token_ids
    );
    _text_streamer.enable();
    return output_token_ids.value_or(std::vector<uint32_t>());
}


std::vector<Eigen::bfloat16> VisionLanguageModel::run_model_for_logits(
    std::span<const uint32_t> input_token_ids
) {
    // Given a list of input token ids, return a list of computed logits for each input token id.
    // The text streamer is disabled for this mode.
    _text_streamer.disable();
    std::vector<Eigen::bfloat16> logits;
    for (size_t i = 0; i < input_token_ids.size(); ++i) {
        _language_model_ptr->run_model_once(1, i, 0, input_token_ids[i], &logits);
    }
    _text_streamer.enable();
    return logits;
}


std::vector<double> VisionLanguageModel::run_model_for_ttnt(
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids
) {
    // Given a list of input token ids, return a list of time in seconds for each generated token.
    // The text streamer is disabled for this mode.
    _text_streamer.disable();

    _language_model_ptr->create_input_buffers(input_token_ids);
    std::vector<double> ttnt;

    ChronoTimer timer(true);

    // Generate the first token.
    auto next_token_ids = _language_model_ptr->run_model(
        input_token_ids, std::nullopt, input_token_ids.size(), override_stop_token_ids
    );
    ttnt.emplace_back(timer.stop(true));
    assert(next_token_ids.has_value() && next_token_ids.value().size() == 1);
    auto next_token_id = next_token_ids.value()[0];

    if (!_language_model_ptr->get_stop_token_ids().contains(next_token_id)) {
        // Override the max_num_tokens and stop_token_ids.
        auto original_max_num_tokens = _language_model_ptr->set_max_num_tokens(
            override_max_num_tokens
        );
        auto original_stop_token_ids = _language_model_ptr->set_stop_token_ids(
            override_stop_token_ids
        );

        // Generate the new tokens until the stop token.
        for (
            size_t i = input_token_ids.size(); i < _language_model_ptr->get_max_num_tokens(); ++i
        ) {
            next_token_id = _language_model_ptr->run_model_once(
                1, i, input_token_ids.size(), next_token_id
            );
            ttnt.emplace_back(timer.stop(true));
            _logger->info("next_token_id: {}", next_token_id);
            if (_language_model_ptr->get_stop_token_ids().contains(next_token_id))
                break;
        }

        // Restore the max_num_tokens and stop_token_ids.
        _language_model_ptr->set_max_num_tokens(original_max_num_tokens);
        _language_model_ptr->set_stop_token_ids(original_stop_token_ids);
    }

    // Clear the token id cache to have correct measurement.
    _language_model_ptr->clear_cached_token_ids();

    _text_streamer.enable();
    return ttnt;
}


void VisionLanguageModel::stop_model() {
    _language_model_ptr->stop_model();
}


}
}
