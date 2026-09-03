#include <spdlog/spdlog.h>

#include <stdexcept>

#include "reasoning_parser.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"

namespace simaai {
namespace llima {

VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template
) : VisionLanguageModel(
    std::move(model_path),
    std::move(system_prompt),
    std::move(chat_template),
    1
) {}


VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    size_t max_kv_cache_slots
) : BaseModel(model_path),
    _vlm_helper(_cfg, _devkit_dir, system_prompt, chat_template),
    _text_streamer(_vlm_helper.get_tokenizer(), std::nullopt, std::nullopt)
{
    _tool_call_format = tool_call_format_for_model(_cfg.model_type);
    if (_cfg.support_image()) {
        _vision_model_ptr = std::make_unique<VisionModel>(model_path);
    }
    _language_model_ptr = std::make_unique<LanguageModel>(
        model_path,
        _vlm_helper.get_stop_token_ids(),
        _vlm_helper.get_image_token_id(),
        _vlm_helper.get_pad_token_id(),
        _text_streamer,
        max_kv_cache_slots
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
        _language_model_ptr->clear_kv_caches();
        _text_streamer.enable();
    }

    // Pre-warm OMP thread pool to avoid libgomp first-call cost.
    warmup_omp();
}


std::optional<std::string> VisionLanguageModel::run_model(
    const Chat& chat, std::optional<uint16_t> max_new_tokens
) {
    return run_model(chat, max_new_tokens, std::nullopt);
}


std::optional<std::string> VisionLanguageModel::run_model(
    const Chat& chat,
    std::optional<uint16_t> max_new_tokens,
    std::optional<std::string> cache_id
) {
    // Acquire lock to ensure only one inference runs at a time
    std::lock_guard<std::mutex> lock(_run_mutex);
    std::vector<std::pair<uint32_t, std::string>> preserved_tokens;
    auto* tokenizer = _vlm_helper.get_tokenizer();
    if (chat.has_tools()) {
        preserved_tokens = resolve_tool_call_special_tokens(_tool_call_format, *tokenizer);
    }
    const auto reasoning_format = reasoning_format_for_model(_cfg.model_type);
    if (chat.get_enable_thinking() && reasoning_format == ReasoningFormat::None) {
        throw std::invalid_argument(
            "Thinking is not supported for model type '" + _cfg.model_type + "'"
        );
    }
    if (chat.get_enable_thinking()) {
        for (const auto marker : reasoning_special_tokens(reasoning_format)) {
            try {
                preserved_tokens.emplace_back(
                    tokenizer->token_to_id(std::string(marker)), marker
                );
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "Reasoning token '" + std::string(marker) + "' is missing from the tokenizer"
                );
            }
        }
    }
    const bool preserve_structural_tokens = !preserved_tokens.empty();
    _text_streamer.set_preserved_token_ids(std::move(preserved_tokens));
    // Keep the existing setter for ABI compatibility; it now gates all
    // request-specific structural markers, not only tool-call markers.
    _text_streamer.set_tool_call_enabled(preserve_structural_tokens);
    
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
            timer_ttft,
            nullptr,
            cache_id
        );
    } else {
        output_token_ids = _language_model_ptr->run_model(
            preprocessed_data.input_token_ids,
            timer_ttft,
            max_num_tokens,
            std::nullopt,
            cache_id
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
    return run_model(
        input_token_ids,
        override_max_num_tokens,
        std::move(override_stop_token_ids),
        std::nullopt
    );
}


std::vector<uint32_t> VisionLanguageModel::run_model(
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids,
    std::optional<std::string> cache_id
) {
    // Given a list of input token ids, return a list of generated token ids. The text streamer is
    // disabled for this mode.
    _text_streamer.set_tool_call_enabled(false);
    _text_streamer.disable();

    _language_model_ptr->create_input_buffers(input_token_ids);
    auto output_token_ids = _language_model_ptr->run_model(
        input_token_ids,
        std::nullopt,
        override_max_num_tokens,
        std::move(override_stop_token_ids),
        std::move(cache_id)
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
    try {
        for (size_t i = 0; i < input_token_ids.size(); ++i) {
            _language_model_ptr->run_model_once(1, i, 0, input_token_ids[i], &logits);
        }
        _text_streamer.enable();
        return logits;
    } catch (...) {
        _text_streamer.enable();
        throw;
    }
}


LogLikelihoodResult VisionLanguageModel::run_model_for_loglikelihood(
    std::span<const uint32_t> input_token_ids,
    size_t continuation_start,
    std::span<const uint32_t> continuation_token_ids,
    bool use_group_prefill
) {
    // Score only the continuation tokens needed by lm-eval.
    _text_streamer.disable();
    try {
        auto result = _language_model_ptr->run_model_for_loglikelihood(
            input_token_ids, continuation_start, continuation_token_ids, use_group_prefill
        );
        _text_streamer.enable();
        return result;
    } catch (...) {
        _text_streamer.enable();
        throw;
    }
}


GenerationPerformanceResult VisionLanguageModel::run_model_for_ttnt(
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids
) {
    // Given a list of input token ids, return a list of time in seconds for each generated token.
    // The text streamer is disabled for this mode.
    _text_streamer.disable();

    _language_model_ptr->create_input_buffers(input_token_ids);
    GenerationPerformanceResult result;

    if (_draft_vlm_ptr != nullptr) {
        auto original_stop_token_ids = _language_model_ptr->set_stop_token_ids(
            override_stop_token_ids
        );
        try {
            auto output_token_ids = _language_model_ptr->run_model_speculative_decoding(
                *_draft_vlm_ptr->_language_model_ptr,
                input_token_ids,
                override_max_num_tokens,
                std::nullopt,
                &result
            );
            if (!output_token_ids.has_value()) {
                throw std::runtime_error(
                    "Speculative performance request leaves insufficient token capacity "
                    "for one verification round; increase --max_new_tokens or reduce "
                    "the input length"
                );
            }
            _language_model_ptr->set_stop_token_ids(original_stop_token_ids);
            _language_model_ptr->clear_cached_token_ids();
            _draft_vlm_ptr->_language_model_ptr->clear_cached_token_ids();
            _text_streamer.enable();
            return result;
        } catch (...) {
            _language_model_ptr->set_stop_token_ids(original_stop_token_ids);
            _text_streamer.enable();
            throw;
        }
    }

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
    result.token_durations = std::move(ttnt);
    result.generated_tokens = result.token_durations.size();
    return result;
}


void VisionLanguageModel::stop_model() {
    _language_model_ptr->stop_model();
}


size_t VisionLanguageModel::kv_cache_count() const {
    return _language_model_ptr->kv_cache_count();
}


bool VisionLanguageModel::remove_kv_cache(const std::string& cache_id) {
    std::lock_guard<std::mutex> lock(_run_mutex);
    const bool removed = _language_model_ptr->remove_kv_cache(cache_id);
    if (_draft_vlm_ptr != nullptr) {
        const bool draft_removed =
            _draft_vlm_ptr->_language_model_ptr->remove_kv_cache(cache_id);
        if (removed != draft_removed) {
            _language_model_ptr->clear_kv_caches();
            _draft_vlm_ptr->_language_model_ptr->clear_kv_caches();
            throw std::runtime_error("EAGLE3 target and draft KV cache pools diverged");
        }
    }
    return removed;
}


void VisionLanguageModel::clear_kv_caches() {
    std::lock_guard<std::mutex> lock(_run_mutex);
    _language_model_ptr->clear_kv_caches();
    if (_draft_vlm_ptr != nullptr) {
        _draft_vlm_ptr->_language_model_ptr->clear_kv_caches();
    }
}


size_t VisionLanguageModel::kv_cache_bytes_per_slot() const {
    size_t bytes = _language_model_ptr->kv_cache_bytes_per_slot();
    if (_draft_vlm_ptr != nullptr) {
        bytes += _draft_vlm_ptr->_language_model_ptr->kv_cache_bytes_per_slot();
    }
    return bytes;
}


}
}
