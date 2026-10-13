#include <spdlog/spdlog.h>

#include <stdexcept>
#include <utility>

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
        std::move(model_path), std::move(system_prompt), std::move(chat_template), nullptr
    ) {}

VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    std::shared_ptr<FileProvider> file_provider
) : BaseModel(model_path, file_provider),
    _vlm_helper(_cfg, *_files, system_prompt, chat_template),
    _text_streamer(_vlm_helper.get_tokenizer(), std::nullopt, std::nullopt)
{
    _tool_call_format = tool_call_format_for_model(_cfg.model_type);
    if (_cfg.support_image() || _cfg.support_audio()) {
        // VisionModel loads its stage1 ELFs and vlm_config.json through the
        // same FileProvider as the language path, so over PCIe they are pulled
        // (fetch -> load -> evict), one ELF on disk at a time.
        _vision_model_ptr = std::make_unique<VisionModel>(model_path, _files);
    }
    _language_model_ptr = std::make_unique<LanguageModel>(
        model_path,
        _vlm_helper.get_stop_token_ids(),
        _vlm_helper.get_image_token_id(),
        _vlm_helper.get_pad_token_id(),
        _text_streamer,
        _files
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
    // gpt_oss always emits harmony channel tokens; preserve them even with
    // thinking off so the parser can strip to the final channel.
    if (chat.get_enable_thinking() || reasoning_format == ReasoningFormat::GptOss) {
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

    struct MediaSpan { uint16_t begin, end; size_t index; bool audio; };
    std::vector<MediaSpan> media;
    LanguageModel::MediaLoader media_loader;
    if (_cfg.vision_bidirectional_attention()) {
        const auto image_token = _vlm_helper.get_image_token_id();
        const auto& ids = preprocessed_data.input_token_ids;
        size_t images = 0, audios = 0;
        for (uint16_t at = 0; at < ids.size();) {
            const bool audio = _cfg.support_audio() && ids[at] == _cfg.audio_cfg->token_id;
            if (!audio && (!image_token || ids[at] != *image_token)) { ++at; continue; }
            uint16_t end = at + 1;
            while (end < ids.size() && ids[end] == ids[at]) ++end;
            if (audio && end - at > _cfg.audio_cfg->max_tokens) {
                throw std::invalid_argument("Audio exceeds the compiled token budget");
            }
            const size_t index = audio ? audios++ : images++;
            const size_t rows = audio
                ? preprocessed_data.audio_tensors.at(index).size() / _cfg.audio_cfg->samples_per_token
                : preprocessed_data.image_positions.at(index).size() / 2;
            if (rows != end - at) throw std::invalid_argument("Media placeholder/count mismatch");
            media.push_back({at, end, index, audio});
            at = end;
        }
        if (images != preprocessed_data.image_tensors.size() || audios != preprocessed_data.audio_tensors.size()) {
            throw std::invalid_argument("Media placeholder/count mismatch");
        }
        media_loader = [&](uint16_t begin, uint16_t end, const MLABufferSlice& input,
                           const MLABufferSlice* scales) {
            for (const auto& span : media) {
                const uint16_t first = std::max(begin, span.begin), last = std::min(end, span.end);
                if (first >= last) continue;
                const uint32_t rows = last - first, offset = first - span.begin;
                const MLABufferSlice destination{input.get_buf_ptr(), {static_cast<uint32_t>(first - begin), 0}, {rows, _cfg.lm_cfg.hidden_size}};
                const MLABufferSlice scale_rows{scales ? scales->get_buf_ptr() : nullptr,
                    {static_cast<uint32_t>(first - begin), 0}, {rows, 1}};
                if (span.audio) {
                    const auto width = _cfg.audio_cfg->samples_per_token;
                    _vision_model_ptr->run_unified_audio(
                        std::span(preprocessed_data.audio_tensors[span.index]).subspan(offset * width, rows * width),
                        destination, scales ? &scale_rows : nullptr);
                } else {
                    const auto side = _cfg.vm_cfg->spatial_patch_size * _cfg.vm_cfg->spatial_merge_size;
                    const auto width = 3 * side * side;
                    _vision_model_ptr->run_unified_model(
                        std::span(preprocessed_data.image_tensors[span.index]).subspan(offset * width, rows * width),
                        std::span(preprocessed_data.image_positions[span.index]).subspan(offset * 2, rows * 2),
                        destination, scales ? &scale_rows : nullptr);
                }
            }
        };
    } else {
        ChronoTimer timer_image_encode(true);
        for (size_t i = 0; i < vision_ofm_maps.size(); ++i) {
            _vision_model_ptr->run_model(preprocessed_data.image_tensors.at(i), &vision_ofm_maps[i]);
        }
        _logger->info("Image encode time: {:.2f}s", timer_image_encode.stop());
    }

    // Decode the text and image embeddings. When a draft VLM is configured
    // (set_draft_vlm), dispatch to speculative decoding; otherwise run the
    // normal language-model decode loop.
    std::optional<uint16_t> max_num_tokens{};
    if (max_new_tokens.has_value())
        max_num_tokens = preprocessed_data.input_token_ids.size() + max_new_tokens.value();

    auto output_token_ids = _run_language_model(
        preprocessed_data.input_token_ids, max_num_tokens, timer_ttft,
        nullptr, preprocessed_data.stable_prefix_token_count, media_loader
    );

    if (output_token_ids.has_value())
        return _vlm_helper.get_tokenizer()->decode(output_token_ids.value(), true);
    return std::nullopt;
}


std::optional<std::vector<uint32_t>> VisionLanguageModel::_run_language_model(
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult* performance_result,
    uint16_t stable_prefix_token_count,
    const LanguageModel::MediaLoader& media_loader
) {
    if (_draft_vlm_ptr == nullptr) {
        return _language_model_ptr->run_model(
            input_token_ids, timer_ttft, max_num_tokens, std::nullopt, stable_prefix_token_count, media_loader
        );
    }
    if (!_cfg.lm_cfg.speculative_decoding_cfg.has_value()) {
        throw std::runtime_error(
            "Draft VLM is configured but target model lacks speculative decoding config"
        );
    }
    if (_cfg.lm_cfg.speculative_decoding_cfg->is_gemma4_mtp()) {
        return _language_model_ptr->run_model_gemma4_mtp(
            *_draft_vlm_ptr->_language_model_ptr, input_token_ids,
            max_num_tokens, timer_ttft, performance_result
        );
    }
    return _language_model_ptr->run_model_speculative_decoding(
        *_draft_vlm_ptr->_language_model_ptr, input_token_ids,
        max_num_tokens, timer_ttft, performance_result
    );
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

    auto original_stop_token_ids = _language_model_ptr->set_stop_token_ids(override_stop_token_ids);
    try {
        _language_model_ptr->create_input_buffers(input_token_ids);
        auto output_token_ids = _run_language_model(input_token_ids, override_max_num_tokens);
        _language_model_ptr->set_stop_token_ids(original_stop_token_ids);
        _text_streamer.enable();
        return output_token_ids.value_or(std::vector<uint32_t>());
    } catch (...) {
        _language_model_ptr->set_stop_token_ids(original_stop_token_ids);
        _text_streamer.enable();
        throw;
    }
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
            auto output_token_ids = _run_language_model(
                input_token_ids, override_max_num_tokens, std::nullopt, &result
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
    auto prefill_result = _language_model_ptr->run_model(
        input_token_ids, std::nullopt, input_token_ids.size(), override_stop_token_ids
    );
    ttnt.emplace_back(timer.stop(true));
    if (!prefill_result.has_value()) {
        _language_model_ptr->clear_cached_token_ids();
        _text_streamer.enable();
        throw std::runtime_error("Performance request was interrupted during prefill");
    }
    auto next_token_id = _language_model_ptr->get_cached_first_generated_token();

    if (prefill_result->empty() && !override_stop_token_ids.value_or(
            _language_model_ptr->get_stop_token_ids()).contains(next_token_id)) {
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


}
}
