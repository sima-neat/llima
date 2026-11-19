#include "embedding_offload.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <utility>
#include <string_view>

#include <Eigen/Dense>
#include <cnpy.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "eagle_helpers.hpp"
#include "language_model.hpp"
#include "language_model_executor.hpp"
#include "language_model_state.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

namespace {

LogLikelihoodResult score_logits(
    std::span<const Eigen::bfloat16> logits, uint32_t target_token_id
) {
    if (target_token_id >= logits.size()) {
        throw std::runtime_error(
            fmt::format(
                "Target token id {} is outside logits vocabulary size {}",
                target_token_id,
                logits.size()
            )
        );
    }

    double max_logit = -std::numeric_limits<double>::infinity();
    uint32_t argmax_token_id = 0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double value = static_cast<float>(logits[i]);
        if (value > max_logit) {
            max_logit = value;
            argmax_token_id = static_cast<uint32_t>(i);
        }
    }

    double exp_sum = 0.0;
    for (const auto logit : logits) {
        exp_sum += std::exp(static_cast<float>(logit) - max_logit);
    }

    const double target_logit = static_cast<float>(logits[target_token_id]);
    return {
        target_logit - (max_logit + std::log(exp_sum)),
        argmax_token_id == target_token_id
    };
}

}

LanguageModel::~LanguageModel() { _finalize(); }

void LanguageModel::stop_model() { _run->cancelled.store(true, std::memory_order_relaxed); }

uint16_t LanguageModel::get_max_num_tokens() const { return _run->max_num_tokens; }

const std::set<uint32_t>& LanguageModel::get_stop_token_ids() const { return _run->stop_token_ids; }

void LanguageModel::clear_cached_token_ids() { _session->cached_token_ids.clear(); }

uint32_t LanguageModel::get_cached_first_generated_token() const {
    return _session->cached_first_generated_token;
}

const std::vector<std::vector<Eigen::bfloat16>>& LanguageModel::get_eagle3_intermediate_hidden_states() const {
    return _workspace->eagle3_intermediate_hidden_states;
}

uint16_t LanguageModel::get_kv_cache_len() const { return _session->kv_cache_len; }

void LanguageModel::set_kv_cache_len(uint16_t len) { _session->kv_cache_len = len; }


LanguageModel::LanguageModel(
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer
) : LanguageModel(
        std::move(model_path), std::move(stop_token_ids), image_token_id, pad_token_id,
        text_streamer, nullptr
    ) {}

LanguageModel::LanguageModel(
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer,
    std::shared_ptr<FileProvider> file_provider
) : LanguageModel(std::move(model_path), std::move(stop_token_ids), image_token_id, pad_token_id,
                  text_streamer, std::move(file_provider), false) {}

LanguageModel::LanguageModel(
    std::filesystem::path model_path, std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id, std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer, std::shared_ptr<FileProvider> file_provider,
    bool defer_session_buffers
) : _resources(LanguageResources::acquire(model_path, std::move(file_provider))),
    _cfg(_resources->cfg),
    _elf_dir(_resources->package->elf_dir),
    _devkit_dir(_resources->package->devkit_dir),
    _files(_resources->package->files),
    _logger(_resources->package->logger),
    _sessions(std::make_unique<SessionManager>()),
    _session(_sessions->get(0)),
    _workspace(std::make_unique<LanguageWorkspace>()),
    _run(std::make_unique<LanguageRunState>(std::move(stop_token_ids),
        _cfg.pipeline_cfg.max_num_tokens, text_streamer)),
    _image_token_id(image_token_id),
    _pad_token_id(pad_token_id)
{
    std::lock_guard execution_lock(RuntimeContext::instance().execution_mutex);
    if (
        _cfg.pipeline_cfg.max_num_tokens == 0
        || _cfg.pipeline_cfg.max_num_tokens % MAX_NUM_TOKENS_ALIGNMENT
    ) {
        throw std::runtime_error(
            "max_num_tokens must be a positive multiple of 1024"
        );
    }
    if (
        _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_eagle3()
        && _cfg.lm_cfg.attn_cfg.swa_enable
    ) {
        throw std::runtime_error(
            "EAGLE3 speculative decoding does not support sliding-window attention"
        );
    }

    _use_group_token_models = (
        _cfg.pipeline_cfg.input_token_group_offsets.has_value()
        && _cfg.pipeline_cfg.input_token_group_offsets.value().size() > 0
        // Gemma4 MTP assistants only execute their speculative-width path. Their
        // serialized pipeline config retains target group offsets, but the compiler
        // intentionally does not emit grouped assistant pre/cache/post models.
        && !_cfg.lm_cfg.is_gemma4_mtp_draft()
    );
    _need_argmax = _cfg.lm_cfg.lm_head_num_splits > 1 || _cfg.pipeline_cfg.return_logits;

    _init_session_layout();

    // EAGLE3: build the constant tree_mask_init (eye(topk)) once. Only the
    // draft uses it (consumed in topk_generate's depth loop seed and concat).
    if (_cfg.lm_cfg.is_eagle3_draft()) {
        const int topk = _cfg.lm_cfg.speculative_decoding_cfg.value().speculative_budget;
        _workspace->eagle3_tree_mask_init.data = std::vector<std::vector<std::vector<std::vector<float>>>>(
            1, std::vector<std::vector<std::vector<float>>>(
                1, std::vector<std::vector<float>>(
                    topk, std::vector<float>(topk, 0.0f)
                )
            )
        );
        for (int i = 0; i < topk; ++i) _workspace->eagle3_tree_mask_init.data[0][0][i][i] = 1.0f;

        _workspace->eagle3_position_ids.assign(topk, 0);
    }

    _initialize(defer_session_buffers);
}


bool LanguageModel::_has_linear_attention_layers() const {
    return std::find(
        _cfg.lm_cfg.layer_types.begin(),
        _cfg.lm_cfg.layer_types.end(),
        "linear_attention"
    ) != _cfg.lm_cfg.layer_types.end();
}


const LinearAttentionConfig& LanguageModel::_linear_attn_cfg() const {
    if (!_cfg.lm_cfg.linear_attn_cfg.has_value()) {
        throw std::runtime_error("linear_attention layers require linear_attn_cfg");
    }
    const auto& linear_cfg = _cfg.lm_cfg.linear_attn_cfg.value();
    if (linear_cfg.conv_kernel_dim <= 1) {
        throw std::runtime_error("linear_attn_cfg.conv_kernel_dim must be greater than 1");
    }
    return linear_cfg;
}


std::vector<std::map<uint8_t, MLABufferSlice>> LanguageModel::create_input_buffers(
    std::span<const uint32_t> input_token_ids
) {
    // Log.
    _logger->info("Input token ids: [{}]", fmt::join(input_token_ids, ", "));

    // Determine the shape of the buffer for input embeds.
    const uint16_t num_input_tokens = input_token_ids.size();
    uint16_t num_padded_input_tokens = num_input_tokens;
    if (_use_group_token_models) {
        for (const auto& offset: _cfg.pipeline_cfg.input_token_group_offsets.value()) {
            if (offset + _cfg.pipeline_cfg.input_token_group_size > num_input_tokens) {
                num_padded_input_tokens = offset + _cfg.pipeline_cfg.input_token_group_size;
                break;
            }
        }
    }

    // Allocate the input embeds with padding to the buffer.
    if (_resources->embedding_offload) {
        _workspace->offload_chunk_staging = !_image_token_id.has_value()
            || std::find(input_token_ids.begin(), input_token_ids.end(), *_image_token_id) == input_token_ids.end();
        if (_workspace->offload_chunk_staging) {
            num_padded_input_tokens = std::max<uint16_t>(
                _cfg.lm_cfg.get_single_num_tokens(),
                _use_group_token_models ? _cfg.pipeline_cfg.input_token_group_size : 1
            );
        }
    }
    define_buffer(
        "input_embeds",
        {num_padded_input_tokens, _cfg.lm_cfg.hidden_size},
        _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16"
    );
    MLABuffer& input_embeds_buf = get_buffer("input_embeds");
    input_embeds_buf.allocate();
    input_embeds_buf.clear();
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        define_buffer("input_embedding_scales", {num_padded_input_tokens, 1});
        auto& scale_buf = get_buffer("input_embedding_scales");
        scale_buf.allocate();
        scale_buf.clear();
    }

    std::vector<std::map<uint8_t, MLABufferSlice>> vision_ofm_maps;
    if (_cfg.vm_cfg.has_value()) {
        // Allocate the deepstack feature caches for qwen3 with the same size as input embeds.
        for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
            auto& buf = get_buffer(fmt::format("deepstack_feature_l{}_cache", i));
            buf.clear();
        }

        // Find all the start idx of the image tokens and create the ofm_maps for each image.
        auto mm_tokens_per_image = _cfg.mm_cfg.value().mm_tokens_per_image;
        uint32_t token_idx = 0;
        do {
            auto it = std::find(
                input_token_ids.begin() + token_idx, input_token_ids.end(), _image_token_id
            );
            if (it == input_token_ids.end())
                break;

            // Move the token_idx to the start of an image.
            token_idx = std::distance(input_token_ids.begin(), it);

            // Fill the ofm map.
            std::map<uint8_t, MLABufferSlice> ofm_map;
            ofm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    &get_buffer("input_embeds"),
                    std::vector<uint32_t>{token_idx, 0},
                    std::vector<uint32_t>{mm_tokens_per_image, _cfg.lm_cfg.hidden_size}
                )
            );
            const uint8_t deepstack_ofm_begin = _cfg.pipeline_cfg.quantize_embeddings ? 2 : 1;
            if (_cfg.pipeline_cfg.quantize_embeddings) {
                ofm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(1),
                    std::forward_as_tuple(
                        &get_buffer("input_embedding_scales"),
                        std::vector<uint32_t>{token_idx, 0},
                        std::vector<uint32_t>{mm_tokens_per_image, 1}
                    )
                );
            }
            for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
                ofm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(i + deepstack_ofm_begin),
                    std::forward_as_tuple(
                        &get_buffer(fmt::format("deepstack_feature_l{}_cache", i)),
                        std::vector<uint32_t>{token_idx, 0},
                        std::vector<uint32_t>{mm_tokens_per_image, _cfg.lm_cfg.hidden_size}
                    )
                );
            }
            vision_ofm_maps.emplace_back(ofm_map);
            token_idx += _cfg.mm_cfg.value().mm_tokens_per_image;
        } while (token_idx < input_token_ids.size());
    }
    return vision_ofm_maps;
}


std::optional<std::vector<uint32_t>> LanguageModel::run_model(
    std::span<const uint32_t> input_token_ids,
    std::optional<ChronoTimer> timer_ttft,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids,
    uint16_t stable_prefix_token_count
) {
    LanguageExecutor::Run run(*this);
    set_max_num_tokens(override_max_num_tokens);
    set_stop_token_ids(override_stop_token_ids);

    std::optional<std::vector<uint32_t>> output_token_ids{};

    // If the input is already greater than the cache size, no need to run the model.
    if (input_token_ids.size() > _run->max_num_tokens) {
        _notify_cache_full();
        output_token_ids = std::vector<uint32_t>();
    } else {
        _init_state_checkpoint_capture(input_token_ids.size(), stable_prefix_token_count);

        // Create input embeds from input token ids and image embeds.
        auto num_cached_tokens = _set_input_text_embeds(input_token_ids);

        // Prefill.
        auto token_id = run_model_prefill(input_token_ids, num_cached_tokens, timer_ttft);
        const auto output_token_id_begin = input_token_ids.size();
        if (_run->cancelled.load(std::memory_order_relaxed)) {
            run.complete();
            return std::nullopt;
        }
        if (input_token_ids.size() == _run->max_num_tokens) {
            _notify_cache_full();
            output_token_ids = std::vector<uint32_t>();
        } else if (_run->stop_token_ids.contains(token_id)) {
            _notify_stop();
            output_token_ids = std::vector<uint32_t>{token_id};
        } else {
            // Decode.
            run_model_decode(input_token_ids.size(), token_id);
            output_token_ids = std::vector<uint32_t>(
                _session->cached_token_ids.begin() + output_token_id_begin, _session->cached_token_ids.end()
            );
        }
    }

    run.complete();
    return _run->cancelled.load(std::memory_order_relaxed) ? std::nullopt : output_token_ids;
}


std::optional<std::vector<uint32_t>> LanguageModel::run_model_speculative_decoding(
    LanguageModel& draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens, std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult* performance_result) {
    return run_model_speculative_decoding(draft_lm, input_token_ids, override_max_num_tokens,
                                          timer_ttft, performance_result, 0);
}

std::optional<std::vector<uint32_t>> LanguageModel::run_model_speculative_decoding(
    LanguageModel& draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens, std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult* performance_result, uint16_t stable_prefix_token_count) {
    LanguageExecutor::Run run(*this, &draft_lm);
    attach_draft(draft_lm);
    const auto& target = *_cfg.lm_cfg.speculative_decoding_cfg;
    const uint16_t limit = std::min(override_max_num_tokens.value_or(_run->max_num_tokens),
                                    _cfg.pipeline_cfg.max_num_tokens);
    std::optional<std::vector<uint32_t>> output;
    if (target.is_gemma4_mtp()) {
        output = run_model_gemma4_mtp(draft_lm, input_token_ids, limit, timer_ttft, performance_result);
    } else if (target.method == "dflash") {
        output = _run_model_dflash_speculative_decoding(draft_lm, input_token_ids, limit, timer_ttft,
                                                       performance_result, stable_prefix_token_count);
    } else {
        output = _run_model_eagle3_speculative_decoding(draft_lm, input_token_ids, limit, timer_ttft,
                                                      performance_result);
    }
    run.complete();
    return _run->cancelled.load(std::memory_order_relaxed) ? std::nullopt : output;
}

void LanguageModel::attach_draft(LanguageModel& draft_lm) {
    std::lock_guard lock(RuntimeContext::instance().execution_mutex);
    if (_workspace->paired_draft.lock() == draft_lm._resources) return;
    if (!_cfg.lm_cfg.speculative_decoding_cfg.has_value() ||
        !draft_lm._cfg.lm_cfg.speculative_decoding_cfg.has_value()) {
        throw std::invalid_argument("Both models must contain speculative-decoding configuration");
    }
    const auto& target = _cfg.lm_cfg.speculative_decoding_cfg.value();
    const auto& draft = draft_lm._cfg.lm_cfg.speculative_decoding_cfg.value();
    if (target.is_draft || !draft.is_draft) {
        throw std::invalid_argument("Speculative decoding requires a target paired with a draft");
    }
    if (target.method != draft.method) {
        throw std::invalid_argument("Target and draft speculative methods do not match");
    }
    if (target.is_gemma4_mtp()) {
        if (!_cfg.lm_cfg.is_gemma4_mtp_target() || !draft_lm._cfg.lm_cfg.is_gemma4_mtp_draft())
            throw std::invalid_argument("Gemma4 MTP requires a target and an assistant");
        if (_cfg.lm_cfg.hidden_size != draft.backbone_hidden_size)
            throw std::invalid_argument("Gemma4 MTP target hidden size must match assistant backbone size");
        const auto width = _cfg.lm_cfg.get_single_num_tokens();
        if (!draft.speculative_budget || width != draft.speculative_budget + 1
            || draft_lm._cfg.lm_cfg.get_single_num_tokens() != 1)
            throw std::invalid_argument("Gemma4 MTP requires an N1 assistant and budget-plus-one target width");
        _workspace->mtp_shared_kv_layers = draft_lm._prepare_gemma4_mtp_shared_kv_layers(*this);
    } else {
    if (_cfg.lm_cfg.hidden_size != draft_lm._cfg.lm_cfg.hidden_size ||
        _cfg.lm_cfg.token_cfg.vocab_size != draft_lm._cfg.lm_cfg.token_cfg.vocab_size) {
        throw std::invalid_argument("Target and draft hidden or vocabulary sizes do not match");
    }
    if (target.method == "dflash") {
        if (target.speculative_budget != draft.speculative_budget ||
            target.target_layer_ids != draft.target_layer_ids) {
            throw std::invalid_argument(
                "DFlash target and draft block configuration does not match");
        }
        if (_cfg.pipeline_cfg.input_token_group_size !=
            draft_lm._cfg.pipeline_cfg.input_token_group_size) {
            throw std::invalid_argument(
                "DFlash target and draft language group sizes do not match");
        }
        if (_cfg.pipeline_cfg.input_token_group_size <= target.speculative_budget) {
    throw std::invalid_argument("DFlash language group size must exceed the block size");
}
        if (draft_lm._cfg.pipeline_cfg.max_num_tokens < _cfg.pipeline_cfg.max_num_tokens) {
    throw std::invalid_argument("DFlash draft cache is smaller than the target cache");
}
        if (_cfg.pipeline_cfg.quantize_embeddings !=
            draft_lm._cfg.pipeline_cfg.quantize_embeddings) {
            throw std::invalid_argument(
                "DFlash target and draft embedding quantization modes do not match");
        }
        if (_cfg.pipeline_cfg.future_token_mask_size <= 1 ||
            draft_lm._cfg.pipeline_cfg.future_token_mask_size <= 1) {
            throw std::invalid_argument(
                "DFlash models require a multi-token future-mask configuration");
        }
        if (target.mask_token_id != draft.mask_token_id || draft.mask_token_id < 0 ||
            static_cast<uint32_t>(draft.mask_token_id) >= _cfg.lm_cfg.token_cfg.vocab_size) {
    throw std::invalid_argument("DFlash target and draft mask token IDs do not match");
}
    } else if (target.method != "eagle3") {
        throw std::invalid_argument("Unsupported speculative-decoding method: " + target.method);
    }
    }
    _workspace->paired_draft = draft_lm._resources;
}

uint32_t LanguageModel::run_model_prefill(
    std::span<const uint32_t> input_token_ids,
    uint16_t num_cached_tokens,
    std::optional<ChronoTimer> timer_ttft
) {
    if (_uses_per_layer_inputs())
        _workspace->prompt_per_layer_token_ids = _get_per_layer_token_ids(input_token_ids);
    if (!timer_ttft.has_value())
        timer_ttft = ChronoTimer{true};
    const uint16_t num_input_tokens = input_token_ids.size();
    const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
    uint16_t token_idx{};
    uint32_t next_token_id{};
    uint16_t last_group_valid_tokens = 0;
    num_cached_tokens = _prepare_state_checkpoints_for_prefill(num_cached_tokens);

    if (num_input_tokens == num_cached_tokens) {
        // All input tokens are already cached.
        if (num_cached_tokens < _session->cached_token_ids.size())
            next_token_id = _session->cached_token_ids[num_cached_tokens];
        else
            next_token_id = _session->cached_first_generated_token;
    } else if (_use_group_token_models) {
        const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
        const auto& num_tokens = _cfg.pipeline_cfg.input_token_group_size;
        auto it = offsets.end();
        while (it != offsets.begin()) {
            --it;
            token_idx = *it;
            if (*it > num_cached_tokens)
                continue;
            if (token_idx <= num_cached_tokens && token_idx + num_tokens > num_cached_tokens)
                break;
            // Number of cached tokens are greater than the supported group offsets.
            token_idx = num_cached_tokens;
            it = offsets.end();
            break;
        }

        while (token_idx < num_input_tokens) {
            if (it != offsets.end()) {
                assert(token_idx >= *it && token_idx < *it + num_tokens);
                token_idx = *it;
                last_group_valid_tokens = std::min(
                    num_input_tokens, static_cast<uint16_t>(token_idx + num_tokens)
                ) - token_idx;
                if (_uses_per_layer_inputs()) {
                    _compute_and_upload_per_layer_inputs_prefill(
                        num_tokens, token_idx, num_input_tokens
                    );
                }
                next_token_id = run_model_once(num_tokens, token_idx, num_input_tokens, 0);
                ++it;
                token_idx += num_tokens;
            } else {
                if (_uses_per_layer_inputs()) {
                    _compute_and_upload_per_layer_inputs_prefill(1, token_idx, num_input_tokens);
                }
                next_token_id = run_model_once(1, token_idx, num_input_tokens, 0);
                token_idx += 1;
            }
            if (_run->cancelled.load(std::memory_order_relaxed)) {
                if (!_cfg.lm_cfg.is_gemma4_mtp_target()) _notify_interrupt();
                return next_token_id;
            }
        }
        if (
            !_session->cached_states.empty()
            && last_group_valid_tokens > 0
            && last_group_valid_tokens < group_size
        ) {
    _move_state_tail_for_decode(last_group_valid_tokens);
}
    } else {
        for (token_idx = num_cached_tokens; token_idx < num_input_tokens; ++token_idx) {
            if (_uses_per_layer_inputs()) {
                _compute_and_upload_per_layer_inputs_prefill(1, token_idx, num_input_tokens);
            }
            next_token_id = run_model_once(1, token_idx, num_input_tokens, 0);
            if (_run->cancelled.load(std::memory_order_relaxed)) {
                if (!_cfg.lm_cfg.is_gemma4_mtp_target()) _notify_interrupt();
                return next_token_id;
            }
        }
    }
    _session->cached_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
    auto duration = timer_ttft.value().stop();
    if (num_input_tokens < _run->max_num_tokens)
        _notify_first_token(next_token_id, duration);
    _session->cached_first_generated_token = next_token_id;
    return next_token_id;
}


void LanguageModel::run_model_decode(
    uint16_t num_input_tokens, uint32_t token_id
) {
    ChronoTimer timer_tps(true);
    uint16_t token_idx;
    for (token_idx = num_input_tokens; token_idx < _run->max_num_tokens; ++token_idx ) {
        if (_run->cancelled.load(std::memory_order_relaxed)) {
            _notify_interrupt();
            return;
        }
        auto next_token_id = run_model_once(1, token_idx, num_input_tokens, token_id);
        if (_run->cancelled.load(std::memory_order_relaxed)) {
            _notify_interrupt();
            return;
        }
        _session->cached_token_ids.emplace_back(token_id);
        token_id = next_token_id;

        auto duration = timer_tps.stop(true);
        if (token_idx + 1 < _run->max_num_tokens) {
            _notify_new_token(next_token_id, duration);
            if (_run->stop_token_ids.contains(token_id)) {
                _notify_stop();
                return;
            }
        }
    }

    // Check if cache is full.
    if (token_idx == _run->max_num_tokens) {
        _notify_cache_full();
    }
}


LogLikelihoodResult LanguageModel::run_model_for_loglikelihood(
    std::span<const uint32_t> input_token_ids,
    size_t continuation_start,
    std::span<const uint32_t> continuation_token_ids,
    bool use_group_prefill
) {
    if (!_cfg.pipeline_cfg.return_logits) {
        throw std::runtime_error(
            "model not compiled with --return_logits; "
            "accuracy/loglikelihood tasks are unsupported"
        );
    }
    if (input_token_ids.empty()) {
        throw std::runtime_error("Loglikelihood request must include at least one input token");
    }
    if (input_token_ids.size() > _run->max_num_tokens) {
        throw std::runtime_error(
            fmt::format(
                "Loglikelihood input length {} exceeds max_num_tokens {}",
                input_token_ids.size(),
                _run->max_num_tokens
            )
        );
    }
    if (continuation_token_ids.empty()) {
        throw std::runtime_error("Loglikelihood request must include continuation token ids");
    }
    if (continuation_start > input_token_ids.size()) {
        throw std::runtime_error(
            fmt::format(
                "Continuation start {} exceeds input length {}",
                continuation_start,
                input_token_ids.size()
            )
        );
    }
    if (continuation_start + continuation_token_ids.size() > input_token_ids.size()) {
        throw std::runtime_error(
            fmt::format(
                "Continuation span [{}:{}) exceeds input length {}",
                continuation_start,
                continuation_start + continuation_token_ids.size(),
                input_token_ids.size()
            )
        );
    }

    LanguageExecutor::Run run(*this);
    double total_logprob = 0.0;
    bool is_greedy = true;
    const size_t continuation_end = continuation_start + continuation_token_ids.size();
    size_t token_idx_begin = 0;

    auto score_current_logits = [this](uint32_t target_token_id) -> LogLikelihoodResult {
        MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
        buf_ptr->invalidate_cache(
            0, static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) *
                sizeof(Eigen::bfloat16)
        );

        const auto* logits_ptr = reinterpret_cast<const Eigen::bfloat16*>(
            buf_ptr->get_virtual_addr()
        );
        return score_logits(
            std::span<const Eigen::bfloat16>(logits_ptr, _cfg.lm_cfg.token_cfg.vocab_size),
            target_token_id
        );
    };

    const size_t group_size = _cfg.pipeline_cfg.input_token_group_size;
    const bool should_group_prefill = (
        use_group_prefill && _use_group_token_models && group_size > 1
    );
    if (!should_group_prefill) {
        // Single-token scoring rewrites the model cache from token zero without going
        // through run_model_prefill(), so the cached-token metadata is no longer valid.
        _session->cached_token_ids.clear();
        _prepare_state_checkpoints_for_prefill(0);
    }
    if (should_group_prefill) {
        create_input_buffers(input_token_ids);

        // Include the final context token so the grouped prefill output scores the first
        // continuation token, matching normal generation prefill behavior for short prompts.
        const size_t prefill_token_count = continuation_start + 1;
        std::span<const uint32_t> prefill_input_tokens(
            input_token_ids.data(), prefill_token_count
        );
        auto num_cached_tokens = _set_input_text_embeds(prefill_input_tokens);
        if (num_cached_tokens >= prefill_token_count && prefill_token_count > 0) {
            // Force the last prefill block to run so n1_buffer4 contains logits for the
            // final context token, not stale logits from a previous request.
            num_cached_tokens = static_cast<uint16_t>(prefill_token_count - 1);
        }
        run_model_prefill(prefill_input_tokens, num_cached_tokens, std::nullopt);
        if (_run->cancelled.load(std::memory_order_relaxed)) {
            throw std::runtime_error("Loglikelihood group prefill was interrupted");
        }
        const auto first_token_score = score_current_logits(continuation_token_ids[0]);
        total_logprob += first_token_score.logprob;
        is_greedy = is_greedy && first_token_score.is_greedy;
        token_idx_begin = prefill_token_count;
    }

    for (size_t token_idx = token_idx_begin; token_idx < input_token_ids.size(); ++token_idx) {
        if (_run->cancelled.load(std::memory_order_relaxed))
            throw std::runtime_error("Loglikelihood request was interrupted");
        if (token_idx >= continuation_start && token_idx < continuation_end) {
            const auto token_score = _run_model_once_for_loglikelihood(
                static_cast<uint16_t>(token_idx),
                input_token_ids[token_idx],
                continuation_token_ids[token_idx - continuation_start]
            );
            total_logprob += token_score.logprob;
            is_greedy = is_greedy && token_score.is_greedy;
        } else {
            _run_model_once_for_loglikelihood_logits(
                static_cast<uint16_t>(token_idx), input_token_ids[token_idx]
            );
        }
    }
    run.complete();
    return {total_logprob, is_greedy};
}


void LanguageModel::_run_model_once_for_loglikelihood_logits(
    uint16_t token_idx, uint32_t input_token_id
) {
    auto execution = LanguageExecutor::lock_execution();
    constexpr uint16_t num_tokens = 1;
    const uint16_t next_token_idx = token_idx + 1;
    _logger->info("Processing token no. {}-{}", token_idx, next_token_idx);

    // Loglikelihood only needs the single-token decode path to refresh n1_buffer4.
    // Keep this separate from run_model_once(), which is shared by normal generation.
    MLABuffer* normal_input_buf = &get_buffer(_resources->embedding_offload ? "offload_decode_embeds" : "embeddings");
    const uint32_t normal_input_row = _resources->embedding_offload ? 0 : input_token_id;
    MLABuffer* normal_scale_buf = _cfg.pipeline_cfg.quantize_embeddings
        ? &get_buffer(_resources->embedding_offload ? "offload_decode_scales" : "embedding_scales") : nullptr;
    if (_resources->embedding_offload) _gather_embedding_rows({&input_token_id, 1}, *normal_input_buf, normal_scale_buf);

    if (_uses_per_layer_inputs()) {
        auto inputs = LanguageExecutor::per_layer_inputs(*this, normal_input_buf,
            normal_scale_buf, normal_input_row, 1, input_token_id);
        _workspace->per_layer_model_map.at({num_tokens, 0, 0}).add_to_queue(&inputs);
    }

    for (uint8_t layer = 0; layer < _cfg.lm_cfg.num_hidden_layers; ++layer) {
        LanguageExecutor::Overrides inputs;
        if (layer == 0) {
            inputs.emplace(0, MLABufferSlice{normal_input_buf, {normal_input_row, 0},
                                             {1, _cfg.lm_cfg.hidden_size}});
        }
        const auto output = layer + 1 == _cfg.lm_cfg.num_hidden_layers
            ? LanguageExecutor::LayerOutput::LogitsRow : LanguageExecutor::LayerOutput::AllRows;
        LanguageExecutor::queue_layer(*this, num_tokens, token_idx, layer, normal_scale_buf,
                                      normal_input_row, {.input = &inputs, .post_input = &inputs}, output);
    }

    MLAModelWithBuffer::run_queue();
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
            get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)).swap_storage(
                get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx))
            );
        }
    }

}


LogLikelihoodResult LanguageModel::_run_model_once_for_loglikelihood(
    uint16_t token_idx, uint32_t input_token_id, uint32_t target_token_id
) {
    if (!_cfg.pipeline_cfg.return_logits) {
        throw std::runtime_error(
            "model not compiled with --return_logits; "
            "accuracy/loglikelihood tasks are unsupported"
        );
    }

    _run_model_once_for_loglikelihood_logits(token_idx, input_token_id);

    MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
    buf_ptr->invalidate_cache(
        0, static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) *
            sizeof(Eigen::bfloat16)
    );

    const auto* logits_ptr = reinterpret_cast<const Eigen::bfloat16*>(
        buf_ptr->get_virtual_addr()
    );
    return score_logits(
        std::span<const Eigen::bfloat16>(logits_ptr, _cfg.lm_cfg.token_cfg.vocab_size),
        target_token_id
    );
}


void LanguageModel::_upload_group_future_token_masks(
    uint16_t num_tokens,
    uint16_t token_idx
) {
    if (
        num_tokens != _cfg.pipeline_cfg.input_token_group_size
        || num_tokens == _cfg.lm_cfg.get_single_num_tokens()
    ) {
        return;
    }

    auto upload_group_mask = [&](
        const std::string& name,
        const std::string& layer_type,
        uint16_t cache_token_idx_begin
    ) {
        const uint16_t effective_context = token_idx + num_tokens - cache_token_idx_begin;
        const uint16_t mask_size = _get_cache_mask_size(
            layer_type, effective_context, true
        );
        if (mask_size <= num_tokens) {
    return;
}
        const uint16_t effective_token_idx = token_idx - cache_token_idx_begin;
        const uint16_t aligned_context = std::min<uint16_t>(
            round_up_to(effective_context, mask_size),
            _cfg.pipeline_cfg.max_num_tokens
        );
        std::vector<Eigen::bfloat16> mask(
            num_tokens * aligned_context,
            std::numeric_limits<Eigen::bfloat16>::lowest()
        );
        for (uint16_t row = 0; row < num_tokens; ++row) {
            std::fill_n(
                mask.begin() + row * aligned_context,
                effective_token_idx + row + 1,
                Eigen::bfloat16{0.0f}
            );
        }
        get_buffer(name).upload_raw(mask.data(), 0, mask.size() * sizeof(Eigen::bfloat16));
    };

    upload_group_mask(
        "group_future_token_mask", "full_attention", 0
    );
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        const uint16_t cache_token_idx_begin = _get_cache_token_idx_begin(
            num_tokens, token_idx, "sliding_attention"
        );
        upload_group_mask(
            "group_sliding_future_token_mask", "sliding_attention", cache_token_idx_begin
        );
    }
}


uint32_t LanguageModel::run_model_once(
    uint16_t num_tokens,
    uint16_t token_idx,
    uint16_t num_input_tokens,
    uint32_t token_id,
    std::vector<Eigen::bfloat16>* logits_ptr
) {
    auto execution = LanguageExecutor::lock_execution();
    if (token_idx == 0 && logits_ptr)
        _prepare_state_checkpoints_for_prefill(0);

    uint16_t next_token_idx;
    if (num_tokens > 1) {
        next_token_idx = std::min(num_input_tokens, uint16_t(token_idx + num_tokens));
    } else {
        next_token_idx = token_idx + 1;
    }
    auto use_input_tokens = token_idx < num_input_tokens;
    _logger->info("Processing token no. {}-{}", token_idx, next_token_idx);

    _upload_group_future_token_masks(num_tokens, token_idx);

    MLABuffer* normal_input_buf;
    MLABuffer* normal_scale_buf = nullptr;
    uint32_t normal_input_row;
    uint32_t normal_input_num_tokens;
    if (use_input_tokens) {
        normal_input_buf = &get_buffer("input_embeds");
        if (_cfg.pipeline_cfg.quantize_embeddings)
            normal_scale_buf = &get_buffer("input_embedding_scales");
        if (_resources->embedding_offload) _prepare_offloaded_prompt(num_tokens, token_idx);
        normal_input_row = _resources->embedding_offload && _workspace->offload_chunk_staging ? 0 : token_idx;
        normal_input_num_tokens = num_tokens;
    } else {
        normal_input_buf = &get_buffer(_resources->embedding_offload ? "offload_decode_embeds" : "embeddings");
        if (_cfg.pipeline_cfg.quantize_embeddings)
            normal_scale_buf = &get_buffer(_resources->embedding_offload ? "offload_decode_scales" : "embedding_scales");
        normal_input_row = _resources->embedding_offload ? 0 : token_id;
        normal_input_num_tokens = 1;
        if (_resources->embedding_offload) _gather_embedding_rows({&token_id, 1}, *normal_input_buf, normal_scale_buf);
    }

    if (_uses_per_layer_inputs()) {
        auto inputs = LanguageExecutor::per_layer_inputs(*this, normal_input_buf,
            normal_scale_buf, normal_input_row, normal_input_num_tokens,
            use_input_tokens ? std::nullopt : std::optional<uint32_t>(token_id));
        _workspace->per_layer_model_map.at({num_tokens, 0, 0}).add_to_queue(&inputs);
    }

    if (num_tokens > 1 && _has_linear_attention_layers()) {
        const uint16_t valid_tokens = next_token_idx - token_idx;
        std::vector<Eigen::bfloat16> valid_mask(num_tokens, Eigen::bfloat16(0.0f));
        std::fill_n(valid_mask.begin(), valid_tokens, Eigen::bfloat16(1.0f));
        get_buffer("linear_valid_mask").upload(valid_mask.data());
    }

    const bool direct_dflash_capture = (
        _cfg.lm_cfg.is_dflash()
        && !_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft
        && num_tokens == _cfg.pipeline_cfg.input_token_group_size
    );
    MLABuffer* dflash_layer_input = nullptr;
    const auto final_output = num_input_tokens > next_token_idx
        ? LanguageExecutor::LayerOutput::CacheOnly : LanguageExecutor::LayerOutput::LogitsRow;
    const uint16_t last_row = next_token_idx - token_idx - 1;
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        const bool last_layer = layer_idx + 1 == _cfg.lm_cfg.num_hidden_layers;
        LanguageExecutor::Overrides inputs;
        if (layer_idx == 0) {
            inputs.emplace(0, MLABufferSlice{normal_input_buf, {normal_input_row, 0},
                                             {normal_input_num_tokens, _cfg.lm_cfg.hidden_size}});
        } else if (dflash_layer_input) {
            inputs.emplace(0, MLABufferSlice{dflash_layer_input, {0, 0},
                                             {num_tokens, _cfg.lm_cfg.hidden_size}});
        }

        MLABuffer* dflash_capture_buffer = direct_dflash_capture && !last_layer
            ? _get_dflash_hidden_buffer(num_tokens, layer_idx) : nullptr;
        LanguageExecutor::Overrides outputs;
        if (dflash_capture_buffer) {
            outputs.emplace(0, MLABufferSlice{dflash_capture_buffer, {0, 0},
                                              {num_tokens, _cfg.lm_cfg.hidden_size}});
        }
        LanguageExecutor::queue_layer(*this, num_tokens, token_idx, layer_idx, normal_scale_buf,
            normal_input_row, {.input = &inputs, .post_input = &inputs, .output = &outputs},
            last_layer ? final_output : LanguageExecutor::LayerOutput::AllRows, last_row);
        dflash_layer_input = dflash_capture_buffer;

        if (
            _cfg.lm_cfg.is_spec_decode()
            && !_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft
            && _cfg.lm_cfg.speculative_decoding_cfg->is_eagle3()
            && layer_idx < _cfg.lm_cfg.num_hidden_layers - 1
        ) {
            const uint8_t num_layers = _cfg.lm_cfg.num_hidden_layers;
            const std::vector<uint8_t> capture_layers{
                2,
                static_cast<uint8_t>(num_layers / 2),
                static_cast<uint8_t>(num_layers - 3),
            };
            const auto it = std::find(
                capture_layers.begin(), capture_layers.end(), layer_idx
            );
            if (it != capture_layers.end()) {
                MLAModelWithBuffer::run_queue();
                const size_t capture_idx = std::distance(capture_layers.begin(), it);
                if (_workspace->eagle3_intermediate_hidden_states.size() < capture_layers.size()) {
                    _workspace->eagle3_intermediate_hidden_states.resize(capture_layers.size());
                }
                auto& buffer = get_buffer(fmt::format("n{}_buffer1", num_tokens));
                auto& capture = _workspace->eagle3_intermediate_hidden_states[capture_idx];
                capture.resize(static_cast<size_t>(num_tokens) * _cfg.lm_cfg.hidden_size);
                buffer.download(capture.data());
            }
        }
    }

    // Run all the queued models.
    MLAModelWithBuffer::run_queue();
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
            get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)).swap_storage(
                get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx))
            );
        }
    }
    // If this run landed exactly on a checkpoint boundary, save the tail.
    const bool save_checkpoint = (
        _workspace->capture_state_checkpoints
        && std::binary_search(
            _session->checkpoint_boundaries.begin(), _session->checkpoint_boundaries.end(), next_token_idx
        )
    );
    if (save_checkpoint) {
        // A partial prefill group can land exactly on a checkpoint boundary.
        _save_state_checkpoint(
            next_token_idx, num_tokens, next_token_idx - token_idx,
            token_idx < num_input_tokens
        );
    }

    if (logits_ptr) {
        assert(num_tokens == 1 && _cfg.pipeline_cfg.return_logits);
        MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
        buf_ptr->invalidate_cache(
            0, static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) *
                sizeof(Eigen::bfloat16)
        );

        // Append the logits.
        auto* ptr = reinterpret_cast<Eigen::bfloat16*>(buf_ptr->get_virtual_addr());
        logits_ptr->insert(logits_ptr->end(), ptr, ptr + _cfg.lm_cfg.token_cfg.vocab_size);

        // Return dummy token id.
        return 0;
    }

    // Find the next token id.
    uint32_t next_token_id;
    if (num_input_tokens <= next_token_idx) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            const auto window = LanguageExecutor::logits_window(_cfg.lm_cfg, num_tokens, last_row);
            const uint32_t row = last_row - window.first;
            const uint16_t logits_num_tokens = window.width;
            const uint32_t vocab_size = _cfg.lm_cfg.get_lm_head_output_size();

            if (_cfg.lm_cfg.lm_head_num_splits == 1) {
                MLABuffer* buf_ptr = &get_buffer(
                    fmt::format("n{}_buffer4", logits_num_tokens)
                );
                const size_t row_bytes =
                    static_cast<size_t>(vocab_size) * sizeof(Eigen::bfloat16);
                const size_t row_offset = static_cast<size_t>(row) * row_bytes;
                buf_ptr->invalidate_cache(row_offset, row_bytes);
                auto* ptr = reinterpret_cast<Eigen::bfloat16*>(
                    static_cast<uint8_t*>(buf_ptr->get_virtual_addr()) + row_offset
                );
                uint32_t best_idx = 0;
                Eigen::bfloat16 best_val = ptr[0];
                for (uint32_t i = 1; i < vocab_size; ++i) {
                    if (ptr[i] > best_val) {
                        best_val = ptr[i];
                        best_idx = i;
                    }
                }
                next_token_id = best_idx;
            } else {
                // Splits: argmax across all split buffers at this row.
                const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
                uint32_t best_idx = 0;
                Eigen::bfloat16 best_val = std::numeric_limits<Eigen::bfloat16>::lowest();
                uint32_t i = 0;
                for (uint32_t split_begin = 0; split_begin < vocab_size; split_begin += split_dim, ++i) {
                    uint32_t split_size = std::min(vocab_size, split_begin + split_dim) - split_begin;
                    MLABuffer* buf_ptr = &get_buffer(
                        fmt::format("n{}_lm_split{}", logits_num_tokens, i)
                    );
                    const size_t row_bytes =
                        static_cast<size_t>(split_size) * sizeof(Eigen::bfloat16);
                    const size_t row_offset = static_cast<size_t>(row) * row_bytes;
                    buf_ptr->invalidate_cache(row_offset, row_bytes);
                    auto* ptr = reinterpret_cast<Eigen::bfloat16*>(
                        static_cast<uint8_t*>(buf_ptr->get_virtual_addr()) + row_offset
                    );
                    for (uint32_t j = 0; j < split_size; ++j) {
                        if (ptr[j] > best_val) {
                            best_val = ptr[j];
                            best_idx = split_begin + j;
                        }
                    }
                }
                next_token_id = best_idx;
            }
            if (_cfg.lm_cfg.is_gemma4_mtp_target()) {
                auto hidden_rows = _read_gemma4_mtp_target_hidden_rows(
                    logits_num_tokens, static_cast<uint16_t>(row + 1)
                );
                const size_t hidden_size = _cfg.lm_cfg.hidden_size;
                const auto row_begin = hidden_rows.begin()
                    + static_cast<std::ptrdiff_t>(row * hidden_size);
                _workspace->gemma4_mtp_prefill_hidden_state.emplace(
                    row_begin,
                    row_begin + static_cast<std::ptrdiff_t>(hidden_size)
                );
            }
        } else {
            // Non-spec: read from single n1_buffer4
            MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
            const size_t result_bytes = _need_argmax
                ? static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) *
                      sizeof(Eigen::bfloat16)
                : sizeof(uint32_t);
            buf_ptr->invalidate_cache(0, result_bytes);
            if (_need_argmax) {
                next_token_id = _calc_next_token_id(buf_ptr);
            } else {
                auto* ptr = reinterpret_cast<uint32_t*>(buf_ptr->get_virtual_addr());
                next_token_id = ptr[0];
            }
        }
    } else {
        next_token_id = 0;
    }
    return next_token_id;
}


void LanguageModel::_finalize() {
    if (!_workspace) return;
    _logger->info("Language model finalize starting ...");
    MLAModelWithBuffer::run_queue();
    // Destroy bindings before their session/workspace buffers and shared resources.
    _workspace.reset();
    _sessions.reset();
    _session = nullptr;
    _resources.reset();
    _logger->info("Language model finalize completed");
}





uint16_t LanguageModel::_get_cache_token_idx_begin(
    uint16_t num_tokens, uint16_t token_idx, const std::string& layer_type
) const {
    if (layer_type != "sliding_attention") {
        return 0;
    }
    // Retain the first query's full window for prefill and verification.
    return std::max(
        0, token_idx + 1 - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
    );
}


LanguageModelMapKey LanguageModel::_get_cache_model_key(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) const {
    const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const uint16_t cache_token_idx_begin = _get_cache_token_idx_begin(
        num_tokens, token_idx, layer_type
    );

    const uint16_t eff_token_idx = token_idx - cache_token_idx_begin;
    const uint16_t eff_num_cached_tokens = token_idx + num_tokens - cache_token_idx_begin;
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    const bool is_single_model = num_tokens == single_num_tokens;
    const uint16_t sliding_window = _cfg.lm_cfg.attn_cfg.sliding_window.value_or(0);
    const bool separate_sliding_cache = (
        layer_type == "sliding_attention"
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.has_value()
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.value() != _cfg.lm_cfg.attn_cfg.head_dim
    );
    // Bucket differences matter only when the sliding window truncates the context.
    const bool sliding_cache_mask_differs = (
        layer_type == "sliding_attention"
        && !separate_sliding_cache
        && sliding_window < _cfg.pipeline_cfg.max_num_tokens
        && (
            _get_cache_mask_size("full_attention", sliding_window, true)
                != _get_cache_mask_size("sliding_attention", sliding_window, true)
            || _get_cache_mask_size("full_attention", sliding_window, false)
                != _get_cache_mask_size("sliding_attention", sliding_window, false)
        )
    );
    const bool use_sliding_cache = (
        separate_sliding_cache
        || (layer_type == "sliding_attention"
            && (!is_single_model || _cfg.lm_cfg.is_spec_decode())
            && token_idx + num_tokens > sliding_window)
        || (is_single_model && sliding_cache_mask_differs
            && eff_num_cached_tokens >= sliding_window)
    );
    const std::string cache_layer_type = (
        use_sliding_cache ? "sliding_attention" : "full_attention"
    );
    const uint16_t cache_mask_size = _get_cache_mask_size(
        cache_layer_type, eff_num_cached_tokens, !is_single_model
    );
    const bool use_single_future_token_mask = is_single_model && cache_mask_size > 1;
    const bool use_group_future_token_mask = (
        !is_single_model && cache_mask_size > num_tokens
    );

    uint16_t aligned_eff_token_idx;
    if (use_single_future_token_mask) {
        aligned_eff_token_idx = std::min(
            round_up_to(eff_num_cached_tokens, cache_mask_size) - 1,
            _cfg.pipeline_cfg.max_num_tokens - 1
        );
    } else if (use_group_future_token_mask) {
        const uint16_t aligned_eff_num_cached_tokens = std::min<uint16_t>(
            round_up_to(eff_num_cached_tokens, cache_mask_size),
            _cfg.pipeline_cfg.max_num_tokens
        );
        aligned_eff_token_idx = aligned_eff_num_cached_tokens - num_tokens;
    } else {
        aligned_eff_token_idx = eff_token_idx;
    }
    if (use_sliding_cache && is_single_model && _cfg.lm_cfg.is_spec_decode()) {
        aligned_eff_token_idx = std::min<uint16_t>(
            aligned_eff_token_idx, sliding_window + num_tokens - 2
        );
    }

    // The second key field is a cache executable kind, not a transformer layer.
    return LanguageModelMapKey{
        num_tokens,
        static_cast<uint8_t>(use_sliding_cache),
        aligned_eff_token_idx
    };
}





void LanguageModel::_init_moe_host_cache() {
    const auto& moe = _cfg.lm_cfg.moe_cfg.value();
    const uint16_t num_experts = moe.num_experts;
    const uint16_t top_k = moe.num_experts_per_tok;

    // Same num_tokens set the MoE buffers were defined for, and the set of moe_nt values
    // _run_moe_post can ask for.
    std::vector<uint16_t> num_tokens_vec{_cfg.lm_cfg.get_single_num_tokens()};
    if (_use_group_token_models) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }

    _moe_host.clear();
    _moe_host.reserve(num_tokens_vec.size());
    for (const auto& nt: num_tokens_vec) {
        MoeHostCache mh;
        mh.num_tokens = nt;
        mh.values = &get_buffer(fmt::format("n{}_router_values", nt));
        mh.indices = &get_buffer(fmt::format("n{}_router_indices", nt));
        mh.weights = &get_buffer(fmt::format("n{}_router_weights", nt));
        const uint16_t num_outputs = nt == 1 ? top_k : num_experts;
        mh.expert_out.reserve(num_outputs);
        for (uint16_t e = 0; e < num_outputs; ++e) {
    mh.expert_out.emplace_back(&get_buffer(fmt::format("n{}_expert{}", nt, e)));
}
        // Only the single-token variant takes the slot-routing path in _run_moe_post.
        if (nt == 1) {
            mh.slot_ofm.resize(top_k);
            for (uint16_t slot = 0; slot < top_k; ++slot) {
                mh.slot_ofm[slot].emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        mh.expert_out[slot],
                        std::vector<uint32_t>{0, 0},
                        std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size}
                    )
                );
            }
        }
        mh.values_scratch.resize(static_cast<size_t>(nt) * top_k);
        mh.indices_scratch.resize(static_cast<size_t>(nt) * top_k);
        mh.weights_scratch.resize(static_cast<size_t>(nt) * num_experts);
        mh.activated_scratch.resize(num_experts);
        _moe_host.emplace_back(std::move(mh));
    }
}


LanguageModel::MoeHostCache& LanguageModel::_get_moe_host(uint16_t moe_nt) {
    // At most two entries (single-token and group), so a scan beats any container.
    for (auto& mh: _moe_host) {
        if (mh.num_tokens == moe_nt) return mh;
    }
    throw std::runtime_error(
        fmt::format("No MoE host cache defined for num_tokens = {}", moe_nt)
    );
}


void LanguageModel::_run_moe_post(
    std::map<uint8_t, MLABufferSlice>* ifm_map,
    uint16_t num_tokens, uint8_t layer_idx
) {
    const auto& moe = _cfg.lm_cfg.moe_cfg.value();
    const uint16_t num_experts = moe.num_experts;
    const uint16_t top_k = moe.num_experts_per_tok;
    const bool is_last = (layer_idx == _cfg.lm_cfg.num_hidden_layers - 1);
    // The last layer processes only the last token (n1), like the non-MoE last post.
    const uint16_t moe_nt = (is_last && !_cfg.lm_cfg.is_spec_decode()) ? 1 : num_tokens;

    const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};

    // Router -> values + indices + residual + norm(h) (TopK+softmax on MLA).
    _workspace->router_model_map.at(model_key).add_to_queue(ifm_map);

    // Flush this layer's pre+cache+router (+ prev layer's experts+ws) so indices are ready.
    MLAModelWithBuffer::run_queue();

    // Host: scatter the k weights into dense router_weights (indices are int32 from the MLA).
    MoeHostCache& mh = _get_moe_host(moe_nt);
    auto& values = mh.values_scratch;
    auto& indices = mh.indices_scratch;
    auto& weights = mh.weights_scratch;
    auto& activated = mh.activated_scratch;
    mh.values->download(values.data());
    mh.indices->download(indices.data());
    std::fill(weights.begin(), weights.end(), Eigen::bfloat16(0.0f));
    std::fill(activated.begin(), activated.end(), uint8_t{0});

    // Scatter each row's selections and record the experts needed by prefill.
    for (uint16_t r = 0; r < moe_nt; ++r) {
        for (uint16_t j = 0; j < top_k; ++j) {
            const size_t sel = static_cast<size_t>(r) * top_k + j;
            const int32_t expert = indices[sel];
            if (expert < 0 || expert >= num_experts) {
                throw std::runtime_error(fmt::format(
                    "MoE router emitted out-of-range expert index {} at selection {} "
                    "(layer {}, num_experts {})",
                    expert, sel, layer_idx, num_experts
                ));
            }
            weights[static_cast<size_t>(r) * num_experts + expert] = values[sel];
            activated[expert] = 1;
        }
    }
    mh.weights->upload(weights.data());

    // Experts. Prefill runs the selected experts (rest zeroed); decode runs the top-k,
    // routing each into the first combine slots.
    if (moe_nt > 1) {
        for (uint16_t e = 0; e < num_experts; ++e) {
            if (activated[e]) {
                _workspace->expert_model_map.at({num_tokens, layer_idx, e}).add_to_queue();
            } else {
                mh.expert_out[e]->clear();
            }
        }
    } else {
        for (uint16_t slot = 0; slot < top_k; ++slot) {
            const uint16_t e = static_cast<uint16_t>(indices[slot]);  // moe_nt==1 row 0
            // Route this expert's output into combine slot `slot` (prebuilt override).
            _workspace->expert_model_map.at({num_tokens, layer_idx, e})
                .add_to_queue(nullptr, &mh.slot_ofm[slot]);
        }
    }

    // Weighted sum -> layer output. No flush: experts+ws ride to the next router flush.
    _workspace->weightedsum_model_map.at(model_key).add_to_queue();
}


void LanguageModel::compact_kv_after_accept(
    std::span<const uint16_t> select_indices, uint16_t prev_input_len
) {
    const size_t n = select_indices.size();
    const uint8_t num_layers = _cfg.lm_cfg.num_hidden_layers;
    const auto& layer_types = _cfg.lm_cfg.layer_types;
    const uint32_t max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
    const uint32_t num_kv_heads = _cfg.lm_cfg.attn_cfg.num_key_value_heads;

    if (static_cast<size_t>(prev_input_len) + n > max_num_tokens) {
        throw std::runtime_error("Accepted EAGLE3 tokens exceed the KV-cache capacity");
    }
    for (const auto index : select_indices) {
        if (index >= max_num_tokens) {
    throw std::runtime_error("EAGLE3 KV selection index exceeds the cache capacity");
}
    }

    if (n == 0) {
        _session->kv_cache_len = prev_input_len;
        return;
    }

    const auto [min_src_it, max_src_it] = std::minmax_element(
        select_indices.begin(), select_indices.end()
    );
    const size_t min_src_pos = *min_src_it;
    const size_t max_src_pos = *max_src_it;

    std::vector<uint8_t> tmp;
    auto compact_buffer = [&](const std::string& name) {
        auto& buf = get_buffer(name);
        const auto& shape = buf.get_shape();
        if (
            shape.size() != 3 || shape[0] != num_kv_heads
            || shape[1] != max_num_tokens
        ) {
            throw std::runtime_error(fmt::format(
                "Unsupported EAGLE3 KV buffer shape for {}", name
            ));
        }
        const size_t row_bytes = buf.get_buf_len(
            std::vector<uint32_t>{1, 1, static_cast<uint32_t>(shape[2])}
        );
        const size_t head_stride_bytes = max_num_tokens * row_bytes;
        tmp.resize(n * row_bytes);

        auto* data = reinterpret_cast<uint8_t*>(buf.get_virtual_addr());
        for (uint32_t head = 0; head < num_kv_heads; ++head) {
            const size_t head_offset = static_cast<size_t>(head) * head_stride_bytes;
            auto* head_base = data + head_offset;

            const size_t source_offset = head_offset + min_src_pos * row_bytes;
            const size_t source_size = (max_src_pos - min_src_pos + 1) * row_bytes;
            buf.invalidate_cache(source_offset, source_size);

            for (size_t row = 0; row < n; ++row) {
                std::memcpy(
                    tmp.data() + row * row_bytes,
                    head_base + static_cast<size_t>(select_indices[row]) * row_bytes,
                    row_bytes
                );
            }
            for (size_t row = 0; row < n; ++row) {
                std::memcpy(
                    head_base + (static_cast<size_t>(prev_input_len) + row) * row_bytes,
                    tmp.data() + row * row_bytes,
                    row_bytes
                );
            }

            const size_t destination_offset =
                head_offset + static_cast<size_t>(prev_input_len) * row_bytes;
            buf.flush_cache(destination_offset, n * row_bytes);
        }
    };

    // Gather accepted K/V rows and their quantization scales into contiguous positions.
    for (uint8_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (
            (layer_types[layer_idx] != "full_attention"
             && layer_types[layer_idx] != "sliding_attention")
            || _cfg.lm_cfg.is_kv_shared_layer(layer_idx)
        ) {
    continue;
}

        for (const std::string& kind : {"cache_key_l", "cache_val_l"}) {
    compact_buffer(fmt::format("{}{}", kind, layer_idx));
}
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            for (const std::string& kind : {"cache_key_scale_l", "cache_val_scale_l"}) {
                compact_buffer(fmt::format("{}{}", kind, layer_idx));
            }
        }
    }

    _session->kv_cache_len = static_cast<uint16_t>(prev_input_len + n);
}


void LanguageModel::invalidate_prompt_reuse() { _session->invalidate_reuse(); }

void LanguageModel::set_image_identity(std::string identity) {
    if (_session->image_identity != identity) {
        _session->invalidate_reuse();
        _session->image_identity = std::move(identity);
    }
}

void LanguageModel::bind_session_adapter() {
    std::lock_guard lock(RuntimeContext::instance().execution_mutex);
    _bind_adapter(_session ? _session->adapter : nullptr);
}

void LanguageModel::_bind_adapter(const std::shared_ptr<LanguageAdapter>& adapter) {
    if (_workspace->bound_adapter == adapter) return;
    using Buffers = std::map<std::string, MLABuffer*>;
    struct Change {
        MLAModelWithBuffer& model;
        Buffers buffers;
        std::shared_ptr<const RuntimeRelocation> next;
        std::shared_ptr<const RuntimeRelocation> previous;
    };
    std::vector<Change> changes;
    const Buffers empty;
    size_t matched = 0;
    for (const auto& [kind, models] : std::initializer_list<std::pair<std::string, LanguageModelMap*>>{
        {"pre", &_workspace->pre_model_map}, {"post", &_workspace->post_model_map},
        {"router", &_workspace->router_model_map}, {"expert", &_workspace->expert_model_map}}) {
        for (auto& [key, model] : *models) {
            const Buffers* desired = &empty;
            if (adapter) {
                const auto found = adapter->bindings.find({kind, key});
                if (found != adapter->bindings.end()) {
                    desired = &found->second;
                    ++matched;
                }
            }
            if (model._reloc_buffers != *desired)
                changes.push_back({model, *desired, {}, model._program->active_reloc});
        }
    }
    if (adapter && matched != adapter->bindings.size())
        throw std::invalid_argument("LoRA names a graph not present in the model");
    for (auto& change : changes)
        change.next = change.model._prepare_reloc(change.buffers, true);

    // All ELF keys and ports are valid. Publish wrapper bindings only after every
    // native update succeeds. A failed driver call leaves existing requests retryable.
    size_t applied = 0;
    try {
        for (auto& change : changes) {
            change.model._program->activate_relocation(change.next);
            ++applied;
        }
    } catch (...) {
        while (applied > 0) {
            auto& change = changes[--applied];
            try {
                change.model._program->activate_relocation(change.previous);
            } catch (const std::exception& error) {
                // The wrapper still owns its original binding and will retry on submission.
                _logger->error("Could not restore MLA relocation: {}", error.what());
            }
        }
        throw;
    }
    for (auto& change : changes) {
        change.model._reloc_buffers.swap(change.buffers);
        change.model._relocation = std::move(change.next);
    }
    _workspace->bound_adapter = adapter;
}

void LanguageModel::set_reloc(const std::string& reloc_name) {
    if (_session->adapter && _session->adapter->name == reloc_name) return;
    if (_files->pulls_files())
        throw std::runtime_error("LoRA requires a local model directory; the asset provider cannot list npy_files");
    std::lock_guard execution_lock(RuntimeContext::instance().execution_mutex);
    std::erase_if(_resources->adapters, [](const auto& entry) { return entry.second.expired(); });
    const auto found = _resources->adapters.find(reloc_name);
    auto adapter = found == _resources->adapters.end() ? nullptr : found->second.lock();
    if (!adapter) {
        adapter = std::make_shared<LanguageAdapter>();
        adapter->name = reloc_name;
        const auto directory = _devkit_dir.parent_path() / "npy_files" / reloc_name;
        const std::regex pattern(R"(_n(\d+)_(pre|post|router)_layer(\d+)(?:_expert(\d+))?_)");
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.path().extension() != ".npy") continue;
            std::smatch match;
            const auto filename = entry.path().filename().string();
            if (!std::regex_search(filename, match, pattern))
                throw std::runtime_error("Invalid file name for relocation: " + filename);
            const auto width = std::stoul(match[1]);
            const auto layer = std::stoul(match[3]);
            const auto expert = match[4].matched ? std::stoul(match[4]) : 0;
            if (width > UINT16_MAX || layer > UINT8_MAX || expert > UINT16_MAX)
                throw std::invalid_argument("Relocation graph index out of range: " + filename);
            if (!_use_group_token_models && width > 1) continue;
            const std::string kind = match[4].matched ? "expert" : match[2].str();
            const auto tensor = cnpy::npy_load(_files->get_path("npy_files/" + reloc_name + "/" + filename));
            if (tensor.word_size != 1 || tensor.fortran_order || tensor.num_vals == 0)
                throw std::invalid_argument("LoRA tensor must contain contiguous row-major bytes: " + filename);
            const auto key = entry.path().stem().string();
            // Compiler-produced arrays already contain the packed MLA section bytes.
            auto [buffer, inserted] = adapter->buffers.try_emplace(
                key, key, std::vector<size_t>{tensor.num_bytes()}, "int8", false);
            buffer->second.allocate();
            buffer->second.upload(tensor.data<void>());
            adapter->bindings[{kind, {width, layer, expert}}].emplace(key, &buffer->second);
        }
        if (adapter->bindings.empty())
            throw std::runtime_error("No LoRA relocation tensors found in " + directory.string());
    }
    auto& cached = _resources->adapters[reloc_name];
    _bind_adapter(adapter);
    cached = adapter;
    _session->adapter = std::move(adapter);
    _session->invalidate_reuse();
    if (_session->draft) _session->draft->invalidate_reuse();
}

void LanguageModel::unset_reloc() {
    if (!_session->adapter) return;
    std::lock_guard execution_lock(RuntimeContext::instance().execution_mutex);
    _bind_adapter(nullptr);
    _session->adapter.reset();
    _session->invalidate_reuse();
    if (_session->draft) _session->draft->invalidate_reuse();
}

uint16_t LanguageModel::set_max_num_tokens(std::optional<uint16_t> max_num_tokens) {
    auto original_max_num_tokens = _run->max_num_tokens;
    if (max_num_tokens.has_value()) {
        _run->max_num_tokens = std::min(max_num_tokens.value(), _cfg.pipeline_cfg.max_num_tokens);
    }
    _logger->info("Setting max_num_tokens: {} -> {}", original_max_num_tokens, _run->max_num_tokens);
    return original_max_num_tokens;
}


std::set<uint32_t> LanguageModel::set_stop_token_ids(
    std::optional<std::set<uint32_t>> stop_token_ids
) {
    auto original_stop_token_ids = _run->stop_token_ids;
    if (stop_token_ids.has_value()) {
        _run->stop_token_ids = stop_token_ids.value();
    }
    _logger->info(
        "Setting stop_token_ids: {} -> {}", original_stop_token_ids, _run->stop_token_ids
    );
    return original_stop_token_ids;
}





void LanguageModel::_prepare_offloaded_prompt(uint16_t num_tokens, uint16_t token_idx) {
    auto& state = *_workspace;
    auto& destination = get_buffer("input_embeds");
    auto* scales = _cfg.pipeline_cfg.quantize_embeddings ? &get_buffer("input_embedding_scales") : nullptr;
    if (state.offload_chunk_staging) {
        destination.clear(false);
        if (scales) scales->clear(false);
    }
    const size_t end = std::min<size_t>(state.offload_prompt_ids.size(), static_cast<size_t>(token_idx) + num_tokens);
    size_t begin = token_idx;
    while (begin < end) {
        if (_image_token_id && state.offload_prompt_ids[begin] == *_image_token_id) {
            ++begin; // Vision has already written this position.
            continue;
        }
        size_t next = begin + 1;
        while (next < end && (!_image_token_id || state.offload_prompt_ids[next] != *_image_token_id)) ++next;
        _gather_embedding_rows(
            std::span<const uint32_t>(state.offload_prompt_ids).subspan(begin, next - begin),
            destination, scales, state.offload_chunk_staging ? begin - token_idx : begin
        );
        begin = next;
    }
    if (state.offload_chunk_staging) {
        destination.flush_cache();
        if (scales) scales->flush_cache();
    }
}




uint16_t LanguageModel::_set_input_text_embeds(std::span<const uint32_t> input_token_ids) {
    // Log.
    _logger->info("Cached token ids: [{}]", fmt::join(_session->cached_token_ids, ", "));

    const uint16_t num_input_tokens = input_token_ids.size();
    if (_resources->embedding_offload) _workspace->offload_prompt_ids.assign(input_token_ids.begin(), input_token_ids.end());
    MLABuffer& buf = get_buffer("input_embeds");
    MLABuffer* scales_buf = _cfg.pipeline_cfg.quantize_embeddings
        ? &get_buffer("input_embedding_scales") : nullptr;
    uint32_t token_idx = 0;
    uint32_t num_images = 0;
    uint16_t num_cached_tokens = 0;
    while (token_idx < num_input_tokens) {
        const auto& token_id = input_token_ids[token_idx];
        if (_image_token_id.has_value() && token_id == _image_token_id.value()) {
            auto next_token_idx = token_idx + _cfg.mm_cfg.value().mm_tokens_per_image;
            ++num_images;
            token_idx = next_token_idx;
        } else {
            auto next_token_idx = token_idx + 1;
            while (
                next_token_idx < num_input_tokens
                && (!_image_token_id
                    || input_token_ids[next_token_idx] != *_image_token_id)
            )
                ++next_token_idx;
            if (!_resources->embedding_offload)
                _gather_embedding_rows(
                    input_token_ids.subspan(token_idx, next_token_idx - token_idx),
                    buf, scales_buf, token_idx
                );
            for (auto idx = token_idx; idx < next_token_idx; ++idx) {
                if (
                    num_images == 0
                    && idx == num_cached_tokens
                    && idx < _session->cached_token_ids.size()
                    && input_token_ids[idx] == _session->cached_token_ids[idx]
                )
                    ++num_cached_tokens;
            }
            token_idx = next_token_idx;
        }
    }
    // Update rope table if needed.
    if (
        _cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "mrope"
        && (_workspace->has_image_token || num_images > 0)
    ) {
        // Previous or current input has image.
        _logger->info("Update rope table");
        auto rope_table = calc_mrope_with_image(
            _cfg, _resources->master_rope_table, _image_token_id.value(), input_token_ids
        );
        get_buffer("global_freq_real").upload(rope_table.re.data());
        get_buffer("global_freq_imag").upload(rope_table.im.data());
    }
    _workspace->has_image_token = num_images > 0;

    _run->text_streamer.push(
        DecodeCallbackType::CACHED_PROMPT_TOKENS, num_cached_tokens, 0.0
    );
    _logger->info("Number of tokens cached: {:d}", num_cached_tokens);
    return num_cached_tokens;
}


std::vector<uint32_t> LanguageModel::_get_per_layer_token_ids(
    std::span<const uint32_t> input_token_ids
) const {
    // Text-only Gemma4 uses per-layer inputs without defining an image token.
    if (!_image_token_id.has_value())
        return std::vector<uint32_t>(input_token_ids.begin(), input_token_ids.end());

    const uint32_t img_id = _image_token_id.value();
    if (std::find(input_token_ids.begin(), input_token_ids.end(), img_id) == input_token_ids.end()) {
        return std::vector<uint32_t>(input_token_ids.begin(), input_token_ids.end());
    }
    std::vector<uint32_t> token_ids(input_token_ids.begin(), input_token_ids.end());
    const uint32_t pad_id = _pad_token_id.value();
    for (auto& token_id : token_ids) {
        if (token_id == img_id)
            token_id = pad_id;
    }
    return token_ids;
}


void LanguageModel::_upload_per_layer_embedding_rows(
    std::span<const uint32_t> token_ids, uint16_t num_tokens,
    uint16_t valid_tokens
) {
    if (!_uses_per_layer_inputs())
        return;
    if (valid_tokens == 0)
        valid_tokens = num_tokens;
    if (valid_tokens > num_tokens || token_ids.size() < valid_tokens)
        throw std::runtime_error("Invalid per-layer embedding staging row count");
    auto& staging = get_buffer(fmt::format("per_layer_emb_staging_n{}", num_tokens));
    const size_t row_size = staging.get_shape().back() * staging.get_elem_size();
    auto* dst = reinterpret_cast<uint8_t*>(staging.get_virtual_addr());
    MLABuffer* scale_staging = nullptr;
    const uint8_t* scale_src_base = nullptr;
    size_t scale_row_size = 0;
    uint8_t* scale_dst = nullptr;
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        scale_staging = &get_buffer(
            fmt::format("per_layer_emb_staging_scale_n{}", num_tokens)
        );
        scale_row_size = scale_staging->get_buf_len(std::vector<uint32_t>{1, 1});
        scale_dst = reinterpret_cast<uint8_t*>(scale_staging->get_virtual_addr());
        scale_src_base = reinterpret_cast<const uint8_t*>(
            get_buffer("per_layer_embedding_scales").get_virtual_addr()
        );
    }
    if (_resources->embedding_offload) {
        _resources->embedding_offload->per_layer->gather(
            token_ids.first(valid_tokens), dst, row_size
        );
    }
    for (uint16_t i = 0; i < valid_tokens; ++i) {
        if (!_resources->embedding_offload) {
            const size_t shard_idx = token_ids[i] / _resources->per_layer_embedding_rows_per_shard;
            const size_t row_in_shard = token_ids[i] % _resources->per_layer_embedding_rows_per_shard;
            const auto* src = reinterpret_cast<const uint8_t*>(
                _resources->per_layer_embedding_shards[shard_idx]->get_virtual_addr()
            ) + row_in_shard * row_size;
            std::memcpy(dst + i * row_size, src, row_size);
        }
        if (scale_staging != nullptr) {
            const auto* scale_src = scale_src_base
                + static_cast<size_t>(token_ids[i]) * scale_row_size;
            std::memcpy(scale_dst + i * scale_row_size, scale_src, scale_row_size);
        }
    }
    staging.flush_cache();
    if (scale_staging != nullptr) {
        scale_staging->flush_cache();
    }
}


void LanguageModel::_compute_and_upload_per_layer_inputs_prefill(
    uint16_t num_tokens, uint16_t token_idx, uint16_t num_input_tokens
) {
    std::vector<uint32_t> token_ids(num_tokens, 0);
    const auto num_prompt_tokens = std::min<uint16_t>(num_tokens, num_input_tokens - token_idx);
    for (uint16_t i = 0; i < num_prompt_tokens; ++i) {
        token_ids[i] = _workspace->prompt_per_layer_token_ids[token_idx + i];
    }
    _upload_per_layer_embedding_rows(token_ids, num_tokens);
}










uint32_t LanguageModel::_calc_next_token_id(MLABuffer* buf_ptr) {
    const auto* values = reinterpret_cast<const Eigen::bfloat16*>(
        buf_ptr->get_virtual_addr()
    );
    const uint32_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
    float best_value = static_cast<float>(values[0]);
    uint32_t best_index = 0;
    #pragma omp parallel for num_threads(4)
    for (uint32_t part = 0; part < 4; ++part) {
        const uint32_t begin = static_cast<uint64_t>(vocab_size) * part / 4;
        const uint32_t end = static_cast<uint64_t>(vocab_size) * (part + 1) / 4;
        const auto [local_value, local_offset] = argmax_bf16(
            values + begin, end - begin
        );
        const uint32_t local_index = begin + local_offset;
        #pragma omp critical
        {
            if (local_value > best_value
                || (local_value == best_value && local_index < best_index)) {
                best_value = local_value;
                best_index = local_index;
            }
        }
    }
    return best_index;
}


void LanguageModel::_notify_first_token(uint32_t token_id, double duration) {
    _logger->info("Time to the first token: {:d} in {:.5f}s", token_id, duration);
    _run->text_streamer.push(DecodeCallbackType::TTFT, token_id, duration);
}


void LanguageModel::_notify_new_token(
    uint32_t token_id, double duration, bool from_draft
) {
    _logger->info("Got token: {:d} in {:.5f}s", token_id, duration);
    _run->text_streamer.push(DecodeCallbackType::TPS, token_id, duration, from_draft);
}


void LanguageModel::_notify_cache_full() const {
    // Use negative token id to indicate that the cache is full.
    _logger->info("Cache full");
    // _py_decode_callback_func(-1, 0);
    _run->text_streamer.push(DecodeCallbackType::CACHE_FULL, 0, 0);
}


void LanguageModel::_notify_stop() const {
    _logger->info("Got stop token");
    _run->text_streamer.push(DecodeCallbackType::STOP, 0, 0);
}


void LanguageModel::_notify_interrupt() const {
    _logger->info("Interrupted by user");
    _run->text_streamer.push(DecodeCallbackType::STOP, 0, 0);
}


}
}
