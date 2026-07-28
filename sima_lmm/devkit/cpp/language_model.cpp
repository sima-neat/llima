#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>
#include <string_view>

#include <Eigen/Dense>
#include <cnpy.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>
#include <simaai/neat/mla/MlaKernelBackend.h>

#include "eagle_helpers.hpp"
#include "language_model.hpp"
#include "mla_execution_plan.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

namespace {
constexpr size_t PER_LAYER_EMBEDDING_MAX_SHARD_SIZE = 1024ULL * 1024 * 1024;

enum class OrdinaryDecodeArm {
    kStagingPlan,
    kStagingDynamic,
    kZeroCopyDynamic,
};

OrdinaryDecodeArm ordinary_decode_arm() {
#if defined(SIMA_LLIMA_ENABLE_DVT_QUEUE_DEPTH_OVERRIDE)
    /*
     * Qualification-only attribution seam for the frozen P1 three-arm A/B.
     * It is compiled out of the product build and does not become a LLiMa,
     * Neat, Backend, or kernel tuning API.
     */
    static const OrdinaryDecodeArm arm = [] {
        const char* value =
            std::getenv("SIMA_LLIMA_DVT_ORDINARY_DECODE_ARM");
        if (!value || std::strcmp(value, "staging_plan") == 0) {
            return OrdinaryDecodeArm::kStagingPlan;
        }
        if (std::strcmp(value, "staging_dynamic") == 0) {
            return OrdinaryDecodeArm::kStagingDynamic;
        }
        if (std::strcmp(value, "zero_copy_dynamic") == 0) {
            return OrdinaryDecodeArm::kZeroCopyDynamic;
        }
        throw std::runtime_error(
            "SIMA_LLIMA_DVT_ORDINARY_DECODE_ARM must be "
            "staging_plan, staging_dynamic, or zero_copy_dynamic"
        );
    }();
    return arm;
#else
    return OrdinaryDecodeArm::kStagingPlan;
#endif
}
}

LanguageModel::LanguageModel(
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer
) : LanguageModel(
        current_mla_execution_session(), std::move(model_path),
        std::move(stop_token_ids), image_token_id, pad_token_id, text_streamer
    ) {}

LanguageModel::LanguageModel(
    std::shared_ptr<MlaExecutionSession> session,
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer
) : BaseModel(model_path, std::move(session)),
    _stop_token_ids(std::move(stop_token_ids)),
    _image_token_id(image_token_id),
    _pad_token_id(pad_token_id),
    _max_num_tokens(_cfg.pipeline_cfg.max_num_tokens),
    _text_streamer(text_streamer),
    _has_image_token(false),
    _is_running(false),
    _reloc_name(std::nullopt)
{
    _use_group_token_models = (
        _cfg.pipeline_cfg.input_token_group_offsets.has_value()
        && _cfg.pipeline_cfg.input_token_group_offsets.value().size() > 0
    );
    _need_argmax = _cfg.lm_cfg.lm_head_num_splits > 1 || _cfg.pipeline_cfg.return_logits;

    if (_uses_cpu_dequantized_embeddings() && !_cfg.pipeline_cfg.embeddings_scale.has_value()) {
        throw std::runtime_error(
            "Quantized multimodal embeddings require pipeline_cfg.embeddings_scale"
        );
    }
    
    // Backwards compatible for models without rope_dimension_count in config
    if (_cfg.lm_cfg.rope_cfg.rope_dimension_count == 0)
        _cfg.lm_cfg.rope_cfg.rope_dimension_count = _cfg.lm_cfg.attn_cfg.head_dim;

    if (_cfg.lm_cfg.layer_types.size() == 0) {
        if (_cfg.lm_cfg.attn_cfg.swa_enable) {
            for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
                if ((layer_idx + 1) % (_cfg.lm_cfg.attn_cfg.swa_ratio + 1)) {
                    _cfg.lm_cfg.layer_types.emplace_back("sliding_attention");
                } else {
                    _cfg.lm_cfg.layer_types.emplace_back("full_attention");
                }
            }
        } else {
            _cfg.lm_cfg.layer_types.resize(_cfg.lm_cfg.num_hidden_layers, "full_attention");
        }
    }

    if (_use_group_token_models) {
        // Collect indices for all stateful layer families.
        std::vector<uint8_t> conv_layer_indices;
        for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
                conv_layer_indices.emplace_back(layer_idx);
            }
        }
        // Future: collect mamba_layer_indices, deltanet_layer_indices, etc.

        // Build _checkpoint_boundaries once if ANY stateful layer family exists.
        // Future: extend the condition with `|| !mamba_layer_indices.empty() || ...`.
        if (!conv_layer_indices.empty()) {
            const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
            const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
            // Checkpoint at each grouped start offset stores the state tail before that block.
            for (const auto& offset: offsets) {
                if (offset) {
                    _checkpoint_boundaries.emplace_back(offset);
                }
            }
            if (!offsets.empty()) {
                // Extra boundary after last grouped block supports reuse beyond grouped offsets.
                uint32_t extra_boundary = offsets.back() + group_size;
                if (extra_boundary < _cfg.pipeline_cfg.max_num_tokens) {
                    _checkpoint_boundaries.emplace_back(extra_boundary);
                }
            }
        }

        //Per-family CachedState init. Each stateful layer family gets its own block.
        if (!conv_layer_indices.empty()) {
            LanguageModel::CachedState conv_state;
            conv_state.buffer_name_prefix = "conv_cache_history_l";
            conv_state.tail_len = std::max<uint16_t>(1, _cfg.lm_cfg.conv_L_cache - 1);
            conv_state.num_elems = _cfg.lm_cfg.hidden_size;
            conv_state.elem_size = sizeof(Eigen::bfloat16);
            conv_state.tail_bytes = static_cast<size_t>(
                conv_state.tail_len * conv_state.num_elems * conv_state.elem_size
            );
            conv_state.layer_indices = std::move(conv_layer_indices);

            const auto num_boundaries = _checkpoint_boundaries.size();
            conv_state.checkpoints.resize(conv_state.layer_indices.size());
            for (auto& checkpoints: conv_state.checkpoints) {
                checkpoints.resize(num_boundaries, std::vector<uint8_t>(conv_state.tail_bytes));
            }
            _cached_states.emplace_back(std::move(conv_state));
        }
        // Future: similar block for mamba/Deltanet families.
    }

    // EAGLE3: build the constant tree_mask_init (eye(topk)) once. Only the
    // draft uses it (consumed in topk_generate's depth loop seed and concat).
    if (_cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft) {
        const int topk = _cfg.lm_cfg.speculative_decoding_cfg.value().speculative_budget;
        _eagle3_tree_mask_init.data = std::vector<std::vector<std::vector<std::vector<float>>>>(
            1, std::vector<std::vector<std::vector<float>>>(
                1, std::vector<std::vector<float>>(
                    topk, std::vector<float>(topk, 0.0f)
                )
            )
        );
        for (int i = 0; i < topk; ++i) _eagle3_tree_mask_init.data[0][0][i][i] = 1.0f;

        _eagle3_position_ids.assign(topk, 0);
    }

    _initialize();
}

LanguageModel::~LanguageModel() {
    _finalize();
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
    const auto& embeddings_buf = get_buffer("embeddings");
    define_buffer(
        "input_embeds",
        {num_padded_input_tokens, _cfg.lm_cfg.hidden_size},
        (_cfg.vm_cfg.has_value() && _cfg.mm_cfg.has_value())
            ? "bfloat16" : embeddings_buf.get_dtype()
    );
    MLABuffer& input_embeds_buf = get_buffer("input_embeds");
    input_embeds_buf.allocate();
    input_embeds_buf.clear();

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
            for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
                ofm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(i + 1),
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
    std::optional<std::set<uint32_t>> override_stop_token_ids
) {
    require_healthy_mla_session();
    // Update the state.
    _is_running = true;

    auto original_max_num_tokens = set_max_num_tokens(override_max_num_tokens);
    auto original_stop_token_ids = set_stop_token_ids(override_stop_token_ids);

    std::optional<std::vector<uint32_t>> output_token_ids{};

    // If the input is already greater than the cache size, no need to run the model.
    if (input_token_ids.size() > _max_num_tokens) {
        _notify_cache_full();
        output_token_ids = std::vector<uint32_t>();
    } else {
        // Create input embeds from input token ids and image embeds.
        auto num_cached_tokens = _set_input_text_embeds(input_token_ids);

        // Prefill.
        auto token_id = run_model_prefill(input_token_ids, num_cached_tokens, timer_ttft);
        auto output_token_id_begin = _cached_token_ids.size() - 1;
        if (_stop_token_ids.contains(token_id)) {
            _notify_stop();
            output_token_ids = std::vector<uint32_t>{token_id};
        } else if (!_is_running.load(std::memory_order_relaxed)) {
            // Do nothing.
        } else {
            // Decode.
            run_model_decode(input_token_ids.size(), token_id);
            output_token_ids = std::vector<uint32_t>(
                _cached_token_ids.begin() + output_token_id_begin, _cached_token_ids.end()
            );
        }
    }

    // Wait until all the streaming finishes.
    _text_streamer.wait_streaming();

    // Restore the original values.
    set_max_num_tokens(original_max_num_tokens);
    set_stop_token_ids(original_stop_token_ids);

    if (_is_running) {
        _is_running = false;
        return output_token_ids;
    } else {
        return std::nullopt;
    }
}


uint32_t LanguageModel::run_model_prefill(
    std::span<const uint32_t> input_token_ids,
    uint16_t num_cached_tokens,
    std::optional<ChronoTimer> timer_ttft
) {
    require_healthy_mla_session();
    if (_uses_per_layer_inputs())
        _prompt_per_layer_token_ids = _get_per_layer_token_ids(input_token_ids);
    if (!timer_ttft.has_value())
        timer_ttft = ChronoTimer{true};
    const uint16_t num_input_tokens = input_token_ids.size();
    const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
    uint16_t token_idx{};
    uint32_t next_token_id{};
    uint16_t last_group_valid_tokens = 0;
    if (!_cached_states.empty()) {
        num_cached_tokens = _prepare_state_checkpoints_for_prefill(num_cached_tokens);
    }

    if (num_input_tokens == num_cached_tokens) {
        // All input tokens are already cached.
        if (num_cached_tokens < _cached_token_ids.size())
            next_token_id = _cached_token_ids[num_cached_tokens];
        else
            next_token_id = _cached_first_generated_token;
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
            if (!_is_running.load(std::memory_order_relaxed)) {
                _notify_interrupt();
                _cached_token_ids.assign(
                    input_token_ids.begin(), input_token_ids.begin() + token_idx
                );
                return next_token_id;
            }
        }
        if (
            !_cached_states.empty() 
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
            if (!_is_running.load(std::memory_order_relaxed)) {
                _notify_interrupt();
                _cached_token_ids.assign(
                    input_token_ids.begin(), input_token_ids.begin() + token_idx + 1
                );
                return next_token_id;
            }
        }
    }
    _cached_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
    auto duration = timer_ttft.value().stop();
    _notify_first_token(next_token_id, duration);
    _cached_first_generated_token = next_token_id;
    return next_token_id;
}


void LanguageModel::run_model_decode(
    uint16_t num_input_tokens, uint32_t token_id
) {
    require_healthy_mla_session();
    ChronoTimer timer_tps(true);
    uint16_t token_idx;
    for (token_idx = num_input_tokens; token_idx < _max_num_tokens; ++token_idx ) {
        auto next_token_id = run_model_once(1, token_idx, num_input_tokens, token_id);
        _cached_token_ids.emplace_back(token_id);
        token_id = next_token_id;

        auto duration = timer_tps.stop(true);
        _notify_new_token(next_token_id, duration);
        if (_stop_token_ids.contains(token_id)) {
            _notify_stop();
            return;
        }
        if (!_is_running.load(std::memory_order_relaxed)) {
            _notify_interrupt();
            return;
        }
    }

    // Check if cache is full.
    if (token_idx == _max_num_tokens) {
        _notify_cache_full();
    }
}


uint32_t LanguageModel::run_model_once(
    uint16_t num_tokens,
    uint16_t token_idx,
    uint16_t num_input_tokens,
    uint32_t token_id,
    std::vector<Eigen::bfloat16>* logits_ptr
) {
    require_healthy_mla_session();
    /*
     * All compiled layers contributing to this state transition share one
     * explicit transaction. Mid-function commits are intentional CPU
     * observation boundaries; the segment retains the session lock across
     * them so another producer cannot interleave dependent KV state.
     */
    MlaExecutionSegment segment(_mla_session);
    uint16_t next_token_idx;
    if (num_tokens > 1) {
        next_token_idx = std::min(num_input_tokens, uint16_t(token_idx + num_tokens));
    } else {
        next_token_idx = token_idx + 1;
    }
    auto use_input_tokens = token_idx < num_input_tokens;
    const bool qualified_ordinary_decode =
        num_tokens == 1 &&
        !use_input_tokens &&
        logits_ptr == nullptr &&
        !_need_argmax &&
        _has_ordinary_decode_recipe(token_idx);
    const OrdinaryDecodeArm decode_arm = ordinary_decode_arm();
    const bool stage_ordinary_decode =
        qualified_ordinary_decode &&
        decode_arm != OrdinaryDecodeArm::kZeroCopyDynamic;
    _logger->info("Processing token no. {}-{}", token_idx, next_token_idx);

    MLABuffer* normal_input_buf;
    uint32_t normal_input_row;
    uint32_t normal_input_num_tokens;
    if (use_input_tokens) {
        normal_input_buf = &get_buffer("input_embeds");
        normal_input_row = token_idx;
        normal_input_num_tokens = num_tokens;
    } else if (stage_ordinary_decode) {
        normal_input_buf = &get_buffer("decode_embedding");
        normal_input_row = 0;
        normal_input_num_tokens = 1;
        _stage_ordinary_decode_embedding(token_id, *normal_input_buf);
    } else if (_uses_cpu_dequantized_embeddings()) {
        /*
         * Unsupported multimodal/per-layer shapes retain develop's generic
         * staging path. They are excluded from the prebound arena, so changing
         * their ownership behavior is neither necessary nor part of P1.
         */
        normal_input_buf = &get_buffer("decode_embedding");
        normal_input_row = 0;
        normal_input_num_tokens = 1;
        _dequantize_embedding_row(token_id, *normal_input_buf);
        normal_input_buf->flush_cache();
    } else {
        normal_input_buf = &get_buffer("embeddings");
        normal_input_row = token_id;
        normal_input_num_tokens = 1;
    }

    // Run the standalone per-layer projection model before the transformer stack (Gemma4 only).
    // IFM0 uses the staging buffer for prefill; decode overrides it with one shard row.
    // IFM1 is the normal embedding input.
    if (_uses_per_layer_inputs()) {
        LanguageModelMapKey per_layer_key{num_tokens, 0, 0};
        std::map<uint8_t, MLABufferSlice> per_layer_ifm_map;
        if (!use_input_tokens) {
            // Decode has one generated token, so use its contiguous shard row directly and
            // avoid copying it through the per-layer staging buffer.
            const uint32_t per_layer_token_id = (
                _image_token_id.has_value() && token_id == _image_token_id.value()
            ) ? _pad_token_id.value() : token_id;
            const size_t shard_idx = (
                per_layer_token_id / _per_layer_embedding_rows_per_shard
            );
            const size_t row_in_shard = (
                per_layer_token_id % _per_layer_embedding_rows_per_shard
            );
            auto* shard = _per_layer_embedding_shards[shard_idx];
            per_layer_ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    shard,
                    std::vector<uint32_t>{static_cast<uint32_t>(row_in_shard), 0},
                    std::vector<uint32_t>{1, static_cast<uint32_t>(shard->get_shape().back())}
                )
            );
        }
        per_layer_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(1),
            std::forward_as_tuple(
                normal_input_buf,
                std::vector<uint32_t>{normal_input_row, 0},
                std::vector<uint32_t>{normal_input_num_tokens, _cfg.lm_cfg.hidden_size}
            )
        );
        _per_layer_model_map.at(per_layer_key).add_to_segment(segment, &per_layer_ifm_map);
    }

    /*
     * Ordinary decode has a construction-time, position-specific recipe.
     * Unlike a global handle list, token_idx selects the exact KV/RoPE/mask
     * cells created for this physical position.  All other execution shapes
     * retain develop's generic traversal below.
     */
    const bool use_ordinary_recipe =
        num_tokens == 1 &&
        !use_input_tokens &&
        _has_ordinary_decode_recipe(token_idx);
    if (use_ordinary_recipe) {
        if (qualified_ordinary_decode &&
            decode_arm == OrdinaryDecodeArm::kStagingPlan &&
            _ordinary_decode_plan &&
            _ordinary_decode_plan->valid(token_idx)) {
            /*
             * segment owns the execution lease acquired before staging. The
             * second validity check inside commit closes the generation race
             * before the first submit without adding metadata locks.
             */
            segment.commit(*_ordinary_decode_plan, token_idx);
        } else {
            _append_ordinary_decode_recipe(
                segment, token_idx, normal_input_buf, normal_input_row
            );
        }
    } else {
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        LanguageModelMapKey model_key(num_tokens, layer_idx, token_idx);

        std::map<uint8_t, MLABufferSlice> ifm_map;
        if (layer_idx == 0) {
            ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    normal_input_buf,
                    std::vector<uint32_t>{normal_input_row, 0},
                    std::vector<uint32_t>{normal_input_num_tokens, _cfg.lm_cfg.hidden_size}
                )
            );
        }
        if (
            _cfg.lm_cfg.layer_types[layer_idx] == "full_attention"
            || _cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention"
        ) {
            _pre_model_map.at(model_key).add_to_segment(segment, &ifm_map);

            if (
                num_input_tokens > next_token_idx
                && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
            ) {
                break;
            }

            _cache_model_map.at(model_key).add_to_segment(segment);

            const bool is_draft = _cfg.lm_cfg.is_spec_decode()
                && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
            const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
            // Using the single post for the target's final layer is valid only when the
            // draft does not require that layer's hidden states. If it does, use the
            // group post for the final layer so all group hidden states remain available.
            const bool use_single_post_for_target_group = (
                _cfg.lm_cfg.is_spec_decode()
                && !is_draft
                && num_tokens != single_num_tokens
                && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
            );

            // Non-spec uses n1 for the final row. Target speculative prefill
            // reuses n16 and binds a contiguous window containing the final row.
            if (num_tokens > 1 && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
                && !_cfg.lm_cfg.is_spec_decode()) {
                ifm_map.clear();
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        nullptr,
                        std::vector<uint32_t>{
                            static_cast<uint32_t>(num_input_tokens - 1 - token_idx), 0
                        },
                        std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size}
                    )
                );
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(1),
                    std::forward_as_tuple(
                        nullptr,
                        std::vector<uint32_t>{
                            static_cast<uint32_t>(num_input_tokens - 1 - token_idx), 0
                        },
                        std::vector<uint32_t>{
                            1,
                            _cfg.lm_cfg.attn_cfg.get_q_size(_cfg.lm_cfg.layer_types[layer_idx])
                        }
                    )
                );
            }
            if (use_single_post_for_target_group) {
                const uint32_t last_valid_row = num_input_tokens - 1 - token_idx;
                const uint32_t slice_start = last_valid_row >= single_num_tokens - 1
                    ? last_valid_row - (single_num_tokens - 1)
                    : 0;

                ifm_map.clear();
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        nullptr,
                        std::vector<uint32_t>{slice_start, 0},
                        std::vector<uint32_t>{single_num_tokens, _cfg.lm_cfg.hidden_size}
                    )
                );
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(1),
                    std::forward_as_tuple(
                        nullptr,
                        std::vector<uint32_t>{slice_start, 0},
                        std::vector<uint32_t>{
                            single_num_tokens,
                            _cfg.lm_cfg.attn_cfg.get_q_size(_cfg.lm_cfg.layer_types[layer_idx])
                        }
                    )
                );
                uint8_t post_ifm_idx = 2;
                if (_uses_per_layer_inputs()) {
                    const uint32_t layer_row_offset =
                        static_cast<uint32_t>(layer_idx) * num_tokens;
                    ifm_map.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(post_ifm_idx++),
                        std::forward_as_tuple(
                            nullptr,
                            std::vector<uint32_t>{layer_row_offset + slice_start, 0},
                            std::vector<uint32_t>{
                                single_num_tokens,
                                _cfg.lm_cfg.hidden_size_per_layer_input
                            }
                        )
                    );
                }
                if (
                    _cfg.vm_cfg.has_value()
                    && layer_idx < _cfg.vm_cfg.value().deepstack_visual_indexes.size()
                ) {
                    const auto buf_name = fmt::format("deepstack_feature_l{}_cache", layer_idx);
                    ifm_map.emplace(
                        std::piecewise_construct,
                        std::forward_as_tuple(post_ifm_idx),
                        std::forward_as_tuple(
                            nullptr,
                            std::vector<uint32_t>{token_idx + slice_start, 0},
                            std::vector<uint32_t>{
                                single_num_tokens,
                                static_cast<uint32_t>(get_buffer(buf_name).get_shape().back())
                            }
                        )
                    );
                }
            }
            if (_uses_per_layer_inputs() && num_tokens > 1
                && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
                && !_cfg.lm_cfg.is_spec_decode()) {
                uint32_t layer_row_offset = static_cast<uint32_t>(layer_idx) * num_tokens;
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(2),
                    std::forward_as_tuple(
                        nullptr,
                        std::vector<uint32_t>{
                            layer_row_offset + static_cast<uint32_t>(num_input_tokens - 1 - token_idx), 0
                        },
                        std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size_per_layer_input}
                    )
                );
            }
            _post_model_map.at(model_key).add_to_segment(segment, &ifm_map);

            // Spec-decoding capture: download n128_buffer1 (this layer's hidden
            // states) for layers 2, N/2, N-3 so the orchestrator can feed them
            // into FC fusion.
            if (
                _cfg.lm_cfg.is_spec_decode()
                && layer_idx < _cfg.lm_cfg.num_hidden_layers - 1
            ) {
                const uint8_t num_layers = _cfg.lm_cfg.num_hidden_layers;
                const std::vector<uint8_t> capture_layers = {
                    2,
                    static_cast<uint8_t>(num_layers / 2),
                    static_cast<uint8_t>(num_layers - 3),
                };
                auto it = std::find(capture_layers.begin(), capture_layers.end(), layer_idx);
                if (it != capture_layers.end()) {
                    segment.commit();
                    if (_eagle3_intermediate_hidden_states.size() < capture_layers.size()) {
                        _eagle3_intermediate_hidden_states.resize(capture_layers.size());
                    }
                    size_t idx = std::distance(capture_layers.begin(), it);
                    auto& buf = get_buffer(fmt::format("n{}_buffer1", num_tokens));
                    const size_t num_elems = static_cast<size_t>(num_tokens) * _cfg.lm_cfg.hidden_size;
                    _eagle3_intermediate_hidden_states[idx].resize(num_elems);
                    buf.download(_eagle3_intermediate_hidden_states[idx].data());
                }
            }
        } else if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
            LanguageModelMapKey conv_model_key(num_tokens, layer_idx, 0);
            _conv_model_map.at(conv_model_key).add_to_segment(segment, &ifm_map);

            if (
                num_input_tokens > next_token_idx
                || layer_idx != (_cfg.lm_cfg.num_hidden_layers - 1)
            ) {
                continue;
            }

            ifm_map.clear();
            if (num_tokens > 1) {
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        nullptr,
                        std::vector<uint32_t>{
                            static_cast<uint32_t>(num_input_tokens - 1 - token_idx), 0
                        },
                        std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size}
                    )
                );
            }
            _conv_final_model_map.at(conv_model_key).add_to_segment(segment, &ifm_map);
        } else {
            throw std::runtime_error(
                std::string("Unsupported layer type: ") + _cfg.lm_cfg.layer_types[layer_idx]
            );
        }
    }
    }

    /*
     * Commit this dependency segment and consume every authoritative terminal
     * CQE before CPU code reads logits/KV state.  The previous run-all path
     * handed a list to MLA-RT/Dispatcher; this segment instead submits a
     * bounded window directly to the kernel while preserving compiler order.
     * Keeping the boundary explicit prevents queue-ahead from crossing a CPU
     * observation or mutation point.
     */
    segment.commit();

    // If this run landed exactly on a checkpoint boundary, save the tail.
    if (!_cached_states.empty()) {
        for (size_t i = 0; i < _checkpoint_boundaries.size(); ++i) {
            if (_checkpoint_boundaries[i] == next_token_idx) {
                // A partial prefill group can land exactly on a checkpoint boundary.
                const uint16_t valid_tokens = next_token_idx - token_idx;
                _save_state_checkpoint(i, num_tokens, valid_tokens);
                break;
            }
        }
    }

    if (logits_ptr) {
        assert(num_tokens == 1 && _cfg.pipeline_cfg.return_logits);
        MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
        buf_ptr->invalidate_cache();

        // Append the logits.
        Eigen::bfloat16* ptr = reinterpret_cast<Eigen::bfloat16*>(buf_ptr->get_virtual_addr());
        logits_ptr->insert(logits_ptr->end(), ptr, ptr + _cfg.lm_cfg.token_cfg.vocab_size);

        // Return dummy token id.
        return 0;
    }

    // Find the next token id.
    uint32_t next_token_id;
    if (num_input_tokens <= next_token_idx) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            const bool is_draft = _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
            const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
            const uint32_t last_valid_row = num_input_tokens - 1 - token_idx;
            const bool used_single_post_for_target_group = (
                !is_draft && num_tokens != single_num_tokens
            );
            const uint32_t slice_start = (
                used_single_post_for_target_group
                && last_valid_row >= single_num_tokens - 1
            ) ? last_valid_row - (single_num_tokens - 1) : 0;
            const uint32_t row = last_valid_row - slice_start;
            const uint16_t logits_num_tokens = used_single_post_for_target_group
                ? single_num_tokens
                : num_tokens;
            const uint32_t vocab_size = _cfg.lm_cfg.get_lm_head_output_size();

            if (_cfg.lm_cfg.lm_head_num_splits == 1) {
                MLABuffer* buf_ptr = &get_buffer(
                    fmt::format("n{}_buffer4", logits_num_tokens)
                );
                buf_ptr->invalidate_cache();
                auto* ptr = reinterpret_cast<Eigen::bfloat16*>(buf_ptr->get_virtual_addr());
                uint32_t best_idx = 0;
                Eigen::bfloat16 best_val = ptr[row * vocab_size];
                for (uint32_t i = 1; i < vocab_size; ++i) {
                    if (ptr[row * vocab_size + i] > best_val) {
                        best_val = ptr[row * vocab_size + i];
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
                    buf_ptr->invalidate_cache();
                    auto* ptr = reinterpret_cast<Eigen::bfloat16*>(buf_ptr->get_virtual_addr());
                    for (uint32_t j = 0; j < split_size; ++j) {
                        if (ptr[row * split_size + j] > best_val) {
                            best_val = ptr[row * split_size + j];
                            best_idx = split_begin + j;
                        }
                    }
                }
                next_token_id = best_idx;
            }
        } else {
            // Non-spec: read from single n1_buffer4
            MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
            /*
             * Direct MLA n1 buffers use SIMAAI_MEM_FLAG_DEFAULT:
             * dma_alloc_coherent() plus a coherent userspace mapping.  The
             * kernel executes dma_rmb() before terminal CQE publication and
             * Backend consumes that CQE through an acquire load.  Keep the
             * final compiler fence explicit, then use ordinary coherent CPU
             * loads.  Develop's dc civac + dsb sy invalidation was required
             * by older transport/cache assumptions and added avoidable work
             * at every serial token boundary.
             *
             * This is intentionally limited to the proven ordinary n1 token
             * output. Speculative logits, checkpoints, Whisper, CVU, and
             * legacy cached allocations retain their existing maintenance.
             */
            std::atomic_thread_fence(std::memory_order_acquire);
            if (_need_argmax) {
                next_token_id = _calc_next_token_id(buf_ptr);
            } else {
                uint32_t* ptr = (uint32_t*)buf_ptr->get_virtual_addr();
                next_token_id = ptr[0];
            }
        }
    } else {
        next_token_id = 0;
    }
    return next_token_id;
}


void LanguageModel::_initialize() {
    _logger->info("Language model initialize starting ...");
    BaseModel::_initialize();

    // Define and load the models in parallel.
    _define_models();
    _mla_model_anchor().load_related_models(
        _elf_dir / _cfg.language_model_name
    );

    // Upload language embeddings (drafts use the target's embeddings, so skip).
    const bool is_draft = _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
    if (!is_draft) {
        auto embeddings_file_name = _devkit_dir / (_cfg.language_model_name + "_embeddings.bin");
        if (std::filesystem::exists(embeddings_file_name)) {
            get_buffer("embeddings").load_file(embeddings_file_name);
        } else {
            // Compatibility with packages generated before raw embedding files were introduced.
            embeddings_file_name = _devkit_dir / (_cfg.language_model_name + "_embeddings.npy");
            auto embeddings_tensor = cnpy::npy_load(embeddings_file_name);
            get_buffer("embeddings").upload(embeddings_tensor.data<void>());
        }
    } else {
        // Load d2t mapping (int64 in npy, narrows to int32 — values fit easily).
        auto d2t_file_name = _devkit_dir / "d2t.npy";
        auto d2t_tensor = cnpy::npy_load(d2t_file_name);
        const int64_t* src = d2t_tensor.data<int64_t>();
        const size_t n = d2t_tensor.num_vals;
        _d2t.resize(n);
        for (size_t i = 0; i < n; ++i) {
            _d2t[i] = static_cast<int32_t>(src[i]);
        }
        _logger->info("Loaded d2t mapping with {} entries", _d2t.size());
    }

    // Upload freq real and imag.
    auto rope_table = calc_freq_real_imag(
        _cfg.pipeline_cfg.max_num_tokens,
        _cfg.lm_cfg.rope_cfg.rope_scaling.rope_type,
        _cfg.lm_cfg.rope_cfg.rope_theta,
        _cfg.lm_cfg.rope_cfg.get_rope_dimension_count("full_attention"),
        _cfg.lm_cfg.attn_cfg.get_head_dim("full_attention"),
        _cfg.lm_cfg.rope_cfg.rope_scaling
    );
    get_buffer("global_freq_real").upload(rope_table.re.data());
    get_buffer("global_freq_imag").upload(rope_table.im.data());
    // Save a pristine host copy for sourcing tree-RoPE rows without re-reading
    // the (mutated) device buffer.
    _global_freq_host = rope_table;
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        rope_table = calc_freq_real_imag(
            _cfg.pipeline_cfg.max_num_tokens,
            "default",
            _cfg.lm_cfg.rope_cfg.rope_local_base_freq,
            _cfg.lm_cfg.rope_cfg.get_rope_dimension_count("sliding_attention"),
            _cfg.lm_cfg.attn_cfg.get_head_dim("sliding_attention"),
            _cfg.lm_cfg.rope_cfg.rope_scaling
        );
        get_buffer("local_freq_real").upload(rope_table.re.data());
        get_buffer("local_freq_imag").upload(rope_table.im.data());
        // Save a pristine host copy for sourcing tree-RoPE rows in spec verify
        // (mirrors _global_freq_host).
        _local_freq_host = rope_table;
    }
    if (_cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "mrope") {
        assert(!_cfg.lm_cfg.attn_cfg.swa_enable);
        _master_rope_table = std::move(rope_table);
    }

    if (_uses_per_layer_inputs()) {
        _load_per_layer_embeddings();
    }

    // Upload the future token mask.
    if (_cfg.pipeline_cfg.future_token_mask_size > 1) {
        std::vector<Eigen::bfloat16> future_token_mask(
            _cfg.pipeline_cfg.future_token_mask_size - 1,
            std::numeric_limits<Eigen::bfloat16>::lowest()
        );
        MLABuffer& future_token_mask_buf = get_buffer("future_token_mask");
        future_token_mask_buf.clear(false);
        future_token_mask_buf.upload(
            future_token_mask.data(),
            _cfg.pipeline_cfg.max_num_tokens * 2,
            future_token_mask.size() * 2,
            true
        );
    }

    // Clear the KV caches and caches states.
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
            get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)).clear();
        } else if (!_cfg.lm_cfg.is_kv_shared_layer(layer_idx)) {
            get_buffer(fmt::format("cache_key_l{}", layer_idx)).clear();
            get_buffer(fmt::format("cache_val_l{}", layer_idx)).clear();
            if (_cfg.pipeline_cfg.quantize_kv_cache) {
                get_buffer(fmt::format("cache_key_scale_l{}", layer_idx)).clear();
                get_buffer(fmt::format("cache_val_scale_l{}", layer_idx)).clear();
            }
        }
    }

    /*
     * Models, all BaseModel buffers, embedding contents and cache/frequency
     * storage are final at this point. Build the candidate arena last so its
     * parent-cookie witnesses describe the exact published generation. A
     * later request may allocate an unrelated input_embeds buffer without
     * invalidating this ordinary-decode-only plan.
     */
    _rebuild_ordinary_decode_plan();

    _logger->info("Language model initialize completed");
}

// Restore the latest checkpoint <= num_cached_tokens so grouped prefill can resume from a saved position.
uint16_t LanguageModel::_prepare_state_checkpoints_for_prefill(uint16_t num_cached_tokens) {
    // Manual clamp semantics: if cache reuse asks past the last grouped block,
    // clamp to last_offset + group_size before selecting a checkpoint boundary.
    const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
    const uint16_t clamp_token_count =
        static_cast<uint16_t>(offsets.back() + _cfg.pipeline_cfg.input_token_group_size);
    if (num_cached_tokens > clamp_token_count) {
        num_cached_tokens = clamp_token_count;
    }

    // Restore latest checkpoint boundary <= requested cache length.
    auto it = std::upper_bound(
        _checkpoint_boundaries.begin(), _checkpoint_boundaries.end(), num_cached_tokens
    );
    if (it == _checkpoint_boundaries.begin()) {
        for (const auto& state: _cached_states) {
            for (const auto& layer_idx: state.layer_indices) {
                get_buffer(fmt::format("{}{}", state.buffer_name_prefix, layer_idx)).clear();
            }
        }
        return 0;
    }
    --it;
    uint16_t restored_token_count = *it;
    auto requested_boundary = std::distance(_checkpoint_boundaries.begin(), it);
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    for (auto& state: _cached_states) {
        const size_t dst_offset = static_cast<size_t>(
            tail_begin * state.num_elems * state.elem_size
        );
        for (size_t layer_slot = 0; layer_slot < state.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layer_indices[layer_slot];
            auto& checkpoints = state.checkpoints[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.buffer_name_prefix, layer_idx));
            buf.upload(
                checkpoints[requested_boundary].data(),
                dst_offset,
                state.tail_bytes,
                true
            );
        }
    }
    return restored_token_count;
}


void LanguageModel::_save_state_checkpoint(
    size_t boundary_idx, uint16_t num_tokens, uint16_t valid_tokens
) {
    // Save the L-1 tail of each stateful layer's MLA buffer
    // so a future prefill can resume from this boundary without replay.
    const uint32_t tail_row_offset = (num_tokens > 1)
        ? static_cast<uint32_t>(valid_tokens - 1)
        : static_cast<uint32_t>(_cfg.pipeline_cfg.input_token_group_size - 1);

    /*
     * These state buffers obey the same uniform allocation contract as the
     * ordinary n1 token output: SIMAAI_MEM_FLAG_DEFAULT is backed by
     * dma_alloc_coherent() and mapped with the coherent pgprot. The final
     * segment CQE is published only after the driver's dma_rmb(), and Backend
     * consumes it with acquire semantics. Repeating the legacy dc civac +
     * dsb sy once per convolution layer here cost about 0.88 ms at the
     * position-320 checkpoint on LFM2.5-VL-450M, even though it cannot make a
     * coherent mapping more current.
     *
     * Keep checkpoint copying synchronous: later decode jobs mutate these
     * exact cache rows, so deferring the memcpy would be a data race. Only the
     * redundant maintenance is removed.
     */
    std::atomic_thread_fence(std::memory_order_acquire);
    for (auto& state: _cached_states) {
        const size_t src_offset_bytes = tail_row_offset * state.num_elems * state.elem_size;
        for (size_t layer_slot = 0; layer_slot < state.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layer_indices[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.buffer_name_prefix, layer_idx));
            auto* ptr = reinterpret_cast<uint8_t*>(buf.get_virtual_addr());
            std::memcpy(
                state.checkpoints[layer_slot][boundary_idx].data(),
                ptr + src_offset_bytes,
                state.tail_bytes
            );
        }
    }
}


void LanguageModel::_move_state_tail_for_decode(uint16_t valid_tokens) {
    // After a partial last group, move the valid tail to the end of the conv buffer,
    //  where the decode MLAModel expects it.
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
    /*
     * This is the reverse ownership transition of checkpoint capture. The
     * grouped-prefill CQE makes coherent MLA writes visible before these CPU
     * reads; the release fence orders the CPU memmoves before subsequent
     * JOB_EXEC submissions. Per-buffer invalidate/flush calls were legacy
     * cached-alias maintenance and are both redundant for the direct path's
     * mandatory coherent DMS0 mapping.
     */
    std::atomic_thread_fence(std::memory_order_acquire);
    for (auto& state: _cached_states) {
        const size_t src_offset_bytes = (valid_tokens - 1) * state.num_elems * state.elem_size;
        const size_t dst_offset_bytes = tail_begin * state.num_elems * state.elem_size;
        for (size_t layer_slot = 0; layer_slot < state.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layer_indices[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.buffer_name_prefix, layer_idx));
            auto* ptr = reinterpret_cast<uint8_t*>(buf.get_virtual_addr());
            std::memmove(ptr + dst_offset_bytes, ptr + src_offset_bytes, state.tail_bytes);
        }
    }
    std::atomic_thread_fence(std::memory_order_release);
}


void LanguageModel::_finalize() {
    _logger->info("Language model finalize starting ...");
    /*
     * BoundExecution retains ModelState and dma-buf registrations. Drop the
     * flat plan before package/import and BaseModel buffer teardown so its
     * ownership cannot extend either lifetime.
     */
    _ordinary_decode_plan.reset();
    _mla_model_anchor().free_related_models(
        _elf_dir / _cfg.language_model_name
    );
    BaseModel::_finalize();
    _logger->info("Language model finalize completed");
}


void LanguageModel::_define_buffer_freq_table(const std::string& name, uint32_t rope_dimension_count) {
    define_buffer(
        fmt::format("{}_freq_real", name),
        {_cfg.pipeline_cfg.max_num_tokens, rope_dimension_count / 2}
    );
    define_buffer(
        fmt::format("{}_freq_imag", name),
        {_cfg.pipeline_cfg.max_num_tokens, rope_dimension_count / 2}
    );
}


void LanguageModel::_define_buffers() {
    // Embedding table. Drafts use the target's embeddings, so skip.
    const bool is_draft = _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
    if (!is_draft) {
        define_buffer(
            "embeddings",
            {_cfg.lm_cfg.token_cfg.vocab_size, _cfg.lm_cfg.hidden_size},
            (_cfg.pipeline_cfg.quantize_embeddings)? "int8" : "bfloat16"
        );
        /*
         * One stable row is the ordinary scalar-decode address for both BF16
         * and quantized embedding tables. Develop bound a changing table row
         * directly for BF16; that zero-copy choice forced two checked dynamic
         * binds at every token. A 2-KiB LFM row copy makes the physical
         * address immutable and enables the flat position plan.
         */
        define_buffer("decode_embedding", {1, _cfg.lm_cfg.hidden_size});
    }

    // Frequency tables.
    _define_buffer_freq_table(
        "global", _cfg.lm_cfg.rope_cfg.get_rope_dimension_count("full_attention")
    );
    if (_cfg.lm_cfg.attn_cfg.swa_enable)
        _define_buffer_freq_table(
            "local", _cfg.lm_cfg.rope_cfg.get_rope_dimension_count("sliding_attention")
        );

    // KV caches.
    std::vector<size_t> cache_shape;
    const uint16_t conv_working_len = _cfg.pipeline_cfg.input_token_group_size + _cfg.lm_cfg.conv_L_cache - 2;
    std::vector<size_t> conv_cache_shape{conv_working_len, _cfg.lm_cfg.hidden_size};
    for (uint8_t i = 0; i < _cfg.lm_cfg.num_hidden_layers; ++i) {
        if (_cfg.lm_cfg.layer_types[i] == "conv") {
            define_buffer(fmt::format("conv_cache_history_l{}", i), conv_cache_shape);
        } else {
            if (_cfg.pipeline_cfg.use_strided_kv_cache) {
                cache_shape = {
                    _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                    _cfg.pipeline_cfg.max_num_tokens,
                    _cfg.lm_cfg.attn_cfg.get_head_dim(_cfg.lm_cfg.layer_types[i])
                };
            } else {
                cache_shape = {
                    _cfg.pipeline_cfg.max_num_tokens,
                    _cfg.lm_cfg.attn_cfg.get_kv_size(_cfg.lm_cfg.layer_types[i])
                };
            }
            if (!_cfg.lm_cfg.is_kv_shared_layer(i)) {
                std::string kv_dtype = _cfg.pipeline_cfg.quantize_kv_cache ? "int8" : "bfloat16";
                define_buffer(fmt::format("cache_key_l{}", i), cache_shape, kv_dtype);
                define_buffer(fmt::format("cache_val_l{}", i), cache_shape, kv_dtype);
                if (_cfg.pipeline_cfg.quantize_kv_cache) {
                    // Scale tensors use HWC16 layout like KV cache:
                    // logical (1, num_kv_heads, max_tokens, 1) bf16
                    // HWC16 pads C=1 bf16 -> C=8 bf16 (C=2 int8 padded to C=16 int8)
                    // 3D strided: {num_kv_heads, max_tokens, 8}
                    std::vector<size_t> scale_shape = {
                        _cfg.lm_cfg.attn_cfg.num_key_value_heads,
                        _cfg.pipeline_cfg.max_num_tokens,
                        8
                    };
                    define_buffer(fmt::format("cache_key_scale_l{}", i), scale_shape);
                    define_buffer(fmt::format("cache_val_scale_l{}", i), scale_shape);
                }
            }
        }
    }

    // Deepstack features for qwen3.
    if (_cfg.vm_cfg.has_value()) {
        for (size_t i = 0; i < _cfg.vm_cfg.value().deepstack_visual_indexes.size(); ++i) {
            define_buffer(
                fmt::format("deepstack_feature_l{}_cache", i),
                {_cfg.pipeline_cfg.max_num_tokens, _cfg.lm_cfg.hidden_size}
            );
        }
    }

    // Other buffers.
    std::vector<uint16_t> num_tokens_vec{_cfg.lm_cfg.get_single_num_tokens()};
    if (_use_group_token_models) {
        const auto& num_tokens = _cfg.pipeline_cfg.input_token_group_size;
        num_tokens_vec.emplace_back(num_tokens);
    }
    if (_cfg.pipeline_cfg.future_token_mask_size > 1) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            define_buffer(
                "future_token_mask",
                {_cfg.lm_cfg.get_single_num_tokens(), _cfg.pipeline_cfg.max_num_tokens}
            );
        } else {
            uint32_t full_token_mask_size = round_up_to_row(
                _cfg.pipeline_cfg.max_num_tokens + _cfg.pipeline_cfg.future_token_mask_size - 1
            );
            define_buffer(
                "future_token_mask",
                {full_token_mask_size}
            );
        }
    }

    for (const auto& num_tokens: num_tokens_vec) {
        // Pre input (for layer_idx > 0), post input and post output.
        define_buffer(
            fmt::format("n{}_buffer1", num_tokens), {num_tokens, _cfg.lm_cfg.hidden_size}
        );

        // Pre output and cache input.
        define_buffer(
            fmt::format("n{}_buffer2", num_tokens),
            {_cfg.lm_cfg.attn_cfg.num_attention_heads, num_tokens, _cfg.lm_cfg.attn_cfg.head_dim}
        );

        // Cache output and post input.
        define_buffer(
            fmt::format("n{}_buffer3", num_tokens),
            {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size("full_attention")}
        );

        if (_uses_per_layer_inputs()) {
            // Output of the standalone per-layer model, consumed by each post layer.
            define_buffer(
                fmt::format("n{}_per_layer_input", num_tokens),
                {_cfg.lm_cfg.num_hidden_layers * num_tokens, _cfg.lm_cfg.hidden_size_per_layer_input}
            );
        }

        // Draft-only buffers: second pre input, draft hidden states output, FC fusion buffers.
        if (is_draft) {
            define_buffer(
                fmt::format("n{}_buffer1a", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
            define_buffer(
                fmt::format("n{}_buffer5", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
            define_buffer(
                fmt::format("fc_n{}_input", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size * 3}
            );
            define_buffer(
                fmt::format("fc_n{}_output", num_tokens),
                {num_tokens, _cfg.lm_cfg.hidden_size}
            );
        }
    }

    if (_uses_per_layer_inputs()) {
        const std::string dtype = _cfg.pipeline_cfg.quantize_embeddings
            ? "int8" : "bfloat16";
        // Vocabulary sharding keeps every token's complete L*H row contiguous.
        const size_t out_dim = static_cast<size_t>(_cfg.lm_cfg.num_hidden_layers)
                             * _cfg.lm_cfg.hidden_size_per_layer_input;
        const size_t elem_size = _cfg.pipeline_cfg.quantize_embeddings
            ? sizeof(int8_t) : sizeof(Eigen::bfloat16);
        _per_layer_embedding_rows_per_shard = std::max<size_t>(
            1, PER_LAYER_EMBEDDING_MAX_SHARD_SIZE / (out_dim * elem_size)
        );
        const size_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
        const size_t num_shards = (
            vocab_size + _per_layer_embedding_rows_per_shard - 1
        ) / _per_layer_embedding_rows_per_shard;
        _per_layer_embedding_shards.clear();
        _per_layer_embedding_shards.reserve(num_shards);
        for (size_t shard_idx = 0; shard_idx < num_shards; ++shard_idx) {
            const size_t row_begin = shard_idx * _per_layer_embedding_rows_per_shard;
            const size_t num_rows = std::min(
                _per_layer_embedding_rows_per_shard, vocab_size - row_begin
            );
            const auto name = fmt::format("per_layer_embeddings_s{}", shard_idx);
            define_buffer(
                name,
                {num_rows, out_dim},
                dtype
            );
            _per_layer_embedding_shards.emplace_back(&get_buffer(name));
        }
        // Input staging for the standalone per-layer model, filled from token-id gathers.
        define_buffer("per_layer_emb_staging_n1", {1, out_dim}, dtype);
        if (_use_group_token_models) {
            define_buffer(
                fmt::format("per_layer_emb_staging_n{}", _cfg.pipeline_cfg.input_token_group_size),
                {_cfg.pipeline_cfg.input_token_group_size, out_dim},
                dtype
            );
        }
    }

    // Post output for last layer.
    if (!_cfg.lm_cfg.is_spec_decode()){
        if (_cfg.lm_cfg.lm_head_num_splits == 1 && !_cfg.pipeline_cfg.return_logits)
            define_buffer("n1_buffer4", {1}, "int32");
        else
            define_buffer("n1_buffer4", {_cfg.lm_cfg.token_cfg.vocab_size});
    } else {
        const auto lm_head_output_size = _cfg.lm_cfg.get_lm_head_output_size();
        const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
        for (const auto& num_tokens: num_tokens_vec) {
            // Target prefill reuses the single-token-group final post model
            // (n16 for EAGLE3). Only drafts need group-width final-post outputs.
            if (!is_draft && num_tokens != _cfg.lm_cfg.get_single_num_tokens()) {
                continue;
            }
            if (_cfg.lm_cfg.lm_head_num_splits == 1 && !_cfg.pipeline_cfg.return_logits) {
                define_buffer(
                    fmt::format("n{}_buffer4", num_tokens), {num_tokens, 1}, "int32"
                );
            } else if (_cfg.lm_cfg.lm_head_num_splits == 1) {
                define_buffer(
                    fmt::format("n{}_buffer4", num_tokens),
                    {num_tokens, lm_head_output_size}
                );
            } else {
                uint32_t i = 0;
                for(uint32_t split_begin = 0; split_begin < lm_head_output_size; split_begin += split_dim, ++i) {
                    uint32_t split_size = std::min(lm_head_output_size, split_begin + split_dim) - split_begin;
                    define_buffer(
                        fmt::format("n{}_lm_split{}", num_tokens, i),
                        {num_tokens, split_size}
                    );
                }
            }
        }
    }
}


void LanguageModel::_define_model(
    const std::string& model_type,
    const LanguageModelMapKey& key,
    const std::filesystem::path& model_path,
    const std::vector<MLABufferSlice>& ifms,
    const std::vector<MLABufferSlice>& ofms
) {
    LanguageModelMap& map = get_model_map(model_type);
    map.emplace(
        key, MLAModelWithBuffer(
            _mla_session, model_path, ifms, ofms,
            _cfg.lm_cfg.has_lora()
        )
    );
}


void LanguageModel::_define_attn_models_iter(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    LanguageModelMapKey model_key{num_tokens, layer_idx, token_idx};
    const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const auto kv_source_layer = _cfg.lm_cfg.get_kv_source_layer(layer_idx);
    const auto& kv_layer_type = _cfg.lm_cfg.layer_types[kv_source_layer];
    std::string freq_prefix;
    uint16_t cache_token_idx_begin;
    if (layer_type == "sliding_attention") {
        freq_prefix = "local";
        cache_token_idx_begin = std::max(
            0,
            token_idx + num_tokens - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
        );
    } else {
        freq_prefix = "global";
        cache_token_idx_begin = 0;
    }

    // Pre model.
    std::vector<uint32_t> pre_kv_cache_shape;
    std::vector<uint32_t> pre_kv_cache_offset;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        pre_kv_cache_offset = {0, token_idx, 0};
        pre_kv_cache_shape = {
            _cfg.lm_cfg.attn_cfg.num_key_value_heads,
            _cfg.pipeline_cfg.max_num_tokens,
            _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
        };
    } else {
        pre_kv_cache_offset = {token_idx, 0};
        pre_kv_cache_shape = {num_tokens, _cfg.lm_cfg.attn_cfg.get_kv_size(layer_type)};
    }

    // Draft pre takes an extra IFM (buffer1a) for the FC fusion output / target hidden state.
    const bool is_draft = _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;

    std::vector<MLABufferSlice> pre_ifms;
    std::vector<MLABufferSlice> pre_ofms;
    if (layer_idx) {
        pre_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
        if (is_draft) {
            pre_ifms.emplace_back(
                MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1a", num_tokens))}
            );
        }
    } else {
        /*
         * Ordinary n1 decode always consumes the fixed staging row. Making it
         * the wrapper's canonical default lets eager package publication
         * create the exact BoundExecution once. Prefill and every excluded
         * shape still supply an explicit checked override.
         */
        if (num_tokens == 1 && !_cfg.lm_cfg.is_spec_decode()) {
            pre_ifms.emplace_back(
                MLABufferSlice{&get_buffer("decode_embedding")}
            );
        } else {
            pre_ifms.emplace_back(MLABufferSlice{});
        }
        if (is_draft) {
            pre_ifms.emplace_back(MLABufferSlice{});
        }
    }
    auto buf_name = fmt::format("{}_freq_real", freq_prefix);
    pre_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(buf_name),
            {token_idx, 0},
            {num_tokens, (uint32_t)get_buffer(buf_name).get_shape().back()}
        }
    );
    buf_name = fmt::format("{}_freq_imag", freq_prefix);
    pre_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(buf_name),
            {token_idx, 0},
            {num_tokens, (uint32_t)get_buffer(buf_name).get_shape().back()}
        }
    );
    pre_ofms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("n{}_buffer2", num_tokens)),
            {0, 0, 0},
            {
                _cfg.lm_cfg.attn_cfg.num_attention_heads,
                num_tokens,
                _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
            }
        }
    );
    if (!_cfg.lm_cfg.is_kv_shared_layer(layer_idx)) {
        pre_ofms.emplace_back(
            MLABufferSlice(
                &get_buffer(fmt::format("cache_key_l{}", layer_idx)),
                pre_kv_cache_offset,
                pre_kv_cache_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            pre_ofms.emplace_back(
                MLABufferSlice(
                    &get_buffer(fmt::format("cache_key_scale_l{}", layer_idx)),
                    {0, token_idx, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 8}
                )
            );
        }
        pre_ofms.emplace_back(
            MLABufferSlice(
                &get_buffer(fmt::format("cache_val_l{}", layer_idx)),
                pre_kv_cache_offset,
                pre_kv_cache_shape
            )
        );
        if (_cfg.pipeline_cfg.quantize_kv_cache) {
            pre_ofms.emplace_back(
                MLABufferSlice(
                    &get_buffer(fmt::format("cache_val_scale_l{}", layer_idx)),
                    {0, token_idx, 0},
                    {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 8}
                )
            );
        }
    }
    _define_model(
        "pre",
        model_key,
        _get_elf_path_pre(num_tokens, layer_idx),
        pre_ifms,
        pre_ofms
    );

    // Cache model.
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    uint16_t eff_token_idx = token_idx - cache_token_idx_begin;
    uint16_t eff_num_cached_tokens = token_idx + num_tokens - cache_token_idx_begin;
    uint16_t aligned_eff_token_idx;
    uint16_t aligned_eff_num_cached_tokens;
    if (num_tokens == single_num_tokens && _cfg.pipeline_cfg.future_token_mask_size > 1) {
        // Round up by total context (eff_num_cached_tokens = past_kv + num_tokens),
        // not by past_kv alone. For spec mode num_tokens=16: past_kv=114 ->
        // total=130 -> bucket=256 (cache_token_255). The old formula used
        // eff_token_idx + 1, which only worked when num_tokens was 1 (non-spec).
        // For spec mode it picked cache_token_127 even when the 16 new tokens
        // wouldn't fit, dropping mask data for the tail tree positions.
        aligned_eff_token_idx = std::min(
            round_up_to(eff_num_cached_tokens, _cfg.pipeline_cfg.future_token_mask_size) - 1,
            _cfg.pipeline_cfg.max_num_tokens - 1
        );
        aligned_eff_num_cached_tokens = aligned_eff_token_idx + 1;
    } else {
        aligned_eff_token_idx = eff_token_idx;
        aligned_eff_num_cached_tokens = eff_num_cached_tokens;
    }
    std::vector<uint32_t> cache_kv_cache_shape;
    std::vector<uint32_t> cache_kv_cache_offset;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        cache_kv_cache_offset = {0, cache_token_idx_begin, 0};
        cache_kv_cache_shape = {
            _cfg.lm_cfg.attn_cfg.num_key_value_heads,
            _cfg.pipeline_cfg.max_num_tokens,
            _cfg.lm_cfg.attn_cfg.get_head_dim(kv_layer_type)
        };
    } else {
        cache_kv_cache_offset = {cache_token_idx_begin, 0};
        cache_kv_cache_shape = {
            aligned_eff_num_cached_tokens,
            _cfg.lm_cfg.attn_cfg.get_kv_size(kv_layer_type)
        };
    }

    std::vector<MLABufferSlice> cache_ifms{
        MLABufferSlice{
            &get_buffer(fmt::format("n{}_buffer2", num_tokens)),
            {0, 0, 0},
            {
                _cfg.lm_cfg.attn_cfg.num_attention_heads,
                num_tokens,
                _cfg.lm_cfg.attn_cfg.get_head_dim(layer_type)
            }
        },
        MLABufferSlice{
            &get_buffer(fmt::format("cache_key_l{}", kv_source_layer)),
            cache_kv_cache_offset,
            cache_kv_cache_shape
        },
    };
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(fmt::format("cache_key_scale_l{}", kv_source_layer)),
                {0, cache_token_idx_begin, 0},
                {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 8}
            }
        );
    }

    if (num_tokens == single_num_tokens && _cfg.pipeline_cfg.future_token_mask_size > 1) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            // Shift the col begin to cache_token_idx_begin so that buffer col
            // p ↔ K-row p for any layer type: full (cache_token_idx_begin = 0)
            // reads cols [0, aligned_eff); sliding reads
            // [cache_token_idx_begin, cache_token_idx_begin + aligned_eff).
            // The mask data layout (col c = mask for K[c]) is the same for
            // both, so no second buffer or layout change is needed.
            cache_ifms.emplace_back(
                MLABufferSlice{
                    &get_buffer("future_token_mask"),
                    {0, cache_token_idx_begin},
                    {num_tokens, aligned_eff_num_cached_tokens}
                }
            );
        } else {
            cache_ifms.emplace_back(
                MLABufferSlice{
                    &get_buffer("future_token_mask"),
                    {(uint32_t)_cfg.pipeline_cfg.max_num_tokens - eff_num_cached_tokens},
                    {aligned_eff_num_cached_tokens}
                }
            );
        }
    }
    cache_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("cache_val_l{}", kv_source_layer)),
            cache_kv_cache_offset,
            cache_kv_cache_shape
        }
    );
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(fmt::format("cache_val_scale_l{}", kv_source_layer)),
                {0, cache_token_idx_begin, 0},
                {_cfg.lm_cfg.attn_cfg.num_key_value_heads, _cfg.pipeline_cfg.max_num_tokens, 8}
            }
        );
    }
    std::vector<MLABufferSlice> cache_ofms{
        MLABufferSlice{
            &get_buffer(fmt::format("n{}_buffer3", num_tokens)),
            {0, 0},
            {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(layer_type)}
        }
    };
    _define_model(
        "cache",
        model_key,
        _get_elf_path_cache(num_tokens, aligned_eff_token_idx, layer_idx),
        cache_ifms,
        cache_ofms
    );

    // Post model. For draft, post IFM 0 is the FC fusion output (pre IFM 1), not the
    // token embeddings (pre IFM 0). For target, pre IFM 0 is the hidden state directly.
    std::vector<MLABufferSlice> post_ifms{is_draft ? pre_ifms[1] : pre_ifms[0], cache_ofms[0]};
    const bool use_single_post_for_target_group = (
        _cfg.lm_cfg.is_spec_decode()
        && !is_draft
        && num_tokens != single_num_tokens
        && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
    );
    if (use_single_post_for_target_group) {
        post_ifms[0] = MLABufferSlice{
            post_ifms[0].get_buf_ptr(),
            {0, 0},
            {single_num_tokens, _cfg.lm_cfg.hidden_size}
        };
        post_ifms[1] = MLABufferSlice{
            post_ifms[1].get_buf_ptr(),
            {0, 0},
            {single_num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size(layer_type)}
        };
    }
    if (_uses_per_layer_inputs()) {
        post_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(fmt::format("n{}_per_layer_input", num_tokens)),
                std::vector<uint32_t>{static_cast<uint32_t>(layer_idx) * num_tokens, 0},
                std::vector<uint32_t>{
                    use_single_post_for_target_group ? single_num_tokens : num_tokens,
                    _cfg.lm_cfg.hidden_size_per_layer_input
                }
            }
        );
    }
    if (
        _cfg.vm_cfg.has_value()
        && layer_idx < _cfg.vm_cfg.value().deepstack_visual_indexes.size()
        && num_tokens > 1
    ) {
        buf_name = fmt::format("deepstack_feature_l{}_cache", layer_idx);
        post_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(buf_name),
                {token_idx, 0},
                {
                    use_single_post_for_target_group ? single_num_tokens : num_tokens,
                    (uint32_t)get_buffer(buf_name).get_shape().back()
                }
            }
        );
    }
    std::vector<MLABufferSlice> post_ofms;
    std::string post_elf_path;
    if (layer_idx < _cfg.lm_cfg.num_hidden_layers - 1) {
        post_ofms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
        post_elf_path = _get_elf_path_post(num_tokens, layer_idx);
    } else {
        // Target speculative prefill only needs the final valid prompt row's
        // logits, so reuse the n16 verification post instead of loading n128.
        const uint16_t post_num_tokens = use_single_post_for_target_group
            ? single_num_tokens
            : (_cfg.lm_cfg.is_spec_decode() ? num_tokens : 1);
        post_elf_path = _get_elf_path_post(post_num_tokens, layer_idx);

        if (_cfg.lm_cfg.lm_head_num_splits == 1) {
            const std::string buf_name = _cfg.lm_cfg.is_spec_decode()
                ? fmt::format("n{}_buffer4", post_num_tokens)
                : std::string("n1_buffer4");
            post_ofms.emplace_back(MLABufferSlice{&get_buffer(buf_name)});
        } else {
            const auto lm_head_output_size = _cfg.lm_cfg.get_lm_head_output_size();
            const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;

            if (_cfg.lm_cfg.is_spec_decode()) {
                uint32_t i = 0;
                for (uint32_t split_begin = 0;
                     split_begin < lm_head_output_size;
                     split_begin += split_dim, ++i)
                {
                    post_ofms.emplace_back(
                        MLABufferSlice{
                            &get_buffer(fmt::format("n{}_lm_split{}", post_num_tokens, i))
                        }
                    );
                }
            } else {
                for (uint32_t split_begin = 0;
                     split_begin < lm_head_output_size;
                     split_begin += split_dim)
                {
                    auto split_size = std::min(lm_head_output_size, split_begin + split_dim) - split_begin;
                    post_ofms.emplace_back(
                        MLABufferSlice{&get_buffer("n1_buffer4"), {split_begin}, {split_size}}
                    );
                }
            }
        }

        // Draft post produces an additional output: hidden states for next iteration.
        if (is_draft) {
            post_ofms.emplace_back(
                MLABufferSlice{&get_buffer(fmt::format("n{}_buffer5", num_tokens))}
            );
        }
    }
    _define_model("post", model_key, post_elf_path, post_ifms, post_ofms);
}


void LanguageModel::_define_state_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
        _define_conv_models_iter(num_tokens, layer_idx);
    }
}


void LanguageModel::_define_conv_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    const uint16_t tail_size = std::max<uint16_t>(1, _cfg.lm_cfg.conv_L_cache - 1);
    const uint16_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;

    // Conv model.
    std::vector<MLABufferSlice> conv_ifms;
    std::vector<MLABufferSlice> conv_ofms;
    if (layer_idx) {
        conv_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
    } else {
        if (num_tokens == 1 && !_cfg.lm_cfg.is_spec_decode()) {
            conv_ifms.emplace_back(
                MLABufferSlice{&get_buffer("decode_embedding")}
            );
        } else {
            conv_ifms.emplace_back(MLABufferSlice{});
        }
    }
    conv_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {tail_begin, 0},
            {tail_size, _cfg.lm_cfg.hidden_size}
        }
    );
    conv_ofms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))});
    conv_ofms.emplace_back(
        MLABufferSlice(
            &get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {num_tokens > 1 ? 0 : tail_begin, 0},
            {
                static_cast<uint32_t>(num_tokens + tail_size - 1),
                _cfg.lm_cfg.hidden_size
            }
        )
    );
    _define_model(
        "conv",
        model_key,
        _get_elf_path_conv(num_tokens, layer_idx),
        conv_ifms,
        conv_ofms
    );

    // Conv final model.
    std::vector<MLABufferSlice> conv_final_ifms{conv_ofms[0]};
    std::vector<MLABufferSlice> conv_final_ofms;
    std::string post_elf_path;
    if (layer_idx < _cfg.lm_cfg.num_hidden_layers - 1) {
        return;
    } else if (_cfg.lm_cfg.lm_head_num_splits == 1) {
        conv_final_ofms.emplace_back(MLABufferSlice{&get_buffer("n1_buffer4")});
    } else {
        const auto& vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
        const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
        for (uint32_t split_begin = 0; split_begin < vocab_size; split_begin += split_dim) {
            auto split_size = std::min(vocab_size, split_begin + split_dim) - split_begin;
            conv_final_ofms.emplace_back(
                MLABufferSlice{&get_buffer("n1_buffer4"), {split_begin}, {split_size}}
            );
        }
    }
    _define_model(
        "conv_final",
        model_key,
        _get_elf_path_conv_final(layer_idx),
        conv_final_ifms,
        conv_final_ofms
    );
}


void LanguageModel::_define_models() {
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {single_num_tokens};
    if (_use_group_token_models) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens: num_tokens_vec) {
        const auto& max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
        const auto& num_hidden_layers = _cfg.lm_cfg.num_hidden_layers;

        if (num_tokens == single_num_tokens) {
            for (uint16_t token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    if (
                        _cfg.lm_cfg.layer_types[layer_idx] == "full_attention"
                        || _cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention"
                    ) {
                        _define_attn_models_iter(num_tokens, token_idx, layer_idx);
                    }
                }
            }
        } else {
            for (const auto& token_idx: _cfg.pipeline_cfg.input_token_group_offsets.value()) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    if (
                        _cfg.lm_cfg.layer_types[layer_idx] == "full_attention"
                        || _cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention"
                    ) {
                        _define_attn_models_iter(num_tokens, token_idx, layer_idx);
                    }
                }
            }
        }

        for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            _define_state_models_iter(num_tokens, layer_idx);
        }
    }
    _define_per_layer_models();

    // Draft-only: FC fusion models.
    const bool is_draft = _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_draft;
    if (is_draft) {
        _define_draft_fc_models();
    }
    _build_ordinary_decode_recipes();
}

void LanguageModel::_build_ordinary_decode_recipes() {
    _ordinary_decode_recipes.clear();

    /*
     * Keep the first optimized shape deliberately narrow.  These conditions
     * describe the same ordinary n1 decode traversal below: no speculative
     * capture boundary, no per-layer CPU staging, and no deepstack input whose
     * topology can vary with the multimodal request.  Every excluded shape
     * continues through the generic builder unchanged.
     */
    const bool has_deepstack =
        _cfg.vm_cfg.has_value() &&
        !_cfg.vm_cfg->deepstack_visual_indexes.empty();
    const bool supported_layers = std::all_of(
        _cfg.lm_cfg.layer_types.begin(),
        _cfg.lm_cfg.layer_types.end(),
        [](const std::string& type) {
            return type == "full_attention" ||
                   type == "sliding_attention" ||
                   type == "conv";
        }
    );
    if (_cfg.lm_cfg.get_single_num_tokens() != 1 ||
        _cfg.lm_cfg.is_spec_decode() ||
        _uses_per_layer_inputs() ||
        has_deepstack ||
        !supported_layers) {
        return;
    }

    const uint16_t max_num_tokens =
        _cfg.pipeline_cfg.max_num_tokens;
    _ordinary_decode_recipes.resize(max_num_tokens);
    for (uint16_t token_idx = 0;
         token_idx < max_num_tokens; ++token_idx) {
        OrdinaryDecodeRecipe& recipe =
            _ordinary_decode_recipes[token_idx];
        /*
         * Twenty-nine map lookups and the topology branches used to run at
         * every LFM token boundary.  Build exactly that order once while all
         * map nodes are stable, but retain only borrowed wrappers: their
         * binding cells still select the current package/LoRA generation.
         */
        recipe.reserve(
            static_cast<std::size_t>(_cfg.lm_cfg.num_hidden_layers) * 3 + 1
        );
        for (uint8_t layer_idx = 0;
             layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            const std::string& type =
                _cfg.lm_cfg.layer_types[layer_idx];
            if (type == "full_attention" ||
                type == "sliding_attention") {
                const LanguageModelMapKey key{
                    1, layer_idx, token_idx
                };
                recipe.push_back({
                    .model = &_pre_model_map.at(key),
                    .embedding_override = layer_idx == 0,
                    .position_sensitive = true,
                });
                recipe.push_back({
                    .model = &_cache_model_map.at(key),
                    .embedding_override = false,
                    .position_sensitive = true,
                });
                recipe.push_back({
                    .model = &_post_model_map.at(key),
                    .embedding_override = layer_idx == 0,
                    .position_sensitive = true,
                });
            } else {
                const LanguageModelMapKey key{1, layer_idx, 0};
                recipe.push_back({
                    .model = &_conv_model_map.at(key),
                    .embedding_override = layer_idx == 0,
                    .position_sensitive = false,
                });
                if (layer_idx ==
                    _cfg.lm_cfg.num_hidden_layers - 1) {
                    recipe.push_back({
                        .model = &_conv_final_model_map.at(key),
                        .embedding_override = layer_idx == 0,
                        .position_sensitive = false,
                    });
                }
            }
        }

        /*
         * Construction-time position oracle: an attention wrapper contains
         * position-specific KV/RoPE/mask slices.  Accidentally substituting
         * the p-1 recipe must fail before packages are published, rather than
         * silently producing plausible but stale token state at runtime.
         */
        if (token_idx != 0) {
            const OrdinaryDecodeRecipe& previous =
                _ordinary_decode_recipes[token_idx - 1];
            if (previous.size() != recipe.size()) {
                throw std::logic_error(
                    "ordinary MLA decode recipe topology changed by position"
                );
            }
            for (std::size_t i = 0; i < recipe.size(); ++i) {
                if (!recipe[i].model ||
                    recipe[i].embedding_override !=
                        previous[i].embedding_override ||
                    recipe[i].position_sensitive !=
                        previous[i].position_sensitive ||
                    (recipe[i].position_sensitive &&
                     recipe[i].model == previous[i].model)) {
                    throw std::logic_error(
                        "ordinary MLA decode recipe reused a stale position cell"
                    );
                }
            }
        }
    }
}

bool LanguageModel::_has_ordinary_decode_recipe(
    uint16_t token_idx
) const {
    return token_idx < _ordinary_decode_recipes.size() &&
           !_ordinary_decode_recipes[token_idx].empty();
}

void LanguageModel::_append_ordinary_decode_recipe(
    MlaExecutionSegment& segment,
    uint16_t token_idx,
    MLABuffer* embedding,
    uint32_t embedding_row
) {
    if (!_has_ordinary_decode_recipe(token_idx) || !embedding) {
        throw std::logic_error(
            "ordinary MLA decode recipe is unavailable"
        );
    }

    for (const OrdinaryDecodeRecipeStep& step :
         _ordinary_decode_recipes[token_idx]) {
        if (!step.model) {
            throw std::logic_error(
                "ordinary MLA decode recipe contains a null model"
            );
        }
        if (step.embedding_override) {
            step.model->_add_embedding_row_to_segment(
                segment, 0, embedding, embedding_row,
                _cfg.lm_cfg.hidden_size
            );
        } else {
            step.model->add_to_segment(segment);
        }
    }
}

void LanguageModel::_append_ordinary_decode_recipe(
    MlaExecutionPlan& plan,
    uint16_t token_idx,
    MLABuffer* embedding,
    uint32_t embedding_row
) {
    if (!_has_ordinary_decode_recipe(token_idx) || !embedding) {
        throw std::logic_error(
            "ordinary MLA decode plan recipe is unavailable"
        );
    }
    plan.begin_position();
    try {
        for (const OrdinaryDecodeRecipeStep& step :
             _ordinary_decode_recipes[token_idx]) {
            if (!step.model) {
                throw std::logic_error(
                    "ordinary MLA decode plan contains a null model"
                );
            }
            /*
             * The ordinary wrapper's canonical IFM0 is the same fixed staging
             * row. Use its eager exact handle; adding an identical override
             * would unnecessarily allocate a second BoundExecution for every
             * physical position.
             */
            (void)embedding;
            (void)embedding_row;
            step.model->_add_to_plan(plan);
        }
        plan.end_position();
    } catch (...) {
        /*
         * The candidate is never published until seal(), so abandoning the
         * local object is the complete rollback. Do not try to close a
         * partially constructed physical position.
         */
        throw;
    }
}

void LanguageModel::_rebuild_ordinary_decode_plan() {
    _ordinary_decode_plan.reset();
    if (_ordinary_decode_recipes.empty() || _need_argmax) {
        return;
    }

    const auto start = std::chrono::steady_clock::now();
    auto candidate =
        std::make_unique<MlaExecutionPlan>(_mla_session);
    MLABuffer* staging = &get_buffer("decode_embedding");
    for (std::size_t position = 0;
         position < _ordinary_decode_recipes.size(); ++position) {
        _append_ordinary_decode_recipe(
            *candidate, static_cast<uint16_t>(position),
            staging, 0
        );
    }
    candidate->seal();
    const auto elapsed_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start
        ).count();
    const std::size_t bytes = candidate->metadata_bytes();
    if (bytes > 16U * 1024U * 1024U) {
        throw std::runtime_error(fmt::format(
            "ordinary MLA decode plan retains {} bytes; release gate is 16 MiB",
            bytes
        ));
    }
    _logger->info(
        "Built ordinary MLA flat plan: positions={} metadata_bytes={} "
        "build_ms={:.3f}",
        candidate->position_count(), bytes, elapsed_ms
    );
    _ordinary_decode_plan = std::move(candidate);
}

void LanguageModel::_stage_ordinary_decode_embedding(
    uint32_t token_id, MLABuffer& staging
) {
    MLABuffer& embeddings = get_buffer("embeddings");
    if (token_id >= embeddings.get_shape().front() ||
        staging.get_shape().empty() ||
        staging.get_shape().back() != _cfg.lm_cfg.hidden_size) {
        throw std::out_of_range(
            "ordinary MLA embedding token or staging shape is invalid"
        );
    }

    simaai::neat::mla::CpuAccessGuard guard;
    const auto begin = begin_mla_buffer_cpu_access(
        _mla_session, &staging,
        simaai::neat::mla::CpuAccessMode::kWrite, &guard
    );
    if (!begin) {
        throw std::runtime_error(fmt::format(
            "begin ordinary MLA embedding CPU ownership failed: {} ({})",
            begin.code, begin.message
        ));
    }

    try {
        if (embeddings.get_dtype() == "int8") {
            if (!_cfg.pipeline_cfg.embeddings_scale.has_value()) {
                throw std::runtime_error(
                    "quantized embeddings require embeddings_scale"
                );
            }
            _dequantize_embedding_row(token_id, staging);
        } else if (
            embeddings.get_dtype() == "bfloat16" &&
            staging.get_dtype() == "bfloat16") {
            const std::size_t row_bytes =
                static_cast<std::size_t>(_cfg.lm_cfg.hidden_size) *
                sizeof(Eigen::bfloat16);
            const auto* source =
                static_cast<const std::uint8_t*>(
                    embeddings.get_virtual_addr()
                ) + static_cast<std::size_t>(token_id) * row_bytes;
            std::memcpy(
                staging.get_virtual_addr(), source, row_bytes
            );
        } else {
            throw std::runtime_error(fmt::format(
                "unsupported ordinary embedding staging {} -> {}",
                embeddings.get_dtype(), staging.get_dtype()
            ));
        }
    } catch (...) {
        (void)guard.end();
        throw;
    }

    /*
     * end() is the release/publication edge for CPU writes and is specified
     * noexcept. It replaces develop's direct flush_cache() call and makes the
     * same helper safe for the later CQ-owned continuation.
     */
    const auto end = guard.end();
    if (!end) {
        throw std::runtime_error(fmt::format(
            "end ordinary MLA embedding CPU ownership failed: {} ({})",
            end.code, end.message
        ));
    }
}


void LanguageModel::_define_draft_fc_models() {
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    std::vector<uint16_t> num_tokens_vec = {single_num_tokens};
    if (_use_group_token_models) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens : num_tokens_vec) {
        auto elf_path = _elf_dir / fmt::format(
            "{}_n{}_draft_fc_stage1_mla.elf",
            _cfg.language_model_name, num_tokens
        );
        std::vector<MLABufferSlice> ifms{
            MLABufferSlice{&get_buffer(fmt::format("fc_n{}_input", num_tokens))}
        };
        std::vector<MLABufferSlice> ofms{
            MLABufferSlice{&get_buffer(fmt::format("fc_n{}_output", num_tokens))}
        };
        _fc_model_map.emplace(
            num_tokens, MLAModelWithBuffer(
                _mla_session, elf_path, ifms, ofms,
                _cfg.lm_cfg.has_lora()
            )
        );
    }
}


void LanguageModel::compact_kv_after_accept(
    std::span<const uint16_t> select_indices, uint16_t prev_input_len
) {
    require_healthy_mla_session();
    const size_t n = select_indices.size();
    const uint8_t num_layers = _cfg.lm_cfg.num_hidden_layers;
    const auto& layer_types = _cfg.lm_cfg.layer_types;
    const uint32_t max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
    const uint32_t num_kv_heads = _cfg.lm_cfg.attn_cfg.num_key_value_heads;

    // For each layer, gather KV at select_indices and scatter to contiguous positions.
    // Strided layout: (num_kv_heads, max_num_tokens, head_dim). Per-position chunk per
    // head is head_dim bf16 values at byte offset (h * max_num_tokens + pos) * head_dim_bytes.
    for (uint8_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        // Skip non-attention layers (e.g., conv layers in some models).
        if (layer_types[layer_idx] != "full_attention"
            && layer_types[layer_idx] != "sliding_attention") {
            continue;
        }

        const uint32_t head_dim = _cfg.lm_cfg.attn_cfg.get_head_dim(layer_types[layer_idx]);
        const size_t head_dim_bytes =
            static_cast<size_t>(head_dim) * sizeof(Eigen::bfloat16);
        const size_t head_stride_bytes =
            static_cast<size_t>(max_num_tokens) * head_dim_bytes;

        for (const std::string& kind : {"cache_key_l", "cache_val_l"}) {
            auto& buf = get_buffer(fmt::format("{}{}", kind, layer_idx));
            buf.invalidate_cache();
            uint8_t* data = reinterpret_cast<uint8_t*>(buf.get_virtual_addr());

            std::vector<uint8_t> tmp(n * head_dim_bytes);
            for (uint32_t h = 0; h < num_kv_heads; ++h) {
                uint8_t* head_base = data + h * head_stride_bytes;

                // Gather rows at select_indices into tmp (handles src/dst overlap).
                for (size_t i = 0; i < n; ++i) {
                    std::memcpy(
                        tmp.data() + i * head_dim_bytes,
                        head_base + static_cast<size_t>(select_indices[i]) * head_dim_bytes,
                        head_dim_bytes
                    );
                }
                // Scatter to contiguous positions [prev_input_len, prev_input_len + n).
                for (size_t i = 0; i < n; ++i) {
                    std::memcpy(
                        head_base
                            + (static_cast<size_t>(prev_input_len) + i) * head_dim_bytes,
                        tmp.data() + i * head_dim_bytes,
                        head_dim_bytes
                    );
                }
            }

            buf.flush_cache();
        }
    }

    _kv_cache_len = static_cast<uint16_t>(prev_input_len + n);
}


// update_inference_inputs implementation moved to language_model_eagle3.cpp
// (alongside the other EAGLE3-specific methods).


void LanguageModel::_define_per_layer_models() {
    if (!_uses_per_layer_inputs())
        return;

    std::vector<uint16_t> num_tokens_vec = {_cfg.lm_cfg.get_single_num_tokens()};
    if (_use_group_token_models)
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);

    for (auto num_tokens : num_tokens_vec) {
        LanguageModelMapKey key{num_tokens, 0, 0};
        std::vector<MLABufferSlice> ifms{
            MLABufferSlice{&get_buffer(fmt::format("per_layer_emb_staging_n{}", num_tokens))},
            MLABufferSlice{}
        };
        std::vector<MLABufferSlice> ofms{
            MLABufferSlice{&get_buffer(fmt::format("n{}_per_layer_input", num_tokens))}
        };
        _define_model("per_layer", key, _get_elf_path_per_layer(num_tokens), ifms, ofms);
    }
}


void LanguageModel::set_reloc(const std::string& reloc_name) {
    require_healthy_mla_session();
    // Swap model adapters from `{devkit}/../npy_files/{reloc_name}` without
    // changing the immutable compiled base model.

    if (_reloc_name == reloc_name) {
        // The model has already been relocated to the same path. Nothing to be done.
        _logger->info("No relocation is needed");
        return;
    }

    // Key to access the reloc addr maps.
    struct RelocMapType {
        std::string model_type;
        uint16_t num_tokens;
        uint8_t layer_idx;
        auto operator<=>(const RelocMapType&) const = default;
    };
    /*
     * Keep dma-buf-backed slices, never raw physical addresses. The direct
     * backend snapshots checked hidden-input BufferViews per accepted job, so
     * switching adapters cannot mutate an older queued descriptor set.
     */
    using RelocMap = std::map<std::string, MLABufferSlice>;

    auto reloc_dir = _devkit_dir / "../npy_files" / reloc_name;
    _logger->info("Relocation with data from: {}", reloc_dir);
    if (!std::filesystem::is_directory(reloc_dir)) {
        throw std::filesystem::filesystem_error(
            "Relocation directory does not exist", reloc_dir,
            std::make_error_code(std::errc::no_such_file_or_directory)
        );
    }
    const uint8_t candidate_bank = _reloc_buffer_bank ^ 1U;
    std::vector<MLAModelWithBuffer*> candidate_models;

    /*
     * set_reloc_set owns the session execution lock for this entire callback.
     * Consequently every previously accepted job has a terminal CQE before an
     * inactive bank is overwritten, and no new snapshot can observe a partial
     * upload. Validation/publishing is performed only after the callback
     * returns its complete candidate vector.
     */
    _mla_model_anchor().set_reloc_set(
        _reloc_models,
        [&]() -> std::vector<MLAModelWithBuffer::RelocUpdate> {
            std::map<RelocMapType, RelocMap> reloc_addr_maps;
            std::vector<std::filesystem::path> adapter_files;
            for (const auto& dir_entry :
                 std::filesystem::directory_iterator(reloc_dir)) {
                if (dir_entry.path().extension() == ".npy") {
                    adapter_files.push_back(dir_entry.path());
                }
            }
            std::sort(adapter_files.begin(), adapter_files.end());
            if (adapter_files.empty()) {
                throw std::invalid_argument(fmt::format(
                    "Adapter directory {} contains no NPY files", reloc_dir
                ));
            }

            /*
             * One dma-buf per NPY made a real 1,888-tensor adapter exceed the
             * process FD limit before publication. Pack the inactive adapter
             * into one immutable arena instead. A conservative 64-KiB start
             * alignment preserves the alignment each old standalone DMS0
             * allocation received; compiler extents still bound every view.
             * The upper bound uses on-disk file sizes (header + payload), so
             * the subsequent raw payload packing cannot exceed it.
             */
            constexpr std::size_t arena_alignment = 64U * 1024U;
            auto align_arena_offset = [](std::size_t value) {
                const std::size_t remainder = value % arena_alignment;
                if (remainder == 0) return value;
                const std::size_t increment = arena_alignment - remainder;
                if (value > std::numeric_limits<std::size_t>::max() - increment) {
                    throw std::overflow_error("LoRA arena alignment overflow");
                }
                return value + increment;
            };
            std::size_t arena_capacity = 0;
            for (const auto& file_name : adapter_files) {
                arena_capacity = align_arena_offset(arena_capacity);
                const std::uintmax_t file_bytes =
                    std::filesystem::file_size(file_name);
                if (file_bytes > std::numeric_limits<std::size_t>::max() ||
                    static_cast<std::size_t>(file_bytes) >
                        std::numeric_limits<std::size_t>::max() - arena_capacity) {
                    throw std::overflow_error("LoRA arena size overflow");
                }
                arena_capacity += static_cast<std::size_t>(file_bytes);
            }

            const std::string arena_name = fmt::format(
                "__reloc_bank{}_arena", candidate_bank
            );
            if (!has_buffer(arena_name)) {
                define_buffer(arena_name, {arena_capacity}, "int8", false);
            } else {
                const auto& existing = get_buffer(arena_name);
                if (existing.get_dtype() != "int8" ||
                    existing.get_shape().size() != 1 ||
                    existing.get_shape()[0] < arena_capacity) {
                    /*
                     * Resizing would invalidate an already imported dma-buf
                     * generation. Compiled adapters for one model must have
                     * the same hidden-port schema, so fail closed and require
                     * session reconstruction instead of leaking an old bank.
                     */
                    throw std::invalid_argument(fmt::format(
                        "Adapter {} does not fit the existing bank {}",
                        reloc_name, arena_name
                    ));
                }
            }
            auto& arena = get_buffer(arena_name);
            arena.try_allocate();
            arena.clear(false);

            std::size_t arena_cursor = 0;
            for (const auto& file_name : adapter_files) {

                std::regex pattern(R"(_n(\d+)_(pre|post)_layer(\d+)_)");
                std::smatch match;
                RelocMapType model_key;
                const std::string file_name_str = file_name.string();
                if (std::regex_search(file_name_str, match, pattern)) {
                    model_key = {
                        match[2].str(),
                        static_cast<uint16_t>(std::stoi(match[1].str())),
                        static_cast<uint8_t>(std::stoi(match[3].str()))
                    };
                } else {
                    throw std::runtime_error(fmt::format(
                        "Invalid file name for relocation: {}", file_name
                    ));
                }
                if (!_use_group_token_models && model_key.num_tokens > 1) {
                    continue;
                }

                auto tensor = cnpy::npy_load(file_name);
                if (tensor.word_size != 1) {
                    throw std::invalid_argument(fmt::format(
                        "Adapter {} is not an int8 NPY payload", file_name
                    ));
                }
                const std::string file_stem = file_name.stem().string();
                /*
                 * Adapter files are named
                 *   <qmla-elf-stem>_reloc.<hidden-port-name>.npy
                 * while the package metadata intentionally exposes only the
                 * compiler's hidden-port name (for example
                 * `reloc.filter....lora_A.weight`).  The old MLA-RT path
                 * discarded names and depended on map order, so carrying the
                 * entire filename stem appeared to work there.  The direct
                 * path validates names before publishing an immutable binding
                 * set; strip exactly the compiler-added filename prefix rather
                 * than weakening that validation or reverting to ordering.
                 */
                constexpr std::string_view reloc_marker = "_mla_reloc.";
                const std::size_t reloc_pos = file_stem.rfind(reloc_marker);
                if (reloc_pos == std::string::npos ||
                    reloc_pos + reloc_marker.size() == file_stem.size()) {
                    throw std::invalid_argument(fmt::format(
                        "Adapter filename has no hidden-port suffix: {}",
                        file_name
                    ));
                }
                const std::string port_name = file_stem.substr(
                    reloc_pos + reloc_marker.size()
                );
                arena_cursor = align_arena_offset(arena_cursor);
                const std::size_t tensor_bytes = tensor.num_bytes();
                if (tensor_bytes == 0 || arena_cursor > arena_capacity ||
                    tensor_bytes > arena_capacity - arena_cursor ||
                    arena_cursor > std::numeric_limits<std::uint32_t>::max() ||
                    tensor_bytes > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::overflow_error(fmt::format(
                        "Adapter payload does not fit {}: {}",
                        arena_name, file_name
                    ));
                }
                arena.upload(
                    tensor.data<void>(), arena_cursor, tensor_bytes, false
                );
                reloc_addr_maps[model_key][port_name] =
                    MLABufferSlice{
                        &arena,
                        {static_cast<std::uint32_t>(arena_cursor)},
                        {static_cast<std::uint32_t>(tensor_bytes)}
                    };
                arena_cursor += tensor_bytes;
            }
            /* Publish all CPU writes to the device once, not once per NPY. */
            arena.flush_cache();
            if (reloc_addr_maps.empty()) {
                throw std::invalid_argument(fmt::format(
                    "Adapter directory {} contains no applicable NPY files",
                    reloc_dir
                ));
            }

            std::vector<MLAModelWithBuffer::RelocUpdate> updates;
            updates.reserve(reloc_addr_maps.size());
            candidate_models.clear();
            for (auto& [reloc_map_type, reloc_addr_map] : reloc_addr_maps) {
                const LanguageModelMapKey model_key{
                    reloc_map_type.num_tokens, reloc_map_type.layer_idx, 0
                };
                MLAModelWithBuffer* model = nullptr;
                if (reloc_map_type.model_type == "pre") {
                    model = &_pre_model_map.at(model_key);
                } else if (reloc_map_type.model_type == "post") {
                    model = &_post_model_map.at(model_key);
                } else {
                    throw std::runtime_error(fmt::format(
                        "Relocate data for {} is not supported",
                        reloc_map_type.model_type
                    ));
                }
                candidate_models.push_back(model);
                updates.push_back({model, std::move(reloc_addr_map)});
            }
            return updates;
        }
    );

    _cached_token_ids.clear();
    _reloc_buffer_bank = candidate_bank;
    _reloc_models = std::move(candidate_models);
    _reloc_name = reloc_name;
    try {
        _rebuild_ordinary_decode_plan();
    } catch (const std::exception& error) {
        /*
         * Adapter publication is already complete and canonical generic
         * bindings are valid. A prebound optimization failure must not roll
         * back or misreport that semantic transaction; leave the plan absent
         * and use checked dynamic recipes until the next cold rebuild.
         */
        _ordinary_decode_plan.reset();
        _logger->warn(
            "Relocation kept generic MLA decode because flat-plan rebuild "
            "failed: {}",
            error.what()
        );
    }
    _logger->info("Relocation completed");
}


void LanguageModel::unset_reloc() {
    require_healthy_mla_session();
    // Unset the new weignts set by the set_reloc function.
    if (!_reloc_name.has_value()) return;

    /*
     * Publish base bindings first, then scrub both inactive adapter banks while
     * the same execution lock is still held. The old implementation merely
     * zeroed storage and left active maps installed, silently selecting a zero
     * adapter on future jobs.
     */
    _mla_model_anchor().clear_reloc_set(_reloc_models, [&]() {
        for (auto& [name, buffer] : _buf_map) {
            if (name.starts_with("__reloc_bank")) {
                buffer.clear();
            }
        }
    });

    _cached_token_ids.clear();
    _reloc_models.clear();
    _reloc_name = std::nullopt;
    try {
        _rebuild_ordinary_decode_plan();
    } catch (const std::exception& error) {
        _ordinary_decode_plan.reset();
        _logger->warn(
            "Base adapter generation kept generic MLA decode because "
            "flat-plan rebuild failed: {}",
            error.what()
        );
    }
    _logger->info("Undo relocation completed");
}


LanguageModelMap& LanguageModel::get_model_map(const std::string& model_type) {
    if (model_type == "pre") {
        return _pre_model_map;
    } else if (model_type == "cache") {
        return _cache_model_map;
    } else if (model_type == "post") {
        return _post_model_map;
    } else if (model_type == "conv") {
        return _conv_model_map;
    } else if (model_type == "conv_final") {
        return _conv_final_model_map;
    } else if (model_type == "per_layer") {
        return _per_layer_model_map;
    } else {
        throw std::runtime_error(std::string("Invalid model type: ") + model_type);
    }
}

MLAModelWithBuffer& LanguageModel::_mla_model_anchor() {
    /*
     * All entries owned by one LanguageModel are constructed while the same
     * explicit session is current and retain that session thereafter.  Pick
     * any existing entry only to reach the session-owned package operation;
     * never consult the reconnectable process-global compatibility factory.
     */
    if (!_pre_model_map.empty()) return _pre_model_map.begin()->second;
    if (!_cache_model_map.empty()) return _cache_model_map.begin()->second;
    if (!_post_model_map.empty()) return _post_model_map.begin()->second;
    if (!_conv_model_map.empty()) return _conv_model_map.begin()->second;
    if (!_conv_final_model_map.empty()) return _conv_final_model_map.begin()->second;
    if (!_per_layer_model_map.empty()) return _per_layer_model_map.begin()->second;
    if (!_fc_model_map.empty()) return _fc_model_map.begin()->second;
    throw std::logic_error("LanguageModel has no MLA model/session anchor");
}


uint16_t LanguageModel::set_max_num_tokens(std::optional<uint16_t> max_num_tokens) {
    auto original_max_num_tokens = _max_num_tokens;
    if (max_num_tokens.has_value()) {
        _max_num_tokens = std::min(max_num_tokens.value(), _cfg.pipeline_cfg.max_num_tokens);
    }
    _logger->info("Setting max_num_tokens: {} -> {}", original_max_num_tokens, _max_num_tokens);
    return original_max_num_tokens;
}


std::set<uint32_t> LanguageModel::set_stop_token_ids(
    std::optional<std::set<uint32_t>> stop_token_ids
) {
    auto original_stop_token_ids = _stop_token_ids;
    if (stop_token_ids.has_value()) {
        _stop_token_ids = stop_token_ids.value();
    }
    _logger->info(
        "Setting stop_token_ids: {} -> {}", original_stop_token_ids, _stop_token_ids
    );
    return original_stop_token_ids;
}


std::filesystem::path LanguageModel::_get_elf_path_pre(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_pre_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_cache(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    std::string cache_name = "cache";
    if (
        _cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention"
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.has_value()
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.value() != _cfg.lm_cfg.attn_cfg.head_dim
    ) {
        cache_name = "sliding_cache";
    }
    auto elf_file_name = fmt::format(
        "{}_n{}_{}_token{}_stage1_mla.elf",
        _cfg.language_model_name,
        num_tokens,
        cache_name,
        token_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_post(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_post_layer{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_conv(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_layer{}_conv_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_conv_final(uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n1_post_layer{}_conv_final_stage1_mla.elf", _cfg.language_model_name, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path LanguageModel::_get_elf_path_per_layer(uint16_t num_tokens) {
    auto elf_file_name = fmt::format(
        "{}_n{}_per_layer_stage1_mla.elf", _cfg.language_model_name, num_tokens
    );
    return _elf_dir / elf_file_name;
}


void LanguageModel::_dequantize_embedding_row(
    uint32_t token_id, MLABuffer& dst, size_t dst_row
) {
    const auto& embeddings = get_buffer("embeddings");
    const size_t hidden_size = embeddings.get_shape().back();
    const auto* src = reinterpret_cast<const int8_t*>(embeddings.get_virtual_addr())
                    + static_cast<size_t>(token_id) * hidden_size;
    auto* dst_row_ptr = reinterpret_cast<Eigen::bfloat16*>(
        reinterpret_cast<uint8_t*>(dst.get_virtual_addr())
        + dst_row * hidden_size * dst.get_elem_size()
    );
    const float scale = _cfg.pipeline_cfg.embeddings_scale.value();
    using Int8Row = Eigen::Array<int8_t, 1, Eigen::Dynamic>;
    Eigen::Map<const Int8Row> src_row(src, static_cast<Eigen::Index>(hidden_size));
    Eigen::Map<ArrayXbf> dst_row_map(dst_row_ptr, static_cast<Eigen::Index>(hidden_size));
    dst_row_map = (src_row.cast<float>() * scale).cast<Eigen::bfloat16>();
}


uint16_t LanguageModel::_set_input_text_embeds(std::span<const uint32_t> input_token_ids) {
    // Log.
    _logger->info("Cached token ids: [{}]", fmt::join(_cached_token_ids, ", "));

    const uint16_t num_input_tokens = input_token_ids.size();
    const auto& embeddings_buf = get_buffer("embeddings");
    const size_t embeddings_elem_size = embeddings_buf.get_elem_size();
    const auto& embeddings_shape = embeddings_buf.get_shape();
    const size_t embeddings_row_size = embeddings_shape.back() * embeddings_elem_size;
    const uint8_t* embeddings_ptr = (
        reinterpret_cast<const uint8_t*>(embeddings_buf.get_virtual_addr())
    );

    MLABuffer& buf = get_buffer("input_embeds");
    const size_t input_row_size = embeddings_shape.back() * buf.get_elem_size();
    const bool dequantize = embeddings_buf.get_dtype() == "int8"
        && buf.get_dtype() == "bfloat16";
    uint32_t token_idx = 0;
    uint32_t num_images = 0;
    uint16_t num_cached_tokens = 0;
    while(token_idx < num_input_tokens) {
        const auto& token_id = input_token_ids[token_idx];
        if (_image_token_id.has_value() && token_id == _image_token_id.value()) {
            auto next_token_idx = token_idx + _cfg.mm_cfg.value().mm_tokens_per_image;
            ++num_images;
            token_idx = next_token_idx;
        } else {
            auto next_token_idx = token_idx + 1;
            if (dequantize) {
                _dequantize_embedding_row(token_id, buf, token_idx);
            } else {
                buf.upload(
                    embeddings_ptr + token_id * embeddings_row_size,
                    token_idx * input_row_size,
                    input_row_size,
                    false
                );
            }
            if (
                num_images == 0
                && token_idx < _cached_token_ids.size()
                && token_id == _cached_token_ids[token_idx]
            )
                ++num_cached_tokens;
            token_idx = next_token_idx;
        }
    }
    buf.flush_cache();

    // Update rope table if needed.
    if (
        _cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "mrope"
        && (_has_image_token || num_images > 0)
    ) {
        // Previous or current input has image.
        _logger->info("Update rope table");
        auto rope_table = calc_mrope_with_image(
            _cfg, _master_rope_table, _image_token_id.value(), input_token_ids
        );
        get_buffer("global_freq_real").upload(rope_table.re.data());
        get_buffer("global_freq_imag").upload(rope_table.im.data());
    }
    _has_image_token = num_images > 0;

    _logger->info("Number of tokens cached: {:d}", num_cached_tokens);
    return num_cached_tokens;
}


std::vector<uint32_t> LanguageModel::_get_per_layer_token_ids(
    std::span<const uint32_t> input_token_ids
) const {
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


void LanguageModel::_load_per_layer_embeddings() {
    const auto file_name = (
        _devkit_dir / (_cfg.language_model_name + "_per_layer_embeddings.bin")
    );
    const size_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
    const size_t out_dim = static_cast<size_t>(_cfg.lm_cfg.num_hidden_layers)
                         * _cfg.lm_cfg.hidden_size_per_layer_input;
    const size_t elem_size = _cfg.pipeline_cfg.quantize_embeddings
        ? sizeof(int8_t) : sizeof(Eigen::bfloat16);
    const size_t token_row_size = out_dim * elem_size;
    const size_t expected_size = vocab_size * token_row_size;
    const size_t actual_size = std::filesystem::file_size(file_name);
    if (actual_size != expected_size) {
        throw std::runtime_error(fmt::format(
            "Invalid size for {}: expected {} bytes, got {}",
            file_name, expected_size, actual_size
        ));
    }

    // Stream contiguous token rows directly into MLA shards to avoid a full host copy.
    std::ifstream stream(file_name, std::ios::binary);
    for (auto* shard : _per_layer_embedding_shards) {
        if (shard->get_buf_len() != shard->get_shape()[0] * token_row_size) {
            throw std::runtime_error(fmt::format(
                "Per-layer embedding rows require unsupported MLA padding: {}",
                shard->get_name()
            ));
        }
        shard->load_stream(stream);
    }
}


void LanguageModel::_upload_per_layer_embedding_rows(
    std::span<const uint32_t> token_ids, uint16_t num_tokens
) {
    if (!_uses_per_layer_inputs())
        return;
    auto& staging = get_buffer(fmt::format("per_layer_emb_staging_n{}", num_tokens));
    const size_t row_size = staging.get_shape().back() * staging.get_elem_size();
    auto* dst = reinterpret_cast<uint8_t*>(staging.get_virtual_addr());
    for (uint16_t i = 0; i < num_tokens; ++i) {
        const size_t shard_idx = token_ids[i] / _per_layer_embedding_rows_per_shard;
        const size_t row_in_shard = token_ids[i] % _per_layer_embedding_rows_per_shard;
        const auto* src = reinterpret_cast<const uint8_t*>(
            _per_layer_embedding_shards[shard_idx]->get_virtual_addr()
        ) + row_in_shard * row_size;
        std::memcpy(dst + i * row_size, src, row_size);
    }
    staging.flush_cache();
}


void LanguageModel::_compute_and_upload_per_layer_inputs_prefill(
    uint16_t num_tokens, uint16_t token_idx, uint16_t num_input_tokens
) {
    std::vector<uint32_t> token_ids(num_tokens, 0);
    const auto num_prompt_tokens = std::min<uint16_t>(num_tokens, num_input_tokens - token_idx);
    for (uint16_t i = 0; i < num_prompt_tokens; ++i) {
        token_ids[i] = _prompt_per_layer_token_ids[token_idx + i];
    }
    _upload_per_layer_embedding_rows(token_ids, num_tokens);
}


uint32_t LanguageModel::_calc_next_token_id(MLABuffer* buf_ptr) {
    Eigen::bfloat16* ptr = (Eigen::bfloat16*)buf_ptr->get_virtual_addr();
    Eigen::bfloat16 max_val = ptr[0];
    uint32_t max_index = 0;
    #pragma omp parallel
    {
        Eigen::bfloat16 thread_max_val = max_val;
        uint32_t thread_max_index = max_index;

        #pragma omp for nowait
        for (uint32_t i = 0; i < _cfg.lm_cfg.token_cfg.vocab_size; ++i) {
            if (ptr[i] > thread_max_val) {
                thread_max_val = ptr[i];
                thread_max_index = i;
            }
        }

        #pragma omp critical
        {
            if (thread_max_val > max_val) {
                max_val = thread_max_val;
                max_index = thread_max_index;
            } else if ((thread_max_val == max_val) && (thread_max_index < max_index)) {
                max_index = thread_max_index;
            }
        }
    }
    return max_index;
}


void LanguageModel::_notify_first_token(uint32_t token_id, double duration) {
    _logger->info("Time to the first token: {:d} in {:.5f}s", token_id, duration);
    _text_streamer.push(DecodeCallbackType::TTFT, token_id, duration);
}


void LanguageModel::_notify_new_token(uint32_t token_id, double duration) {
    _logger->info("Got token: {:d} in {:.5f}s", token_id, duration);
    _text_streamer.push(DecodeCallbackType::TPS, token_id, duration);
}


void LanguageModel::_notify_cache_full() const {
    // Use negative token id to indicate that the cache is full.
    _logger->info("Cache full");
    // _py_decode_callback_func(-1, 0);
    _text_streamer.push(DecodeCallbackType::CACHE_FULL, 0, 0);
}


void LanguageModel::_notify_stop() const {
    _logger->info("Got stop token");
    _text_streamer.push(DecodeCallbackType::STOP, 0, 0);
}


void LanguageModel::_notify_interrupt() const {
    _logger->info("Interrupted by user");
    _text_streamer.push(DecodeCallbackType::STOP, 0, 0);
}


}
}
