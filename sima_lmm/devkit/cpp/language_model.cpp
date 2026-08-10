#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <regex>
#include <set>
#include <stdexcept>

#include <Eigen/Dense>
#include <cnpy.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "eagle_helpers.hpp"
#include "language_model.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

namespace {
constexpr size_t PER_LAYER_EMBEDDING_MAX_SHARD_SIZE = 1024ULL * 1024 * 1024;
}

LanguageModel::LanguageModel(
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer
) : BaseModel(model_path),
    _stop_token_ids(std::move(stop_token_ids)),
    _image_token_id(image_token_id),
    _pad_token_id(pad_token_id),
    _max_num_tokens(_cfg.pipeline_cfg.max_num_tokens),
    _text_streamer(text_streamer),
    _has_image_token(false),
    _is_running(false),
    _reloc_name(std::nullopt)
{
    if (
        _cfg.pipeline_cfg.max_num_tokens == 0
        || _cfg.pipeline_cfg.max_num_tokens % MAX_NUM_TOKENS_ALIGNMENT
    ) {
        throw std::runtime_error(
            "max_num_tokens must be a positive multiple of 1024"
        );
    }
    if (_cfg.lm_cfg.is_spec_decode() && _cfg.lm_cfg.attn_cfg.swa_enable) {
        throw std::runtime_error(
            "EAGLE3 speculative decoding does not support sliding-window attention"
        );
    }

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
        std::vector<uint8_t> linear_layer_indices;
        for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
            if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
                conv_layer_indices.emplace_back(layer_idx);
            } else if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
                linear_layer_indices.emplace_back(layer_idx);
            }
        }

        // Build _checkpoint_boundaries once if ANY stateful layer family exists.
        if (!conv_layer_indices.empty() || !linear_layer_indices.empty()) {
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
        if (!linear_layer_indices.empty()) {
            const auto& linear_cfg = _linear_attn_cfg();
            LanguageModel::CachedState linear_conv_state;
            linear_conv_state.buffer_name_prefix = "linear_conv_cache_history_l";
            linear_conv_state.tail_len = static_cast<uint16_t>(linear_cfg.conv_kernel_dim - 1);
            linear_conv_state.num_elems = linear_cfg.get_conv_dim();
            linear_conv_state.elem_size = sizeof(Eigen::bfloat16);
            linear_conv_state.tail_bytes = static_cast<size_t>(
                linear_conv_state.tail_len
                * linear_conv_state.num_elems
                * linear_conv_state.elem_size
            );
            linear_conv_state.layer_indices = linear_layer_indices;

            const auto num_boundaries = _checkpoint_boundaries.size();
            linear_conv_state.checkpoints.resize(linear_conv_state.layer_indices.size());
            for (auto& checkpoints: linear_conv_state.checkpoints) {
                checkpoints.resize(
                    num_boundaries, std::vector<uint8_t>(linear_conv_state.tail_bytes)
                );
            }
            _cached_states.emplace_back(std::move(linear_conv_state));

            LanguageModel::CachedState linear_delta_state;
            linear_delta_state.buffer_name_prefix = "linear_delta_state_history_l";
            linear_delta_state.tail_len = 1;
            linear_delta_state.num_elems = linear_cfg.get_recurrent_state_size();
            linear_delta_state.elem_size = sizeof(Eigen::bfloat16);
            linear_delta_state.prefill_single_output = true;
            linear_delta_state.tail_bytes = static_cast<size_t>(
                linear_delta_state.tail_len
                * linear_delta_state.num_elems
                * linear_delta_state.elem_size
            );
            linear_delta_state.layer_indices = std::move(linear_layer_indices);

            linear_delta_state.checkpoints.resize(linear_delta_state.layer_indices.size());
            for (auto& checkpoints: linear_delta_state.checkpoints) {
                checkpoints.resize(
                    num_boundaries, std::vector<uint8_t>(linear_delta_state.tail_bytes)
                );
            }
            _cached_states.emplace_back(std::move(linear_delta_state));
        }
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
                _kv_cache_len = static_cast<uint16_t>(_cached_token_ids.size());
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
                _kv_cache_len = static_cast<uint16_t>(_cached_token_ids.size());
                return next_token_id;
            }
        }
    }
    _cached_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
    _kv_cache_len = static_cast<uint16_t>(_cached_token_ids.size());
    auto duration = timer_ttft.value().stop();
    _notify_first_token(next_token_id, duration);
    _cached_first_generated_token = next_token_id;
    return next_token_id;
}


void LanguageModel::run_model_decode(
    uint16_t num_input_tokens, uint32_t token_id
) {
    ChronoTimer timer_tps(true);
    uint16_t token_idx;
    for (token_idx = num_input_tokens; token_idx < _max_num_tokens; ++token_idx ) {
        auto next_token_id = run_model_once(1, token_idx, num_input_tokens, token_id);
        _cached_token_ids.emplace_back(token_id);
        _kv_cache_len = static_cast<uint16_t>(_cached_token_ids.size());
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
        get_buffer(name).upload(mask.data(), 0, mask.size() * sizeof(Eigen::bfloat16));
    };

    upload_group_mask(
        "group_future_token_mask", "full_attention", 0
    );
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        const uint16_t cache_token_idx_begin = std::max(
            0,
            token_idx + num_tokens
                - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
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
    uint32_t normal_input_row;
    uint32_t normal_input_num_tokens;
    if (use_input_tokens) {
        normal_input_buf = &get_buffer("input_embeds");
        normal_input_row = token_idx;
        normal_input_num_tokens = num_tokens;
    } else if (_uses_cpu_dequantized_embeddings()) {
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
        _per_layer_model_map.at(per_layer_key).add_to_queue(&per_layer_ifm_map);
    }

    if (num_tokens > 1 && _has_linear_attention_layers()) {
        const uint16_t valid_tokens = next_token_idx - token_idx;
        std::vector<Eigen::bfloat16> valid_mask(num_tokens, Eigen::bfloat16(0.0f));
        std::fill_n(valid_mask.begin(), valid_tokens, Eigen::bfloat16(1.0f));
        get_buffer("linear_valid_mask").upload(valid_mask.data());
    }

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
            _pre_model_map.at(model_key).add_to_queue(&ifm_map);

            if (
                num_input_tokens > next_token_idx
                && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1
            ) {
                break;
            }

            _cache_model_map.at(model_key).add_to_queue();

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
            _post_model_map.at(model_key).add_to_queue(&ifm_map);

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
                    MLAModelWithBuffer::run_queue();
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
            _conv_model_map.at(conv_model_key).add_to_queue(&ifm_map);

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
            _conv_final_model_map.at(conv_model_key).add_to_queue(&ifm_map);
        } else if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
            LanguageModelMapKey linear_model_key(num_tokens, layer_idx, 0);
            _linear_model_map.at(linear_model_key).add_to_queue(&ifm_map);
        } else {
            throw std::runtime_error(
                std::string("Unsupported layer type: ") + _cfg.lm_cfg.layer_types[layer_idx]
            );
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
            buf_ptr->invalidate_cache();
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
    MLAModelWithBuffer::load_all_models(_elf_dir / _cfg.language_model_name);

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
    const uint16_t max_future_token_mask_size = _get_max_future_token_mask_size();
    if (max_future_token_mask_size > 1) {
        std::vector<Eigen::bfloat16> future_token_mask(
            max_future_token_mask_size - 1,
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
        } else if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
            get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx)).clear();
            get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)).clear();
            get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx)).clear();
        } else if (!_cfg.lm_cfg.is_kv_shared_layer(layer_idx)) {
            get_buffer(fmt::format("cache_key_l{}", layer_idx)).clear();
            get_buffer(fmt::format("cache_val_l{}", layer_idx)).clear();
            if (_cfg.pipeline_cfg.quantize_kv_cache) {
                get_buffer(fmt::format("cache_key_scale_l{}", layer_idx)).clear();
                get_buffer(fmt::format("cache_val_scale_l{}", layer_idx)).clear();
            }
        }
    }

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
        const uint32_t restore_row_offset = state.prefill_single_output ? 0 : tail_begin;
        const size_t dst_offset = static_cast<size_t>(
            restore_row_offset * state.num_elems * state.elem_size
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
    const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;

    for (auto& state: _cached_states) {
        const uint32_t tail_row_offset = state.prefill_single_output
            ? 0
            : (
                num_tokens > 1
                    ? static_cast<uint32_t>(valid_tokens - 1)
                    : tail_begin
            );
        const size_t src_offset_bytes = tail_row_offset * state.num_elems * state.elem_size;
        for (size_t layer_slot = 0; layer_slot < state.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layer_indices[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.buffer_name_prefix, layer_idx));
            buf.invalidate_cache();
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
    for (auto& state: _cached_states) {
        if (state.prefill_single_output) {
            continue;
        }
        const size_t src_offset_bytes = (valid_tokens - 1) * state.num_elems * state.elem_size;
        const size_t dst_offset_bytes = tail_begin * state.num_elems * state.elem_size;
        for (size_t layer_slot = 0; layer_slot < state.layer_indices.size(); ++layer_slot) {
            auto layer_idx = state.layer_indices[layer_slot];
            auto& buf = get_buffer(fmt::format("{}{}", state.buffer_name_prefix, layer_idx));
            buf.invalidate_cache();
            auto* ptr = reinterpret_cast<uint8_t*>(buf.get_virtual_addr());
            std::memmove(ptr + dst_offset_bytes, ptr + src_offset_bytes, state.tail_bytes);
            buf.flush_cache();
        }
    }
}


void LanguageModel::_finalize() {
    _logger->info("Language model finalize starting ...");
    MLAModelWithBuffer::free_all_models(_elf_dir / _cfg.language_model_name);
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
        if (_uses_cpu_dequantized_embeddings()) {
            define_buffer("decode_embedding", {1, _cfg.lm_cfg.hidden_size});
        }
    }

    // Frequency tables.
    _define_buffer_freq_table(
        "global", _cfg.lm_cfg.rope_cfg.get_rope_dimension_count("full_attention")
    );
    if (_cfg.lm_cfg.attn_cfg.swa_enable)
        _define_buffer_freq_table(
            "local", _cfg.lm_cfg.rope_cfg.get_rope_dimension_count("sliding_attention")
        );

    if (_use_group_token_models && _has_linear_attention_layers()) {
        define_buffer(
            "linear_valid_mask",
            {_cfg.pipeline_cfg.input_token_group_size, 1},
            "bfloat16",
            false
        );
    }

    // KV caches.
    std::vector<size_t> cache_shape;
    const uint16_t conv_working_len = _cfg.pipeline_cfg.input_token_group_size + _cfg.lm_cfg.conv_L_cache - 2;
    std::vector<size_t> conv_cache_shape{conv_working_len, _cfg.lm_cfg.hidden_size};
    for (uint8_t i = 0; i < _cfg.lm_cfg.num_hidden_layers; ++i) {
        if (_cfg.lm_cfg.layer_types[i] == "conv") {
            define_buffer(fmt::format("conv_cache_history_l{}", i), conv_cache_shape);
        } else if (_cfg.lm_cfg.layer_types[i] == "linear_attention") {
            const auto& linear_cfg = _linear_attn_cfg();
            const uint16_t linear_conv_working_len = static_cast<uint16_t>(
                _cfg.pipeline_cfg.input_token_group_size + linear_cfg.conv_kernel_dim - 2
            );
            define_buffer(
                fmt::format("linear_conv_cache_history_l{}", i),
                {linear_conv_working_len, linear_cfg.get_conv_dim()}
            );
            define_buffer(
                fmt::format("linear_delta_state_history_l{}", i),
                {
                    1,
                    linear_cfg.get_recurrent_state_size()
                }
            );
            // Ping-pong state avoids MLA read-after-write corruption during grouped prefill.
            define_buffer(
                fmt::format("linear_delta_state_history_alt_l{}", i),
                {1, linear_cfg.get_recurrent_state_size()}
            );
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
    const uint16_t max_future_token_mask_size = _get_max_future_token_mask_size();
    if (max_future_token_mask_size > 1) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            define_buffer(
                "future_token_mask",
                {_cfg.lm_cfg.get_single_num_tokens(), _cfg.pipeline_cfg.max_num_tokens}
            );
        } else {
            uint32_t full_token_mask_size = round_up_to_row(
                _cfg.pipeline_cfg.max_num_tokens + max_future_token_mask_size - 1
            );
            define_buffer(
                "future_token_mask",
                {full_token_mask_size}
            );
        }
    }
    const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
    if (
        _use_group_token_models
        && _get_cache_mask_size(
            "full_attention", _cfg.pipeline_cfg.max_num_tokens, true
        ) > group_size
    ) {
        define_buffer(
            "group_future_token_mask",
            {
                static_cast<size_t>(group_size) * _cfg.pipeline_cfg.max_num_tokens
            },
            "bfloat16",
            false
        );
    }
    if (
        _use_group_token_models
        && _cfg.lm_cfg.attn_cfg.swa_enable
        && _get_cache_mask_size(
            "sliding_attention", _cfg.pipeline_cfg.max_num_tokens, true
        ) > group_size
    ) {
        define_buffer(
            "group_sliding_future_token_mask",
            {static_cast<size_t>(group_size) * _cfg.pipeline_cfg.max_num_tokens},
            "bfloat16",
            false
        );
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

        // Qwen3.5: transient gate buffer, written by Pre and consumed by Post within the same layer.
        if (_cfg.lm_cfg.attn_cfg.attn_output_gate) {
            define_buffer(
                fmt::format("n{}_buffer_gate", num_tokens),
                {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size("full_attention")}
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
    map.emplace(key, MLAModelWithBuffer(model_path, ifms, ofms));
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
        pre_ifms.emplace_back(MLABufferSlice{});
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
    if (_cfg.lm_cfg.attn_cfg.attn_output_gate) {
        pre_ofms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer_gate", num_tokens))}
        );
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
    const bool is_single_model = num_tokens == single_num_tokens;
    const uint16_t sliding_window = _cfg.lm_cfg.attn_cfg.sliding_window.value_or(0);
    const bool separate_sliding_cache = (
        layer_type == "sliding_attention"
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.has_value()
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.value() != _cfg.lm_cfg.attn_cfg.head_dim
    );
    const bool sliding_cache_mask_differs = (
        layer_type == "sliding_attention"
        && !separate_sliding_cache
        && (
            _get_cache_mask_size("full_attention", sliding_window, true)
                != _get_cache_mask_size("sliding_attention", sliding_window, true)
            || _get_cache_mask_size("full_attention", sliding_window, false)
                != _get_cache_mask_size("sliding_attention", sliding_window, false)
        )
    );
    const bool use_sliding_cache = (
        separate_sliding_cache
        || (
            sliding_cache_mask_differs
            && eff_num_cached_tokens >= sliding_window
        )
    );
    const std::string cache_layer_type = (
        use_sliding_cache ? "sliding_attention" : "full_attention"
    );
    const uint16_t cache_mask_size = _get_cache_mask_size(
        cache_layer_type, eff_num_cached_tokens, !is_single_model
    );
    const bool use_single_future_token_mask = (
        is_single_model && cache_mask_size > 1
    );
    const bool use_group_future_token_mask = (
        !is_single_model
        && cache_mask_size > num_tokens
    );
    if (use_single_future_token_mask) {
        // Round up by total context (eff_num_cached_tokens = past_kv + num_tokens),
        // not by past_kv alone. For spec mode num_tokens=16: past_kv=114 ->
        // total=130 -> bucket=256 (cache_token_255). The old formula used
        // eff_token_idx + 1, which only worked when num_tokens was 1 (non-spec).
        // For spec mode it picked cache_token_127 even when the 16 new tokens
        // wouldn't fit, dropping mask data for the tail tree positions.
        aligned_eff_token_idx = std::min(
            round_up_to(eff_num_cached_tokens, cache_mask_size) - 1,
            _cfg.pipeline_cfg.max_num_tokens - 1
        );
        aligned_eff_num_cached_tokens = aligned_eff_token_idx + 1;
    } else if (use_group_future_token_mask) {
        aligned_eff_num_cached_tokens = std::min<uint16_t>(
            round_up_to(eff_num_cached_tokens, cache_mask_size),
            _cfg.pipeline_cfg.max_num_tokens
        );
        aligned_eff_token_idx = aligned_eff_num_cached_tokens - num_tokens;
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

    if (use_group_future_token_mask) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer(
                    cache_layer_type == "sliding_attention"
                        ? "group_sliding_future_token_mask"
                        : "group_future_token_mask"
                ),
                {0},
                {static_cast<uint32_t>(num_tokens) * aligned_eff_num_cached_tokens}
            }
        );
    } else if (use_single_future_token_mask) {
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
        _get_elf_path_cache(num_tokens, aligned_eff_token_idx, use_sliding_cache),
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
    if (_cfg.lm_cfg.attn_cfg.attn_output_gate) {
        post_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer_gate", num_tokens))}
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
    } else if (_cfg.lm_cfg.layer_types[layer_idx] == "linear_attention") {
        _define_linear_models_iter(num_tokens, layer_idx);
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
        conv_ifms.emplace_back(MLABufferSlice{});
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


void LanguageModel::_define_linear_models_iter(uint16_t num_tokens, uint8_t layer_idx) {
    const auto& linear_cfg = _linear_attn_cfg();
    LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    const uint16_t conv_tail_size = static_cast<uint16_t>(linear_cfg.conv_kernel_dim - 1);
    const uint16_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;

    std::vector<MLABufferSlice> linear_ifms;
    std::vector<MLABufferSlice> linear_ofms;
    if (layer_idx) {
        linear_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
    } else {
        linear_ifms.emplace_back(MLABufferSlice{});
    }
    linear_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx)),
            {tail_begin, 0},
            {conv_tail_size, linear_cfg.get_conv_dim()}
        }
    );
    if (num_tokens > 1) {
        linear_ifms.emplace_back(MLABufferSlice{&get_buffer("linear_valid_mask")});
    }
    linear_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)),
            {0, 0},
            {1, linear_cfg.get_recurrent_state_size()}
        }
    );

    linear_ofms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))});
    linear_ofms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx)),
            {num_tokens > 1 ? 0 : tail_begin, 0},
            {
                static_cast<uint32_t>(num_tokens + conv_tail_size - 1),
                linear_cfg.get_conv_dim()
            }
        }
    );
    linear_ofms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx)),
            {0, 0},
            {1, linear_cfg.get_recurrent_state_size()}
        }
    );
    _define_model(
        "linear",
        model_key,
        _get_elf_path_linear(num_tokens, layer_idx),
        linear_ifms,
        linear_ofms
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
            num_tokens, MLAModelWithBuffer(elf_path, ifms, ofms)
        );
    }
}


void LanguageModel::compact_kv_after_accept(
    std::span<const uint16_t> select_indices, uint16_t prev_input_len
) {
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
    // Swap the model weights with the data from the `{_devkit_dir}/../npy_files/{reloc_name}`.
    // Instead of overwrite the dram space allocated by the mla-rt, allocate new buffers populated
    // with the data read from the npy files and relocates the dram addresses of the dma
    // descriptors so that the models use the weights from the new buffers.

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
    using RelocMap = std::map<std::string, uint64_t>;

    // Allocate the memory for the new content to be relocated and collect the maps of the addresses
    // to relocate the models' dma descriptors.
    auto reloc_dir = _devkit_dir / "../npy_files" / reloc_name;
    _logger->info("Relocation with data from: {}", reloc_dir);
    if (!std::filesystem::is_directory(reloc_dir)) {
        throw std::filesystem::filesystem_error(
            "Relocation directory does not exist", reloc_dir,
            std::make_error_code(std::errc::no_such_file_or_directory)
        );
    }
    std::map<RelocMapType, RelocMap> reloc_addr_maps;
    for (const auto& dir_entry: std::filesystem::directory_iterator(reloc_dir)) {
        auto file_name = dir_entry.path();
        auto file_name_str = file_name.string();
        if (file_name.extension() != ".npy")
            continue;

        // Extract the num_tokens and layer_idx from the file name.
        std::regex pattern(R"(_n(\d+)_(pre|post)_layer(\d+)_)");
        std::smatch match;
        RelocMapType model_key;
        if (std::regex_search(file_name_str, match, pattern)) {
            uint16_t num_tokens = std::stoi(match[1].str());
            std::string model_type = match[2].str();
            uint8_t layer_idx = std::stoi(match[3].str());
            model_key = {model_type, num_tokens, layer_idx};
        } else {
            auto msg = fmt::format("Invalid file name for relocation: {}", file_name);
            throw std::runtime_error(msg);
        }

        // Skip the file for group token models if not used.
        if (!_use_group_token_models && model_key.num_tokens > 1)
            continue;

        // Upload the tensor to the buffer.
        auto tensor = cnpy::npy_load(file_name);
        auto buffer_name = file_name.stem();

        // If the buffer does not exist, define and allocate it first.
        if (!has_buffer(buffer_name)) {
            define_buffer(buffer_name, tensor.shape, "int8");
        }
        auto& buf = get_buffer(buffer_name);
        buf.try_allocate();

        // Upload the tensor.
        buf.upload(tensor.data<void>());

        // Append the reloc addr map.
        reloc_addr_maps[model_key][buffer_name] = buf.get_buf_addr();
    }

    // Relocation the dma descriptors
    for (const auto& [reloc_map_type, reloc_addr_map]: reloc_addr_maps) {
        // For each pre/post model, we only need to relocate the unique model once.
        LanguageModelMapKey model_key{reloc_map_type.num_tokens, reloc_map_type.layer_idx, 0};

        MLAModelWithBuffer* model_ptr;
        if (reloc_map_type.model_type == "pre") {
            model_ptr = &_pre_model_map.at(model_key);
        } else if (reloc_map_type.model_type == "post") {
            model_ptr = &_post_model_map.at(model_key);
        } else {
            auto msg = fmt::format(
                "Relocate data for {} is not supported", reloc_map_type.model_type
            );
            throw std::runtime_error(msg);
        }
        model_ptr->update_reloc(reloc_addr_map);
    }

    _cached_token_ids.clear();
    _reloc_name = reloc_name;
    _logger->info("Relocation completed");
}


void LanguageModel::unset_reloc() {
    // Unset the new weignts set by the set_reloc function.
    if (!_reloc_name.has_value()) return;

    // Find the buffers with `reloc` in the name and unset the buffer data.
    for (auto& [name, buffer]: _buf_map) {
        if (name.find("reloc") == std::string::npos) continue;

        // Zero out the buffer data.
        buffer.clear();
    }

    _cached_token_ids.clear();
    _reloc_name = std::nullopt;
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
    } else if (model_type == "linear") {
        return _linear_model_map;
    } else {
        throw std::runtime_error(std::string("Invalid model type: ") + model_type);
    }
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
    uint16_t num_tokens,
    uint16_t token_idx,
    bool use_sliding_cache
) {
    const std::string cache_name = use_sliding_cache ? "sliding_cache" : "cache";
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


std::filesystem::path LanguageModel::_get_elf_path_linear(uint16_t num_tokens, uint8_t layer_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_layer{}_linear_stage1_mla.elf", _cfg.language_model_name, num_tokens, layer_idx
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
    bool cache_prefix_matches = true;
    while (token_idx < num_input_tokens) {
        const auto& token_id = input_token_ids[token_idx];
        if (_image_token_id.has_value() && token_id == _image_token_id.value()) {
            auto next_token_idx = token_idx + _cfg.mm_cfg.value().mm_tokens_per_image;
            ++num_images;
            cache_prefix_matches = false;
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
                && cache_prefix_matches
                && token_idx < _cached_token_ids.size()
                && token_id == _cached_token_ids[token_idx]
            ) {
                ++num_cached_tokens;
            } else {
                cache_prefix_matches = false;
            }
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
