//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#include <regex>

#include <Eigen/Dense>
#include <cnpy.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "language_model.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

LanguageModel::LanguageModel(
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    TextStreamer& text_streamer,
    bool do_parallel_load
) : BaseModel(model_path),
    _stop_token_ids(std::move(stop_token_ids)),
    _image_token_id(image_token_id),
    _max_num_tokens(_cfg.pipeline_cfg.max_num_tokens),
    _text_streamer(text_streamer),
    _do_parallel_load(do_parallel_load),
    _has_image_token(false),
    _is_running(false),
    _reloc_name(std::nullopt)
{
    _use_group_token_models = (
        _cfg.pipeline_cfg.input_token_group_offsets.has_value()
        && _cfg.pipeline_cfg.input_token_group_offsets.value().size() > 0
    );
    _need_argmax = _cfg.lm_cfg.lm_head_num_splits > 1 || _cfg.pipeline_cfg.return_logits;
    
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
    _initialize();
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
        embeddings_buf.get_dtype()
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
    if (!timer_ttft.has_value())
        timer_ttft = ChronoTimer{true};
    const uint16_t num_input_tokens = input_token_ids.size();
    uint16_t token_idx{};
    uint32_t next_token_id{};

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
            token_idx = num_cached_tokens;
            break;
        }

        while (token_idx < num_input_tokens) {
            if (it != offsets.end()) {
                assert(token_idx >= *it && token_idx < *it + num_tokens);
                token_idx = *it;
                next_token_id = run_model_once(num_tokens, token_idx, num_input_tokens, 0);
                ++it;
                token_idx += num_tokens;
            } else {
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
    } else {
        for (token_idx = num_cached_tokens; token_idx < num_input_tokens; ++token_idx) {
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
    uint16_t next_token_idx;
    if (num_tokens > 1) {
        next_token_idx = std::min(num_input_tokens, uint16_t(token_idx + num_tokens));
    } else {
        next_token_idx = token_idx + 1;
    }
    auto use_input_tokens = token_idx < num_input_tokens;
    _logger->info("Processing token no. {}-{}", token_idx, next_token_idx);

    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        LanguageModelMapKey model_key(num_tokens, layer_idx, token_idx);

        std::map<uint8_t, MLABufferSlice> ifm_map;
        if (layer_idx == 0) {
            if (use_input_tokens) {
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        &get_buffer("input_embeds"),
                        std::vector<uint32_t>{token_idx, 0},
                        std::vector<uint32_t>{num_tokens, _cfg.lm_cfg.hidden_size}
                    )
                );
            } else {
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        &get_buffer("embeddings"),
                        std::vector<uint32_t>{token_id, 0},
                        std::vector<uint32_t>{1, _cfg.lm_cfg.hidden_size}
                    )
                );
            }
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

            if (num_tokens > 1 && layer_idx == _cfg.lm_cfg.num_hidden_layers - 1) {
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
                        std::vector<uint32_t>{1, _cfg.lm_cfg.attn_cfg.get_q_size()}
                    )
                );
            }
            _post_model_map.at(model_key).add_to_queue(&ifm_map);
        } else if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
            _conv_model_map.at(model_key).add_to_queue(&ifm_map);

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
            _conv_final_model_map.at(model_key).add_to_queue(&ifm_map);
        } else {
            throw std::runtime_error(
                std::string("Unsupported layer type: ") + _cfg.lm_cfg.layer_types[layer_idx]
            );
        }
    }

    // Run all the queued models.
    MLAModelWithBuffer::run_queue();

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
        MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
        buf_ptr->invalidate_cache();

        if (_need_argmax) {
            next_token_id = _calc_next_token_id(buf_ptr);
        } else {
            uint32_t* ptr = (uint32_t*)buf_ptr->get_virtual_addr();
            next_token_id = ptr[0];
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
    MLAModelWithBuffer::load_all_models(_do_parallel_load, _elf_dir / _cfg.language_model_name);

    // Upload language embeddings.
    auto embeddings_file_name = _devkit_dir / (_cfg.language_model_name + "_embeddings.npy");
    auto embeddings_tensor = cnpy::npy_load(embeddings_file_name);
    get_buffer("embeddings").upload(embeddings_tensor.data<void>());

    // Upload freq real and imag.
    auto rope_table = calc_freq_real_imag(
        _cfg.pipeline_cfg.max_num_tokens,
        _cfg.lm_cfg.rope_cfg.rope_scaling.rope_type,
        _cfg.lm_cfg.rope_cfg.rope_theta,
        _cfg.lm_cfg.rope_cfg.rope_dimension_count,
        _cfg.lm_cfg.rope_cfg.rope_scaling
    );
    get_buffer("global_freq_real").upload(rope_table.re.data());
    get_buffer("global_freq_imag").upload(rope_table.im.data());
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        rope_table = calc_freq_real_imag(
            _cfg.pipeline_cfg.max_num_tokens,
            "default",
            _cfg.lm_cfg.rope_cfg.rope_local_base_freq,
            _cfg.lm_cfg.rope_cfg.rope_dimension_count,
            _cfg.lm_cfg.rope_cfg.rope_scaling
        );
        get_buffer("local_freq_real").upload(rope_table.re.data());
        get_buffer("local_freq_imag").upload(rope_table.im.data());
    }
    if (_cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "mrope") {
        assert(!_cfg.lm_cfg.attn_cfg.swa_enable);
        _master_rope_table = std::move(rope_table);
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

    // Clear the KV caches.
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        if (_cfg.lm_cfg.layer_types[layer_idx] == "conv") {
            get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)).clear();
        } else {
            get_buffer(fmt::format("cache_key_l{}", layer_idx)).clear();
            get_buffer(fmt::format("cache_val_l{}", layer_idx)).clear();
        }
    }

    _logger->info("Language model initialize completed");
}


void LanguageModel::_finalize() {
    _logger->info("Language model finalize starting ...");
    MLAModelWithBuffer::free_all_models(_elf_dir / _cfg.language_model_name);
    BaseModel::_finalize();
    _logger->info("Language model finalize completed");
}


void LanguageModel::_define_buffer_freq_table(const std::string& name) {
    define_buffer(
        fmt::format("{}_freq_real", name),
        {_cfg.pipeline_cfg.max_num_tokens, _cfg.lm_cfg.rope_cfg.rope_dimension_count / 2}
    );
    define_buffer(
        fmt::format("{}_freq_imag", name),
        {_cfg.pipeline_cfg.max_num_tokens, _cfg.lm_cfg.rope_cfg.rope_dimension_count / 2}
    );
}


void LanguageModel::_define_buffers() {
    // Embedding table.
    std::string embeddings_dtype = (_cfg.pipeline_cfg.quantize_embeddings)? "int8" : "bfloat16";
    define_buffer(
        "embeddings",
        {_cfg.lm_cfg.token_cfg.vocab_size, _cfg.lm_cfg.hidden_size},
        (_cfg.pipeline_cfg.quantize_embeddings)? "int8" : "bfloat16"
    );

    // Frequency tables.
    _define_buffer_freq_table("global");
    if (_cfg.lm_cfg.attn_cfg.swa_enable)
        _define_buffer_freq_table("local");

    // KV caches.
    std::vector<size_t> cache_shape;
    if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        cache_shape = {
            _cfg.lm_cfg.attn_cfg.num_key_value_heads,
            _cfg.pipeline_cfg.max_num_tokens,
            _cfg.lm_cfg.attn_cfg.head_dim
        };
    } else {
        cache_shape = {
            _cfg.pipeline_cfg.max_num_tokens,
            _cfg.lm_cfg.attn_cfg.get_kv_size()
        };
    }
    std::vector<size_t> conv_cache_shape{
        _cfg.pipeline_cfg.max_num_tokens + _cfg.lm_cfg.conv_L_cache - 1,
        _cfg.lm_cfg.hidden_size
    };
    for (uint8_t i = 0; i < _cfg.lm_cfg.num_hidden_layers; ++i) {
        if (_cfg.lm_cfg.layer_types[i] == "conv") {
            define_buffer(fmt::format("conv_cache_history_l{}", i), conv_cache_shape);
        } else {
            define_buffer(fmt::format("cache_key_l{}", i), cache_shape);
            define_buffer(fmt::format("cache_val_l{}", i), cache_shape);
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
    std::vector<uint16_t> num_tokens_vec{1};
    if (_use_group_token_models) {
        const auto& num_tokens = _cfg.pipeline_cfg.input_token_group_size;
        num_tokens_vec.emplace_back(num_tokens);
    }
    if (_cfg.pipeline_cfg.future_token_mask_size > 1) {
        uint32_t full_token_mask_size = round_up_to_row(
            _cfg.pipeline_cfg.max_num_tokens + _cfg.pipeline_cfg.future_token_mask_size - 1
        );
        define_buffer("future_token_mask", {full_token_mask_size});
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
            fmt::format("n{}_buffer3", num_tokens), {num_tokens, _cfg.lm_cfg.attn_cfg.get_q_size()}
        );
    }

    // Post output for last layer.
    if (_cfg.lm_cfg.lm_head_num_splits == 1 && !_cfg.pipeline_cfg.return_logits)
        define_buffer("n1_buffer4", {1}, "int32");
    else
        define_buffer("n1_buffer4", {_cfg.lm_cfg.token_cfg.vocab_size});
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


void LanguageModel::_define_models_iter(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    if (_cfg.lm_cfg.layer_types[layer_idx] == "conv")
        _define_conv_models_iter(num_tokens, token_idx, layer_idx);
    else
        _define_attn_models_iter(num_tokens, token_idx, layer_idx);
}


void LanguageModel::_define_attn_models_iter(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    LanguageModelMapKey model_key{num_tokens, layer_idx, token_idx};
    std::string freq_prefix;
    uint16_t cache_token_idx_begin;
    if (_cfg.lm_cfg.layer_types[layer_idx] == "sliding_attention") {
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
            _cfg.lm_cfg.attn_cfg.head_dim
        };
    } else {
        pre_kv_cache_offset = {token_idx, 0};
        pre_kv_cache_shape = {num_tokens, _cfg.lm_cfg.attn_cfg.get_kv_size()};
    }

    std::vector<MLABufferSlice> pre_ifms;
    std::vector<MLABufferSlice> pre_ofms;
    if (layer_idx) {
        pre_ifms.emplace_back(
            MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))}
        );
    } else {
        pre_ifms.emplace_back(MLABufferSlice{});
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
    pre_ofms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_buffer2", num_tokens))});
    pre_ofms.emplace_back(
        MLABufferSlice(
            &get_buffer(fmt::format("cache_key_l{}", layer_idx)),
            pre_kv_cache_offset,
            pre_kv_cache_shape
        )
    );
    pre_ofms.emplace_back(
        MLABufferSlice(
            &get_buffer(fmt::format("cache_val_l{}", layer_idx)),
            pre_kv_cache_offset,
            pre_kv_cache_shape
        )
    );
    _define_model(
        "pre",
        model_key,
        _get_elf_path_pre(num_tokens, layer_idx),
        pre_ifms,
        pre_ofms
    );

    // Cache model.
    uint16_t eff_token_idx = token_idx - cache_token_idx_begin;
    uint16_t eff_num_cached_tokens = token_idx + num_tokens - cache_token_idx_begin;
    uint16_t aligned_eff_token_idx;
    uint16_t aligned_eff_num_cached_tokens;
    if (num_tokens == 1 && _cfg.pipeline_cfg.future_token_mask_size > 1) {
        aligned_eff_token_idx = std::min(
            round_up_to(eff_token_idx + 1, _cfg.pipeline_cfg.future_token_mask_size) - 1,
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
            _cfg.lm_cfg.attn_cfg.head_dim
        };
    } else {
        cache_kv_cache_offset = {cache_token_idx_begin, 0};
        cache_kv_cache_shape = {aligned_eff_num_cached_tokens, _cfg.lm_cfg.attn_cfg.get_kv_size()};
    }

    std::vector<MLABufferSlice> cache_ifms{
        MLABufferSlice{&get_buffer(fmt::format("n{}_buffer2", num_tokens))},
        MLABufferSlice{
            &get_buffer(fmt::format("cache_key_l{}", layer_idx)),
            cache_kv_cache_offset,
            cache_kv_cache_shape
        },
    };

    if (num_tokens == 1 && _cfg.pipeline_cfg.future_token_mask_size > 1) {
        cache_ifms.emplace_back(
            MLABufferSlice{
                &get_buffer("future_token_mask"),
                {(uint32_t)_cfg.pipeline_cfg.max_num_tokens - eff_num_cached_tokens},
                {aligned_eff_num_cached_tokens}
            }
        );
    }
    cache_ifms.emplace_back(
        MLABufferSlice{
            &get_buffer(fmt::format("cache_val_l{}", layer_idx)),
            cache_kv_cache_offset,
            cache_kv_cache_shape
        }
    );
    std::vector<MLABufferSlice> cache_ofms{
        MLABufferSlice{&get_buffer(fmt::format("n{}_buffer3", num_tokens))}
    };
    _define_model(
        "cache",
        model_key,
        _get_elf_path_cache(num_tokens, aligned_eff_token_idx),
        cache_ifms,
        cache_ofms
    );

    // Post model.
    std::vector<MLABufferSlice> post_ifms{pre_ifms[0], cache_ofms[0]};
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
                {num_tokens, (uint32_t)get_buffer(buf_name).get_shape().back()}
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
    } else if (_cfg.lm_cfg.lm_head_num_splits == 1) {
        post_ofms.emplace_back(MLABufferSlice{&get_buffer("n1_buffer4")});
        post_elf_path = _get_elf_path_post(1, layer_idx);
    } else {
        post_elf_path = _get_elf_path_post(1, layer_idx);
        const auto& vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
        const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
        for (uint32_t split_begin = 0; split_begin < vocab_size; split_begin += split_dim) {
            auto split_size = std::min(vocab_size, split_begin + split_dim) - split_begin;
            post_ofms.emplace_back(
                MLABufferSlice{&get_buffer("n1_buffer4"), {split_begin}, {split_size}}
            );
        }
    }
    _define_model("post", model_key, post_elf_path, post_ifms, post_ofms);
}


void LanguageModel::_define_conv_models_iter(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) {
    LanguageModelMapKey model_key{num_tokens, layer_idx, token_idx};

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
            {token_idx, 0},
            {_cfg.lm_cfg.conv_L_cache - 1, _cfg.lm_cfg.hidden_size}
        }
    );
    conv_ofms.emplace_back(MLABufferSlice{&get_buffer(fmt::format("n{}_buffer1", num_tokens))});
    conv_ofms.emplace_back(
        MLABufferSlice(
            &get_buffer(fmt::format("conv_cache_history_l{}", layer_idx)),
            {token_idx + _cfg.lm_cfg.conv_L_cache - 1, 0},
            {num_tokens, _cfg.lm_cfg.hidden_size}
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
    std::vector<uint16_t> num_tokens_vec = {1};
    if (_use_group_token_models) {
        num_tokens_vec.emplace_back(_cfg.pipeline_cfg.input_token_group_size);
    }
    for (const auto& num_tokens: num_tokens_vec) {
        const auto& max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
        const auto& num_hidden_layers = _cfg.lm_cfg.num_hidden_layers;
        if (num_tokens == 1) {
            for (uint16_t token_idx = 0; token_idx < max_num_tokens; ++token_idx) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    _define_models_iter(num_tokens, token_idx, layer_idx);
                }
            }
        } else {
            for (const auto& token_idx: _cfg.pipeline_cfg.input_token_group_offsets.value()) {
                for (uint8_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
                    _define_models_iter(num_tokens, token_idx, layer_idx);
                }
            }
        }
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


std::filesystem::path LanguageModel::_get_elf_path_cache(uint16_t num_tokens, uint16_t token_idx) {
    auto elf_file_name = fmt::format(
        "{}_n{}_cache_token{}_stage1_mla.elf", _cfg.language_model_name, num_tokens, token_idx
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


uint16_t LanguageModel::_set_input_text_embeds(std::span<const uint32_t> input_token_ids) {
    // Log.
    _logger->info("Cached token ids: [{}]", fmt::join(_cached_token_ids, ", "));

    const uint16_t num_input_tokens = input_token_ids.size();
    const auto& embeddings_buf = get_buffer("embeddings");
    const auto elem_size = embeddings_buf.get_elem_size();
    const auto& embeddings_shape = embeddings_buf.get_shape();
    const uint16_t embed_size = embeddings_shape.back() * elem_size;
    const uint8_t* embeddings_ptr = (
        reinterpret_cast<const uint8_t*>(embeddings_buf.get_virtual_addr())
    );

    MLABuffer& buf = get_buffer("input_embeds");
    uint32_t token_idx = 0;
    uint32_t num_images = 0;
    uint16_t num_cached_tokens = 0;
    while(token_idx < num_input_tokens) {
        const auto& token_id = input_token_ids[token_idx];
        if (_image_token_id.has_value() && token_id == _image_token_id.value()) {
            auto next_token_idx = token_idx + _cfg.mm_cfg.value().mm_tokens_per_image;
            if (next_token_idx == num_input_tokens)
                buf.flush_cache();
            ++num_images;
            token_idx = next_token_idx;
        } else {
            auto next_token_idx = token_idx + 1;
            buf.upload(
                embeddings_ptr + token_id * embed_size,
                token_idx * embed_size,
                embed_size,
                next_token_idx == num_input_tokens
            );
            if (
                num_images == 0
                && token_idx < _cached_token_ids.size()
                && token_id == _cached_token_ids[token_idx]
            )
                ++num_cached_tokens;
            token_idx = next_token_idx;
        }
    }

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
