#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include <simaai_memory.h>

#include "language_model.hpp"

namespace simaai {
namespace llima {

namespace {

bool is_attention(std::string_view layer_type) {
  return layer_type == "full_attention" || layer_type == "sliding_attention";
}

} // namespace

void LanguageModel::_commit_dflash_linear_state(uint16_t prefix_tokens) {
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens == 0 || prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash linear-state prefix length");
  }

  const auto &linear_cfg = _linear_attn_cfg();
  const uint32_t prefix_index = prefix_tokens - 1;
  const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] != "linear_attention") {
      continue;
    }

    auto copy_slice =
        [&](MLABuffer &destination, const std::vector<uint32_t> &dst_begin,
            MLABuffer &source, const std::vector<uint32_t> &src_begin,
            const std::vector<uint32_t> &shape) {
          if (simaai_memcpy_part(destination.get_simaai_memory(),
                                 destination.get_buf_addr_offset(dst_begin),
                                 source.get_simaai_memory(),
                                 source.get_buf_addr_offset(src_begin),
                                 source.get_buf_len(shape)) == nullptr) {
            throw std::runtime_error(
                "Failed to commit DFlash linear-attention state");
          }
        };

    auto &conv_state =
        get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx));
    auto &conv_prefix =
        get_buffer(fmt::format("linear_conv_prefix_states_l{}", layer_idx));
    copy_slice(conv_state, {tail_begin, 0}, conv_prefix, {prefix_index, 0},
               {linear_cfg.conv_kernel_dim - 1, linear_cfg.get_conv_dim()});

    auto &delta_state =
        get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx));
    auto &delta_prefix =
        get_buffer(fmt::format("linear_delta_prefix_states_l{}", layer_idx));
    copy_slice(delta_state, {0, 0}, delta_prefix, {prefix_index, 0},
               {1, linear_cfg.get_recurrent_state_size()});
  }
}

std::optional<std::vector<uint32_t>>
LanguageModel::run_model_speculative_decoding(
    LanguageModel &draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult *performance_result) {
  if (!_cfg.lm_cfg.speculative_decoding_cfg.has_value() ||
      !draft_lm._cfg.lm_cfg.speculative_decoding_cfg.has_value()) {
    throw std::invalid_argument(
        "Both models must contain speculative-decoding configuration");
  }
  const auto &target = _cfg.lm_cfg.speculative_decoding_cfg.value();
  const auto &draft = draft_lm._cfg.lm_cfg.speculative_decoding_cfg.value();
  if (target.is_draft || !draft.is_draft) {
    throw std::invalid_argument(
        "Speculative decoding requires a target paired with a draft");
  }
  if (target.method != draft.method) {
    throw std::invalid_argument(
        "Target and draft speculative methods do not match");
  }
  if (_cfg.lm_cfg.hidden_size != draft_lm._cfg.lm_cfg.hidden_size ||
      _cfg.lm_cfg.token_cfg.vocab_size !=
          draft_lm._cfg.lm_cfg.token_cfg.vocab_size) {
    throw std::invalid_argument(
        "Target and draft hidden or vocabulary sizes do not match");
  }
  if (target.method == "dflash") {
    if (target.speculative_budget != draft.speculative_budget ||
        target.target_layer_ids != draft.target_layer_ids) {
      throw std::invalid_argument(
          "DFlash target and draft block configuration does not match");
    }
    return _run_model_dflash_speculative_decoding(
        draft_lm, input_token_ids, override_max_num_tokens, timer_ttft,
        performance_result);
  }
  if (target.method == "eagle3") {
    return _run_model_eagle3_speculative_decoding(
        draft_lm, input_token_ids, override_max_num_tokens, timer_ttft,
        performance_result);
  }
  throw std::invalid_argument("Unsupported speculative-decoding method: " +
                              target.method);
}

void LanguageModel::_append_dflash_context(LanguageModel &target_lm,
                                           uint16_t num_tokens,
                                           uint16_t token_idx,
                                           uint16_t valid_tokens) {
  const auto &captures = target_lm._eagle3_intermediate_hidden_states;
  const auto expected =
      _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids.size();
  if (captures.size() != expected) {
    throw std::runtime_error(
        "DFlash target did not produce all configured hidden-state taps");
  }
  if (token_idx + num_tokens > _cfg.pipeline_cfg.max_num_tokens) {
    throw std::runtime_error("DFlash context write exceeds the draft cache");
  }

  for (size_t index = 0; index < captures.size(); ++index) {
    auto &input = get_buffer(fmt::format("fc_n{}_input{}", num_tokens, index));
    input.upload(captures[index].data());
  }
  _fc_model_map.at(num_tokens).add_to_queue();

  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const char *freq_prefix =
        layer_type == "sliding_attention" ? "local" : "global";
    auto &model = _dflash_context_model_map.at(
        LanguageModelMapKey{num_tokens, layer_idx, 0});
    model._bind_ifm(1, &get_buffer(fmt::format("{}_freq_real", freq_prefix)),
                    {token_idx, 0});
    model._bind_ifm(2, &get_buffer(fmt::format("{}_freq_imag", freq_prefix)),
                    {token_idx, 0});
    uint8_t output = 0;
    auto bind_kv = [&](const char *kind) {
      auto &buffer = get_buffer(fmt::format("cache_{}_l{}", kind, layer_idx));
      if (_cfg.pipeline_cfg.use_strided_kv_cache) {
        model._bind_ofm(output++, &buffer, {0, token_idx, 0});
      } else {
        model._bind_ofm(output++, &buffer, {token_idx, 0});
      }
    };
    bind_kv("key");
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
      model._bind_ofm(
          output++, &get_buffer(fmt::format("cache_key_scale_l{}", layer_idx)),
          {0, token_idx, 0});
    }
    bind_kv("val");
    if (_cfg.pipeline_cfg.quantize_kv_cache) {
      model._bind_ofm(
          output, &get_buffer(fmt::format("cache_val_scale_l{}", layer_idx)),
          {0, token_idx, 0});
    }
    model.add_to_queue();
  }
  MLAModelWithBuffer::run_queue();
  _kv_cache_len = token_idx + valid_tokens;
}

std::vector<uint32_t> LanguageModel::_run_dflash_draft(LanguageModel &target_lm,
                                                       uint32_t anchor_token,
                                                       uint16_t token_idx) {
  const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
  const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
  const auto mask_token =
      _cfg.lm_cfg.speculative_decoding_cfg.value().mask_token_id;
  std::vector<uint32_t> input_ids(num_tokens,
                                  static_cast<uint32_t>(mask_token));
  input_ids.front() = anchor_token;

  const bool quantized_embeddings = _cfg.pipeline_cfg.quantize_embeddings;
  auto &input =
      quantized_embeddings
          ? get_buffer(fmt::format("eagle3_input_embeds_n{}", num_tokens))
          : get_buffer(fmt::format("n{}_buffer1", num_tokens));
  MLABuffer *input_scale =
      quantized_embeddings
          ? &get_buffer(
                fmt::format("eagle3_input_embedding_scales_n{}", num_tokens))
          : nullptr;
  _stage_embedding_rows(target_lm, input_ids, input, input_scale);

  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const uint16_t cache_begin =
        layer_type == "sliding_attention"
            ? std::max(0, token_idx + num_tokens -
                              static_cast<int>(
                                  _cfg.lm_cfg.attn_cfg.sliding_window.value()))
            : 0;
    const uint16_t visible_end = token_idx + num_tokens;
    const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
    std::vector<Eigen::bfloat16> mask(static_cast<size_t>(num_tokens) *
                                          _cfg.pipeline_cfg.max_num_tokens,
                                      neg_inf);
    for (uint16_t query = 0; query < num_tokens; ++query) {
      const uint16_t query_position = token_idx + query;
      const uint16_t end = layer_type == "full_attention"
                               ? visible_end
                               : static_cast<uint16_t>(query_position + 1);
      const uint16_t begin =
          layer_type == "sliding_attention"
              ? std::max<uint16_t>(
                    cache_begin,
                    static_cast<uint16_t>(
                        end -
                        std::min<uint16_t>(
                            end, _cfg.lm_cfg.attn_cfg.sliding_window.value())))
              : 0;
      std::fill(
          mask.begin() +
              static_cast<size_t>(query) * _cfg.pipeline_cfg.max_num_tokens +
              begin,
          mask.begin() +
              static_cast<size_t>(query) * _cfg.pipeline_cfg.max_num_tokens +
              end,
          Eigen::bfloat16{0.0f});
    }
    get_buffer("future_token_mask").upload(mask.data());

    const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    const auto cache_key =
        _bind_attn_models(num_tokens, token_idx, layer_idx, input_scale);
    std::map<uint8_t, MLABufferSlice> pre_inputs;
    if (layer_idx == 0) {
      pre_inputs.emplace(
          0, MLABufferSlice{&input, {0, 0}, {num_tokens, hidden_size}});
    }
    _pre_model_map.at(model_key).add_to_queue(&pre_inputs);
    _cache_model_map.at(cache_key).add_to_queue();
    _post_model_map.at(model_key).add_to_queue(layer_idx == 0 ? &pre_inputs
                                                              : nullptr);
    // The last draft layer is bidirectional while the preceding layers are
    // causal/sliding, so each layer must consume the mask uploaded above.
    MLAModelWithBuffer::run_queue();
  }

  const uint32_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
  std::vector<uint32_t> proposals(num_tokens - 1);
  std::vector<float> best(proposals.size(),
                          -std::numeric_limits<float>::infinity());
  if (_cfg.lm_cfg.lm_head_num_splits == 1) {
    auto &buffer = get_buffer(fmt::format("n{}_buffer4", num_tokens));
    std::vector<Eigen::bfloat16> logits(static_cast<size_t>(num_tokens - 1) *
                                        vocab_size);
    buffer.download(logits.data());
    for (size_t row = 0; row < proposals.size(); ++row) {
      for (uint32_t token = 0; token < vocab_size; ++token) {
        const float value =
            static_cast<float>(logits[row * vocab_size + token]);
        if (value > best[row]) {
          best[row] = value;
          proposals[row] = token;
        }
      }
    }
  } else {
    for (uint32_t split = 0, begin = 0; begin < vocab_size;
         begin += _cfg.lm_cfg.lm_head_split_dim, ++split) {
      const uint32_t width =
          std::min(_cfg.lm_cfg.lm_head_split_dim, vocab_size - begin);
      std::vector<Eigen::bfloat16> logits(static_cast<size_t>(num_tokens - 1) *
                                          width);
      get_buffer(fmt::format("n{}_lm_split{}", num_tokens, split))
          .download(logits.data());
      for (size_t row = 0; row < proposals.size(); ++row) {
        for (uint32_t column = 0; column < width; ++column) {
          const float value = static_cast<float>(logits[row * width + column]);
          if (value > best[row]) {
            best[row] = value;
            proposals[row] = begin + column;
          }
        }
      }
    }
  }

  // Drop the noisy block. The next accepted target context overwrites its rows.
  _kv_cache_len = token_idx;
  return proposals;
}

std::vector<uint32_t>
LanguageModel::_run_dflash_target_verify(std::span<const uint32_t> input_ids,
                                         uint16_t token_idx) {
  const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
  if (input_ids.size() != num_tokens) {
    throw std::invalid_argument(
        "DFlash verification input must match the block size");
  }
  const bool quantized_embeddings = _cfg.pipeline_cfg.quantize_embeddings;
  auto &input =
      quantized_embeddings
          ? get_buffer(fmt::format("eagle3_input_embeds_n{}", num_tokens))
          : get_buffer(fmt::format("n{}_buffer1", num_tokens));
  MLABuffer *input_scale =
      quantized_embeddings
          ? &get_buffer(
                fmt::format("eagle3_input_embedding_scales_n{}", num_tokens))
          : nullptr;
  _stage_embedding_rows(*this, input_ids, input, input_scale);
  if (_has_linear_attention_layers()) {
    std::vector<Eigen::bfloat16> valid(num_tokens, Eigen::bfloat16{1.0f});
    get_buffer("linear_valid_mask").upload(valid.data());
  }
  const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
  std::vector<Eigen::bfloat16> causal_mask(static_cast<size_t>(num_tokens) *
                                               _cfg.pipeline_cfg.max_num_tokens,
                                           neg_inf);
  for (uint16_t query = 0; query < num_tokens; ++query) {
    std::fill_n(causal_mask.begin() + static_cast<size_t>(query) *
                                          _cfg.pipeline_cfg.max_num_tokens,
                token_idx + query + 1, Eigen::bfloat16{0.0f});
  }
  get_buffer("future_token_mask").upload(causal_mask.data());

  const auto &capture_layers =
      _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids;
  _eagle3_intermediate_hidden_states.assign(capture_layers.size(), {});
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
    std::map<uint8_t, MLABufferSlice> inputs;
    if (layer_idx == 0) {
      inputs.emplace(0, MLABufferSlice{&input,
                                       {0, 0},
                                       {num_tokens, _cfg.lm_cfg.hidden_size}});
    }
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    if (is_attention(layer_type)) {
      const auto cache_key =
          _bind_attn_models(num_tokens, token_idx, layer_idx, input_scale);
      _pre_model_map.at(model_key).add_to_queue(&inputs);
      _cache_model_map.at(cache_key).add_to_queue();
      _post_model_map.at(model_key).add_to_queue(layer_idx == 0 ? &inputs
                                                                : nullptr);
    } else if (layer_type == "linear_attention") {
      auto &model = _linear_model_map.at(model_key);
      if (quantized_embeddings && layer_idx == 0) {
        model._bind_ifm(1, input_scale, {0, 0});
      }
      model.add_to_queue(&inputs);
    } else if (layer_type == "conv") {
      auto &model = _conv_model_map.at(model_key);
      if (quantized_embeddings && layer_idx == 0) {
        model._bind_ifm(1, input_scale, {0, 0});
      }
      model.add_to_queue(&inputs);
    } else {
      throw std::runtime_error("Unsupported DFlash target layer type: " +
                               layer_type);
    }

    const auto capture =
        std::find(capture_layers.begin(), capture_layers.end(), layer_idx);
    if (capture != capture_layers.end()) {
      MLAModelWithBuffer::run_queue();
      auto &destination = _eagle3_intermediate_hidden_states[std::distance(
          capture_layers.begin(), capture)];
      destination.resize(static_cast<size_t>(num_tokens) *
                         _cfg.lm_cfg.hidden_size);
      get_buffer(fmt::format("n{}_buffer1", num_tokens))
          .download(destination.data());
    }
  }
  MLAModelWithBuffer::run_queue();

  const uint32_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
  std::vector<uint32_t> next(num_tokens);
  std::vector<float> best(num_tokens, -std::numeric_limits<float>::infinity());
  if (_cfg.lm_cfg.lm_head_num_splits == 1) {
    std::vector<Eigen::bfloat16> logits(static_cast<size_t>(num_tokens) *
                                        vocab_size);
    get_buffer(fmt::format("n{}_buffer4", num_tokens)).download(logits.data());
    for (uint16_t row = 0; row < num_tokens; ++row) {
      for (uint32_t token = 0; token < vocab_size; ++token) {
        const float value =
            static_cast<float>(logits[row * vocab_size + token]);
        if (value > best[row]) {
          best[row] = value;
          next[row] = token;
        }
      }
    }
  } else {
    for (uint32_t split = 0, begin = 0; begin < vocab_size;
         begin += _cfg.lm_cfg.lm_head_split_dim, ++split) {
      const uint32_t width =
          std::min(_cfg.lm_cfg.lm_head_split_dim, vocab_size - begin);
      std::vector<Eigen::bfloat16> logits(static_cast<size_t>(num_tokens) *
                                          width);
      get_buffer(fmt::format("n{}_lm_split{}", num_tokens, split))
          .download(logits.data());
      for (uint16_t row = 0; row < num_tokens; ++row) {
        for (uint32_t column = 0; column < width; ++column) {
          const float value = static_cast<float>(logits[row * width + column]);
          if (value > best[row]) {
            best[row] = value;
            next[row] = begin + column;
          }
        }
      }
    }
  }
  return next;
}

std::optional<std::vector<uint32_t>>
LanguageModel::_run_model_dflash_speculative_decoding(
    LanguageModel &draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult *performance_result) {
  _is_running = true;
  draft_lm._is_running = true;
  if (!timer_ttft.has_value()) {
    timer_ttft = ChronoTimer{true};
  }
  if (performance_result != nullptr) {
    *performance_result = GenerationPerformanceResult{};
    performance_result->accepted_draft_tokens = 0;
  }
  const uint16_t max_length = override_max_num_tokens.value_or(_max_num_tokens);
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (input_token_ids.empty() || input_token_ids.size() >= max_length ||
      input_token_ids.size() + block_size > _cfg.pipeline_cfg.max_num_tokens) {
    _is_running = false;
    draft_lm._is_running = false;
    return std::nullopt;
  }

  auto clear_state = [](LanguageModel &model) {
    for (uint8_t layer = 0; layer < model._cfg.lm_cfg.num_hidden_layers;
         ++layer) {
      const auto &type = model._cfg.lm_cfg.layer_types[layer];
      if (is_attention(type) && !model._cfg.lm_cfg.is_kv_shared_layer(layer)) {
        model.get_buffer(fmt::format("cache_key_l{}", layer)).clear();
        model.get_buffer(fmt::format("cache_val_l{}", layer)).clear();
        if (model._cfg.pipeline_cfg.quantize_kv_cache) {
          model.get_buffer(fmt::format("cache_key_scale_l{}", layer)).clear();
          model.get_buffer(fmt::format("cache_val_scale_l{}", layer)).clear();
        }
      } else if (type == "linear_attention") {
        model.get_buffer(fmt::format("linear_conv_cache_history_l{}", layer))
            .clear();
        model.get_buffer(fmt::format("linear_delta_state_history_l{}", layer))
            .clear();
        model
            .get_buffer(
                fmt::format("linear_delta_state_history_alt_l{}", layer))
            .clear();
      }
    }
    model._kv_cache_len = 0;
    model._cached_token_ids.clear();
  };
  clear_state(*this);
  clear_state(draft_lm);

  _set_input_text_embeds(input_token_ids);
  const uint16_t prompt_len = input_token_ids.size();
  const uint16_t prefill_width = _use_group_token_models
                                     ? _cfg.pipeline_cfg.input_token_group_size
                                     : block_size;
  uint32_t anchor = 0;
  for (uint16_t offset = 0; offset < prompt_len; offset += prefill_width) {
    const uint16_t valid =
        std::min<uint16_t>(prefill_width, prompt_len - offset);
    if (prefill_width == block_size) {
      const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
      std::vector<Eigen::bfloat16> mask(static_cast<size_t>(block_size) *
                                            _cfg.pipeline_cfg.max_num_tokens,
                                        neg_inf);
      for (uint16_t query = 0; query < block_size; ++query) {
        std::fill_n(mask.begin() + static_cast<size_t>(query) *
                                       _cfg.pipeline_cfg.max_num_tokens,
                    offset + query + 1, Eigen::bfloat16{0.0f});
      }
      get_buffer("future_token_mask").upload(mask.data());
    }
    anchor = run_model_once(prefill_width, offset, prompt_len, 0);
    if (!_is_running.load(std::memory_order_relaxed)) {
      draft_lm._is_running = false;
      _notify_interrupt();
      _text_streamer.wait_streaming();
      return std::nullopt;
    }
    draft_lm._append_dflash_context(*this, prefill_width, offset, valid);
    if (valid < prefill_width && prefill_width != block_size &&
        _has_linear_attention_layers()) {
      _move_state_tail_for_decode(valid);
    }
  }
  _kv_cache_len = prompt_len;
  draft_lm._kv_cache_len = prompt_len;
  _cached_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
  draft_lm._cached_token_ids = _cached_token_ids;
  _cached_first_generated_token = anchor;

  std::vector<uint32_t> output{anchor};
  const double first_duration = timer_ttft->stop();
  _notify_first_token(anchor, first_duration);
  if (performance_result != nullptr) {
    performance_result->token_durations.push_back(first_duration);
    performance_result->generated_tokens = 1;
  }
  bool stopped = _stop_token_ids.contains(anchor);
  bool cache_full = false;

  while (!stopped && _is_running.load(std::memory_order_relaxed) &&
         output.size() + prompt_len < max_length) {
    if (_kv_cache_len + block_size > _cfg.pipeline_cfg.max_num_tokens) {
      cache_full = true;
      break;
    }
    ChronoTimer iteration_timer(true);
    const uint16_t verify_start = _kv_cache_len;
    auto proposals = draft_lm._run_dflash_draft(*this, anchor, verify_start);
    std::vector<uint32_t> verify_input(block_size, anchor);
    std::copy(proposals.begin(), proposals.end(), verify_input.begin() + 1);
    auto target_tokens = _run_dflash_target_verify(verify_input, verify_start);

    size_t accepted = 0;
    while (accepted < proposals.size() &&
           proposals[accepted] == target_tokens[accepted]) {
      ++accepted;
    }
    const uint16_t committed_rows = static_cast<uint16_t>(accepted + 1);
    _commit_dflash_linear_state(committed_rows);
    _kv_cache_len = verify_start + committed_rows;
    draft_lm._kv_cache_len = verify_start;
    draft_lm._append_dflash_context(*this, block_size, verify_start,
                                    committed_rows);

    std::vector<uint32_t> emitted(proposals.begin(),
                                  proposals.begin() + accepted);
    emitted.push_back(target_tokens[accepted]);
    const double duration = iteration_timer.stop();
    for (const auto token : emitted) {
      if (prompt_len + output.size() >= max_length) {
        cache_full = true;
        break;
      }
      output.push_back(token);
      _cached_token_ids.push_back(token);
      draft_lm._cached_token_ids.push_back(token);
      _notify_new_token(token, duration / emitted.size());
      if (performance_result != nullptr) {
        performance_result->token_durations.push_back(duration /
                                                      emitted.size());
        ++performance_result->generated_tokens;
      }
      if (_stop_token_ids.contains(token)) {
        stopped = true;
        break;
      }
    }
    if (performance_result != nullptr) {
      performance_result->accepted_draft_tokens.value() += accepted;
    }
    anchor = emitted.back();
  }

  if (stopped) {
    _notify_stop();
  } else if (cache_full || prompt_len + output.size() >= max_length) {
    _notify_cache_full();
  } else if (!_is_running.load(std::memory_order_relaxed)) {
    _notify_interrupt();
  }
  _text_streamer.wait_streaming();
  const bool completed = _is_running.load(std::memory_order_relaxed);
  _is_running = false;
  draft_lm._is_running = false;
  return completed ? std::optional<std::vector<uint32_t>>(std::move(output))
                   : std::nullopt;
}

} // namespace llima
} // namespace simaai
