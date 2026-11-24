#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include <fmt/format.h>
#include "language/language_model.hpp"
#include "language/layer_executor.hpp"
#include "language/language_model_state.hpp"

namespace simaai {
namespace llima {

namespace {

bool is_attention(LayerType layer_type) {
  return layer_type == LayerType::FullAttention || layer_type == LayerType::SlidingAttention;
}



} // namespace



MLABuffer* LanguageModel::_get_dflash_hidden_buffer(uint16_t num_tokens, uint8_t layer_idx) {
  const auto& layers = _cfg.lm_cfg.speculative_decoding_cfg->target_layer_ids;
  const auto layer = std::find(layers.begin(), layers.end(), layer_idx);
  return layer == layers.end() ? nullptr : &get_buffer(fmt::format(
      "dflash_target_hidden_n{}_{}", num_tokens, std::distance(layers.begin(), layer)));
}

void LanguageModel::_bind_dflash_linear_conv_state(uint16_t prefix_tokens) {
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash convolution-state prefix length");
  }
  if (!_has_linear_attention_layers()) {
    return;
  }

  const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] != LayerType::LinearAttention) {
      continue;
    }
    auto &model = _workspace->linear_model_map.at(
        LanguageModelMapKey{block_size, layer_idx, 0});
    const uint8_t conv_ifm =
        1 + (_cfg.pipeline_cfg.quantize_embeddings && layer_idx == 0);
    if (prefix_tokens == 0) {
      model._bind_ifm(
          conv_ifm,
          &get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx)),
          {tail_begin, 0});
    } else {
      model._bind_ifm(
          conv_ifm,
          &get_buffer(fmt::format("linear_conv_prefix_states_l{}", layer_idx)),
          {static_cast<uint32_t>(prefix_tokens - 1), 0});
    }
  }
}

void LanguageModel::_resolve_dflash_linear_state(uint16_t prefix_tokens) {
  auto execution = LayerExecutor::lock_execution();
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens == 0 || prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash linear-state prefix length");
  }
  if (!_has_linear_attention_layers() ||
      _workspace->dflash_resolved_prefix == prefix_tokens) {
    return;
  }
  if (prefix_tokens == 1) {
    _workspace->dflash_resolved_prefix = prefix_tokens;
    return;
  }

  const uint32_t decay_row = prefix_tokens - 1;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] != LayerType::LinearAttention) {
      continue;
    }
    auto &model = _workspace->dflash_state_resolver_model_map.at(
        LanguageModelMapKey{block_size, layer_idx, 0});
    model._bind_ifm(
        3,
        &get_buffer(fmt::format("linear_delta_resolver_decay_mask_l{}",
                                layer_idx)),
        {0, decay_row, 0});
    model.add_to_queue();
  }
  MLAModelWithBuffer::run_queue();
  _workspace->dflash_resolved_prefix = prefix_tokens;
}

void LanguageModel::_commit_dflash_linear_state(uint16_t prefix_tokens) {
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens == 0 || prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash linear-state prefix length");
  }
  if (!_has_linear_attention_layers()) {
    return;
  }
  _resolve_dflash_linear_state(prefix_tokens);

  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] != LayerType::LinearAttention) {
      continue;
    }

    auto &delta_state =
        get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx));
    const auto delta_source_name =
        prefix_tokens == 1
            ? fmt::format("linear_delta_state_history_alt_l{}", layer_idx)
            : fmt::format("linear_delta_resolver_output_l{}", layer_idx);
    auto &delta_source = get_buffer(delta_source_name);
    delta_state.swap_storage(delta_source);
  }
  _bind_dflash_linear_conv_state(prefix_tokens);
}

void LanguageModel::_save_dflash_state_checkpoint(uint16_t token_count,
                                                   uint16_t prefix_tokens) {
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens == 0 || prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash checkpoint prefix length");
  }
  if (_cache->cached_states.empty()) {
    return;
  }

  const auto checkpoint_slot =
      _select_state_checkpoint_slot(token_count, false);
  if (!checkpoint_slot.has_value()) {
    return;
  }
  _resolve_dflash_linear_state(prefix_tokens);
  const uint32_t prefix_index = prefix_tokens - 1;
  for (auto &state : _cache->cached_states) {
    std::string source_prefix;
    std::vector<uint32_t> source_begin;
    if (state.layout.buffer_name_prefix == "linear_conv_cache_history_l") {
      source_prefix = "linear_conv_prefix_states_l";
      source_begin = {prefix_index, 0};
    } else if (state.layout.buffer_name_prefix == "linear_delta_state_history_l") {
      source_prefix = prefix_tokens == 1
                          ? "linear_delta_state_history_alt_l"
                          : "linear_delta_resolver_output_l";
      source_begin = {0, 0};
    } else {
      throw std::runtime_error(
          "Unsupported state family in DFlash checkpoint capture");
    }

    for (size_t layer_slot = 0; layer_slot < state.layout.layer_indices.size();
         ++layer_slot) {
      const auto layer_idx = state.layout.layer_indices[layer_slot];
      auto &source = get_buffer(fmt::format("{}{}", source_prefix, layer_idx));
      const size_t source_offset = source.get_buf_addr_offset(source_begin);
      source.invalidate_cache(source_offset, state.layout.tail_bytes);
      auto *source_ptr = reinterpret_cast<const uint8_t *>(
          source.get_virtual_addr());
      auto &snapshot = state.checkpoints[layer_slot][*checkpoint_slot];
      snapshot.assign(source_ptr + source_offset, source_ptr + source_offset + state.layout.tail_bytes);
    }
  }
  _cache->state_checkpoint_positions[*checkpoint_slot] = token_count;
}

void LanguageModel::_upload_dflash_attention_mask(uint16_t num_tokens,
                                                  uint16_t token_idx,
                                                  uint8_t layer_idx,
                                                  bool bidirectional,
                                                  std::vector<Eigen::bfloat16>
                                                      &mask) {
  const auto &layer_type = _cfg.lm_cfg.layer_types.at(layer_idx);
  const uint16_t cache_begin =
      _get_cache_token_idx_begin(num_tokens, token_idx, layer_type);
  const auto cache_key = _get_cache_model_key(num_tokens, token_idx, layer_idx);
  const uint16_t aligned_context = std::get<2>(cache_key) + 1;
  const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
  mask.assign(static_cast<size_t>(num_tokens) * aligned_context, neg_inf);

  for (uint16_t query = 0; query < num_tokens; ++query) {
    const uint16_t query_position = token_idx + query;
    const uint16_t end = bidirectional
                             ? token_idx + num_tokens
                             : static_cast<uint16_t>(query_position + 1);
    const uint16_t begin =
        layer_type == LayerType::SlidingAttention
            ? std::max<uint16_t>(
                  cache_begin,
                  end - std::min<uint16_t>(
                            end, _cfg.lm_cfg.attn_cfg.sliding_window.value()))
            : 0;
    std::fill(mask.begin() + static_cast<size_t>(query) * aligned_context +
                  begin - cache_begin,
              mask.begin() + static_cast<size_t>(query) * aligned_context +
                  end - cache_begin,
              Eigen::bfloat16{0.0f});
  }
  get_buffer("future_token_mask")
      .upload_raw(mask.data(), 0, mask.size() * sizeof(Eigen::bfloat16));
}

void LanguageModel::_append_dflash_context(LanguageModel &target_lm,
                                           uint16_t num_tokens,
                                           uint16_t token_idx,
                                           uint16_t valid_tokens) {
  auto execution = LayerExecutor::lock_execution();
  const auto expected =
      _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids.size();
  if (token_idx + num_tokens > _cfg.pipeline_cfg.max_num_tokens) {
    throw std::runtime_error("DFlash context write exceeds the draft cache");
  }

  auto &fc_model = _workspace->fc_model_map.at(num_tokens);
  for (size_t index = 0; index < expected; ++index) {
    fc_model._bind_ifm(
        static_cast<uint8_t>(index),
        &target_lm.get_buffer(fmt::format(
            "dflash_target_hidden_n{}_{}", num_tokens, index)),
        {0, 0});
  }
  fc_model.add_to_queue();

  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const char *freq_prefix =
        layer_type == LayerType::SlidingAttention ? "local" : "global";
    auto &model = _workspace->dflash_context_model_map.at(
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
  _cache->kv_cache_len = token_idx + valid_tokens;
}

const std::vector<uint32_t> &LanguageModel::_run_dflash_draft(
    LanguageModel &target_lm, uint32_t anchor_token, uint16_t token_idx,
    DFlashScratch &scratch) {
  auto execution = LayerExecutor::lock_execution();
  const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
  const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
  const auto mask_token =
      _cfg.lm_cfg.speculative_decoding_cfg.value().mask_token_id;
  auto &input_ids = scratch.draft_input_ids;
  input_ids.assign(num_tokens, static_cast<uint32_t>(mask_token));
  input_ids.front() = anchor_token;

  const auto embedding = LayerExecutor::stage_tokens(*this, target_lm, input_ids, num_tokens);
  auto& input = *embedding.values;
  auto* input_scale = embedding.scales;

  std::optional<LayerType> active_mask_type;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const bool bidirectional = layer_type == LayerType::FullAttention;
    if (!active_mask_type.has_value() || *active_mask_type != layer_type) {
      MLAModelWithBuffer::run_queue();
      _upload_dflash_attention_mask(num_tokens, token_idx, layer_idx,
                                    bidirectional, scratch.attention_mask);
      active_mask_type = layer_type;
    }
    std::map<uint8_t, MLABufferSlice> pre_inputs;
    if (layer_idx == 0) {
      pre_inputs.emplace(0, MLABufferSlice{&input, {0, 0}, {num_tokens, hidden_size}});
    }
    LayerExecutor::queue_layer(*this, num_tokens, token_idx, layer_idx, input_scale, 0,
        {.input = &pre_inputs, .post_input = layer_idx == 0 ? &pre_inputs : nullptr});
  }
  MLAModelWithBuffer::run_queue();

  auto &proposals = scratch.proposals;
  proposals.resize(num_tokens - 1);
  auto &views = scratch.logit_views;
  _prepare_lm_head_logit_views(num_tokens, proposals.size(), views);
  for (const auto &view : views) {
    view.buffer->invalidate_cache();
  }

#pragma omp parallel for schedule(static)
  for (int row = 0; row < static_cast<int>(proposals.size()); ++row) {
    proposals[row] = _argmax_lm_head_row(views, row);
  }

  // Drop the noisy block. The next accepted target context overwrites its rows.
  _cache->kv_cache_len = token_idx;
  return proposals;
}

std::pair<uint16_t, uint32_t>
LanguageModel::_run_dflash_target_verify(std::span<const uint32_t> input_ids,
                                         uint16_t token_idx,
                                         DFlashScratch &scratch) {
  auto execution = LayerExecutor::lock_execution();
  const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
  if (input_ids.size() != num_tokens) {
    throw std::invalid_argument(
        "DFlash verification input must match the block size");
  }
  _workspace->dflash_resolved_prefix = 0;
  const auto embedding = LayerExecutor::stage_tokens(*this, *this, input_ids, num_tokens);
  auto& input = *embedding.values;
  auto* input_scale = embedding.scales;

  MLABuffer *layer_input = nullptr;
  std::optional<LayerType> active_mask_type;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    std::map<uint8_t, MLABufferSlice> inputs;
    if (layer_idx == 0) {
      inputs.emplace(0, MLABufferSlice{&input,
                                       {0, 0},
                                       {num_tokens, _cfg.lm_cfg.hidden_size}});
    } else if (layer_input != nullptr) {
      inputs.emplace(0, MLABufferSlice{layer_input,
                                       {0, 0},
                                       {num_tokens, _cfg.lm_cfg.hidden_size}});
    }
    MLABuffer *capture_buffer = _get_dflash_hidden_buffer(num_tokens, layer_idx);
    std::map<uint8_t, MLABufferSlice> outputs;
    if (capture_buffer != nullptr) {
      outputs.emplace(0, MLABufferSlice{capture_buffer,
                                        {0, 0},
                                        {num_tokens, _cfg.lm_cfg.hidden_size}});
    }
    auto *input_overrides = inputs.empty() ? nullptr : &inputs;
    auto *output_overrides = outputs.empty() ? nullptr : &outputs;
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    if (is_attention(layer_type)) {
      if (!active_mask_type.has_value() || *active_mask_type != layer_type) {
        MLAModelWithBuffer::run_queue();
        _upload_dflash_attention_mask(num_tokens, token_idx, layer_idx, false,
                                      scratch.attention_mask);
        active_mask_type = layer_type;
      }
    }
    if (!is_attention(layer_type) && layer_type != LayerType::LinearAttention) {
      throw std::runtime_error(fmt::format("Unsupported DFlash target layer type: {}", layer_type_name(layer_type)));
    }
    LayerExecutor::queue_layer(*this, num_tokens, token_idx, layer_idx, input_scale, 0,
        {.input = input_overrides, .post_input = input_overrides, .output = output_overrides});

    layer_input = capture_buffer;
  }
  MLAModelWithBuffer::run_queue();

  const auto token_ids = _argmax_lm_head_rows(num_tokens, num_tokens, input_ids.subspan(1));
  return {static_cast<uint16_t>(token_ids.size() - 1), token_ids.back()};
}

std::optional<std::vector<uint32_t>>
LanguageModel::_run_model_dflash_speculative_decoding(
    LanguageModel &draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult *performance_result,
    uint16_t stable_prefix_token_count) {
  _run->require_active_request();
  draft_lm._run->require_active_request();
  if (!timer_ttft.has_value()) {
    timer_ttft = ChronoTimer{true};
  }
  if (performance_result != nullptr) {
    *performance_result = GenerationPerformanceResult{};
    performance_result->accepted_draft_tokens = 0;
  }
  const uint16_t max_length =
      override_max_num_tokens.value_or(_run->limits().max_num_tokens);
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (input_token_ids.empty() || input_token_ids.size() >= max_length ||
      input_token_ids.size() > _cfg.pipeline_cfg.max_num_tokens) {
    _notify_cache_full();
    return std::nullopt;
  }

  const auto cancelled = [this] { return _run->cancelled(); };
  // A prior DFlash request may leave the verification model bound to its
  // accepted scratch tail. Prefill restores or clears the persistent state.
  _bind_dflash_linear_conv_state(0);

  _init_state_checkpoint_capture(input_token_ids.size(), stable_prefix_token_count);
  uint16_t cached_tokens = _set_input_text_embeds(input_token_ids);
  uint16_t draft_cached_tokens = 0;
  while (draft_cached_tokens < input_token_ids.size() &&
         draft_cached_tokens < draft_lm._cache->cached_token_ids.size() &&
         input_token_ids[draft_cached_tokens] ==
             draft_lm._cache->cached_token_ids[draft_cached_tokens]) {
    ++draft_cached_tokens;
  }
  cached_tokens = _prepare_state_checkpoints_for_prefill(
      std::min(cached_tokens, draft_cached_tokens));
  if (_use_group_token_models && cached_tokens < input_token_ids.size()) {
    const auto &offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
    const auto next_offset =
        std::upper_bound(offsets.begin(), offsets.end(), cached_tokens);
    cached_tokens = next_offset == offsets.begin() ? 0 : *std::prev(next_offset);
  }
  _cache->kv_cache_len = cached_tokens;
  draft_lm._cache->kv_cache_len = cached_tokens;

  const uint16_t prompt_len = input_token_ids.size();
  const uint16_t prefill_width =
      _use_group_token_models ? _cfg.pipeline_cfg.input_token_group_size
                              : block_size;
  uint32_t anchor = 0;
  if (cached_tokens == prompt_len) {
    anchor = cached_tokens < _cache->cached_token_ids.size()
                 ? _cache->cached_token_ids[cached_tokens]
                 : _cache->cached_first_generated_token;
  }
  for (uint16_t offset = cached_tokens; offset < prompt_len;
       offset += prefill_width) {
    const uint16_t valid =
        std::min<uint16_t>(prefill_width, prompt_len - offset);
    anchor = run_model_once(prefill_width, offset, prompt_len, 0);
    if (cancelled()) {
      _notify_interrupt();
      return std::nullopt;
    }
    draft_lm._append_dflash_context(*this, prefill_width, offset, valid);
    if (valid < prefill_width && prefill_width != block_size &&
        _has_linear_attention_layers()) {
      _move_state_tail_for_decode(valid);
    }
  }
  if (cancelled()) {
    _notify_interrupt();
    return std::nullopt;
  }
  _cache->kv_cache_len = prompt_len;
  draft_lm._cache->kv_cache_len = prompt_len;
  _cache->cached_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
  draft_lm._cache->cached_token_ids = _cache->cached_token_ids;
  _cache->cached_first_generated_token = anchor;
  draft_lm._cache->cached_first_generated_token = anchor;

  std::vector<uint32_t> output{anchor};
  const double first_duration = timer_ttft->stop();
  _notify_first_token(anchor, first_duration);
  if (performance_result != nullptr) {
    performance_result->token_durations.push_back(first_duration);
    performance_result->generated_tokens = 1;
  }
  bool stopped = _run->limits().stop_token_ids.contains(anchor);
  bool cache_full = false;
  auto& scratch = _workspace->dflash;
  scratch.verify_input.resize(block_size);
  scratch.emitted.reserve(block_size);

  while (!stopped && !cancelled() &&
         output.size() + prompt_len < max_length) {
    if (_cache->kv_cache_len + block_size > _cfg.pipeline_cfg.max_num_tokens) {
      // A fixed-width DFlash package cannot safely run a partial final block.
      cache_full = true;
      break;
    }
    ChronoTimer iteration_timer(true);
    const uint16_t verify_start = _cache->kv_cache_len;
    const auto &proposals =
        draft_lm._run_dflash_draft(*this, anchor, verify_start, scratch);
    if (cancelled()) {
      break;
    }
    auto &verify_input = scratch.verify_input;
    verify_input.front() = anchor;
    std::copy(proposals.begin(), proposals.end(), verify_input.begin() + 1);
    const auto [accepted, bonus_token] =
        _run_dflash_target_verify(verify_input, verify_start, scratch);
    if (cancelled()) {
      break;
    }
    auto &emitted = scratch.emitted;
    emitted.assign(proposals.begin(), proposals.begin() + accepted);
    emitted.push_back(bonus_token);
    size_t produced = std::min<size_t>(
        emitted.size(), max_length - prompt_len - output.size());
    for (size_t index = 0; index < produced; ++index) {
      if (_run->limits().stop_token_ids.contains(emitted[index])) {
        produced = index + 1;
        stopped = true;
        break;
      }
    }
    const uint16_t committed_rows = static_cast<uint16_t>(produced);
    if (_workspace->capture_state_checkpoints) {
      auto boundary = std::upper_bound(_cache->checkpoint_boundaries.begin(),
                                       _cache->checkpoint_boundaries.end(),
                                       verify_start + committed_rows);
      if (boundary != _cache->checkpoint_boundaries.begin()) {
        --boundary;
        if (*boundary > verify_start) {
          _save_dflash_state_checkpoint(*boundary, *boundary - verify_start);
        }
      }
    }
    if (cancelled())
      break;
    _commit_dflash_linear_state(committed_rows);
    _cache->kv_cache_len = verify_start + committed_rows;
    draft_lm._cache->kv_cache_len = verify_start;
    draft_lm._append_dflash_context(*this, block_size, verify_start,
                                    committed_rows);
    const double duration = iteration_timer.stop();
    for (size_t index = 0; index < produced && !cancelled(); ++index) {
      const auto token = emitted[index];
      output.push_back(token);
      _notify_new_token(token, duration / produced, index < accepted);
      if (performance_result != nullptr) {
        performance_result->token_durations.push_back(duration / produced);
        ++performance_result->generated_tokens;
        if (index < accepted)
          ++performance_result->accepted_draft_tokens.value();
      }
    }
    if (cancelled())
      break;
    for (auto* model : {this, &draft_lm}) {
      model->_cache->cached_token_ids.insert(model->_cache->cached_token_ids.end(), verify_input.begin(),
                                      verify_input.begin() + committed_rows);
    }
    _cache->cached_first_generated_token = emitted[produced - 1];
    draft_lm._cache->cached_first_generated_token = _cache->cached_first_generated_token;
    anchor = emitted[produced - 1];
    if (produced < emitted.size()) {
      cache_full = true;
    }
  }

  if (cancelled()) {
    _notify_interrupt();
    return std::nullopt;
  }
  if (stopped) {
    _notify_stop();
  } else if (cache_full || prompt_len + output.size() >= max_length) {
    _notify_cache_full();
  }
  return output;
}

} // namespace llima
} // namespace simaai
