#include <algorithm>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
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

struct DFlashLogitView {
  const uint8_t *data;
  size_t row_stride;
  uint32_t vocab_begin;
  uint32_t width;
};

class ScopeExit {
public:
  explicit ScopeExit(std::function<void()> callback)
      : _callback(std::move(callback)) {}
  ~ScopeExit() noexcept { _callback(); }

private:
  std::function<void()> _callback;
};

void copy_buffer_slice(MLABuffer &destination,
                       const std::vector<uint32_t> &dst_begin,
                       MLABuffer &source,
                       const std::vector<uint32_t> &src_begin,
                       const std::vector<uint32_t> &shape,
                       const char *operation) {
  if (simaai_memcpy_part(destination.get_simaai_memory(),
                         destination.get_buf_addr_offset(dst_begin),
                         source.get_simaai_memory(),
                         source.get_buf_addr_offset(src_begin),
                         source.get_buf_len(shape)) == nullptr) {
    throw std::runtime_error(operation);
  }
}

} // namespace

struct LanguageModel::DFlashScratch {
  std::vector<uint32_t> draft_input_ids;
  std::vector<uint32_t> proposals;
  std::vector<uint32_t> verify_input;
  std::vector<uint32_t> emitted;
  std::vector<DFlashLogitView> logit_views;
  std::vector<Eigen::bfloat16> attention_mask;
  std::vector<Eigen::bfloat16> linear_valid_mask;
};

void LanguageModel::_capture_dflash_hidden_state(uint16_t num_tokens,
                                                  size_t capture_idx) {
  auto &source = get_buffer(fmt::format("n{}_buffer1", num_tokens));
  auto &destination = get_buffer(
      fmt::format("dflash_target_hidden_n{}_{}", num_tokens, capture_idx));
  copy_buffer_slice(destination, {0, 0}, source, {0, 0},
                    {num_tokens, _cfg.lm_cfg.hidden_size},
                    "Failed to retain DFlash target hidden state");
}

void LanguageModel::_resolve_dflash_linear_state(uint16_t prefix_tokens) {
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens == 0 || prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash linear-state prefix length");
  }
  if (prefix_tokens == block_size || _dflash_resolved_prefix == prefix_tokens) {
    return;
  }

  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] != "linear_attention") {
      continue;
    }
    _dflash_state_resolver_model_map
        .at(LanguageModelMapKey{block_size, layer_idx, prefix_tokens})
        .add_to_queue();
  }
  MLAModelWithBuffer::run_queue();
  _dflash_resolved_prefix = prefix_tokens;
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

  const auto &linear_cfg = _linear_attn_cfg();
  const uint32_t prefix_index = prefix_tokens - 1;
  const uint32_t tail_begin = _cfg.pipeline_cfg.input_token_group_size - 1;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    if (_cfg.lm_cfg.layer_types[layer_idx] != "linear_attention") {
      continue;
    }

    auto &conv_state =
        get_buffer(fmt::format("linear_conv_cache_history_l{}", layer_idx));
    auto &conv_prefix =
        get_buffer(fmt::format("linear_conv_prefix_states_l{}", layer_idx));
    copy_buffer_slice(
        conv_state, {tail_begin, 0}, conv_prefix, {prefix_index, 0},
        {linear_cfg.conv_kernel_dim - 1, linear_cfg.get_conv_dim()},
        "Failed to commit DFlash convolution state");

    auto &delta_state =
        get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx));
    const auto delta_source_name = prefix_tokens == block_size
                                       ? fmt::format(
                                             "linear_delta_state_history_alt_l{}",
                                             layer_idx)
                                       : fmt::format(
                                             "linear_delta_resolver_output_l{}",
                                             layer_idx);
    auto &delta_source = get_buffer(delta_source_name);
    delta_state.swap_storage(delta_source);
  }
}

void LanguageModel::_save_dflash_state_checkpoint(uint16_t token_count,
                                                   uint16_t prefix_tokens,
                                                   bool is_prefill) {
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (prefix_tokens == 0 || prefix_tokens > block_size) {
    throw std::invalid_argument("Invalid DFlash checkpoint prefix length");
  }
  if (_cached_states.empty()) {
    return;
  }

  const auto checkpoint_slot =
      _select_state_checkpoint_slot(token_count, is_prefill);
  if (!checkpoint_slot.has_value()) {
    return;
  }
  _resolve_dflash_linear_state(prefix_tokens);
  const uint32_t prefix_index = prefix_tokens - 1;
  for (auto &state : _cached_states) {
    std::string source_prefix;
    std::vector<uint32_t> source_begin;
    if (state.buffer_name_prefix == "linear_conv_cache_history_l") {
      source_prefix = "linear_conv_prefix_states_l";
      source_begin = {prefix_index, 0};
    } else if (state.buffer_name_prefix == "linear_delta_state_history_l") {
      source_prefix = prefix_tokens == block_size
                          ? "linear_delta_state_history_alt_l"
                          : "linear_delta_resolver_output_l";
      source_begin = {0, 0};
    } else {
      throw std::runtime_error(
          "Unsupported state family in DFlash checkpoint capture");
    }

    for (size_t layer_slot = 0; layer_slot < state.layer_indices.size();
         ++layer_slot) {
      const auto layer_idx = state.layer_indices[layer_slot];
      auto &source = get_buffer(fmt::format("{}{}", source_prefix, layer_idx));
      const size_t source_offset = source.get_buf_addr_offset(source_begin);
      source.invalidate_cache(source_offset, state.tail_bytes);
      auto *source_ptr = reinterpret_cast<const uint8_t *>(
          source.get_virtual_addr());
      std::memcpy(
          state.checkpoints[layer_slot][*checkpoint_slot].data(),
          source_ptr + source_offset, state.tail_bytes);
    }
  }
  _state_checkpoint_positions[*checkpoint_slot] = token_count;
}

void LanguageModel::_upload_dflash_attention_mask(uint16_t num_tokens,
                                                  uint16_t token_idx,
                                                  uint8_t layer_idx,
                                                  bool bidirectional,
                                                  std::vector<Eigen::bfloat16>
                                                      &mask) {
  const auto &layer_type = _cfg.lm_cfg.layer_types.at(layer_idx);
  const uint16_t cache_begin =
      layer_type == "sliding_attention"
          ? std::max(0, token_idx + num_tokens -
                            static_cast<int>(
                                _cfg.lm_cfg.attn_cfg.sliding_window.value()))
          : 0;
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
        layer_type == "sliding_attention"
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

std::optional<std::vector<uint32_t>>
LanguageModel::run_model_speculative_decoding(
    LanguageModel &draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult *performance_result) {
  return run_model_speculative_decoding(draft_lm, input_token_ids,
                                        override_max_num_tokens, timer_ttft,
                                        performance_result, 0);
}

std::optional<std::vector<uint32_t>>
LanguageModel::run_model_speculative_decoding(
    LanguageModel &draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult *performance_result,
    uint16_t stable_prefix_token_count) {
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
    if (_cfg.pipeline_cfg.input_token_group_size !=
        draft_lm._cfg.pipeline_cfg.input_token_group_size) {
      throw std::invalid_argument(
          "DFlash target and draft language group sizes do not match");
    }
    if (_cfg.pipeline_cfg.input_token_group_size <
        target.speculative_budget) {
      throw std::invalid_argument(
          "DFlash language group size is smaller than the block size");
    }
    if (draft_lm._cfg.pipeline_cfg.max_num_tokens <
        _cfg.pipeline_cfg.max_num_tokens) {
      throw std::invalid_argument(
          "DFlash draft cache is smaller than the target cache");
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
    if (target.mask_token_id != draft.mask_token_id ||
        draft.mask_token_id < 0 ||
        static_cast<uint32_t>(draft.mask_token_id) >=
            _cfg.lm_cfg.token_cfg.vocab_size) {
      throw std::invalid_argument(
          "DFlash target and draft mask token IDs do not match");
    }
    return _run_model_dflash_speculative_decoding(
        draft_lm, input_token_ids, override_max_num_tokens, timer_ttft,
        performance_result, stable_prefix_token_count);
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
  const auto expected =
      _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids.size();
  if (token_idx + num_tokens > _cfg.pipeline_cfg.max_num_tokens) {
    throw std::runtime_error("DFlash context write exceeds the draft cache");
  }

  auto &fc_model = _fc_model_map.at(num_tokens);
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

const std::vector<uint32_t> &LanguageModel::_run_dflash_draft(
    LanguageModel &target_lm, uint32_t anchor_token, uint16_t token_idx,
    DFlashScratch &scratch) {
  const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
  const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
  const auto mask_token =
      _cfg.lm_cfg.speculative_decoding_cfg.value().mask_token_id;
  auto &input_ids = scratch.draft_input_ids;
  input_ids.assign(num_tokens, static_cast<uint32_t>(mask_token));
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

  std::optional<std::pair<std::string_view, bool>> active_mask;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const auto &layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const bool bidirectional = layer_type == "full_attention";
    const auto required_mask = std::pair{std::string_view(layer_type),
                                         bidirectional};
    if (!active_mask.has_value() || *active_mask != required_mask) {
      MLAModelWithBuffer::run_queue();
      _upload_dflash_attention_mask(num_tokens, token_idx, layer_idx,
                                    bidirectional, scratch.attention_mask);
      active_mask = required_mask;
    }
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
  }
  MLAModelWithBuffer::run_queue();

  const uint32_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
  auto &proposals = scratch.proposals;
  proposals.resize(num_tokens - 1);
  auto &views = scratch.logit_views;
  views.clear();
  auto add_view = [&](MLABuffer &buffer, uint32_t begin, uint32_t width) {
    if (buffer.get_dtype() != "bfloat16" || buffer.get_shape().size() != 2 ||
        buffer.get_shape().front() != proposals.size() ||
        buffer.get_shape().back() != width) {
      throw std::runtime_error("Invalid DFlash draft logit buffer");
    }
    buffer.invalidate_cache();
    views.push_back(DFlashLogitView{
        static_cast<const uint8_t *>(buffer.get_virtual_addr()),
        buffer.get_buf_len(std::vector<uint32_t>{1, width}), begin, width});
  };
  if (_cfg.lm_cfg.lm_head_num_splits == 1) {
    add_view(get_buffer(fmt::format("n{}_buffer4", num_tokens)), 0,
             vocab_size);
  } else {
    for (uint32_t split = 0, begin = 0; begin < vocab_size;
         begin += _cfg.lm_cfg.lm_head_split_dim, ++split) {
      const uint32_t width =
          std::min(_cfg.lm_cfg.lm_head_split_dim, vocab_size - begin);
      add_view(get_buffer(fmt::format("n{}_lm_split{}", num_tokens, split)),
               begin, width);
    }
  }

#pragma omp parallel for schedule(static)
  for (int row = 0; row < static_cast<int>(proposals.size()); ++row) {
    float best = -std::numeric_limits<float>::infinity();
    uint32_t best_token = 0;
    for (const auto &view : views) {
      const auto *logits = reinterpret_cast<const Eigen::bfloat16 *>(
          view.data + static_cast<size_t>(row) * view.row_stride);
      for (uint32_t column = 0; column < view.width; ++column) {
        const float value = static_cast<float>(logits[column]);
        if (value > best) {
          best = value;
          best_token = view.vocab_begin + column;
        }
      }
    }
    proposals[row] = best_token;
  }

  // Drop the noisy block. The next accepted target context overwrites its rows.
  _kv_cache_len = token_idx;
  return proposals;
}

std::pair<uint16_t, uint32_t>
LanguageModel::_run_dflash_target_verify(std::span<const uint32_t> input_ids,
                                         uint16_t token_idx,
                                         DFlashScratch &scratch) {
  const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
  if (input_ids.size() != num_tokens) {
    throw std::invalid_argument(
        "DFlash verification input must match the block size");
  }
  _dflash_resolved_prefix = 0;
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
    auto &valid = scratch.linear_valid_mask;
    valid.assign(num_tokens, Eigen::bfloat16{1.0f});
    get_buffer("linear_valid_mask")
        .upload_raw(valid.data(), 0, valid.size() * sizeof(Eigen::bfloat16));
  }

  const auto &capture_layers =
      _cfg.lm_cfg.speculative_decoding_cfg.value().target_layer_ids;
  MLABuffer *layer_input = nullptr;
  std::optional<std::string_view> active_mask_type;
  for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers;
       ++layer_idx) {
    const LanguageModelMapKey model_key{num_tokens, layer_idx, 0};
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
    const auto capture =
        std::find(capture_layers.begin(), capture_layers.end(), layer_idx);
    MLABuffer *capture_buffer = nullptr;
    std::map<uint8_t, MLABufferSlice> outputs;
    if (capture != capture_layers.end()) {
      const auto capture_idx = std::distance(capture_layers.begin(), capture);
      capture_buffer = &get_buffer(fmt::format(
          "dflash_target_hidden_n{}_{}", num_tokens, capture_idx));
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
      const auto cache_key =
          _bind_attn_models(num_tokens, token_idx, layer_idx, input_scale);
      _pre_model_map.at(model_key).add_to_queue(input_overrides);
      _cache_model_map.at(cache_key).add_to_queue();
      _post_model_map.at(model_key).add_to_queue(input_overrides,
                                                  output_overrides);
    } else if (layer_type == "linear_attention") {
      auto &model = _linear_model_map.at(model_key);
      if (quantized_embeddings && layer_idx == 0) {
        model._bind_ifm(1, input_scale, {0, 0});
      }
      model.add_to_queue(input_overrides, output_overrides);
    } else if (layer_type == "conv") {
      auto &model = _conv_model_map.at(model_key);
      if (quantized_embeddings && layer_idx == 0) {
        model._bind_ifm(1, input_scale, {0, 0});
      }
      model.add_to_queue(input_overrides, output_overrides);
    } else {
      throw std::runtime_error("Unsupported DFlash target layer type: " +
                               layer_type);
    }

    layer_input = capture_buffer;
  }
  MLAModelWithBuffer::run_queue();

  const uint32_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
  auto &views = scratch.logit_views;
  views.clear();
  auto add_view = [&](MLABuffer &buffer, uint32_t begin, uint32_t width) {
    if (buffer.get_dtype() != "bfloat16" || buffer.get_shape().size() != 2 ||
        buffer.get_shape().front() != num_tokens ||
        buffer.get_shape().back() != width) {
      throw std::runtime_error("Invalid DFlash target logit buffer");
    }
    buffer.invalidate_cache();
    views.push_back(DFlashLogitView{
        static_cast<const uint8_t *>(buffer.get_virtual_addr()),
        buffer.get_buf_len(std::vector<uint32_t>{1, width}), begin, width});
  };
  if (_cfg.lm_cfg.lm_head_num_splits == 1) {
    add_view(get_buffer(fmt::format("n{}_buffer4", num_tokens)), 0,
             vocab_size);
  } else {
    for (uint32_t split = 0, begin = 0; begin < vocab_size;
         begin += _cfg.lm_cfg.lm_head_split_dim, ++split) {
      const uint32_t width =
          std::min(_cfg.lm_cfg.lm_head_split_dim, vocab_size - begin);
      add_view(get_buffer(fmt::format("n{}_lm_split{}", num_tokens, split)),
               begin, width);
    }
  }

  for (uint16_t row = 0; row < num_tokens; ++row) {
    float best = -std::numeric_limits<float>::infinity();
    uint32_t target_token = 0;
    for (const auto &view : views) {
      const auto *logits = reinterpret_cast<const Eigen::bfloat16 *>(
          view.data + static_cast<size_t>(row) * view.row_stride);
      for (uint32_t column = 0; column < view.width; ++column) {
        const float value = static_cast<float>(logits[column]);
        if (value > best) {
          best = value;
          target_token = view.vocab_begin + column;
        }
      }
    }
    if (row == num_tokens - 1 || target_token != input_ids[row + 1]) {
      return {row, target_token};
    }
  }
  throw std::runtime_error("DFlash verification did not produce a bonus token");
}

std::optional<std::vector<uint32_t>>
LanguageModel::_run_model_dflash_speculative_decoding(
    LanguageModel &draft_lm, std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult *performance_result,
    uint16_t stable_prefix_token_count) {
  _is_running = true;
  draft_lm._is_running = true;
  const int uncaught_exceptions = std::uncaught_exceptions();
  ScopeExit cleanup([&]() {
    const bool failed = std::uncaught_exceptions() > uncaught_exceptions;
    if (failed) {
      _cached_token_ids.clear();
      draft_lm._cached_token_ids.clear();
    }
    _rolling_checkpoint_slot = 0;
    _writable_checkpoint_slots = 0;
    _capture_state_checkpoints = false;
    _is_running = false;
    draft_lm._is_running = false;
    if (failed) {
      try {
        _text_streamer.wait_streaming();
      } catch (...) {
      }
    }
  });
  if (!timer_ttft.has_value()) {
    timer_ttft = ChronoTimer{true};
  }
  if (performance_result != nullptr) {
    *performance_result = GenerationPerformanceResult{};
    performance_result->accepted_draft_tokens = 0;
  }
  const uint16_t max_length =
      override_max_num_tokens.value_or(_max_num_tokens);
  const uint16_t block_size = _cfg.lm_cfg.get_single_num_tokens();
  if (input_token_ids.empty() || input_token_ids.size() >= max_length ||
      input_token_ids.size() > _cfg.pipeline_cfg.max_num_tokens) {
    return std::nullopt;
  }

  if (!_cached_states.empty()) {
    _capture_state_checkpoints = true;
    _rolling_checkpoint_slot = 0;
    _writable_checkpoint_slots = 0;
    const auto system_boundary = std::upper_bound(
        _checkpoint_boundaries.begin(), _checkpoint_boundaries.end(),
        std::min<size_t>(stable_prefix_token_count, input_token_ids.size()));
    _system_checkpoint_position =
        system_boundary == _checkpoint_boundaries.begin()
            ? 0
            : *std::prev(system_boundary);
    if (_system_checkpoint_position &&
        _state_checkpoint_positions[0] != _system_checkpoint_position) {
      _state_checkpoint_positions[0] = 0;
    }
  }
  uint16_t cached_tokens = _set_input_text_embeds(input_token_ids);
  uint16_t draft_cached_tokens = 0;
  while (draft_cached_tokens < input_token_ids.size() &&
         draft_cached_tokens < draft_lm._cached_token_ids.size() &&
         input_token_ids[draft_cached_tokens] ==
             draft_lm._cached_token_ids[draft_cached_tokens]) {
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
  _kv_cache_len = cached_tokens;
  draft_lm._kv_cache_len = cached_tokens;

  const uint16_t prompt_len = input_token_ids.size();
  const uint16_t prefill_width =
      _use_group_token_models ? _cfg.pipeline_cfg.input_token_group_size
                              : block_size;
  uint32_t anchor = 0;
  if (cached_tokens == prompt_len) {
    anchor = cached_tokens < _cached_token_ids.size()
                 ? _cached_token_ids[cached_tokens]
                 : _cached_first_generated_token;
  }
  for (uint16_t offset = cached_tokens; offset < prompt_len;
       offset += prefill_width) {
    const uint16_t valid =
        std::min<uint16_t>(prefill_width, prompt_len - offset);
    anchor = run_model_once(prefill_width, offset, prompt_len, 0);
    if (!_is_running.load(std::memory_order_relaxed)) {
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
  draft_lm._cached_first_generated_token = anchor;

  std::vector<uint32_t> output{anchor};
  const double first_duration = timer_ttft->stop();
  _notify_first_token(anchor, first_duration);
  if (performance_result != nullptr) {
    performance_result->token_durations.push_back(first_duration);
    performance_result->generated_tokens = 1;
  }
  bool stopped = _stop_token_ids.contains(anchor);
  bool cache_full = false;
  DFlashScratch scratch;
  scratch.verify_input.resize(block_size);
  scratch.emitted.reserve(block_size);

  while (!stopped && _is_running.load(std::memory_order_relaxed) &&
         output.size() + prompt_len < max_length) {
    if (_kv_cache_len + block_size > _cfg.pipeline_cfg.max_num_tokens) {
      // A fixed-width DFlash package cannot safely run a partial final block.
      cache_full = true;
      break;
    }
    ChronoTimer iteration_timer(true);
    const uint16_t verify_start = _kv_cache_len;
    const auto &proposals =
        draft_lm._run_dflash_draft(*this, anchor, verify_start, scratch);
    auto &verify_input = scratch.verify_input;
    std::fill(verify_input.begin(), verify_input.end(), anchor);
    std::copy(proposals.begin(), proposals.end(), verify_input.begin() + 1);
    const auto [accepted, bonus_token] =
        _run_dflash_target_verify(verify_input, verify_start, scratch);
    auto &emitted = scratch.emitted;
    emitted.assign(proposals.begin(), proposals.begin() + accepted);
    emitted.push_back(bonus_token);
    size_t produced = std::min<size_t>(
        emitted.size(), max_length - prompt_len - output.size());
    for (size_t index = 0; index < produced; ++index) {
      if (_stop_token_ids.contains(emitted[index])) {
        produced = index + 1;
        stopped = true;
        break;
      }
    }
    const uint16_t committed_rows = static_cast<uint16_t>(produced);
    if (_capture_state_checkpoints) {
      auto boundary = std::upper_bound(_checkpoint_boundaries.begin(),
                                       _checkpoint_boundaries.end(),
                                       verify_start + committed_rows);
      if (boundary != _checkpoint_boundaries.begin()) {
        --boundary;
        if (*boundary > verify_start) {
          _save_dflash_state_checkpoint(*boundary, *boundary - verify_start,
                                        false);
        }
      }
    }
    _commit_dflash_linear_state(committed_rows);
    _kv_cache_len = verify_start + committed_rows;
    draft_lm._kv_cache_len = verify_start;
    draft_lm._append_dflash_context(*this, block_size, verify_start,
                                    committed_rows);
    const double duration = iteration_timer.stop();
    for (size_t index = 0; index < produced; ++index) {
      const auto token = emitted[index];
      output.push_back(token);
      _notify_new_token(token, duration / produced, index < accepted);
      if (performance_result != nullptr) {
        performance_result->token_durations.push_back(duration / produced);
        ++performance_result->generated_tokens;
      }
    }
    _cached_token_ids.insert(_cached_token_ids.end(), verify_input.begin(),
                             verify_input.begin() + committed_rows);
    draft_lm._cached_token_ids = _cached_token_ids;
    _cached_first_generated_token = emitted[produced - 1];
    draft_lm._cached_first_generated_token = _cached_first_generated_token;
    if (performance_result != nullptr) {
      performance_result->accepted_draft_tokens.value() +=
          std::min<size_t>(accepted, produced);
    }
    anchor = emitted[produced - 1];
    if (produced < emitted.size()) {
      cache_full = true;
    }
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
  return completed ? std::optional<std::vector<uint32_t>>(std::move(output))
                   : std::nullopt;
}

} // namespace llima
} // namespace simaai
