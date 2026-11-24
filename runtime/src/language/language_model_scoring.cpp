#include "language/layer_executor.hpp"
#include "language/request_scope.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace simaai::llima {

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

} // namespace

LogLikelihoodResult LanguageModel::run_model_for_loglikelihood(
    std::span<const uint32_t> input_token_ids,
    size_t continuation_start,
    std::span<const uint32_t> continuation_token_ids,
    bool use_group_prefill
) {
    RequestScope request(*this);
    auto result = _score(input_token_ids, continuation_start, continuation_token_ids, use_group_prefill);
    request.finish();
    return result;
}

LogLikelihoodResult LanguageModel::_score(
    std::span<const uint32_t> input_token_ids,
    size_t continuation_start,
    std::span<const uint32_t> continuation_token_ids,
    bool use_group_prefill
) {
    _run->require_active_request();
    if (!_cfg.pipeline_cfg.return_logits) {
        throw std::runtime_error(
            "model not compiled with --return_logits; "
            "accuracy/loglikelihood tasks are unsupported"
        );
    }
    if (input_token_ids.empty()) {
        throw std::runtime_error("Loglikelihood request must include at least one input token");
    }
    if (input_token_ids.size() > _run->limits().max_num_tokens) {
        throw std::runtime_error(
            fmt::format(
                "Loglikelihood input length {} exceeds max_num_tokens {}",
                input_token_ids.size(),
                _run->limits().max_num_tokens
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
        // through _prefill(), so the cached-token metadata is no longer valid.
        _cache->cached_token_ids.clear();
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
        _prefill(prefill_input_tokens, num_cached_tokens, std::nullopt, nullptr);
        if (_run->cancelled()) {
            throw std::runtime_error("Loglikelihood group prefill was interrupted");
        }
        const auto first_token_score = score_current_logits(continuation_token_ids[0]);
        total_logprob += first_token_score.logprob;
        is_greedy = is_greedy && first_token_score.is_greedy;
        token_idx_begin = prefill_token_count;
    }

    for (size_t token_idx = token_idx_begin; token_idx < input_token_ids.size(); ++token_idx) {
        if (_run->cancelled())
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
    return {total_logprob, is_greedy};
}


void LanguageModel::_run_model_once_for_loglikelihood_logits(
    uint16_t token_idx, uint32_t input_token_id
) {
    auto execution = LayerExecutor::lock_execution();
    constexpr uint16_t num_tokens = 1;
    const uint16_t next_token_idx = token_idx + 1;
    // PrefillTrace tests parse this message prefix.
    _logger->info("Processing token no. {}-{}", token_idx, next_token_idx);

    // Loglikelihood only needs the single-token decode path to refresh n1_buffer4.
    // Keep this separate from run_model_once(), which is shared by normal generation.
    MLABuffer* normal_input_buf = &get_buffer(_resources->embedding_offload ? "offload_decode_embeds" : "embeddings");
    const uint32_t normal_input_row = _resources->embedding_offload ? 0 : input_token_id;
    MLABuffer* normal_scale_buf = _cfg.pipeline_cfg.quantize_embeddings
        ? &get_buffer(_resources->embedding_offload ? "offload_decode_scales" : "embedding_scales") : nullptr;
    if (_resources->embedding_offload) _gather_embedding_rows({&input_token_id, 1}, *normal_input_buf, normal_scale_buf);

    if (_uses_per_layer_inputs()) {
        auto inputs = LayerExecutor::per_layer_inputs(*this, normal_input_buf,
            normal_scale_buf, normal_input_row, 1, input_token_id);
        _workspace->per_layer_model_map.at({num_tokens, 0, 0}).add_to_queue(&inputs);
    }

    for (uint8_t layer = 0; layer < _cfg.lm_cfg.num_hidden_layers; ++layer) {
        LayerExecutor::Overrides inputs;
        if (layer == 0) {
            inputs.emplace(0, MLABufferSlice{normal_input_buf, {normal_input_row, 0},
                                             {1, _cfg.lm_cfg.hidden_size}});
        }
        const auto output = layer + 1 == _cfg.lm_cfg.num_hidden_layers
            ? LayerExecutor::LayerOutput::LogitsRow : LayerExecutor::LayerOutput::AllRows;
        LayerExecutor::queue_layer(*this, num_tokens, token_idx, layer, normal_scale_buf,
                                      normal_input_row, {.input = &inputs, .post_input = &inputs}, output);
    }

    MLAModelWithBuffer::run_queue();
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        if (_cfg.lm_cfg.layer_types[layer_idx] == LayerType::LinearAttention) {
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

} // namespace simaai::llima
