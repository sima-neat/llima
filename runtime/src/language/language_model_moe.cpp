#include "language/language_model_state.hpp"

#include <algorithm>
#include <stdexcept>

namespace simaai::llima {

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

} // namespace simaai::llima
