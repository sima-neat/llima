//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2026 SiMa.ai, All Rights Reserved.  ||
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

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <set>

#include <fmt/ranges.h>

#include "language_model.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {


// Draft model forward pass. `this` is the draft LanguageModel. Two input
// shapes: stacked target captures (prefill, triggers FC fusion) or already-
// projected hidden_size hidden_states (decode, FC skipped).
LanguageModel::DraftForwardResult LanguageModel::run_eagle3_draft_model(
    LanguageModel& target_lm,
    std::vector<Eigen::bfloat16> hidden_states,
    std::vector<uint32_t> input_ids,
    std::optional<std::vector<uint8_t>> attention_mask,
    std::optional<std::vector<int32_t>> position_ids,
    int num_cached_tokens,
    bool is_prefill
) {
    const size_t seq_length = input_ids.size();
    size_t seq_len_with_past = seq_length;
    int past_key_values_length = 0;
    if (num_cached_tokens > 0) {
        past_key_values_length = num_cached_tokens;
        seq_len_with_past = seq_length + past_key_values_length;
    }

    // Freq upload: sliced at position_ids if given, else pristine linear host copy.
    auto& freq_real_buf = get_buffer("global_freq_real");
    auto& freq_imag_buf = get_buffer("global_freq_imag");
    if (position_ids.has_value()) {
        const auto& positions = position_ids.value();
        const uint32_t freq_dim = freq_real_buf.get_shape().back();
        const uint16_t max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;

        std::vector<Eigen::bfloat16> freq_real_padded(
            static_cast<size_t>(max_num_tokens) * freq_dim, Eigen::bfloat16{0.0f}
        );
        std::vector<Eigen::bfloat16> freq_imag_padded(
            static_cast<size_t>(max_num_tokens) * freq_dim, Eigen::bfloat16{0.0f}
        );
        for (size_t i = 0; i < positions.size(); ++i) {
            const size_t pos = static_cast<size_t>(positions[i]);
            for (size_t k = 0; k < freq_dim; ++k) {
                freq_real_padded[i * freq_dim + k] = _global_freq_host.re[pos * freq_dim + k];
                freq_imag_padded[i * freq_dim + k] = _global_freq_host.im[pos * freq_dim + k];
            }
        }
        freq_real_buf.upload(freq_real_padded.data());
        freq_imag_buf.upload(freq_imag_padded.data());
    } else {
        freq_real_buf.upload(_global_freq_host.re.data());
        freq_imag_buf.upload(_global_freq_host.im.data());
    }

    // Default mask is all-ones; tree overlay applied by prepare_attention_mask.
    if (!attention_mask.has_value()) {
        attention_mask = std::vector<uint8_t>(seq_len_with_past, 1);
    }
    auto mask = eagle_helpers::prepare_attention_mask(
        attention_mask, seq_length, past_key_values_length, _eagle3_tree_mask
    );

    // Decode mask stride must match the cache ELF's compiled bucket; a wider
    // stride misaligns rows >= 1 and collapses the tree. Prefill mask is baked
    // into the ELF, so this local buffer is unused.
    const uint16_t num_tokens = is_prefill ? 128 : 5;
    const uint16_t mask_bucket = _get_cache_mask_size(
        "full_attention", seq_len_with_past, false
    );
    const size_t pad_rows = num_tokens;
    const size_t pad_cols = is_prefill
        ? static_cast<size_t>(num_tokens)
        : static_cast<size_t>(round_up_to(static_cast<uint32_t>(seq_len_with_past), mask_bucket));
    const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
    std::vector<Eigen::bfloat16> padded_mask(pad_rows * pad_cols, neg_inf);
    // Clamp c to pad_cols — multi-group prefill (offset > 0) has
    // seq_len_with_past > 128 and unclamped writes overrun the buffer.
    for (size_t r = 0; r < seq_length; ++r) {
        for (size_t c = 0; c < std::min(seq_len_with_past, pad_cols); ++c) {
            padded_mask[r * pad_cols + c] = mask[r * seq_len_with_past + c];
        }
    }

    // Decode (n5) needs a runtime mask upload; prefill (n128) has its mask
    // baked into the graph.
    if (!is_prefill) {
        get_buffer("future_token_mask").upload(padded_mask.data());
    }

    // Drafts share the target's embedding table — read rows from target's buffer.
    const auto& target_embeddings_buf = target_lm.get_buffer("embeddings");
    const uint32_t embed_size = target_embeddings_buf.get_shape().back();  // = hidden_size
    const auto* embeddings_ptr = reinterpret_cast<const Eigen::bfloat16*>(
        target_embeddings_buf.get_virtual_addr()
    );
    std::vector<Eigen::bfloat16> input_embeds(seq_length * embed_size);
    const size_t row_bytes = embed_size * sizeof(Eigen::bfloat16);
    for (size_t i = 0; i < seq_length; ++i) {
        const uint32_t token_id = input_ids[i];
        std::memcpy(
            input_embeds.data() + i * embed_size,
            embeddings_ptr + token_id * embed_size,
            row_bytes
        );
    }

    // FC fusion: folds (seq_length, 3*hidden_size) stacked target captures down
    // to (seq_length, hidden_size). Skipped when input is already hidden_size wide.
    const size_t hidden_cols = hidden_states.size() / seq_length;
    if (hidden_cols != embed_size) {
        const size_t fc_in_elems  = static_cast<size_t>(num_tokens) * 3 * embed_size;
        const size_t fc_out_elems = static_cast<size_t>(num_tokens) * embed_size;
        // Copy only what hidden_states holds — cross-round has fewer rows than
        // input_ids (trailing bonus_token's FC input row stays zero-padded).
        const size_t valid_bytes = hidden_states.size() * sizeof(Eigen::bfloat16);
        std::vector<Eigen::bfloat16> hidden_states_padded(fc_in_elems, Eigen::bfloat16{0.0f});
        std::memcpy(hidden_states_padded.data(), hidden_states.data(), valid_bytes);

        auto& fc_input = get_buffer(fmt::format("fc_n{}_input", num_tokens));
        fc_input.upload(hidden_states_padded.data());

        auto it = _fc_model_map.find(num_tokens);
        if (it == _fc_model_map.end()) {
            throw std::runtime_error(
                "run_eagle3_draft_model: FC model not defined for n="
                + std::to_string(num_tokens)
            );
        }
        // IFM override forces the ELF to read the full (num_tokens, 3*hidden_size)
        // shape instead of its static default (which may be smaller).
        std::map<uint8_t, MLABufferSlice> ifm_map;
        ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(0),
            std::forward_as_tuple(
                &fc_input,
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_tokens, 3 * embed_size}
            )
        );

        // add_to_queue runs synchronously when SIMA_LLIMA_RUN_DISABLE_QUEUE=1.
        it->second.add_to_queue(&ifm_map);

        auto& fc_output = get_buffer(fmt::format("fc_n{}_output", num_tokens));
        fc_output.invalidate_cache();
        std::vector<Eigen::bfloat16> fc_out_full(fc_out_elems);
        fc_output.download(fc_out_full.data());
        hidden_states.assign(
            fc_out_full.begin(),
            fc_out_full.begin() + seq_length * embed_size
        );
    } else {
        // FC skipped, but pre_model IFM[1] is statically wired to
        // fc_n{N}_output — upload hidden_states there to overwrite stale data.
        std::vector<Eigen::bfloat16> hidden_states_padded(
            static_cast<size_t>(num_tokens) * embed_size, Eigen::bfloat16{0.0f}
        );
        std::memcpy(
            hidden_states_padded.data(),
            hidden_states.data(),
            seq_length * embed_size * sizeof(Eigen::bfloat16)
        );
        auto& fc_output = get_buffer(fmt::format("fc_n{}_output", num_tokens));
        fc_output.upload(hidden_states_padded.data());
    }
    // Stage token embeds for pre_model IFM[0]; IFM[1] is the FC output above.
    std::vector<Eigen::bfloat16> input_embeds_padded(
        static_cast<size_t>(num_tokens) * embed_size, Eigen::bfloat16{0.0f}
    );
    std::memcpy(
        input_embeds_padded.data(),
        input_embeds.data(),
        seq_length * embed_size * sizeof(Eigen::bfloat16)
    );
    auto& token_embeds_buf = get_buffer(fmt::format("n{}_buffer1", num_tokens));
    token_embeds_buf.upload(input_embeds_padded.data());

    // model_key = (num_tokens, 0, past_kv_len) — picks the ELF with the right
    // KV-cache write offset, so no OFM override needed.
    const uint8_t  layer_idx = 0;
    const uint16_t token_idx = static_cast<uint16_t>(past_key_values_length);
    const LanguageModelMapKey model_key{num_tokens, layer_idx, token_idx};
    auto& fc_output_buf = get_buffer(fmt::format("fc_n{}_output", num_tokens));

    // pre_model dispatch with explicit IFM begins/shapes for token embeds
    // (IFM[0]) and FC output (IFM[1]).
    std::map<uint8_t, MLABufferSlice> pre_ifm_map;
    pre_ifm_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(0),
        std::forward_as_tuple(
            &token_embeds_buf,
            std::vector<uint32_t>{0, 0},
            std::vector<uint32_t>{num_tokens, embed_size}
        )
    );
    pre_ifm_map.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(1),
        std::forward_as_tuple(
            &fc_output_buf,
            std::vector<uint32_t>{0, 0},
            std::vector<uint32_t>{num_tokens, embed_size}
        )
    );
    // Override freq IFMs (2, 3) to read from row 0 — both the sliced and the
    // pristine linear paths upload starting at row 0.
    {
        const uint32_t freq_dim = freq_real_buf.get_shape().back();
        pre_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(2),
            std::forward_as_tuple(
                &freq_real_buf,
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_tokens, freq_dim}
            )
        );
        pre_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(3),
            std::forward_as_tuple(
                &freq_imag_buf,
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_tokens, freq_dim}
            )
        );
    }
    _pre_model_map.at(model_key).add_to_queue(&pre_ifm_map);

    // cache_model dispatch. IFMs/OFMs are all statically wired, no overrides.
    _cache_model_map.at(model_key).add_to_queue();

    // post_model dispatch (lm_head fused in). Override IFM[0] to fc_n{N}_output
    // (same buffer pre_model used as IFM[1]); IFM[1] is statically wired.
    std::map<uint8_t, MLABufferSlice> post_ifm_map;
    post_ifm_map.emplace(0, MLABufferSlice{&fc_output_buf});
    _post_model_map.at(model_key).add_to_queue(&post_ifm_map);

    // Download both outputs: hidden_states (n{N}_buffer5) and logits
    // (n{N}_buffer4, lm_head fused into post_model; single split for draft).
    const uint32_t draft_vocab_size = _cfg.lm_cfg.get_lm_head_output_size();
    DraftForwardResult result;
    {
        auto& hidden_buf = get_buffer(fmt::format("n{}_buffer5", num_tokens));
        hidden_buf.invalidate_cache();
        const size_t hidden_elems = static_cast<size_t>(num_tokens) * embed_size;
        result.hidden_states.resize(hidden_elems);
        hidden_buf.download(result.hidden_states.data());
        // Slice to seq_length rows (drop padded rows).
        result.hidden_states.resize(seq_length * embed_size);
    }
    {
        auto& logits_buf = get_buffer(fmt::format("n{}_buffer4", num_tokens));
        logits_buf.invalidate_cache();
        const size_t logits_elems = static_cast<size_t>(num_tokens) * draft_vocab_size;
        result.logits.resize(logits_elems);
        logits_buf.download(result.logits.data());
    }

    return result;
}


// Target verify forward pass. K tree candidates with absolute position_ids
// (= tree_position_ids + input_ids.shape[-1]). `this` is the target. Loops
// through all target layers, captures hidden states at layers 2, N/2, N-3,
// downloads fused lm_head logits at the end.
LanguageModel::TargetVerifyResult LanguageModel::run_eagle3_target_verify(
    std::vector<uint32_t> input_ids,
    std::vector<int32_t> position_ids
) {
    const int num_cached_tokens = static_cast<int>(_eagle3_stable_kv);

    // num_tokens = 16 for target verify; seq_length is the valid candidate count.
    const uint16_t num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    const uint8_t  num_layers = _cfg.lm_cfg.num_hidden_layers;
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    const auto&    layer_types = _cfg.lm_cfg.layer_types;
    const size_t   seq_length = input_ids.size();
    const int      past_key_values_length = num_cached_tokens;
    const size_t   seq_len_with_past = seq_length + past_key_values_length;

    if (position_ids.size() != seq_length) {
        throw std::runtime_error(fmt::format(
            "run_eagle3_target_verify: position_ids size {} != input_ids size {}",
            position_ids.size(), seq_length
        ));
    }

    // Freq upload: gather rows at position_ids from pristine host, zero-pad to
    // (max_num_tokens, freq_dim), upload to row 0. When SWA is enabled, do the
    // same for the local freq table so sliding-attention layers see the right
    // rotary base.
    auto upload_freq_rows = [&](
        const RopeTable& host, const std::string& real_name, const std::string& imag_name
    ) {
        auto& real_buf = get_buffer(real_name);
        auto& imag_buf = get_buffer(imag_name);
        const uint32_t freq_dim = real_buf.get_shape().back();
        const uint16_t max_num_tokens = _cfg.pipeline_cfg.max_num_tokens;
        std::vector<Eigen::bfloat16> real_padded(
            static_cast<size_t>(max_num_tokens) * freq_dim, Eigen::bfloat16{0.0f}
        );
        std::vector<Eigen::bfloat16> imag_padded(
            static_cast<size_t>(max_num_tokens) * freq_dim, Eigen::bfloat16{0.0f}
        );
        for (size_t i = 0; i < seq_length; ++i) {
            const size_t pos = static_cast<size_t>(position_ids[i]);
            for (size_t k = 0; k < freq_dim; ++k) {
                real_padded[i * freq_dim + k] = host.re[pos * freq_dim + k];
                imag_padded[i * freq_dim + k] = host.im[pos * freq_dim + k];
            }
        }
        real_buf.upload(real_padded.data());
        imag_buf.upload(imag_padded.data());
    };
    upload_freq_rows(_global_freq_host, "global_freq_real", "global_freq_imag");
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        upload_freq_rows(_local_freq_host, "local_freq_real", "local_freq_imag");
    }
    auto& freq_real_buf = get_buffer("global_freq_real");
    auto& freq_imag_buf = get_buffer("global_freq_imag");

    // Attention mask: ones over seq_len_with_past, overlay tree_mask, pad to
    // (16, aligned_cols), upload to future_token_mask.
    auto attention_mask = std::optional<std::vector<uint8_t>>{
        std::vector<uint8_t>(seq_len_with_past, 1)
    };
    auto mask = eagle_helpers::prepare_attention_mask(
        attention_mask, seq_length, past_key_values_length, _eagle3_tree_mask
    );

    // pad_cols must match the cache ELF's compiled stride: rounding up
    // seq_len_with_past to mask_bucket. Wider strides misalign rows >= 1.
    const uint16_t mask_bucket = _get_cache_mask_size(
        "full_attention", seq_len_with_past, false
    );
    const size_t pad_rows = num_tokens;
    const size_t pad_cols = static_cast<size_t>(
        round_up_to(static_cast<uint32_t>(seq_len_with_past), mask_bucket)
    );
    const Eigen::bfloat16 neg_inf{-std::numeric_limits<float>::infinity()};
    std::vector<Eigen::bfloat16> padded_mask(pad_rows * pad_cols, neg_inf);
    for (size_t r = 0; r < seq_length; ++r) {
        for (size_t c = 0; c < seq_len_with_past; ++c) {
            padded_mask[r * pad_cols + c] = mask[r * seq_len_with_past + c];
        }
    }
    get_buffer("future_token_mask").upload(padded_mask.data());

    // Stage input embeddings. The same `input_embeds` buffer is reused for both
    // n128 prefill and n16 verify; we write the first K rows here.
    const auto& embeddings_buf = get_buffer("embeddings");
    const auto* embeddings_ptr = reinterpret_cast<const Eigen::bfloat16*>(
        embeddings_buf.get_virtual_addr()
    );
    std::vector<Eigen::bfloat16> input_embeds_padded(
        static_cast<size_t>(num_tokens) * hidden_size, Eigen::bfloat16{0.0f}
    );
    const size_t row_bytes = hidden_size * sizeof(Eigen::bfloat16);
    for (size_t i = 0; i < seq_length; ++i) {
        const uint32_t token_id = input_ids[i];
        std::memcpy(
            input_embeds_padded.data() + i * hidden_size,
            embeddings_ptr + token_id * hidden_size,
            row_bytes
        );
    }
    auto& input_embeds_buf = get_buffer("input_embeds");
    input_embeds_buf.upload(input_embeds_padded.data());

    // Per-layer loop; capture hidden states at layers {2, N/2, N-3}. model_key
    // bakes the KV write offset, so no OFM override needed.
    const uint16_t token_idx = static_cast<uint16_t>(past_key_values_length);
    const std::vector<uint8_t> capture_layers = {
        2,
        static_cast<uint8_t>(num_layers / 2),
        static_cast<uint8_t>(num_layers - 3),
    };
    TargetVerifyResult result;
    result.hidden_states.resize(capture_layers.size());

    for (uint8_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        if (layer_types[layer_idx] != "full_attention"
            && layer_types[layer_idx] != "sliding_attention") {
            // Target verify is attention-only; conv layers don't apply here.
            continue;
        }

        const LanguageModelMapKey model_key{num_tokens, layer_idx, token_idx};

        // Slot 0 (input_embeds for layer 0) is shared between pre and post —
        // both have empty-placeholder bindings, so build the override once.
        std::map<uint8_t, MLABufferSlice> ifm_map;
        if (layer_idx == 0) {
            ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    &input_embeds_buf,
                    std::vector<uint32_t>{0, 0},
                    std::vector<uint32_t>{num_tokens, hidden_size}
                )
            );
        }

        // Freq IFMs (slots 1, 2) need pre-only overrides — redirect static
        // (past_kv_len, 0) read to row 0. Leaking these into post would
        // corrupt its slots 1, 2 (cache_ofms / per_layer_input).
        // Pick local vs global freq buffers based on layer type so sliding
        // layers consume rope_local_base_freq and full layers consume rope_theta.
        const bool is_sliding = layer_types[layer_idx] == "sliding_attention";
        auto& layer_freq_real = is_sliding
            ? get_buffer("local_freq_real") : freq_real_buf;
        auto& layer_freq_imag = is_sliding
            ? get_buffer("local_freq_imag") : freq_imag_buf;
        const uint32_t layer_freq_dim = layer_freq_real.get_shape().back();
        auto pre_ifm_map = ifm_map;
        pre_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(1),
            std::forward_as_tuple(
                &layer_freq_real,
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_tokens, layer_freq_dim}
            )
        );
        pre_ifm_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(2),
            std::forward_as_tuple(
                &layer_freq_imag,
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_tokens, layer_freq_dim}
            )
        );

        _pre_model_map.at(model_key).add_to_queue(&pre_ifm_map);
        _cache_model_map.at(model_key).add_to_queue();
        _post_model_map.at(model_key).add_to_queue(&ifm_map);

        // post_model.ofms[0] writes n{N}_buffer1 (next layer's IFM[0]), so
        // reading it here yields this layer's output.
        auto it = std::find(capture_layers.begin(), capture_layers.end(), layer_idx);
        if (it != capture_layers.end()) {
            const size_t cap_idx = std::distance(capture_layers.begin(), it);
            auto& buf = get_buffer(fmt::format("n{}_buffer1", num_tokens));
            buf.invalidate_cache();
            const size_t num_elems = static_cast<size_t>(num_tokens) * hidden_size;
            std::vector<Eigen::bfloat16> full_cap(num_elems);
            buf.download(full_cap.data());
            // Slice to seq_length rows (drop padding).
            result.hidden_states[cap_idx].assign(
                full_cap.begin(), full_cap.begin() + seq_length * hidden_size
            );
        }
    }

    // Download lm_head logits (fused into the last layer's post_model);
    // multi-split heads concat the n16_lm_split{i} buffers along vocab.
    const uint32_t lm_head_output_size = _cfg.lm_cfg.get_lm_head_output_size();
    const auto num_splits = _cfg.lm_cfg.lm_head_num_splits;
    const uint32_t split_dim = _cfg.lm_cfg.lm_head_split_dim;
    result.logits.resize(static_cast<size_t>(num_tokens) * lm_head_output_size);

    for (uint32_t s = 0; s < num_splits; ++s) {
        const std::string buf_name = (num_splits == 1)
            ? fmt::format("n{}_buffer4", num_tokens)
            : fmt::format("n{}_lm_split{}", num_tokens, s);
        auto& buf = get_buffer(buf_name);
        buf.invalidate_cache();
        const uint32_t this_split_dim = (num_splits == 1)
            ? lm_head_output_size
            : std::min<uint32_t>(split_dim, lm_head_output_size - s * split_dim);
        std::vector<Eigen::bfloat16> split_data(
            static_cast<size_t>(num_tokens) * this_split_dim
        );
        buf.download(split_data.data());
        for (uint16_t r = 0; r < num_tokens; ++r) {
            std::memcpy(
                result.logits.data() + r * lm_head_output_size + s * split_dim,
                split_data.data() + r * this_split_dim,
                this_split_dim * sizeof(Eigen::bfloat16)
            );
        }
    }

    return result;
}


// Wraps run_eagle3_target_verify with the position_ids shift and the
// retrieve_indices logit gather. `this` is the target.
LanguageModel::TreeDecodingResult LanguageModel::tree_decoding(
    std::vector<uint32_t> tree_candidates,
    std::vector<int32_t> tree_position_ids,
    std::span<const uint32_t> input_ids,
    const std::vector<std::vector<int32_t>>& retrieve_indices
) {

    // Absolute position_ids = tree_position_ids + input_ids length.
    const int32_t shift = static_cast<int32_t>(input_ids.size());
    std::vector<int32_t> position_ids(tree_position_ids.size());
    for (size_t i = 0; i < tree_position_ids.size(); ++i) {
        position_ids[i] = tree_position_ids[i] + shift;
    }

    auto target_out = run_eagle3_target_verify(
        std::move(tree_candidates),
        std::move(position_ids)
    );

    // Gather logits[p][d] = tree_logits[retrieve_indices[p][d]]. -1 entries
    // (path-padding) map to the last row of tree_logits.
    const uint32_t vocab_size = _cfg.lm_cfg.get_lm_head_output_size();
    const uint16_t K = _cfg.lm_cfg.get_single_num_tokens();
    const size_t n_paths   = retrieve_indices.size();
    const size_t max_depth = n_paths ? retrieve_indices[0].size() : 0;

    TreeDecodingResult result;
    result.vocab_size = vocab_size;
    result.logits.resize(n_paths * max_depth * vocab_size);
    for (size_t p = 0; p < n_paths; ++p) {
        for (size_t d = 0; d < max_depth; ++d) {
            int32_t idx = retrieve_indices[p][d];
            if (idx < 0) idx = static_cast<int32_t>(K) - 1;
            std::memcpy(
                result.logits.data() + (p * max_depth + d) * vocab_size,
                target_out.logits.data() + static_cast<size_t>(idx) * vocab_size,
                vocab_size * sizeof(Eigen::bfloat16)
            );
        }
    }

    // Concat the 3 layer captures along the feature axis:
    // row i = [cap0[i], cap1[i], cap2[i]].
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    const size_t valid_K = target_out.hidden_states.empty()
        ? 0
        : target_out.hidden_states[0].size() / hidden_size;
    result.hidden_states.resize(valid_K * 3 * hidden_size);
    for (size_t i = 0; i < valid_K; ++i) {
        for (size_t cap = 0; cap < target_out.hidden_states.size(); ++cap) {
            std::memcpy(
                result.hidden_states.data() + (i * 3 + cap) * hidden_size,
                target_out.hidden_states[cap].data() + i * hidden_size,
                hidden_size * sizeof(Eigen::bfloat16)
            );
        }
    }

    return result;
}


// Per-iteration update after target verify + accept/reject. `this` is the
// target; `draft_lm` is the draft (for the next topk_generate call).
LanguageModel::UpdateInferenceInputsResult LanguageModel::update_inference_inputs(
    LanguageModel& draft_lm,
    std::vector<uint32_t> input_ids,
    const std::vector<std::vector<int32_t>>& candidates,
    size_t best_candidate,
    int32_t accept_length,
    const std::vector<std::vector<int32_t>>& retrieve_indices,
    int32_t new_token,
    const std::vector<Eigen::bfloat16>& hidden_state_new,
    const std::vector<Eigen::bfloat16>& sample_p
) {
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    const size_t prev_input_len = input_ids.size();
    const size_t take = static_cast<size_t>(accept_length) + 1;

    // Absolute KV positions of the accepted path.
    std::vector<uint16_t> select_indices(take);
    for (size_t i = 0; i < take; ++i) {
        select_indices[i] = static_cast<uint16_t>(
            retrieve_indices[best_candidate][i] + static_cast<int32_t>(prev_input_len)
        );
    }

    // Extend input_ids with the accepted path.
    input_ids.reserve(prev_input_len + take + 1);
    for (size_t i = 0; i < take; ++i) {
        input_ids.push_back(
            static_cast<uint32_t>(candidates[best_candidate][i])
        );
    }

    // Move accepted-path KVs to contiguous positions [prev_input_len, +take).
    compact_kv_after_accept(select_indices, static_cast<uint16_t>(prev_input_len));

    // Gather `take` rows from hidden_state_new (K × 3*hidden_size, row-major).
    std::vector<Eigen::bfloat16> accept_hidden_state_new(
        take * 3 * hidden_size
    );
    for (size_t i = 0; i < take; ++i) {
        const int32_t src_row = retrieve_indices[best_candidate][i];
        std::memcpy(
            accept_hidden_state_new.data() + i * 3 * hidden_size,
            hidden_state_new.data() + static_cast<size_t>(src_row) * 3 * hidden_size,
            3 * hidden_size * sizeof(Eigen::bfloat16)
        );
    }

    auto bonus_it = std::max_element(sample_p.begin(), sample_p.end(),
        [](const Eigen::bfloat16& a, const Eigen::bfloat16& b) {
            return static_cast<float>(a) < static_cast<float>(b);
        });
    const uint32_t bonus_token = static_cast<uint32_t>(
        std::distance(sample_p.begin(), bonus_it)
    );

    // Inject bonus_token into topk_generate only; persisting it in input_ids
    // would shift next tree_decoding's RoPE and break topk_generate's slice.
    std::vector<uint32_t> input_ids_with_bonus = input_ids;
    input_ids_with_bonus.push_back(bonus_token);

    // Build next round's tree from the accepted hidden states. Caller must sync
    // target's _eagle3_stable_kv with draft's after this returns.
    auto tg = draft_lm.topk_generate(
        *this,
        std::move(input_ids_with_bonus),
        std::move(accept_hidden_state_new),
        /*num_cached_tokens=*/static_cast<int>(draft_lm._eagle3_stable_kv),
        /*is_prefill=*/false
    );

    new_token += accept_length + 1;

    // Cache for next-turn prefix matching. input_ids excludes the bonus, so
    // it matches target's _kv_cache_len after compact_kv_after_accept.
    _cached_token_ids.assign(input_ids.begin(), input_ids.end());
    draft_lm._cached_token_ids.assign(input_ids.begin(), input_ids.end());

    UpdateInferenceInputsResult result;
    result.input_ids         = std::move(input_ids);
    result.draft_tokens      = std::move(tg.draft_tokens);
    result.retrieve_indices  = std::move(tg.retrieve_indices);
    result.tree_mask         = std::move(tg.tree_mask);
    result.tree_position_ids = std::move(tg.tree_position_ids);
    result.new_token         = new_token;
    result.bonus_token       = bonus_token;
    return result;
}


// EAGLE3 draft-side tree expansion. One draft prefill (or cross-round decode)
// seeded by target captures, then `depth - 1` topk + cumulative-score growth
// iterations, then final tree-mask / position-ids / retrieve-indices assembly.
// `this` is the draft.
LanguageModel::TopkGenerateResult LanguageModel::topk_generate(
    LanguageModel& target_lm,
    std::vector<uint32_t> input_ids,
    std::vector<Eigen::bfloat16> hidden_states,
    int num_cached_tokens,
    bool is_prefill
) {
    TopkGenerateResult result;

    // total_tokens = target's speculative_budget - 1 (typically 15 from 16).
    // topk = draft's speculative_budget (typically 5). depth is hardcoded for now.
    const int total_tokens = static_cast<int>(
        target_lm._cfg.lm_cfg.speculative_decoding_cfg.value().speculative_budget
    ) - 1;
    const int depth = 4;
    const int topk  = static_cast<int>(
        _cfg.lm_cfg.speculative_decoding_cfg.value().speculative_budget
    );

    const uint32_t sample_token = input_ids.back();

    // Per-iter rows: first append is 1×K from the prefill topk, depth iters
    // append K×K cu_scores. Concatenated below for tree finalization.
    std::vector<std::vector<std::vector<float>>>    scores_list;
    std::vector<std::vector<int32_t>>               parents_list;
    std::vector<std::vector<std::vector<uint32_t>>> ss_token;

    // EAGLE shift: drop input_ids[0]; draft K cache row i corresponds to
    // input position i+1.
    input_ids.erase(input_ids.begin());

    // Tracks the position of the most recent K candidates; the depth loop
    // increments by 1 per iter.
    size_t len_posi = input_ids.size();

    // Clear stale tree mask on both target and draft (shared shared_ptr).
    _eagle3_tree_mask.reset();
    target_lm._eagle3_tree_mask.reset();

    // Cross-round input_ids has one MORE entry than hidden_states (bonus_token
    // has no hidden state yet). Use hidden_rows, not input_ids.size(), for
    // stable_kv advance and last_hidden indexing.
    const uint32_t embed_size_for_kv = _cfg.lm_cfg.hidden_size;
    const size_t hidden_rows = hidden_states.size() / (3 * embed_size_for_kv);
    const size_t stable_kv_before = _eagle3_stable_kv;
    input_ids.erase(input_ids.begin(), input_ids.begin() + _eagle3_stable_kv);

    DraftForwardResult draft_out;
    size_t last_row;

    if (is_prefill) {
        // Multi-group draft prefill: input_ids can exceed the n=128 batch. Each
        // chunk writes K-cache at row `offset`; only the LAST chunk's outputs
        // are consumed. Per-chunk position_ids pick the correct freq slice.
        const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
        const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
        const uint16_t total_tokens = static_cast<uint16_t>(input_ids.size());
        const size_t row_stride_3h = static_cast<size_t>(3) * embed_size_for_kv;
        uint16_t last_valid_in_chunk = 0;

        for (size_t i = 0; i < offsets.size(); ++i) {
            const uint16_t offset = offsets[i];
            if (offset >= total_tokens) break;
            // Skip chunks fully covered by the prefix cache; the partial chunk
            // straddling the boundary is the last one we run.
            if (offset + group_size <= static_cast<uint16_t>(num_cached_tokens)) {
                continue;
            }
            const uint16_t valid = std::min<uint16_t>(group_size, total_tokens - offset);

            std::vector<uint32_t> chunk_input_ids(
                input_ids.begin() + offset,
                input_ids.begin() + offset + valid
            );
            std::vector<Eigen::bfloat16> chunk_hidden(
                hidden_states.begin() + static_cast<size_t>(offset) * row_stride_3h,
                hidden_states.begin() + static_cast<size_t>(offset + valid) * row_stride_3h
            );
            std::vector<int32_t> chunk_position_ids(valid);
            for (uint16_t j = 0; j < valid; ++j) {
                chunk_position_ids[j] = static_cast<int32_t>(offset + j);
            }

            draft_out = run_eagle3_draft_model(
                target_lm,
                std::move(chunk_hidden),
                chunk_input_ids,
                /*attention_mask=*/std::nullopt,
                /*position_ids=*/chunk_position_ids,
                /*num_cached_tokens=*/static_cast<int>(offset),
                /*is_prefill=*/true
            );
            last_valid_in_chunk = valid;
        }

        hidden_states = std::move(draft_out.hidden_states);
        // Last valid row of the last chunk = where lm_head logits for the
        // final input position live.
        last_row = static_cast<size_t>(last_valid_in_chunk) - 1;
        _eagle3_stable_kv += hidden_rows;
    } else {
        // Cross-round single dispatch (n=5). Absolute position_ids are required
        // so the per-token freq slice matches stable_kv+1..+K, not 0..K-1.
        std::vector<int32_t> pos(input_ids.size());
        for (size_t i = 0; i < pos.size(); ++i) {
            pos[i] = static_cast<int32_t>(stable_kv_before + 1 + i);
        }
        std::optional<std::vector<int32_t>> first_call_position_ids = std::move(pos);

        draft_out = run_eagle3_draft_model(
            target_lm,
            std::move(hidden_states),
            input_ids,
            /*attention_mask=*/std::nullopt,
            /*position_ids=*/first_call_position_ids,
            /*num_cached_tokens=*/static_cast<int>(_eagle3_stable_kv),
            /*is_prefill=*/false
        );
        hidden_states = std::move(draft_out.hidden_states);
        // Trim trailing bonus_token row (zero-FC slot, uninformative).
        hidden_states.resize(hidden_rows * embed_size_for_kv);
        last_row = hidden_rows - 1;
        _eagle3_stable_kv += hidden_rows;
    }

    // lm_head logits row for the final input position.
    const uint32_t draft_vocab_size = _cfg.lm_cfg.get_lm_head_output_size();
    const Eigen::bfloat16* lm_output_draft =
        draft_out.logits.data() + last_row * draft_vocab_size;

    // Top-K over the lm_head logits row, then logsoftmax. Both shapes are
    // (1, topk) — leading 1 mirrors lm_output_draft's (1, vocab_size).
    auto topk_result = simaai::llima::topk<Eigen::bfloat16>(
        lm_output_draft, /*rows=*/1, /*cols=*/draft_vocab_size, topk
    );
    std::vector<std::vector<int32_t>> topk_index = std::move(topk_result.indices);
    std::vector<std::vector<Eigen::bfloat16>> topk_p_raw = std::move(topk_result.values);

    std::vector<std::vector<float>> topk_p(topk_p_raw.size());
    for (size_t r = 0; r < topk_p_raw.size(); ++r) {
        topk_p[r] = logsoftmax<Eigen::bfloat16>(
            topk_p_raw[r].data(), topk_p_raw[r].size()
        );
    }

    // Log topk values (convert bf16 → float for fmt).
    std::vector<std::vector<float>> topk_p_raw_f(topk_p_raw.size());
    for (size_t r = 0; r < topk_p_raw.size(); ++r) {
        topk_p_raw_f[r].resize(topk_p_raw[r].size());
        for (size_t i = 0; i < topk_p_raw[r].size(); ++i)
            topk_p_raw_f[r][i] = static_cast<float>(topk_p_raw[r][i]);
    }

    // scores must be a copy, not a reference: the depth loop rebinds it later
    // and we don't want to mutate topk_p[0].
    auto scores = topk_p[0];

    scores_list.push_back({scores});       // (1, K)
    parents_list.push_back({0});           // (1,)

    // Map draft-vocab top-K indices to target-vocab tokens via d2t.
    std::vector<uint32_t> target_tokens(topk_index[0].size());
    for (size_t i = 0; i < topk_index[0].size(); ++i) {
        const int32_t draft_idx = topk_index[0][i];
        target_tokens[i] = static_cast<uint32_t>(draft_idx + _d2t[draft_idx]);
    }
    ss_token.push_back({target_tokens});
    input_ids = std::move(target_tokens);  // (K,) for next draft_forward

    // last_hidden = last row of the draft forward's hidden_states.
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    const size_t hs_rows = hidden_states.size() / hidden_size;
    std::vector<Eigen::bfloat16> last_hidden(
        hidden_states.begin() + (hs_rows - 1) * hidden_size,
        hidden_states.end()
    );

    // Broadcast last_hidden across all K rows for the depth loop.
    std::vector<Eigen::bfloat16> input_hidden(
        static_cast<size_t>(topk) * hidden_size
    );
    for (int i = 0; i < topk; ++i) {
        std::memcpy(
            input_hidden.data() + i * hidden_size,
            last_hidden.data(),
            hidden_size * sizeof(Eigen::bfloat16)
        );
    }

    // Copy because the depth-loop concat rebinds `tree_mask` later.
    auto tree_mask = _eagle3_tree_mask_init;

    std::vector<int32_t> topk_cs_index(topk);
    std::iota(topk_cs_index.begin(), topk_cs_index.end(), 0);

    // Reuse the function parameter slot; the depth loop advances it by topk
    // each iter.
    num_cached_tokens = static_cast<int>(_eagle3_stable_kv);

    for (int i = 0; i < depth - 1; ++i) {
        // Share the current tree mask with the target via shared_ptr.
        _eagle3_tree_mask = std::make_shared<eagle_helpers::EagleTreeMask>(tree_mask);
        target_lm._eagle3_tree_mask = _eagle3_tree_mask;

        std::vector<int32_t> position_ids(_eagle3_position_ids.size());
        for (size_t k = 0; k < _eagle3_position_ids.size(); ++k) {
            position_ids[k] = static_cast<int32_t>(len_posi) + _eagle3_position_ids[k];
        }

        auto draft_out_d = run_eagle3_draft_model(
            target_lm,
            std::move(input_hidden),
            input_ids,
            /*attention_mask=*/std::nullopt,
            /*position_ids=*/position_ids,
            /*num_cached_tokens=*/static_cast<int>(num_cached_tokens),
            /*is_prefill=*/false
        );
        auto out_hidden = std::move(draft_out_d.hidden_states);

        len_posi += 1;
        num_cached_tokens += topk;

        // Bias bookkeeping for finalize-time indexing into flat ss_token /
        // scores. iter i: parents = topk_cs_index + (1 + topk^2*max(0,i-1)
        // + (topk if i > 0 else 0)).
        const int32_t bias1 = (i > 0) ? topk : 0;
        const int32_t bias2 = std::max(0, i - 1);
        const int32_t bias  = 1 + topk * topk * bias2 + bias1;
        std::vector<int32_t> parents(topk_cs_index.size());
        for (size_t k = 0; k < topk_cs_index.size(); ++k) {
            parents[k] = topk_cs_index[k] + bias;
        }
        parents_list.push_back(parents);

        // Per-candidate top-K + logsoftmax. lm_head is fused into post_model,
        // so logits live in draft_out_d.logits at shape (topk, draft_vocab).
        auto last_p_topk = simaai::llima::topk<Eigen::bfloat16>(
            draft_out_d.logits.data(),
            /*rows=*/static_cast<size_t>(topk),
            /*cols=*/draft_vocab_size,
            topk
        );
        std::vector<std::vector<int32_t>> topk_index_d = std::move(last_p_topk.indices);
        std::vector<std::vector<Eigen::bfloat16>> topk_p_raw_d = std::move(last_p_topk.values);

        std::vector<std::vector<float>> topk_p_d(topk_p_raw_d.size());
        for (size_t r = 0; r < topk_p_raw_d.size(); ++r) {
            topk_p_d[r] = logsoftmax<Eigen::bfloat16>(
                topk_p_raw_d[r].data(), topk_p_raw_d[r].size()
            );
        }

        // Log (convert bf16 → float for fmt).
        std::vector<std::vector<float>> topk_p_raw_d_f(topk_p_raw_d.size());
        for (size_t r = 0; r < topk_p_raw_d.size(); ++r) {
            topk_p_raw_d_f[r].resize(topk_p_raw_d[r].size());
            for (size_t k = 0; k < topk_p_raw_d[r].size(); ++k)
                topk_p_raw_d_f[r][k] = static_cast<float>(topk_p_raw_d[r][k]);
        }

        // cu_scores[r][k] = topk_p_d[r][k] + scores[r]. Shape (topk, topk).
        std::vector<std::vector<float>> cu_scores(topk_p_d.size());
        for (size_t r = 0; r < topk_p_d.size(); ++r) {
            cu_scores[r].resize(topk_p_d[r].size());
            for (size_t k = 0; k < topk_p_d[r].size(); ++k) {
                cu_scores[r][k] = topk_p_d[r][k] + scores[r];
            }
        }

        // Top-K over flattened (topk × topk) cu_scores — the K best
        // candidate continuations across all K parent rows.
        std::vector<float> flat_cu(static_cast<size_t>(topk) * topk);
        for (int r = 0; r < topk; ++r) {
            for (int k = 0; k < topk; ++k) {
                flat_cu[r * topk + k] = cu_scores[r][k];
            }
        }
        auto cs_topk = simaai::llima::topk<float>(
            flat_cu.data(),
            /*rows=*/1,
            /*cols=*/static_cast<size_t>(topk) * topk,
            topk
        );
        std::vector<int32_t> topk_cs_index = std::move(cs_topk.indices[0]);
        std::vector<float>   topk_cs_p     = std::move(cs_topk.values[0]);
        scores = topk_cs_p;

        // out_ids[k] = parent row in cu_scores for the k-th selected candidate.
        std::vector<int32_t> out_ids(topk_cs_index.size());
        for (size_t k = 0; k < topk_cs_index.size(); ++k) {
            out_ids[k] = topk_cs_index[k] / topk;
        }

        // Gather hidden rows by out_ids → shape (topk, hidden_size).
        std::vector<Eigen::bfloat16> new_input_hidden(
            static_cast<size_t>(topk) * embed_size_for_kv
        );
        for (size_t k = 0; k < topk_cs_index.size(); ++k) {
            const size_t src_row = static_cast<size_t>(out_ids[k]);
            std::memcpy(
                new_input_hidden.data() + k * embed_size_for_kv,
                out_hidden.data() + src_row * embed_size_for_kv,
                embed_size_for_kv * sizeof(Eigen::bfloat16)
            );
        }
        input_hidden = std::move(new_input_hidden);

        // Next-iter input_ids: gather draft top-K by topk_cs_index, then d2t.
        input_ids.resize(topk_cs_index.size());
        for (size_t k = 0; k < topk_cs_index.size(); ++k) {
            const size_t flat_idx = static_cast<size_t>(topk_cs_index[k]);
            const int32_t draft_idx = topk_index_d[flat_idx / topk][flat_idx % topk];
            input_ids[k] = static_cast<uint32_t>(draft_idx + _d2t[draft_idx]);
        }

        // Map all (K, K) draft indices to target tokens via d2t.
        std::vector<std::vector<uint32_t>> ss_token_entry_d(topk_index_d.size());
        for (size_t r = 0; r < topk_index_d.size(); ++r) {
            ss_token_entry_d[r].resize(topk_index_d[r].size());
            for (size_t k = 0; k < topk_index_d[r].size(); ++k) {
                const int32_t draft_idx = topk_index_d[r][k];
                ss_token_entry_d[r][k] = static_cast<uint32_t>(draft_idx + _d2t[draft_idx]);
            }
        }
        ss_token.push_back(ss_token_entry_d);
        scores_list.push_back(cu_scores);

        // Grow tree_mask: gather K rows by out_ids, then append the K-identity
        // (tree_mask_init) along the col axis → (K, prev_cols + K).
        {
            const size_t prev_cols = tree_mask.data[0][0][0].size();
            const size_t new_cols  = prev_cols + static_cast<size_t>(topk);
            std::vector<std::vector<std::vector<std::vector<float>>>> new_tree_data(
                1, std::vector<std::vector<std::vector<float>>>(
                    1, std::vector<std::vector<float>>(
                        static_cast<size_t>(topk), std::vector<float>(new_cols, 0.0f)
                    )
                )
            );
            for (size_t r = 0; r < static_cast<size_t>(topk); ++r) {
                const size_t src_row = static_cast<size_t>(out_ids[r]);
                for (size_t c = 0; c < prev_cols; ++c) {
                    new_tree_data[0][0][r][c] = tree_mask.data[0][0][src_row][c];
                }
                for (size_t c = 0; c < static_cast<size_t>(topk); ++c) {
                    new_tree_data[0][0][r][prev_cols + c] =
                        _eagle3_tree_mask_init.data[0][0][r][c];
                }
            }
            tree_mask.data = std::move(new_tree_data);
        }
    }

    // Tree finalization. Flatten scores_list and ss_token to 1D arrays of size
    // K + (depth-1)*K*K (= 55 for K=5, depth=3).
    std::vector<float> scores_list_flat;
    scores_list_flat.reserve(topk + 2 * topk * topk);  // = 55
    for (const auto& entry : scores_list) {
        for (const auto& row : entry) {
            scores_list_flat.insert(scores_list_flat.end(), row.begin(), row.end());
        }
    }

    std::vector<uint32_t> ss_token_list;
    ss_token_list.reserve(topk + 2 * topk * topk);
    for (const auto& entry : ss_token) {
        for (const auto& row : entry) {
            ss_token_list.insert(ss_token_list.end(), row.begin(), row.end());
        }
    }

    // Top `total_tokens` (= speculative_budget - 1) of the flat scores.
    auto top_scores = simaai::llima::topk<float>(
        scores_list_flat.data(),
        /*rows=*/1,
        /*cols=*/scores_list_flat.size(),
        total_tokens
    );
    std::vector<int32_t> top_scores_index = std::move(top_scores.indices[0]);
    std::vector<float>   top_score_values = std::move(top_scores.values[0]);

    std::sort(top_scores_index.begin(), top_scores_index.end());

    // Build the surviving draft tokens: gather ss_token_list by top_scores_index,
    // then prepend the root sample_token.
    std::vector<uint32_t> draft_tokens(top_scores_index.size());
    for (size_t i = 0; i < top_scores_index.size(); ++i) {
        draft_tokens[i] = ss_token_list[top_scores_index[i]];
    }
    draft_tokens.insert(draft_tokens.begin(), sample_token);

    // Flatten parents_list: size = 1 + (depth-1)*K.
    std::vector<int32_t> draft_parents_flat;
    draft_parents_flat.reserve(1 + (depth - 1) * topk);
    for (const auto& entry : parents_list) {
        draft_parents_flat.insert(draft_parents_flat.end(), entry.begin(), entry.end());
    }

    // For each survivor, look up its parent index (integer-div by topk).
    std::vector<int64_t> draft_parents(top_scores_index.size());
    for (size_t i = 0; i < top_scores_index.size(); ++i) {
        draft_parents[i] = static_cast<int64_t>(
            draft_parents_flat[top_scores_index[i] / topk]
        );
    }

    // For each parent, find its position in the sorted top_scores_index —
    // a back-pointer into the surviving entries.
    std::vector<int32_t> queries(draft_parents.size());
    for (size_t i = 0; i < draft_parents.size(); ++i) {
        queries[i] = static_cast<int32_t>(draft_parents[i] - 1);
    }
    auto mask_index_raw = eagle_helpers::searchsorted<int32_t>(top_scores_index, queries);
    std::vector<int64_t> mask_index(mask_index_raw.size());
    for (size_t i = 0; i < mask_index_raw.size(); ++i) {
        mask_index[i] = static_cast<int64_t>(mask_index_raw[i]);
    }

    // Root-parent (draft_parents == 0) sentinel: shift so mask_index[i] = 0 means
    // root. Set -1 first, then +1 across the board.
    for (size_t i = 0; i < draft_parents.size(); ++i) {
        if (draft_parents[i] == 0) mask_index[i] = -1;
    }
    for (size_t i = 0; i < mask_index.size(); ++i) mask_index[i] += 1;

    // Build tree_mask: identity + col 0 = 1, then for each non-root i propagate
    // its parent's row via OR.
    const int N = total_tokens + 1;
    std::vector<std::vector<int>> tree_mask_2d(N, std::vector<int>(N, 0));
    for (int i = 0; i < N; ++i) tree_mask_2d[i][i] = 1;  // identity
    for (int i = 0; i < N; ++i) tree_mask_2d[i][0] = 1;  // col 0 = 1
    for (int i = 0; i < total_tokens; ++i) {
        const int src = static_cast<int>(mask_index[i]);
        for (int j = 0; j < N; ++j) {
            tree_mask_2d[i + 1][j] = tree_mask_2d[i + 1][j] | tree_mask_2d[src][j];
        }
    }

    // Per-row count of 1s minus 1 → node depth in the tree.
    std::vector<int32_t> tree_position_ids(N);
    for (int r = 0; r < N; ++r) {
        int sum = 0;
        for (int c = 0; c < N; ++c) sum += tree_mask_2d[r][c];
        tree_position_ids[r] = sum - 1;
    }

    // Wrap as 4D (1, 1, N, N) float tensor.
    eagle_helpers::EagleTreeMask final_tree_mask;
    final_tree_mask.data = std::vector<std::vector<std::vector<std::vector<float>>>>(
        1, std::vector<std::vector<std::vector<float>>>(
            1, std::vector<std::vector<float>>(N, std::vector<float>(N, 0.0f))
        )
    );
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            final_tree_mask.data[0][0][r][c] = static_cast<float>(tree_mask_2d[r][c]);
        }
    }

    std::vector<std::vector<uint32_t>> draft_tokens_2d{draft_tokens};

    // Build retrieve_indices: for each leaf node, walk back to the root via
    // mask_index and fill in the path right-to-left.
    const int max_depth = *std::max_element(
        tree_position_ids.begin(), tree_position_ids.end()
    ) + 1;
    std::set<int64_t> noleaf_set(mask_index.begin(), mask_index.end());
    const int noleaf_num = static_cast<int>(noleaf_set.size()) - 1;
    const int leaf_num = total_tokens - noleaf_num;

    std::vector<std::vector<int32_t>> retrieve_indices(
        leaf_num, std::vector<int32_t>(max_depth, -1)
    );

    int rid = 0;
    for (int i = 0; i < total_tokens + 1; ++i) {
        if (noleaf_set.find(static_cast<int64_t>(i)) != noleaf_set.end()) continue;
        int cid = i;
        const int depth_i = tree_position_ids[i];
        for (int j = depth_i; j >= 0; --j) {
            retrieve_indices[rid][j] = cid;
            if (cid > 0) {
                cid = static_cast<int>(mask_index[cid - 1]);
            }
        }
        ++rid;
    }

    result.draft_tokens = std::move(draft_tokens);
    result.retrieve_indices = std::move(retrieve_indices);
    result.tree_mask = std::move(final_tree_mask);
    result.tree_position_ids = std::move(tree_position_ids);

    return result;
}


// Target prefill (per-layer loop with hidden-state captures at layers
// 2, N/2, N-3) followed by topk_generate to seed the initial draft tree.
// `this` is the target; `draft_lm` is the draft.
LanguageModel::InitTreeResult LanguageModel::initialize_tree(
    LanguageModel& draft_lm,
    std::vector<uint32_t> input_ids,
    int num_cached_tokens
) {
    InitTreeResult result;

    const uint16_t num_tokens = _cfg.pipeline_cfg.input_token_group_size;
    const uint16_t num_input_tokens = static_cast<uint16_t>(input_ids.size());

    get_buffer("global_freq_real").upload(_global_freq_host.re.data());
    get_buffer("global_freq_imag").upload(_global_freq_host.im.data());

    // Uploads token embeds and returns the prefix-match length against
    // _cached_token_ids.
    const uint16_t prefix_cached = _set_input_text_embeds(input_ids);

    // Full-match early-out: every input token is already cached on both target
    // and draft sides — restore the tree state saved last turn.
    if (prefix_cached == num_input_tokens && !_cached_draft_tokens.empty()) {
        result.token              = _cached_first_generated_token;
        result.draft_tokens       = _cached_draft_tokens;
        result.retrieve_indices   = _cached_retrieve_indices;
        result.tree_mask          = _cached_tree_mask;
        result.tree_position_ids  = _cached_tree_position_ids;
        _eagle3_stable_kv          = _cached_eagle3_stable_kv;
        draft_lm._eagle3_stable_kv = draft_lm._cached_eagle3_stable_kv;
        return result;
    }

    // Multi-group target prefill. Per-group capture of intermediate hidden
    // states (layers 2, N/2, N-3) must be copied out before the next dispatch
    // overwrites the buffer — queue disabled for synchronous per-layer download.
    const uint32_t hidden_size = _cfg.lm_cfg.hidden_size;
    const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();

    // Pre-allocate the 3 capture buffers at full input length; each group's
    // memcpy fills its slice.
    result.hidden_states.resize(3);
    for (auto& cap : result.hidden_states) {
        cap.assign(static_cast<size_t>(num_input_tokens) * hidden_size, Eigen::bfloat16{0.0f});
    }

    for (size_t i = 0; i < offsets.size(); ++i) {
        const uint16_t offset = offsets[i];
        if (offset >= num_input_tokens) break;
        // Skip chunks fully covered by the prefix cache; lm_head row comes
        // from the last RUN chunk (draft mirrors this in topk_generate).
        if (static_cast<size_t>(offset) + num_tokens
            <= static_cast<size_t>(prefix_cached)) {
            continue;
        }
        const uint16_t valid_rows = std::min<uint16_t>(num_tokens, num_input_tokens - offset);

        // result.token is only meaningful at the LAST group; earlier ones get
        // overwritten by subsequent dispatches.
        result.token = run_model_once(num_tokens, offset, num_input_tokens, 0);

        // Copy this group's captures into the accumulator before next dispatch.
        for (size_t cap_idx = 0; cap_idx < result.hidden_states.size(); ++cap_idx) {
            std::memcpy(
                result.hidden_states[cap_idx].data()
                    + static_cast<size_t>(offset) * hidden_size,
                _eagle3_intermediate_hidden_states[cap_idx].data(),
                static_cast<size_t>(valid_rows) * hidden_size * sizeof(Eigen::bfloat16)
            );
        }
    }

    _logger->info("root token: {}", result.token);

    // Cache for next-turn prefix matching. Must run BEFORE push_back so
    // input_ids still matches the pre-root-token prefill length.
    _cached_token_ids.assign(input_ids.begin(), input_ids.end());
    draft_lm._cached_token_ids.assign(input_ids.begin(), input_ids.end());
    _cached_first_generated_token = result.token;

    input_ids.push_back(result.token);

    // Concat the 3 layer captures along the feature axis:
    // row i = [cap0[i], cap1[i], cap2[i]].
    std::vector<Eigen::bfloat16> hidden_states_concat(
        static_cast<size_t>(num_input_tokens) * 3 * hidden_size
    );
    for (uint16_t i = 0; i < num_input_tokens; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            std::memcpy(
                hidden_states_concat.data() + (i * 3 + c) * hidden_size,
                result.hidden_states[c].data() + i * hidden_size,
                hidden_size * sizeof(Eigen::bfloat16)
            );
        }
    }

    // Pass prefix_cached (not the caller's num_cached_tokens, which is always 0)
    // so the draft skips the same cached chunks the target just skipped.
    auto topk_result = draft_lm.topk_generate(
        *this, std::move(input_ids), std::move(hidden_states_concat),
        static_cast<int>(prefix_cached), /*is_prefill=*/true
    );
    result.draft_tokens       = std::move(topk_result.draft_tokens);
    result.retrieve_indices   = std::move(topk_result.retrieve_indices);
    result.tree_mask          = std::move(topk_result.tree_mask);
    result.tree_position_ids  = std::move(topk_result.tree_position_ids);

    // Cache tree state for the next turn's full-match early-out.
    // _cached_first_generated_token was set above (before the push_back).
    _cached_draft_tokens       = result.draft_tokens;
    _cached_retrieve_indices   = result.retrieve_indices;
    _cached_tree_mask          = result.tree_mask;
    _cached_tree_position_ids  = result.tree_position_ids;
    _cached_eagle3_stable_kv          = _eagle3_stable_kv;
    draft_lm._cached_eagle3_stable_kv = draft_lm._eagle3_stable_kv;

    return result;
}


// EAGLE3 speculative-decoding driver. Loops initialize_tree → tree_decoding →
// update_inference_inputs while streaming accepted tokens. Caller serializes
// invocations (VLM wrapper holds the run mutex).
std::optional<std::vector<uint32_t>> LanguageModel::run_model_speculative_decoding(
    LanguageModel& draft_lm,
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens
) {
    std::vector<uint32_t> input_ids(input_token_ids.begin(), input_token_ids.end());
    const size_t input_len = input_ids.size();
    const int num_cached_tokens = 0;

    const uint16_t max_length = override_max_num_tokens.has_value()
        ? override_max_num_tokens.value()
        : _cfg.pipeline_cfg.max_num_tokens;
    const size_t spec_budget =
        _cfg.lm_cfg.speculative_decoding_cfg.value().speculative_budget;

    // Pre-check: if the input itself already leaves no room for a verify
    // dispatch, bail out before initialize_tree. Going further would lead to
    // topk_generate's cross-round decode using a token_idx beyond the
    // registered range, throwing map::at.
    if (input_len + spec_budget > max_length) {
        _logger->info(
            "stop (pre-init): input_len {} + spec_budget {} > max_length {}",
            input_len, spec_budget, max_length
        );
        _text_streamer.push(DecodeCallbackType::CACHE_FULL, 0, 0);
        _text_streamer.wait_streaming();
        return std::nullopt;
    }

    draft_lm._eagle3_stable_kv = 0;

    // Paired-clear keeps target and draft shared_ptr both null so neither
    // sees a stale mask from a previous run.
    _eagle3_tree_mask.reset();
    draft_lm._eagle3_tree_mask.reset();

    std::vector<uint16_t> total_generated_tokens;

    auto init = initialize_tree(draft_lm, input_ids, num_cached_tokens);

    // Sync target's _eagle3_stable_kv with draft's — the draft incremented it
    // inside initialize_tree's first topk_generate, and tree_decoding reads it.
    _eagle3_stable_kv = draft_lm._eagle3_stable_kv;

    // Loop-carried state: seeded from initialize_tree, rebound each iter by
    // update_inference_inputs.
    auto draft_tokens       = std::move(init.draft_tokens);
    auto tree_position_ids  = std::move(init.tree_position_ids);
    auto retrieve_indices   = std::move(init.retrieve_indices);
    auto tree_mask          = std::move(init.tree_mask);
    int32_t new_token       = 0;

    // TPS timer starts here, after all prefill work.
    const auto run_model_begin = std::chrono::steady_clock::now();
    auto iter_begin = run_model_begin;
    bool first_token = true;
    bool cache_full = false;

    size_t idx = 0;
    for (; idx < max_length; ++idx) {
        // Cache-full guard up front: tree_decoding would otherwise dispatch
        // with token_idx >= max_num_tokens for which no model_key is
        // registered, leading to a map::at throw.
        if (input_ids.size() + spec_budget > max_length) {
            _logger->info(
                "[iter {}] stop (pre-verify): input_ids.size + spec_budget > max_length", idx
            );
            cache_full = true;
            break;
        }
        // Share the current tree_mask across target and draft via shared_ptr.
        _eagle3_tree_mask = std::make_shared<eagle_helpers::EagleTreeMask>(tree_mask);
        draft_lm._eagle3_tree_mask = _eagle3_tree_mask;

        auto td = tree_decoding(
            draft_tokens, tree_position_ids, input_ids, retrieve_indices
        );

        // candidates[p][d] = draft_tokens[retrieve_indices[p][d]]; -1 entries
        // mean "path padding" and stay as -1 in the output.
        const size_t n_paths   = retrieve_indices.size();
        const size_t max_depth = retrieve_indices[0].size();
        std::vector<std::vector<int32_t>> candidates_2d(
            n_paths, std::vector<int32_t>(max_depth, -1)
        );
        std::vector<int32_t> candidates_flat(n_paths * max_depth, -1);
        for (size_t p = 0; p < n_paths; ++p) {
            for (size_t d = 0; d < max_depth; ++d) {
                const int32_t r_idx = retrieve_indices[p][d];
                if (r_idx >= 0) {
                    const int32_t tok =
                        static_cast<int32_t>(draft_tokens[r_idx]);
                    candidates_2d[p][d] = tok;
                    candidates_flat[p * max_depth + d] = tok;
                }
            }
        }

        auto post = eagle_helpers::evaluate_posterior(
            td.logits.data(),
            candidates_flat.data(),
            n_paths,
            max_depth,
            td.vocab_size
        );
        total_generated_tokens.push_back(static_cast<uint16_t>(post.accept_length + 1));

        auto upd = update_inference_inputs(
            draft_lm,
            input_ids,
            candidates_2d,
            post.best_candidate,
            post.accept_length,
            retrieve_indices,
            new_token,
            td.hidden_states,
            post.sample_p
        );

        // Sync target's stable_kv with draft's — same pattern as after
        // initialize_tree.
        _eagle3_stable_kv = draft_lm._eagle3_stable_kv;

        const size_t prev_size = input_ids.size();
        input_ids          = std::move(upd.input_ids);
        draft_tokens       = std::move(upd.draft_tokens);
        retrieve_indices   = std::move(upd.retrieve_indices);
        tree_mask          = std::move(upd.tree_mask);
        tree_position_ids  = std::move(upd.tree_position_ids);
        new_token          = upd.new_token;

        // Stream the newly-accepted tokens. Per-token duration = iter time /
        // accepted count; first emitted token gets TTFT, rest get TPS.
        const auto iter_end = std::chrono::steady_clock::now();
        const double iter_duration = std::chrono::duration<double>(
            iter_end - iter_begin
        ).count();
        const size_t accepted = input_ids.size() - prev_size;
        const double per_token = accepted > 0
            ? iter_duration / static_cast<double>(accepted)
            : 0.0;
        for (size_t i = prev_size; i < input_ids.size(); ++i) {
            const auto type = first_token
                ? DecodeCallbackType::TTFT
                : DecodeCallbackType::TPS;
            // The first emitted token per round (offset 0) is the tree's seed
            // — initialize_tree's root in round 1, or the previous round's
            // bonus in later rounds — and is target-generated, not from draft.
            // Offsets 1..accept_length are the draft-accepted path.
            const size_t offset = i - prev_size;
            const bool from_draft = (offset > 0) && (offset <= post.accept_length);
            _text_streamer.push(type, input_ids[i], per_token, from_draft);
            first_token = false;
        }
        iter_begin = iter_end;

        // Stop when any configured stop token appears in the generated tail.
        // Cache-overflow is checked at the top of the next iteration.
        const auto& stop_token_ids = get_stop_token_ids();
        if (std::any_of(
                input_ids.begin() + input_len, input_ids.end(),
                [&](uint32_t t) { return stop_token_ids.contains(t); }
            )) {
            _logger->info("[iter {}] stop: stop token in generated tokens", idx);
            break;
        }
    }

    // TPS summary.
    const auto run_model_duration = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - run_model_begin
    ).count();
    const size_t rounds = idx + 1;
    const size_t total_tokens = std::accumulate(
        total_generated_tokens.begin(), total_generated_tokens.end(), size_t{0}
    );
    _logger->info(
        "EAGLE generated {} tokens in {:.2f}s. TPS = {:.2f}",
        total_tokens, run_model_duration,
        run_model_duration > 0.0
            ? static_cast<double>(total_tokens) / run_model_duration
            : 0.0
    );
    _logger->info(
        "DECODING completed in {} rounds ({:.2f} tokens generated per round) | {:.2f}s per round",
        rounds,
        static_cast<double>(total_tokens) / static_cast<double>(rounds),
        run_model_duration / static_cast<double>(rounds)
    );

    // Signal end-of-stream and wait for the streamer thread to flush.
    // Use CACHE_FULL when generation stopped because the K-cache was exhausted,
    // mirroring non-spec; the streamer prints the "Cache full" notice.
    _text_streamer.push(
        cache_full ? DecodeCallbackType::CACHE_FULL : DecodeCallbackType::STOP, 0, 0
    );
    _text_streamer.wait_streaming();

    return std::vector<uint32_t>(input_ids.begin() + input_len, input_ids.end());
}


}  // namespace llima
}  // namespace simaai
