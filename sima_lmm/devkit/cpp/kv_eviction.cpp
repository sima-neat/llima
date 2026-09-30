#include "kv_eviction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "mla_buffer.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

namespace {

[[noreturn]] void reject(const std::string& reason) {
    throw std::runtime_error("kv_eviction: " + reason);
}

// RoPE types whose rotation depends only on the position.
bool is_supported_rope_type(const std::string& rope_type) {
    return rope_type.empty() || rope_type == "default" || rope_type == "linear"
        || rope_type == "llama3";
}

// The trained context length from the Hugging Face config, if known.
std::optional<uint32_t> read_trained_max_positions(const std::filesystem::path& devkit_dir) {
    std::ifstream stream(devkit_dir / "config.json");
    const auto config = nlohmann::json::parse(stream, nullptr, false);
    const auto it = config.find("max_position_embeddings");
    if (it == config.end() || !it->is_number_unsigned() || it->get<uint64_t>() == 0) {
        return std::nullopt;
    }
    return static_cast<uint32_t>(
        std::min<uint64_t>(it->get<uint64_t>(), std::numeric_limits<uint32_t>::max())
    );
}

} // namespace


KvEvictionLimits resolve_kv_eviction_limits(
    const VlmConfig& cfg,
    const KvEvictionConfig& kv_eviction,
    std::optional<uint32_t> trained_max_positions
) {
    const auto& lm_cfg = cfg.lm_cfg;
    if (cfg.is_multimodal()) {
        reject("vision-language models are not supported yet; use a text-only model");
    }
    if (lm_cfg.is_spec_decode()) {
        reject("speculative decoding (EAGLE3/MTP) is not supported with eviction");
    }
    if (lm_cfg.attn_cfg.swa_enable) {
        reject("models with sliding-window attention are not supported yet");
    }
    for (const auto& layer_type : lm_cfg.layer_types) {
        if (layer_type != "full_attention") {
            reject(fmt::format(
                "layer type '{}' is not supported; every layer must use full attention",
                layer_type
            ));
        }
    }
    if (lm_cfg.num_kv_shared_layers > 0) {
        reject("models with shared KV layers are not supported");
    }
    if (lm_cfg.hidden_size_per_layer_input > 0) {
        reject("models with per-layer inputs are not supported");
    }
    const auto& rope_type = lm_cfg.rope_cfg.rope_scaling.rope_type;
    if (!is_supported_rope_type(rope_type)) {
        reject(fmt::format(
            "RoPE type '{}' is not supported; supported types are default, linear and llama3",
            rope_type
        ));
    }

    const auto& pipeline_cfg = cfg.pipeline_cfg;
    const bool use_groups = pipeline_cfg.input_token_group_offsets.has_value()
        && !pipeline_cfg.input_token_group_offsets->empty();

    KvEvictionLimits limits{};
    limits.capacity = pipeline_cfg.max_num_tokens;
    limits.group_size = use_groups ? pipeline_cfg.input_token_group_size : 1;
    const uint32_t capacity = limits.capacity;
    const uint32_t group_size = limits.group_size;

    // The recent window covers a group: prefill may restart at the previous group boundary.
    const uint32_t min_recent = std::max<uint32_t>(group_size, KV_EVICTION_MIN_RECENT_TOKENS);
    const uint32_t min_budget = KV_EVICTION_SINK_TOKENS + 2 * min_recent;

    // Prefill continues after an eviction with a group at slot `budget`.
    auto& offsets = limits.group_offsets;
    if (use_groups) {
        for (const auto offset : pipeline_cfg.input_token_group_offsets.value()) {
            if (offset + group_size <= capacity) {
                offsets.push_back(offset);
            }
        }
    }
    auto is_valid_budget = [&](uint32_t value) {
        return use_groups
            ? std::find(offsets.begin(), offsets.end(), value) != offsets.end()
            : value < capacity;
    };

    uint32_t budget = 0;
    if (kv_eviction.budget_tokens.has_value()) {
        budget = kv_eviction.budget_tokens.value();
        if (!is_valid_budget(budget)) {
            if (use_groups) {
                reject(fmt::format(
                    "budget_tokens {} must be a compiled prefill group offset that leaves room "
                    "for one group of {} tokens; valid values: {}",
                    budget, group_size, fmt::join(offsets, ", ")
                ));
            }
            reject(fmt::format(
                "budget_tokens {} must be smaller than the compiled cache ({} tokens)",
                budget, capacity
            ));
        }
    } else {
        // Measured on Modalix: keydiff recalls as well with half the cache and evicts less.
        const uint32_t headroom = kv_eviction.policy == "keydiff"
            ? capacity / 2
            : std::max<uint32_t>(group_size, capacity / 8);
        const uint32_t target = capacity - headroom;
        if (use_groups) {
            for (const uint32_t offset : offsets) {
                if (offset <= target) budget = std::max(budget, offset);
            }
        } else {
            budget = target;
        }
    }
    if (budget < min_budget) {
        reject(fmt::format(
            "budget of {} tokens is too small for this model (compiled cache {} tokens, "
            "prefill group {}); use at least {}",
            budget, capacity, group_size, min_budget
        ));
    }
    limits.budget = static_cast<uint16_t>(budget);
    limits.recent_tokens = static_cast<uint16_t>(std::clamp<uint32_t>(
        budget / 4, min_recent, std::max<uint32_t>(min_recent, KV_EVICTION_MAX_RECENT_TOKENS)
    ));

    // Token indices are uint16_t, with room for one padded group.
    const uint32_t type_limit = std::numeric_limits<uint16_t>::max() - group_size;
    const uint32_t max_tokens = std::min(trained_max_positions.value_or(type_limit), type_limit);
    if (max_tokens <= capacity) {
        reject(fmt::format(
            "the model's trained context ({} tokens) already fits in the compiled cache ({} "
            "tokens), so eviction cannot extend it",
            max_tokens, capacity
        ));
    }
    limits.max_tokens = static_cast<uint16_t>(max_tokens);
    return limits;
}


namespace {

// int8 rows are loaded without their per-row scale, which cosine similarity ignores.
void load_key_row(const KvHeadView& head, uint16_t slot, std::span<float> out) {
    const uint8_t* row = head.keys + static_cast<size_t>(slot) * head.row_stride;
    for (uint16_t d = 0; d < head.head_dim; ++d) {
        if (head.keys_int8) {
            out[d] = static_cast<float>(reinterpret_cast<const int8_t*>(row)[d]);
        } else {
            uint16_t bfloat16_bits;
            std::memcpy(&bfloat16_bits, row + 2 * static_cast<size_t>(d), sizeof(bfloat16_bits));
            const uint32_t float_bits = static_cast<uint32_t>(bfloat16_bits) << 16;
            std::memcpy(&out[d], &float_bits, sizeof(float));
        }
    }
}

float dot(std::span<const float> a, std::span<const float> b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

// StreamingLLM-style baseline: keep the newest tokens.
class SinkWindowPolicy final : public KvEvictionPolicy {
    public:
        std::string_view name() const override { return "sink_window"; }
        void score(
            const KvHeadView&, uint16_t begin, uint16_t end, std::span<float> scores
        ) const override {
            for (uint16_t slot = begin; slot < end; ++slot) {
                scores[slot - begin] = static_cast<float>(slot);
            }
        }
};

// KeyDiff (arXiv 2504.15364): score = -cos(key, mean of the L2-normalized keys).
class KeyDiffPolicy final : public KvEvictionPolicy {
    public:
        std::string_view name() const override { return "keydiff"; }
        void score(
            const KvHeadView& head, uint16_t begin, uint16_t end, std::span<float> scores
        ) const override {
            std::vector<float> key(head.head_dim);
            std::vector<float> anchor(head.head_dim, 0.0f);
            for (uint16_t slot = 0; slot < head.num_valid; ++slot) {
                load_key_row(head, slot, key);
                const float norm = std::sqrt(dot(key, key));
                if (norm > 0.0f) {
                    for (uint16_t d = 0; d < head.head_dim; ++d) anchor[d] += key[d] / norm;
                }
            }
            // The sum has the direction of the mean.
            const float anchor_norm = std::sqrt(dot(anchor, anchor));
            for (uint16_t slot = begin; slot < end; ++slot) {
                load_key_row(head, slot, key);
                const float norms = std::sqrt(dot(key, key)) * anchor_norm;
                scores[slot - begin] = norms > 0.0f ? -dot(key, anchor) / norms : 0.0f;
            }
        }
};

} // namespace


std::unique_ptr<KvEvictionPolicy> make_kv_eviction_policy(std::string_view name) {
    if (name == "sink_window") {
        return std::make_unique<SinkWindowPolicy>();
    }
    if (name == "keydiff") {
        return std::make_unique<KeyDiffPolicy>();
    }
    throw std::runtime_error(fmt::format(
        "kv_eviction: unknown policy '{}'; use off, sink_window or keydiff", name
    ));
}


void select_kv_slots(
    const KvEvictionPolicy& policy,
    const KvHeadView& head,
    uint16_t pinned,
    uint16_t recent,
    uint16_t keep,
    std::vector<float>& scores,
    std::vector<uint16_t>& kept_slots
) {
    const uint16_t middle_begin = pinned;
    const uint16_t middle_end = head.num_valid - recent;
    const uint16_t middle_keep = keep - pinned - recent;

    kept_slots.clear();
    kept_slots.reserve(keep);
    for (uint16_t slot = 0; slot < pinned; ++slot) {
        kept_slots.push_back(slot);
    }
    if (middle_keep > 0) {
        const uint16_t middle_size = middle_end - middle_begin;
        scores.resize(middle_size);
        policy.score(head, middle_begin, middle_end, scores);
        std::vector<uint16_t> order(middle_size);
        std::iota(order.begin(), order.end(), 0);
        const auto better = [&](uint16_t a, uint16_t b) {
            return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
        };
        std::nth_element(order.begin(), order.begin() + middle_keep, order.end(), better);
        order.resize(middle_keep);
        std::sort(order.begin(), order.end());
        for (const auto offset : order) {
            kept_slots.push_back(middle_begin + offset);
        }
    }
    for (uint16_t slot = middle_end; slot < head.num_valid; ++slot) {
        kept_slots.push_back(slot);
    }
}


namespace {

// Rows of one KV head inside a cache or scale buffer.
struct KvHeadRows {
    uint8_t* base;
    size_t stride;     // Bytes between consecutive cache slots.
    size_t row_bytes;  // Bytes to move per slot.
};

// Strided caches and all scale buffers are [kv_heads, max_num_tokens, dim]. Non-strided caches
// are [max_num_tokens, kv_heads * head_dim], with each head's head_dim values adjacent.
KvHeadRows kv_head_rows(const MLABuffer& buf, uint32_t head, uint32_t head_dim) {
    auto* data = reinterpret_cast<uint8_t*>(buf.get_virtual_addr());
    const auto& shape = buf.get_shape();
    if (shape.size() == 3) {
        const size_t begin = buf.get_buf_addr_offset(std::vector<uint32_t>{head, 0, 0});
        return {
            data + begin,
            buf.get_buf_addr_offset(std::vector<uint32_t>{head, 1, 0}) - begin,
            buf.get_buf_len(std::vector<uint32_t>{1, 1, static_cast<uint32_t>(shape[2])}),
        };
    }
    return {
        data + buf.get_buf_addr_offset(std::vector<uint32_t>{0, head * head_dim}),
        buf.get_buf_addr_offset(std::vector<uint32_t>{1, 0}),
        static_cast<size_t>(head_dim) * buf.get_elem_size(),
    };
}

// Moves the kept rows (ascending) to the front. kept[i] >= i, so the forward copy is safe.
void compact_kv_head_rows(const KvHeadRows& rows, const std::vector<uint16_t>& kept) {
    for (size_t i = 0; i < kept.size(); ++i) {
        if (kept[i] != i) {
            std::memcpy(
                rows.base + i * rows.stride, rows.base + kept[i] * rows.stride, rows.row_bytes
            );
        }
    }
}

} // namespace


KvEviction::KvEviction(
    const VlmConfig& cfg,
    const KvEvictionConfig& settings,
    const std::filesystem::path& devkit_dir,
    BufferLookup get_buffer,
    std::shared_ptr<spdlog::logger> logger
) : _get_buffer(std::move(get_buffer)), _logger(std::move(logger)) {
    _policy = make_kv_eviction_policy(settings.policy);
    const auto trained_max_positions = read_trained_max_positions(devkit_dir);
    _limits = resolve_kv_eviction_limits(cfg, settings, trained_max_positions);

    // The same table as the model's global RoPE buffers.
    const auto& lm_cfg = cfg.lm_cfg;
    const auto rope_table = [&](uint16_t num_positions) {
        auto rope_scaling = lm_cfg.rope_cfg.rope_scaling;
        return calc_freq_real_imag(
            num_positions,
            rope_scaling.rope_type,
            lm_cfg.rope_cfg.rope_theta,
            lm_cfg.rope_cfg.get_rope_dimension_count("full_attention"),
            lm_cfg.attn_cfg.get_head_dim("full_attention"),
            rope_scaling
        );
    };
    _rope = rope_table(_limits.max_tokens + _limits.group_size);
    _uploaded_rope = rope_table(_limits.capacity);

    _num_layers = lm_cfg.num_hidden_layers;
    _num_kv_heads = lm_cfg.attn_cfg.num_key_value_heads;
    _head_dim = lm_cfg.attn_cfg.get_head_dim("full_attention");
    _keys_int8 = cfg.pipeline_cfg.quantize_kv_cache;

    if (!trained_max_positions.has_value()) {
        _logger->warn(
            "KV eviction: devkit/config.json has no max_position_embeddings; conversations are "
            "limited to {} tokens only by the runtime", _limits.max_tokens
        );
    }
    _logger->info(
        "KV eviction enabled: policy={} cache={} budget={} recent={} max_tokens={}",
        settings.policy, _limits.capacity, _limits.budget, _limits.recent_tokens,
        _limits.max_tokens
    );
}


uint16_t KvEviction::resume_or_reset(uint16_t num_cached_tokens, uint16_t pinned_tokens) {
    _pinned_tokens = std::max(KV_EVICTION_SINK_TOKENS, pinned_tokens);
    const uint16_t min_recent = std::max(_limits.group_size, KV_EVICTION_MIN_RECENT_TOKENS);
    if (static_cast<uint32_t>(_pinned_tokens) + min_recent > _limits.budget) {
        throw std::runtime_error(fmt::format(
            "kv_eviction: the pinned prefix ({} tokens) and the newest {} tokens do not fit in "
            "the budget of {} tokens; raise budget_tokens or shorten the system prompt",
            _pinned_tokens, min_recent, _limits.budget
        ));
    }
    // Prefill restarts up to one group before the first uncached token.
    const uint16_t restart_token_idx = num_cached_tokens
        - std::min<uint16_t>(num_cached_tokens, _limits.group_size - 1);
    if (_state.can_restart_at(restart_token_idx)) {
        return num_cached_tokens;
    }
    _logger->info(
        "KV eviction: the conversation changed before token {}, which is no longer cached in "
        "order; re-processing the whole conversation", restart_token_idx
    );
    reset();
    return 0;
}


uint16_t KvEviction::reserve_slots(uint16_t token_idx, uint16_t num_tokens) {
    if (token_idx == 0) {
        // A new sequence starts from an empty cache.
        reset();
    }
    if (num_tokens == 1 && _state.slot(token_idx) >= _limits.capacity) {
        evict(token_idx);
    }
    const uint16_t slot = _state.slot(token_idx);
    if (_state.evicted == 0) {
        // Row r still holds the RoPE values of position r.
        return slot;
    }
    for (const auto& [name, table] : {
        std::pair{"global_freq_real", &_rope.re}, std::pair{"global_freq_imag", &_rope.im}
    }) {
        auto& buf = _get_buffer(name);
        const uint32_t freq_dim = buf.get_shape().back();
        const size_t host_row_bytes = static_cast<size_t>(freq_dim) * sizeof(Eigen::bfloat16);
        const size_t device_row_bytes = buf.get_buf_len(std::vector<uint32_t>{1, freq_dim});
        const auto* src = table->data() + static_cast<size_t>(token_idx) * freq_dim;
        const size_t dst = buf.get_buf_addr_offset(std::vector<uint32_t>{slot, 0});
        if (device_row_bytes == host_row_bytes) {
            buf.upload_raw(src, dst, num_tokens * host_row_bytes);
        } else {
            for (uint16_t row = 0; row < num_tokens; ++row) {
                buf.upload_raw(
                    src + static_cast<size_t>(row) * freq_dim,
                    dst + row * device_row_bytes, host_row_bytes
                );
            }
        }
    }
    return slot;
}


bool KvEviction::evict_for_prefill(uint16_t token_idx, uint16_t num_input_tokens) {
    const uint16_t free_slots = _limits.capacity - _state.slot(token_idx);
    if (_state.evicted == 0 && num_input_tokens - token_idx <= free_slots) {
        return false;
    }
    // Group prefill is faster than single steps, and an evicting conversation overflows again.
    evict(token_idx);
    return true;
}


void KvEviction::evict(uint16_t next_token_idx) {
    ChronoTimer timer(true);
    const uint16_t valid = _state.slot(next_token_idx);
    const uint16_t pinned = _pinned_tokens;
    const uint16_t recent = std::min<uint16_t>(_limits.recent_tokens, _limits.budget - pinned);

    // Per layer: key, value, key scale and value scale (no scales without int8 caches).
    std::vector<std::array<MLABuffer*, 4>> layers;
    for (uint32_t layer_idx = 0; layer_idx < _num_layers; ++layer_idx) {
        const auto buffer = [&](const char* name) {
            return &_get_buffer(fmt::format("{}_l{}", name, layer_idx));
        };
        layers.push_back({
            buffer("cache_key"),
            buffer("cache_val"),
            _keys_int8 ? buffer("cache_key_scale") : nullptr,
            _keys_int8 ? buffer("cache_val_scale") : nullptr,
        });
    }

    // Layers are independent.
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < layers.size(); ++i) {
        const auto& buffers = layers[i];
        std::vector<float> scores;
        std::vector<uint16_t> kept;
        for (MLABuffer* buf : buffers) {
            if (buf) buf->invalidate_cache();
        }
        for (uint32_t head = 0; head < _num_kv_heads; ++head) {
            const auto keys = kv_head_rows(*buffers[0], head, _head_dim);
            const KvHeadView view{
                keys.base, keys.stride, _keys_int8, static_cast<uint16_t>(_head_dim), valid
            };
            select_kv_slots(*_policy, view, pinned, recent, _limits.budget, scores, kept);
            for (MLABuffer* buf : buffers) {
                if (buf) compact_kv_head_rows(kv_head_rows(*buf, head, _head_dim), kept);
            }
        }
        for (MLABuffer* buf : buffers) {
            if (buf) buf->flush_cache();
        }
    }

    _state.on_evict(next_token_idx, _limits.budget, recent);
    _logger->info(
        "KV eviction ({}): {} -> {} slots before token {} (pinned {}, recent {}); "
        "{} tokens evicted so far; {:.1f} ms",
        _policy->name(), valid, _limits.budget, next_token_idx, pinned, recent,
        _state.evicted, timer.stop() * 1000.0
    );
}


void KvEviction::reset() {
    if (_state.evicted == 0) {
        return;
    }
    _get_buffer("global_freq_real").upload(_uploaded_rope.re.data());
    _get_buffer("global_freq_imag").upload(_uploaded_rope.im.data());
    _state.reset();
}

}
}
