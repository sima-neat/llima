#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>

#include "vlm_config.hpp"

namespace simaai::llima {

class MLABuffer;

// Borrowed KV, scale and attention-sink buffers for one layer.
// Resolves shared-KV layers to their source buffers.
struct CacheLayerBuffers {
    MLABuffer* key = nullptr;
    MLABuffer* value = nullptr;
    MLABuffer* key_scale = nullptr;
    MLABuffer* value_scale = nullptr;
    MLABuffer* sinks = nullptr;

    template<class Lookup>
    static CacheLayerBuffers for_layer(const VlmConfig& cfg, uint8_t layer, Lookup lookup) {
        CacheLayerBuffers buffers;
        const auto& model = cfg.lm_cfg;
        const auto type = model.layer_types.at(layer);
        if (type != LayerType::FullAttention && type != LayerType::SlidingAttention)
            return buffers;
        if (model.uses_attention_sinks())
            buffers.sinks = lookup(fmt::format("sinks_l{}", layer));
        // Shared assistant inputs are supplied by the paired target at execution.
        if (model.is_gemma4_mtp_draft() && model.is_kv_shared_layer(layer))
            return buffers;

        const auto source = model.get_kv_source_layer(layer);
        buffers.key = lookup(fmt::format("cache_key_l{}", source));
        buffers.value = lookup(fmt::format("cache_val_l{}", source));
        if (cfg.pipeline_cfg.quantize_kv_cache) {
            buffers.key_scale = lookup(fmt::format("cache_key_scale_l{}", source));
            buffers.value_scale = lookup(fmt::format("cache_val_scale_l{}", source));
        }
        return buffers;
    }
};

// Identifies which cached KV rows are visible and how many query tokens use them.
struct CacheWindow {
    uint32_t begin;             // First visible KV row in the backing buffer.
    uint32_t length;            // Compiled cache ELF's aligned context width.
    uint32_t effective_length;  // Valid context, before alignment.
    uint32_t queries;
};

enum class CacheMask { None, Group, Single, Speculative };

// Defines cache-stage input positions and slice layouts from the compiled configuration.
// Used by both binding construction and execution to keep optional inputs consistent.
struct CachePorts {
    static constexpr uint8_t query = 0;
    static constexpr uint8_t key = 1;
    std::optional<uint8_t> key_scale, mask, sinks, value_scale;
    uint8_t value;
    uint8_t count;
    CacheMask mask_kind;

    CachePorts(const VlmConfig& config, uint16_t queries, uint16_t mask_size)
        : mask_kind(CacheMask::None) {
        const auto& language = config.lm_cfg;
        if (queries == language.get_single_num_tokens()) {
            if (mask_size > 1)
                mask_kind = language.is_spec_decode() ? CacheMask::Speculative : CacheMask::Single;
        } else if (mask_size > queries) {
            mask_kind = CacheMask::Group;
        }
        uint8_t next = key + 1;
        if (config.pipeline_cfg.quantize_kv_cache) key_scale = next++;
        if (mask_kind != CacheMask::None) mask = next++;
        if (language.uses_attention_sinks()) sinks = next++;
        value = next++;
        if (config.pipeline_cfg.quantize_kv_cache) value_scale = next++;
        count = next;
    }

    uint32_t context_width(uint16_t aligned_index, uint16_t queries) const {
        const bool single_mask = mask_kind == CacheMask::Single || mask_kind == CacheMask::Speculative;
        return static_cast<uint32_t>(aligned_index) + (single_mask ? 1u : queries);
    }

    // The builder materializes slices. The executor can update just pointers and offsets.
    using Dimensions = std::initializer_list<uint32_t>;
    template<class Bind>
    void bind_layer(const CacheLayerBuffers& buffers, const VlmConfig& config, LayerType type,
                    CacheWindow window, Bind bind) const {
        const auto& attention = config.lm_cfg.attn_cfg;
        const uint32_t heads = attention.num_key_value_heads;
        const uint32_t capacity = config.pipeline_cfg.max_num_tokens;
        auto bind_pair = [&](Dimensions offset, Dimensions shape) {
            bind(key, buffers.key, offset, shape);
            bind(value, buffers.value, offset, shape);
        };
        if (config.pipeline_cfg.use_strided_kv_cache) {
            bind_pair({0, window.begin, 0}, {heads, capacity, attention.get_head_dim(type)});
        } else {
            bind_pair({window.begin, 0}, {window.length, attention.get_kv_size(type)});
        }
        if (key_scale) {
            bind(*key_scale, buffers.key_scale, {0, window.begin, 0}, {heads, capacity, 1});
            bind(*value_scale, buffers.value_scale, {0, window.begin, 0}, {heads, capacity, 1});
        }
        if (sinks)
            bind(*sinks, buffers.sinks, {0, 0}, {window.queries, attention.num_attention_heads});
    }

    template<class Bind>
    void bind_mask(MLABuffer* buffer, CacheWindow window, uint32_t capacity,
                   bool packed_rows, Bind bind) const {
        switch (mask_kind) {
        case CacheMask::None:
            break;
        case CacheMask::Group:
            bind(*mask, buffer, {0}, {window.queries * window.length});
            break;
        case CacheMask::Single:
            bind(*mask, buffer, {capacity - window.effective_length}, {window.length});
            break;
        case CacheMask::Speculative:
            // DFlash and MTP upload packed rows relative to the bound KV slice.
            bind(*mask, buffer, {0, packed_rows ? 0u : window.begin}, {window.queries, window.length});
            break;
        }
    }
};

} // namespace simaai::llima
