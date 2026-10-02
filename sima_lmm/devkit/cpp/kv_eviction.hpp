#ifndef _SIMA_LLIMA_KV_EVICTION_
#define _SIMA_LLIMA_KV_EVICTION_

// Runtime KV cache eviction; not part of the installed API. A token index is a position in the
// conversation; after evictions, token t is stored in cache slot t - evicted.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "rope_utils.hpp"
#include "vlm_config.hpp"

namespace spdlog {
class logger;
}

namespace simaai {
namespace llima {

class MLABuffer;

inline constexpr uint16_t KV_EVICTION_SINK_TOKENS = 4;
inline constexpr uint16_t KV_EVICTION_MIN_RECENT_TOKENS = 128;
inline constexpr uint16_t KV_EVICTION_MAX_RECENT_TOKENS = 512;

struct KvEvictionLimits {
    uint16_t capacity;       // Compiled max_num_tokens.
    uint16_t group_size;     // 1 without group models.
    uint16_t budget;         // Slots kept by an eviction; a group offset.
    uint16_t recent_tokens;  // Newest tokens always kept.
    uint16_t max_tokens;     // Longest supported conversation.
    std::vector<uint16_t> group_offsets;  // Group offsets whose group fits in the cache.
};

// Validates the model and settings and resolves the budgets.
KvEvictionLimits resolve_kv_eviction_limits(
    const VlmConfig& cfg,
    const KvEvictionConfig& kv_eviction,
    std::optional<uint32_t> trained_max_positions
);

// Keys of one KV head in one layer: slot i starts at keys + i * row_stride.
struct KvHeadView {
    const uint8_t* keys = nullptr;
    size_t row_stride = 0;
    bool keys_int8 = false;  // Otherwise bfloat16.
    uint16_t head_dim = 0;
    uint16_t num_valid = 0;
};

class KvEvictionPolicy {
    public:
        virtual ~KvEvictionPolicy() = default;
        virtual std::string_view name() const = 0;
        // Scores slots [begin, end); higher scores are kept.
        virtual void score(
            const KvHeadView& head, uint16_t begin, uint16_t end, std::span<float> scores
        ) const = 0;
};

std::unique_ptr<KvEvictionPolicy> make_kv_eviction_policy(std::string_view name);

// Keeps [0, pinned), the newest `recent` slots and the best-scored slots in between, `keep` in
// total (pinned + recent <= keep <= num_valid), in ascending order. Ties go to the lower slot.
void select_kv_slots(
    const KvEvictionPolicy& policy,
    const KvHeadView& head,
    uint16_t pinned,
    uint16_t recent,
    uint16_t keep,
    std::vector<float>& scores,
    std::vector<uint16_t>& kept_slots
);

struct KvEvictionState {
    uint16_t evicted = 0;
    // Tokens from here on are stored in order on every head.
    uint16_t identity_start = 0;

    uint16_t slot(uint16_t token_idx) const { return token_idx - evicted; }
    void on_evict(uint16_t next_token_idx, uint16_t budget, uint16_t recent) {
        evicted = next_token_idx - budget;
        identity_start = next_token_idx - recent;
    }
    bool can_restart_at(uint16_t token_idx) const {
        return evicted == 0 || token_idx >= identity_start;
    }
    void reset() { *this = KvEvictionState{}; }
};

// Eviction for one language model. Buffers are looked up when used, so this can be created
// before the model allocates them.
class KvEviction {
    public:
        using BufferLookup = std::function<MLABuffer&(const std::string&)>;

        KvEviction(
            const VlmConfig& cfg,
            const KvEvictionConfig& settings,
            const std::filesystem::path& devkit_dir,
            BufferLookup get_buffer,
            std::shared_ptr<spdlog::logger> logger
        );

        const KvEvictionLimits& limits() const { return _limits; }
        uint16_t evicted() const { return _state.evicted; }
        uint16_t slot(uint16_t token_idx) const { return _state.slot(token_idx); }

        // Pins the system prompt and returns how many cached tokens prefill may reuse; resets
        // and returns 0 when the conversation changed where the cache is no longer in order.
        uint16_t resume_or_reset(uint16_t num_cached_tokens, uint16_t pinned_tokens);
        // Returns the first cache slot for the tokens about to be written; evicts if a single
        // token does not fit and uploads the RoPE rows of their true positions.
        uint16_t reserve_slots(uint16_t token_idx, uint16_t num_tokens);
        // Evicts when no group fits and the prompt does not fit either (or already evicted).
        bool evict_for_prefill(uint16_t token_idx, uint16_t num_input_tokens);
        // Forgets all evictions and restores the RoPE rows.
        void reset();

    private:
        void evict(uint16_t next_token_idx);

        KvEvictionLimits _limits;
        std::unique_ptr<KvEvictionPolicy> _policy;
        KvEvictionState _state;
        uint16_t _pinned_tokens = KV_EVICTION_SINK_TOKENS;
        RopeTable _rope;           // Rows for token indices up to max_tokens + one group.
        RopeTable _uploaded_rope;  // The model's table: position = slot.
        BufferLookup _get_buffer;
        uint32_t _num_layers;
        uint32_t _num_kv_heads;
        uint32_t _head_dim;
        bool _keys_int8;
        std::shared_ptr<spdlog::logger> _logger;
};

}
}

#endif
