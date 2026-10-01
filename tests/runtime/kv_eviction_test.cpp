#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "kv_eviction.hpp"

using namespace simaai::llima;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

VlmConfig make_config(uint16_t group_size = 128, std::vector<uint16_t> offsets = {}) {
    if (offsets.empty()) {
        for (uint32_t offset = 0; offset + group_size <= 2048; offset += group_size) {
            offsets.push_back(offset);
        }
    }
    return nlohmann::json({
        {"model_type", "llm-qwen3"},
        {"lm_cfg", {
            {"num_hidden_layers", 2}, {"hidden_size", 256},
            {"layer_types", {"full_attention", "full_attention"}},
            {"attn_cfg", {{"num_attention_heads", 4}, {"num_key_value_heads", 2}, {"head_dim", 64}}},
            {"rope_cfg", {{"rope_scaling", {{"rope_type", "default"}}}}},
        }},
        {"pipeline_cfg", {
            {"max_num_tokens", 2048}, {"input_token_group_size", group_size},
            {"input_token_group_offsets", offsets}, {"future_token_mask_size", 128},
        }},
    }).get<VlmConfig>();
}

KvEvictionConfig policy(const std::string& name, std::optional<uint16_t> budget = std::nullopt) {
    return {name, budget};
}

void expect_rejected(const std::function<void()>& action, const std::string& message_part) {
    try {
        action();
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find(message_part) != std::string::npos, "unexpected error: " + message);
        return;
    }
    throw std::runtime_error("expected an error containing '" + message_part + "'");
}

void test_config_parsing() {
    auto json = nlohmann::json(make_config());
    const auto parse = [&] { return json.get<VlmConfig>().pipeline_cfg.kv_eviction; };
    require(!parse().enabled(), "a missing field is not off");
    // A null budget selects the default.
    json["pipeline_cfg"]["kv_eviction"] = {{"policy", "off"}, {"budget_tokens", nullptr}};
    require(!parse().enabled(), "the serialized default is not off");
    json["pipeline_cfg"]["kv_eviction"] = {{"policy", "keydiff"}, {"budget_tokens", 1536}};
    require(parse().policy == "keydiff" && parse().budget_tokens == 1536, "keydiff not parsed");
}

void test_budgets() {
    // {config, policy, trained positions, budget, recent tokens, max tokens}
    const std::vector<uint16_t> llama_offsets = {0, 320, 640, 704, 960, 1280, 1600, 1920};
    for (const auto& [config, kv_eviction, trained, budget, recent, max_tokens] : {
        std::tuple{make_config(), policy("keydiff"), std::optional<uint32_t>{40960}, 1792, 448, 40960},
        std::tuple{make_config(), policy("sink_window"), std::optional<uint32_t>{40960}, 1792, 448, 40960},
        std::tuple{make_config(), policy("keydiff", 1536), std::optional<uint32_t>{}, 1536, 384, 65535 - 128},
        std::tuple{make_config(320), policy("keydiff"), std::optional<uint32_t>{131072}, 1600, 400, 65535 - 320},
        // Deployed Llama-3.2-3B offsets: 704 is irregular, the group at 1920 overruns the cache.
        std::tuple{make_config(320, llama_offsets), policy("sink_window"), std::optional<uint32_t>{131072}, 1600, 400, 65535 - 320},
        std::tuple{make_config(320, llama_offsets), policy("keydiff", 704), std::optional<uint32_t>{131072}, 704, 320, 65535 - 320},
    }) {
        const auto limits = resolve_kv_eviction_limits(config, kv_eviction, trained);
        require(limits.budget == budget && limits.recent_tokens == recent, "wrong budget");
        require(limits.max_tokens == max_tokens, "wrong max_tokens");
    }
    for (const auto& [budget, message] : {
        std::pair{1000, "must be a compiled prefill group offset"},
        std::pair{2048, "must be a compiled prefill group offset"},
        std::pair{256, "too small for this model (compiled cache 2048 tokens, prefill group 128); use at least 384"},
    }) {
        expect_rejected(
            [&] { resolve_kv_eviction_limits(make_config(), policy("keydiff", budget), 40960); },
            message
        );
    }
    expect_rejected(
        [&] { resolve_kv_eviction_limits(make_config(320, llama_offsets), policy("keydiff", 1920), 131072); },
        "valid values: 0, 320, 640, 704, 960, 1280, 1600"
    );
}

void test_rejects_unsupported_models() {
    const std::vector<std::pair<std::function<void(VlmConfig&)>, std::string>> cases = {
        {[](VlmConfig& c) { c.lm_cfg.layer_types[1] = "sliding_attention"; }, "sliding_attention"},
        {[](VlmConfig& c) { c.lm_cfg.layer_types[0] = "linear_attention"; }, "linear_attention"},
        {[](VlmConfig& c) { c.lm_cfg.attn_cfg.swa_enable = true; }, "sliding-window"},
        {[](VlmConfig& c) { c.lm_cfg.num_kv_shared_layers = 1; }, "shared KV"},
        {[](VlmConfig& c) { c.lm_cfg.speculative_decoding_cfg = SpeculativeDecodingConfig{}; }, "speculative"},
        {[](VlmConfig& c) { c.vm_cfg = VisionModelConfig{}; }, "vision-language"},
        {[](VlmConfig& c) { c.lm_cfg.rope_cfg.rope_scaling.rope_type = "longrope"; }, "longrope"},
    };
    for (const auto& [modify, message] : cases) {
        auto config = make_config();
        modify(config);
        expect_rejected([&] { resolve_kv_eviction_limits(config, policy("keydiff"), 40960); }, message);
    }
    expect_rejected([] { make_kv_eviction_policy("h2o"); }, "unknown policy 'h2o'");
    expect_rejected(
        [] { resolve_kv_eviction_limits(make_config(), policy("keydiff"), 2048); }, "already fits"
    );
}

constexpr uint16_t kNumValid = 64, kHeadDim = 16, kNumHeads = 2, kCapacity = 80;
constexpr uint16_t kPinned = 4, kRecent = 8, kKeep = 24;

// Small integers, exact in bfloat16 and int8, with repeated rows (ties).
int synthetic_key(int head, int slot, int dim) {
    return slot % 9 == 0 ? (dim % 5) - 2 + head : ((slot * 37 + dim * 11 + head * 5) % 23) - 11;
}

// Cache bytes in the non-strided [slot][head][dim] or strided [head][slot][dim] layout, as
// bfloat16 or as int8 multiplied by `scale` (a per-row scale must not change the scores).
struct SyntheticCache {
    bool int8, strided;
    std::vector<uint8_t> bytes;

    SyntheticCache(bool int8_rows, bool strided_layout, int scale,
                   const std::function<int(int, int, int)>& key)
        : int8(int8_rows), strided(strided_layout),
          bytes(static_cast<size_t>(kNumHeads) * kCapacity * kHeadDim * elem()) {
        for (int head = 0; head < kNumHeads; ++head) {
            for (int slot = 0; slot < kNumValid; ++slot) {
                for (int dim = 0; dim < kHeadDim; ++dim) {
                    const float value = static_cast<float>(key(head, slot, dim) * scale);
                    uint32_t bits;
                    std::memcpy(&bits, &value, sizeof(bits));
                    const int8_t int8_value = static_cast<int8_t>(value);
                    const uint16_t bf16_value = static_cast<uint16_t>(bits >> 16);  // exact here
                    std::memcpy(row(head, slot) + dim * elem(),
                                int8 ? static_cast<const void*>(&int8_value) : &bf16_value, elem());
                }
            }
        }
    }
    size_t elem() const { return int8 ? 1 : 2; }
    size_t stride() const { return (strided ? 1 : kNumHeads) * kHeadDim * elem(); }
    uint8_t* row(int head, int slot) {
        return bytes.data() + (strided ? (head * kCapacity + slot) * stride()
                                       : slot * stride() + head * kHeadDim * elem());
    }
    KvHeadView view(int head) { return {row(head, 0), stride(), int8, kHeadDim, kNumValid}; }
};

// KeyDiff in double precision: -cos(key, mean of the L2-normalized keys).
std::vector<double> reference_scores(int head) {
    const auto key = [head](int slot, int dim) { return double(synthetic_key(head, slot, dim)); };
    std::vector<double> anchor(kHeadDim, 0.0), norms(kNumValid, 0.0), scores(kNumValid, 0.0);
    for (int slot = 0; slot < kNumValid; ++slot) {
        for (int dim = 0; dim < kHeadDim; ++dim) norms[slot] += key(slot, dim) * key(slot, dim);
        norms[slot] = std::sqrt(norms[slot]);
        for (int dim = 0; dim < kHeadDim; ++dim) anchor[dim] += key(slot, dim) / norms[slot] / kNumValid;
    }
    double anchor_norm = 0.0;
    for (const double value : anchor) anchor_norm += value * value;
    for (int slot = 0; slot < kNumValid; ++slot) {
        for (int dim = 0; dim < kHeadDim; ++dim) scores[slot] -= key(slot, dim) * anchor[dim];
        scores[slot] /= norms[slot] * std::sqrt(anchor_norm);
    }
    return scores;
}

std::vector<uint16_t> range(uint16_t begin, uint16_t end, std::vector<uint16_t> slots = {}) {
    for (uint16_t slot = begin; slot < end; ++slot) slots.push_back(slot);
    return slots;
}

// Pinned slots, the best-scored middle slots (ties to the lower slot) and the recent slots.
std::vector<uint16_t> reference_kept(const std::vector<double>& scores) {
    auto middle = range(kPinned, kNumValid - kRecent);
    std::stable_sort(middle.begin(), middle.end(), [&](uint16_t a, uint16_t b) {
        return scores[a] > scores[b];
    });
    middle.resize(kKeep - kPinned - kRecent);
    std::sort(middle.begin(), middle.end());
    auto kept = range(0, kPinned);
    kept.insert(kept.end(), middle.begin(), middle.end());
    return range(kNumValid - kRecent, kNumValid, kept);
}

std::vector<uint16_t> select(const std::string& name, SyntheticCache& cache, int head) {
    std::vector<float> scores;
    std::vector<uint16_t> kept;
    select_kv_slots(*make_kv_eviction_policy(name), cache.view(head), kPinned, kRecent, kKeep,
                    scores, kept);
    return kept;
}

void test_keydiff_matches_reference() {
    SyntheticCache bf16(false, false, 1, synthetic_key);
    for (int head = 0; head < kNumHeads; ++head) {
        const auto expected = reference_scores(head);
        std::vector<float> scores(kNumValid);
        make_kv_eviction_policy("keydiff")->score(bf16.view(head), 0, kNumValid, scores);
        for (int slot = 0; slot < kNumValid; ++slot) {
            require(std::abs(scores[slot] - expected[slot]) < 1e-5, "score differs from reference");
        }
        for (const auto& [int8, strided, scale] : {
            std::tuple{false, false, 1}, std::tuple{false, true, 1}, std::tuple{true, false, 3},
            std::tuple{true, true, 2},
        }) {
            SyntheticCache cache(int8, strided, scale, synthetic_key);
            require(select("keydiff", cache, head) == reference_kept(expected), "wrong kept slots");
        }
    }
}

void test_sink_window_and_ties() {
    SyntheticCache cache(false, false, 1, synthetic_key);
    require(select("sink_window", cache, 0) == range(44, 64, range(0, 4)), "not the newest");

    SyntheticCache equal_keys(false, false, 1, [](int, int, int) { return 1; });
    require(select("keydiff", equal_keys, 0) == range(56, 64, range(0, 16)), "ties not lower");
}

void test_state_bookkeeping() {
    KvEvictionState state;
    require(state.slot(1000) == 1000 && state.can_restart_at(0), "fresh state not identity");
    state.on_evict(2048, 1792, 448);  // full at 2048 slots; keep 1792 incl. the newest 448
    require(state.evicted == 256 && state.identity_start == 1600, "wrong first eviction");
    require(state.slot(2048) == 1792 && state.slot(1600) == 1344, "wrong slots");
    require(state.can_restart_at(1600) && !state.can_restart_at(1599), "wrong restart point");
    state.on_evict(2304, 1792, 448);
    require(state.evicted == 512 && state.identity_start == 1856, "wrong second eviction");
    state.reset();
    require(state.evicted == 0 && state.identity_start == 0, "reset kept state");
}

} // namespace

int main() {
    try {
        test_config_parsing();
        test_budgets();
        test_rejects_unsupported_models();
        test_keydiff_matches_reference();
        test_sink_window_and_ties();
        test_state_bookkeeping();
    } catch (const std::exception& error) {
        std::cerr << "kv eviction tests failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "kv eviction tests passed\n";
    return 0;
}
