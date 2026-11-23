#include "attention_cache.hpp"
#include "gemma4_mtp_helpers.hpp"

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// The planner only passes buffer identities to its caller. This executable has no MLA dependency.
namespace simaai::llima {
class MLABuffer {
public:
    std::string name;
};
}

namespace {
using namespace simaai::llima;
using enum LayerType;

void require(bool passed, const std::string& message) {
    if (!passed) throw std::runtime_error(message);
}

void layer_config() {
    const nlohmann::json names = {"full_attention", "sliding_attention", "linear_attention", "conv"};
    const auto model = nlohmann::json{{"layer_types", names}}.get<LanguageModelConfig>();
    require(model.layer_types == LayerTypes{FullAttention, SlidingAttention, LinearAttention, Conv},
            "parse all four layer types");
    require(nlohmann::json(model).at("layer_types") == names, "preserve JSON layer names");
    for (const auto& json : {nlohmann::json::object(), nlohmann::json{{"layer_types", nullptr}},
                             nlohmann::json{{"layer_types", nlohmann::json::array()}}})
        require(json.get<LanguageModelConfig>().layer_types.empty(), "allow legacy layer type inference");

    for (const nlohmann::json& invalid : {nlohmann::json("new_attention"), nlohmann::json(42)}) {
        const nlohmann::json bad = {{"layer_types", {"conv", invalid}}};
        try {
            (void)bad.get<LanguageModelConfig>();
        } catch (const std::invalid_argument& error) {
            const std::string message = error.what();
            require(message.find("layer_types[1]") != message.npos &&
                    message.find(invalid.dump()) != message.npos, "report layer index and offending value");
            continue;
        }
        throw std::runtime_error("accepted invalid layer type");
    }
}

VlmConfig config(LayerTypes types) {
    VlmConfig cfg{};
    cfg.lm_cfg.layer_types = std::move(types);
    cfg.lm_cfg.num_hidden_layers = cfg.lm_cfg.layer_types.size();
    cfg.lm_cfg.attn_cfg.num_attention_heads = 8;
    cfg.lm_cfg.attn_cfg.num_key_value_heads = 2;
    cfg.lm_cfg.attn_cfg.head_dim = 32;
    cfg.lm_cfg.attn_cfg.sliding_head_dim = 16;
    cfg.pipeline_cfg.max_num_tokens = 1024;
    cfg.pipeline_cfg.use_strided_kv_cache = false;
    return cfg;
}

struct NamedBuffers {
    std::map<std::string, MLABuffer> buffers;
    MLABuffer* operator()(const std::string& name) {
        return &buffers.try_emplace(name, MLABuffer{name}).first->second;
    }
};

struct Slice {
    std::string buffer;
    std::vector<uint32_t> begin, shape;
    bool operator==(const Slice&) const = default;
};

using Slots = std::map<uint8_t, Slice>;

void check_bindings(const char* name, const CachePorts& ports, const CacheLayerBuffers& buffers,
                    const VlmConfig& cfg, LayerType type, CacheWindow window, bool packed,
                    const Slots& expected) {
    Slots actual;
    auto record = [&](uint8_t port, MLABuffer* buffer, CachePorts::Dimensions begin,
                      CachePorts::Dimensions shape) {
        require(buffer != nullptr, fmt::format("{}: null buffer at port {}", name, port));
        require(actual.emplace(port, Slice{buffer->name, begin, shape}).second,
                fmt::format("{}: port {} bound twice", name, port));
    };
    MLABuffer mask{"mask"};
    ports.bind_layer(buffers, cfg, type, window, record);
    ports.bind_mask(&mask, window, cfg.pipeline_cfg.max_num_tokens, packed, record);
    require(ports.query == 0 && ports.count == expected.size() + 1, fmt::format("{}: query or port count", name));
    require(actual.size() == expected.size(), fmt::format("{}: unexpected number of bound inputs", name));
    for (const auto& [port, slice] : expected) {
        require(actual.contains(port) && actual.at(port) == slice,
                fmt::format("{}: expected {} at port {} with the specified offset and shape", name, slice.buffer, port));
    }
}

void ordinary_cache() {
    NamedBuffers names;
    auto cfg = config({FullAttention, SlidingAttention});
    const CacheWindow single{37, 64, 57, 1};
    auto check = [&](const char* name, uint8_t layer, CacheWindow window, uint16_t mask_size,
                     const Slots& expected) {
        const auto buffers = CacheLayerBuffers::for_layer(cfg, layer, std::ref(names));
        const CachePorts ports(cfg, window.queries, mask_size);
        check_bindings(name, ports, buffers, cfg, cfg.lm_cfg.layer_types[layer], window, false, expected);
    };

    check("contiguous, no optional inputs", 0, single, 1, {
        {1, {"cache_key_l0", {37, 0}, {64, 64}}},
        {2, {"cache_val_l0", {37, 0}, {64, 64}}}
    });
    cfg.lm_cfg.arch = "gpt_oss";
    check("sinks without mask or quantization", 1, single, 1, {
        {1, {"cache_key_l1", {37, 0}, {64, 32}}},
        {2, {"sinks_l1", {0, 0}, {1, 8}}},
        {3, {"cache_val_l1", {37, 0}, {64, 32}}}
    });
    cfg.pipeline_cfg.quantize_kv_cache = true;
    check("quantized, single mask and sinks", 1, single, 64, {
        {1, {"cache_key_l1", {37, 0}, {64, 32}}},
        {2, {"cache_key_scale_l1", {0, 37, 0}, {2, 1024, 1}}},
        {3, {"mask", {967}, {64}}},
        {4, {"sinks_l1", {0, 0}, {1, 8}}},
        {5, {"cache_val_l1", {37, 0}, {64, 32}}},
        {6, {"cache_val_scale_l1", {0, 37, 0}, {2, 1024, 1}}}
    });
    cfg.lm_cfg.arch.clear();
    cfg.pipeline_cfg.use_strided_kv_cache = true;
    check("strided quantized, group mask", 0, {37, 64, 57, 4}, 64, {
        {1, {"cache_key_l0", {0, 37, 0}, {2, 1024, 32}}},
        {2, {"cache_key_scale_l0", {0, 37, 0}, {2, 1024, 1}}},
        {3, {"mask", {0}, {256}}},
        {4, {"cache_val_l0", {0, 37, 0}, {2, 1024, 32}}},
        {5, {"cache_val_scale_l0", {0, 37, 0}, {2, 1024, 1}}}
    });
    check("group mask cutoff", 0, {37, 64, 57, 4}, 4, {
        {1, {"cache_key_l0", {0, 37, 0}, {2, 1024, 32}}},
        {2, {"cache_key_scale_l0", {0, 37, 0}, {2, 1024, 1}}},
        {3, {"cache_val_l0", {0, 37, 0}, {2, 1024, 32}}},
        {4, {"cache_val_scale_l0", {0, 37, 0}, {2, 1024, 1}}}
    });
    cfg.pipeline_cfg.quantize_kv_cache = false;
    cfg.pipeline_cfg.use_strided_kv_cache = false;
    cfg.lm_cfg.speculative_decoding_cfg = SpeculativeDecodingConfig{.speculative_budget = 4};
    const CachePorts spec_ports(cfg, 4, 64);
    require(spec_ports.context_width(63, 4) == 64, "speculative verification uses the aligned context width");
    check("speculative mask with absolute KV columns", 1, {37, 64, 57, 4}, 64, {
        {1, {"cache_key_l1", {37, 0}, {64, 32}}},
        {2, {"mask", {0, 37}, {4, 64}}},
        {3, {"cache_val_l1", {37, 0}, {64, 32}}}
    });
}

void mtp_shared_cache() {
    NamedBuffers names;
    auto target = config({FullAttention, SlidingAttention, SlidingAttention, FullAttention,
                          SlidingAttention, FullAttention});
    target.lm_cfg.num_kv_shared_layers = 2;
    target.pipeline_cfg.quantize_kv_cache = true;
    target.pipeline_cfg.use_strided_kv_cache = true;
    auto assistant = target;
    assistant.lm_cfg.layer_types = {FullAttention, SlidingAttention, FullAttention};
    assistant.lm_cfg.num_hidden_layers = assistant.lm_cfg.num_kv_shared_layers = 3;
    assistant.lm_cfg.speculative_decoding_cfg = SpeculativeDecodingConfig{.method = "gemma4_mtp", .is_draft = true};

    std::vector<CacheLayerBuffers> target_layers;
    for (uint8_t layer = 0; layer < target.lm_cfg.num_hidden_layers; ++layer)
        target_layers.push_back(CacheLayerBuffers::for_layer(target, layer, std::ref(names)));
    require(target_layers[4].key == names("cache_key_l2") && target_layers[5].key == names("cache_key_l3"),
            "target layers sharing KV must resolve to the owning layer");
    // Expose an erroneous selection of a trailing shared layer even when it aliases the right owner.
    target_layers[4] = target_layers[5] = {};
    const auto sources = gemma4_mtp_helpers::resolve_target_kv(target, assistant, target_layers);
    require(sources.size() == 3, "one target binding per assistant layer");
    for (uint8_t layer = 0; layer < sources.size(); ++layer) {
        const auto placeholders = CacheLayerBuffers::for_layer(assistant, layer, [](const auto&) -> MLABuffer* {
            throw std::runtime_error("looked up a shared KV buffer on the assistant");
        });
        require(!placeholders.key && !placeholders.value, "assistant cache program requires target overrides");
    }
    const CachePorts ports(assistant, 1, 64);
    const Slots full{
        {1, {"cache_key_l3", {0, 0, 0}, {2, 1024, 32}}},
        {2, {"cache_key_scale_l3", {0, 0, 0}, {2, 1024, 1}}},
        {3, {"mask", {0, 0}, {1, 128}}},
        {4, {"cache_val_l3", {0, 0, 0}, {2, 1024, 32}}},
        {5, {"cache_val_scale_l3", {0, 0, 0}, {2, 1024, 1}}}
    };
    check_bindings("assistant full", ports, sources[0], assistant, FullAttention, {0, 128, 95, 1}, true, full);
    check_bindings("assistant sliding", ports, sources[1], assistant, SlidingAttention, {37, 64, 58, 1}, true, {
        {1, {"cache_key_l2", {0, 37, 0}, {2, 1024, 16}}},
        {2, {"cache_key_scale_l2", {0, 37, 0}, {2, 1024, 1}}},
        {3, {"mask", {0, 0}, {1, 64}}},
        {4, {"cache_val_l2", {0, 37, 0}, {2, 1024, 16}}},
        {5, {"cache_val_scale_l2", {0, 37, 0}, {2, 1024, 1}}}
    });
    check_bindings("assistant full again", ports, sources[2], assistant, FullAttention, {0, 128, 95, 1}, true, full);
}

} // namespace

int main() {
    try {
        layer_config();
        ordinary_cache();
        mtp_shared_cache();
        std::cout << "Layer configuration and attention cache bindings passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
