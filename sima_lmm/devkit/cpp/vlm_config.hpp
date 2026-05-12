//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
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

#ifndef _SIMA_LLIMA_VLM_CONFIG_
#define _SIMA_LLIMA_VLM_CONFIG_

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "utils.hpp"

namespace simaai {
namespace llima {


// Redefines the *Config python classes in sima_lmm/config/vlm_config.py here. Only the fields that
// are used in the evaluation time is defined.

struct VisionModelConfig {
    std::vector<uint16_t> image_sizes;
    bool cls_embed;
    uint16_t temporal_patch_size;
    uint16_t spatial_patch_size;
    uint16_t spatial_merge_size;
    std::vector<uint8_t> deepstack_visual_indexes;

    std::vector<uint16_t> num_spatial_patches;
};


inline void to_json(nlohmann::json& j, const VisionModelConfig& v) {
    if (v.image_sizes[0] == v.image_sizes[1]) {
        j["image_size"] = v.image_sizes[0];
    } else {
        j["image_size"] = v.image_sizes;
    }
    j["patch_size"] = v.spatial_patch_size;
    j["temporal_patch_size"] = v.temporal_patch_size;
    j["deepstack_visual_indexes"] = v.deepstack_visual_indexes;
}


inline void from_json(const nlohmann::json& j, VisionModelConfig& v) {
    if (j.contains("image_size")) {
        auto image_size_json = j["image_size"];
        if (image_size_json.is_number_integer()) {
            uint16_t image_size = image_size_json.get<uint16_t>();
            v.image_sizes = std::vector<uint16_t>{image_size, image_size};
        } else if (image_size_json.is_array()) {
            v.image_sizes = image_size_json.get<std::vector<uint16_t>>();
        } else {
            throw std::runtime_error("Failed to parse image size: " + image_size_json.dump());
        }
    } else {
        throw std::runtime_error("Cannot find image_size in json");
    }
    if (j.contains("patch_size")) {
        v.spatial_patch_size = j["patch_size"].get<uint16_t>();
    } else {
        throw std::runtime_error("Cannot find patch_size in json");
    }
    v.temporal_patch_size = j.value("temporal_patch_size", 1);
    v.cls_embed = j.value("cls_embed", false);
    v.spatial_merge_size = j.value("spatial_merge_size", 1);

    if (j.contains("deepstack_visual_indexes"))
        v.deepstack_visual_indexes = j["deepstack_visual_indexes"].get<std::vector<uint8_t>>();

    v.num_spatial_patches = {
        static_cast<uint16_t>(v.image_sizes[0] / v.spatial_patch_size),
        static_cast<uint16_t>(v.image_sizes[1] / v.spatial_patch_size)
    };
}


struct MMConnectionConfig {
    uint16_t mm_tokens_per_image;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MMConnectionConfig, mm_tokens_per_image)


struct TokenEmbedConfig {
    uint32_t vocab_size;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TokenEmbedConfig, vocab_size)


struct RopeScalingConfig {
    double factor;
    double low_freq_factor;
    double high_freq_factor;
    uint32_t original_max_position_embeddings;
    std::optional<std::vector<double>> long_factor;
    std::optional<std::vector<double>> short_factor;
    std::string rope_type;
    std::optional<std::vector<size_t>> mrope_section;
    bool mrope_interleaved = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    RopeScalingConfig, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings,
    long_factor, short_factor, rope_type, mrope_section, mrope_interleaved
)


struct RoPEConfig {
    double rope_theta;
    double rope_local_base_freq;
    uint32_t rope_dimension_count = 0;
    RopeScalingConfig rope_scaling;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    RoPEConfig, rope_theta, rope_local_base_freq, rope_dimension_count, rope_scaling
)


struct AttentionBlockConfig {
    uint8_t num_attention_heads;
    uint8_t num_key_value_heads;
    uint32_t head_dim;
    bool swa_enable;
    uint8_t swa_ratio = 0;  // Deprecated in 2.1.
    std::optional<uint32_t> sliding_window;


    uint32_t get_q_size() const { return num_attention_heads * head_dim; }
    uint32_t get_kv_size() const { return num_key_value_heads * head_dim; }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    AttentionBlockConfig, num_attention_heads, num_key_value_heads, head_dim, swa_enable, swa_ratio,
    sliding_window
)

struct LayerTypes : std::vector<std::string> { using std::vector<std::string>::vector; };

struct LanguageModelConfig {
    TokenEmbedConfig token_cfg;
    RoPEConfig rope_cfg;
    AttentionBlockConfig attn_cfg;
    uint32_t hidden_size;
    uint8_t num_hidden_layers;
    uint16_t lm_head_num_splits;
    uint32_t lm_head_split_dim;
    LayerTypes layer_types = {};
    uint32_t conv_L_cache = 3;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    LanguageModelConfig, token_cfg, rope_cfg, attn_cfg, num_hidden_layers, hidden_size,
    lm_head_num_splits, lm_head_split_dim, layer_types, conv_L_cache
)


inline void to_json(nlohmann::json& j, const LayerTypes& v) {
    j = static_cast<const std::vector<std::string>&>(v);
}


inline void from_json(const nlohmann::json& j, LayerTypes& v) {
    if (j.is_null()) {
        v.clear();
    } else {
        j.get_to(static_cast<std::vector<std::string>&>(v));
    }
}


struct PipelineConfig {
    std::optional<std::string> system_prompt;
    std::optional<std::string> chat_template;
    uint16_t max_num_tokens;
    uint16_t input_token_group_size;
    std::optional<std::vector<uint16_t>> input_token_group_offsets;
    uint16_t future_token_mask_size;
    bool return_logits;
    bool use_strided_kv_cache;
    bool enable_filter_sharing;
    bool quantize_embeddings = false;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    PipelineConfig, system_prompt, chat_template, max_num_tokens, input_token_group_size,
    input_token_group_offsets, future_token_mask_size, return_logits, use_strided_kv_cache,
    enable_filter_sharing, quantize_embeddings
)


struct VlmConfig {
    std::string model_name;
    std::string model_type;
    std::optional<VisionModelConfig> vm_cfg;
    std::optional<MMConnectionConfig> mm_cfg;
    LanguageModelConfig lm_cfg;
    PipelineConfig pipeline_cfg;
    std::string vision_model_name = "";
    std::string language_model_name;
    std::string gguf_file_name = "";

    bool is_multimodal() const { return vm_cfg.has_value(); }
    bool support_image() const {
        static bool disable_vision = get_env_var("SIMA_LLIMA_RUN_DISABLE_VISION", false);
        return (
            is_multimodal()
            && !(model_type == "vlm-gemma3" && vm_cfg.value().image_sizes[0] > 448)
            && !disable_vision
        );
    }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
    VlmConfig, model_name, model_type, vm_cfg, mm_cfg, lm_cfg, pipeline_cfg, vision_model_name,
    language_model_name, gguf_file_name
)

}
}

#endif
