#ifndef _SIMA_LLIMA_WHISPER_CONFIG_
#define _SIMA_LLIMA_WHISPER_CONFIG_

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace simaai {
namespace llima {

struct WhisperConfig{
    std::string model_type;
    uint32_t d_model;
    uint32_t encoder_attention_heads;
    uint32_t encoder_layers;
    uint32_t decoder_attention_heads;
    uint32_t decoder_layers;
    uint32_t max_source_positions;
    uint32_t max_target_positions;
    uint32_t num_mel_bins;
    uint32_t vocab_size;

    // Fields added by sima-lmm during whisper_config.json file generation.
    std::string model_name;
    bool decoder_use_future_token_mask;
    std::vector<uint32_t> language_token_ids;
    std::vector<std::string> language_codes;

    uint32_t get_encoder_head_dim() const { return d_model / encoder_attention_heads; };
    uint32_t get_decoder_head_dim() const { return d_model / decoder_attention_heads; };
};

inline void from_json(const nlohmann::json& j, WhisperConfig& v) {
    j.at("model_name").get_to(v.model_name);
    j.at("model_type").get_to(v.model_type);
    j.at("d_model").get_to(v.d_model);
    j.at("encoder_attention_heads").get_to(v.encoder_attention_heads);
    j.at("encoder_layers").get_to(v.encoder_layers);
    j.at("decoder_attention_heads").get_to(v.decoder_attention_heads);
    j.at("decoder_layers").get_to(v.decoder_layers);
    j.at("max_source_positions").get_to(v.max_source_positions);
    j.at("max_target_positions").get_to(v.max_target_positions);
    j.at("num_mel_bins").get_to(v.num_mel_bins);
    j.at("vocab_size").get_to(v.vocab_size);
    j.at("decoder_use_future_token_mask").get_to(v.decoder_use_future_token_mask);

    j.at("language_token_ids").get_to(v.language_token_ids);
    j.at("language_codes").get_to(v.language_codes);
}


}
}

#endif
