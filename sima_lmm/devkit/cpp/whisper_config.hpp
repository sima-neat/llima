#ifndef _SIMA_LLIMA_WHISPER_CONFIG_
#define _SIMA_LLIMA_WHISPER_CONFIG_

#include <string>

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

    uint32_t get_encoder_head_dim() const { return d_model / encoder_attention_heads; };
    uint32_t get_decoder_head_dim() const { return d_model / decoder_attention_heads; };
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(
    WhisperConfig, model_name, model_type, d_model, encoder_attention_heads, encoder_layers,
    decoder_attention_heads, decoder_layers, max_source_positions, max_target_positions,
    num_mel_bins, vocab_size, decoder_use_future_token_mask
)


}
}

#endif
