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
    bool log_probe_enabled = false;
    std::vector<uint32_t> language_token_ids;
    std::vector<std::string> language_codes;

    uint32_t get_encoder_head_dim() const { return d_model / encoder_attention_heads; };
    uint32_t get_decoder_head_dim() const { return d_model / decoder_attention_heads; };
};

/*
 * Keep the package reader backwards compatible with compiled Whisper drops.
 * `log_probe_enabled`, `language_token_ids`, and `language_codes` were added
 * by newer compiler/runtime revisions, but the public SiMa Whisper package
 * predates them.  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE uses `at()` for every
 * field and therefore defeats the defaults above: an older, otherwise valid
 * package fails before any MLA work is submitted.  Required shape/architecture
 * fields continue to use `at()` so a malformed package still fails closed;
 * only versioned optional metadata uses `value()`.
 */
inline void from_json(const nlohmann::json& json, WhisperConfig& config) {
    json.at("model_name").get_to(config.model_name);
    json.at("model_type").get_to(config.model_type);
    json.at("d_model").get_to(config.d_model);
    json.at("encoder_attention_heads").get_to(config.encoder_attention_heads);
    json.at("encoder_layers").get_to(config.encoder_layers);
    json.at("decoder_attention_heads").get_to(config.decoder_attention_heads);
    json.at("decoder_layers").get_to(config.decoder_layers);
    json.at("max_source_positions").get_to(config.max_source_positions);
    json.at("max_target_positions").get_to(config.max_target_positions);
    json.at("num_mel_bins").get_to(config.num_mel_bins);
    json.at("vocab_size").get_to(config.vocab_size);
    json.at("decoder_use_future_token_mask")
        .get_to(config.decoder_use_future_token_mask);

    config.log_probe_enabled = json.value("log_probe_enabled", false);
    config.language_token_ids = json.value(
        "language_token_ids", std::vector<uint32_t>{}
    );
    config.language_codes = json.value(
        "language_codes", std::vector<std::string>{}
    );
    /*
     * The two arrays are one versioned lookup table. Older published packages
     * omit both, while current develop emits both. Accept those two coherent
     * states only: accepting a half-upgraded/mismatched pair defers the error
     * until Whisper indexes one vector with an iterator from the other.
     */
    if (config.language_token_ids.size() != config.language_codes.size()) {
        throw nlohmann::json::out_of_range::create(
            401,
            "Whisper language_token_ids and language_codes must have equal lengths",
            &json
        );
    }
}

inline void to_json(nlohmann::json& json, const WhisperConfig& config) {
    json = nlohmann::json{
        {"model_name", config.model_name},
        {"model_type", config.model_type},
        {"d_model", config.d_model},
        {"encoder_attention_heads", config.encoder_attention_heads},
        {"encoder_layers", config.encoder_layers},
        {"decoder_attention_heads", config.decoder_attention_heads},
        {"decoder_layers", config.decoder_layers},
        {"max_source_positions", config.max_source_positions},
        {"max_target_positions", config.max_target_positions},
        {"num_mel_bins", config.num_mel_bins},
        {"vocab_size", config.vocab_size},
        {"decoder_use_future_token_mask",
         config.decoder_use_future_token_mask},
        {"log_probe_enabled", config.log_probe_enabled},
        {"language_token_ids", config.language_token_ids},
        {"language_codes", config.language_codes},
    };
}


}
}

#endif
