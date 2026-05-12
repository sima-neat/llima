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
