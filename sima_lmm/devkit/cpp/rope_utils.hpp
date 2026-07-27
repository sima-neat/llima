#ifndef _SIMA_LLIMA_ROPE_UTILS_
#define _SIMA_LLIMA_ROPE_UTILS_

#include <cassert>
#include <vector>

#include "eigen_types.hpp"
#include "vlm_config.hpp"

namespace simaai {
namespace llima {

struct RopeTable {
  std::vector<Eigen::bfloat16> re;
  std::vector<Eigen::bfloat16> im;
};

RopeTable calc_freq_real_imag(uint16_t max_num_tokens, const std::string& rope_type, double theta,
                              uint16_t rope_dimension_count, uint16_t layer_head_dim,
                              RopeScalingConfig& rope_scaling_cfg);

RopeTable calc_mrope_with_image(const VlmConfig& vlm_cfg, const RopeTable& master_rope_table,
                                uint32_t image_token_id, std::span<const uint32_t> input_token_ids);

} // namespace llima
} // namespace simaai

#endif
