#include <algorithm>
#include <fstream>
#include <iostream>
#include <numbers>

#include <fmt/format.h>

#include "rope_utils.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

static Eigen::ArrayXf calc_rope_scaling_llama3(const Eigen::ArrayXf& inv_freq,
                                               const RopeScalingConfig& rope_scaling_cfg) {
  const auto& factor = rope_scaling_cfg.factor;
  const auto& low_freq_factor = rope_scaling_cfg.low_freq_factor;
  const auto& high_freq_factor = rope_scaling_cfg.high_freq_factor;
  const auto& original_max_position_embeddings =
      (rope_scaling_cfg.original_max_position_embeddings);
  double low_freq_wavelen = original_max_position_embeddings / low_freq_factor;
  double high_freq_wavelen = original_max_position_embeddings / high_freq_factor;
  auto wavelen = (2 * std::numbers::pi) / inv_freq;
  // wavelen < high_freq_wavelen: do nothing
  // wavelen > low_freq_wavelen: divide by factor
  // otherwise: interpolate between the two, using a smooth factor
  Eigen::ArrayXf smooth_factor = ((original_max_position_embeddings / wavelen - low_freq_factor) /
                                  (high_freq_factor - low_freq_factor));
  Eigen::ArrayXf smoothed_inv_freq =
      ((1 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq);
  Eigen::ArrayXf inv_freq_llama =
      (wavelen < high_freq_wavelen)
          .select(inv_freq,
                  (wavelen > low_freq_wavelen).select(inv_freq / factor, smoothed_inv_freq));
  return inv_freq_llama;
}

static Eigen::ArrayXf calc_rope_scaling_longrope(const Eigen::ArrayXf& inv_freq,
                                                 RopeScalingConfig& rope_scaling_cfg,
                                                 uint16_t max_num_tokens) {
  if (max_num_tokens > rope_scaling_cfg.original_max_position_embeddings) {
    assert(rope_scaling_cfg.long_factor.has_value());
    auto long_factor = Eigen::Map<Eigen::ArrayXd>(rope_scaling_cfg.long_factor.value().data(),
                                                  rope_scaling_cfg.long_factor.value().size())
                           .cast<float>();
    assert(long_factor.size() <= inv_freq.size());
    return inv_freq.segment(0, long_factor.size()) / long_factor;
  } else {
    assert(rope_scaling_cfg.short_factor.has_value());
    auto short_factor = Eigen::Map<Eigen::ArrayXd>(rope_scaling_cfg.short_factor.value().data(),
                                                   rope_scaling_cfg.short_factor.value().size())
                            .cast<float>();
    assert(short_factor.size() <= inv_freq.size());
    return inv_freq.segment(0, short_factor.size()) / short_factor;
  }
}

// rope_dimension_count controls the active width of the rotary table.
// Proportional RoPE spaces the active frequencies over the full layer head dim;
// other RoPE variants space them over the active rotary dim itself.
RopeTable calc_freq_real_imag(uint16_t max_num_tokens, const std::string& rope_type, double theta,
                              uint16_t rope_dimension_count, uint16_t layer_head_dim,
                              RopeScalingConfig& rope_scaling_cfg) {
  assert(rope_dimension_count % 2 == 0);
  assert(layer_head_dim % 2 == 0);
  uint16_t freq_dim = rope_dimension_count / 2;
  uint16_t frequency_denominator =
      (rope_type == "proportional" ? layer_head_dim : rope_dimension_count);
  auto inv_freq_seq = Eigen::ArrayXd::LinSpaced(freq_dim, 0, rope_dimension_count - 2);
  Eigen::ArrayXf inv_freq =
      Eigen::pow(theta, inv_freq_seq / frequency_denominator).inverse().cast<float>();
  Eigen::ArrayXf scaled_inv_freq;
  if (rope_type == "" || rope_type == "default" || rope_type == "mrope" ||
      rope_type == "proportional") {
    scaled_inv_freq = inv_freq;
  } else if (rope_type == "linear") {
    scaled_inv_freq = inv_freq / rope_scaling_cfg.factor;
  } else if (rope_type == "llama3") {
    scaled_inv_freq = calc_rope_scaling_llama3(inv_freq, rope_scaling_cfg);
  } else if (rope_type == "longrope") {
    scaled_inv_freq = calc_rope_scaling_longrope(inv_freq, rope_scaling_cfg, max_num_tokens);
  } else {
    throw std::runtime_error(fmt::format("{} rope type is not supported", rope_type));
  }

  auto num_scaled_freq = scaled_inv_freq.size();
  auto token_idx_seq = Eigen::ArrayXf::LinSpaced(max_num_tokens, 0, max_num_tokens - 1);

  // Use row major type.
  MatrixXXf scaled_inv_freq_span = (token_idx_seq.matrix() * scaled_inv_freq.matrix().transpose());
  MatrixXXf re_float = scaled_inv_freq_span.array().cos().matrix();
  MatrixXXf im_float = scaled_inv_freq_span.array().sin().matrix();
  if (rope_type == "longrope" && rope_scaling_cfg.attention_factor.has_value()) {
    re_float *= rope_scaling_cfg.attention_factor.value();
    im_float *= rope_scaling_cfg.attention_factor.value();
  }
  MatrixXXbf re = re_float.cast<Eigen::bfloat16>();
  MatrixXXbf im = im_float.cast<Eigen::bfloat16>();

  std::vector<Eigen::bfloat16> re_vec(max_num_tokens * freq_dim);
  std::vector<Eigen::bfloat16> im_vec(max_num_tokens * freq_dim);
  if (num_scaled_freq < freq_dim) {
    // Unscaled dimensions must use identity rotation (cos=1, sin=0), not zero.
    MatrixXXbf padded_re = MatrixXXbf::Ones(max_num_tokens, freq_dim);
    MatrixXXbf padded_im = MatrixXXbf::Zero(max_num_tokens, freq_dim);
    padded_re.block(0, 0, max_num_tokens, num_scaled_freq) = re;
    padded_im.block(0, 0, max_num_tokens, num_scaled_freq) = im;
    std::copy_n(padded_re.data(), re_vec.size(), re_vec.data());
    std::copy_n(padded_im.data(), im_vec.size(), im_vec.data());
  } else {
    std::copy_n(re.data(), re_vec.size(), re_vec.data());
    std::copy_n(im.data(), im_vec.size(), im_vec.data());
  }
  return {std::move(re_vec), std::move(im_vec)};
}

RopeTable calc_mrope_with_image(const VlmConfig& vlm_cfg, const RopeTable& master_rope_table,
                                uint32_t image_token_id,
                                std::span<const uint32_t> input_token_ids) {
  // Map the pre-calculated rope table in std::vector back to Eigen array.
  const auto& max_num_tokens = vlm_cfg.pipeline_cfg.max_num_tokens;
  uint16_t freq_dim = master_rope_table.re.size() / max_num_tokens;
  Eigen::Map<const ArrayXXbf> master_re(master_rope_table.re.data(), max_num_tokens, freq_dim);
  Eigen::Map<const ArrayXXbf> master_im(master_rope_table.im.data(), max_num_tokens, freq_dim);

  // Since image size is fixed at the compile time, so the number of patches (or grids) per each
  // side is fixed.
  const auto& vm_cfg = vlm_cfg.vm_cfg.value();
  size_t num_grids_h = vm_cfg.num_spatial_patches[0] / vm_cfg.spatial_merge_size;
  size_t num_grids_w = vm_cfg.num_spatial_patches[1] / vm_cfg.spatial_merge_size;
  const auto& mm_tokens_per_image = vlm_cfg.mm_cfg.value().mm_tokens_per_image;
  assert(num_grids_h * num_grids_w == mm_tokens_per_image);
  size_t max_num_grids_hw = std::max(num_grids_h, num_grids_w);

  // Get mrope parameters.
  const auto& rope_scaling = vlm_cfg.lm_cfg.rope_cfg.rope_scaling;
  assert(rope_scaling.mrope_section.has_value());
  const auto& mrope_sections = rope_scaling.mrope_section.value();
  const auto& mrope_interleaved = rope_scaling.mrope_interleaved;

  // Construct the updated rope table.
  RopeTable updated_rope_table;
  updated_rope_table.re.resize(max_num_tokens * freq_dim);
  updated_rope_table.im.resize(max_num_tokens * freq_dim);
  Eigen::Map<ArrayXXbf> updated_re(updated_rope_table.re.data(), max_num_tokens, freq_dim);
  Eigen::Map<ArrayXXbf> updated_im(updated_rope_table.im.data(), max_num_tokens, freq_dim);

  size_t pos_idx = 0;
  size_t token_idx = 0;
  size_t image_idx = 0;
  while (true) {
    assert(token_idx <= input_token_ids.size());

    // Find the first image token.
    auto it = std::find(input_token_ids.begin() + token_idx, input_token_ids.end(), image_token_id);

    // Fill the table for text tokens.
    size_t num_text_tokens;
    if (it == input_token_ids.end()) {
      num_text_tokens = max_num_tokens - token_idx;
    } else {
      num_text_tokens = std::distance(input_token_ids.begin(), it) - token_idx;
    }
    if (num_text_tokens > 0) {
      updated_re.block(token_idx, 0, num_text_tokens, freq_dim) =
          (master_re.block(pos_idx, 0, num_text_tokens, freq_dim));
      updated_im.block(token_idx, 0, num_text_tokens, freq_dim) =
          (master_im.block(pos_idx, 0, num_text_tokens, freq_dim));

      if (it == input_token_ids.end())
        break;

      pos_idx += num_text_tokens;
      token_idx += num_text_tokens;
    }

    if (token_idx + mm_tokens_per_image > input_token_ids.size())
      throw std::runtime_error("Not enough of input tokens for image " + std::to_string(image_idx));

    // Fill the table for image tokens.
    if (mrope_interleaved) {
      assert(mrope_sections[0] >= mrope_sections[1] && mrope_sections[0] >= mrope_sections[2]);
      size_t temporal_offset = 0;
      size_t height_offset = mrope_sections[0];
      size_t width_offset = height_offset + mrope_sections[1];
      size_t curr_token_idx = token_idx;

      for (size_t i = 0; i < num_grids_h; ++i) {
        for (size_t j = 0; j < num_grids_w; ++j) {
          // Temporal.
          updated_re.row(curr_token_idx) = master_re.row(pos_idx);
          // Height.
          updated_re(curr_token_idx, Eigen::seqN(1, mrope_sections[1], Eigen::fix<3>)) =
              master_re(pos_idx + i, Eigen::seqN(1, mrope_sections[1], Eigen::fix<3>));
          // Width.
          updated_re(curr_token_idx, Eigen::seqN(2, mrope_sections[2], Eigen::fix<3>)) =
              master_re(pos_idx + j, Eigen::seqN(2, mrope_sections[2], Eigen::fix<3>));
          ++curr_token_idx;
        }
      }
      curr_token_idx = token_idx;
      for (size_t i = 0; i < num_grids_h; ++i) {
        for (size_t j = 0; j < num_grids_w; ++j) {
          // Temporal.
          updated_im.row(curr_token_idx) = master_im.row(pos_idx);
          // Height.
          updated_im(curr_token_idx, Eigen::seqN(1, mrope_sections[1], Eigen::fix<3>)) =
              master_im(pos_idx + i, Eigen::seqN(1, mrope_sections[1], Eigen::fix<3>));
          // Width.
          updated_im(curr_token_idx, Eigen::seqN(2, mrope_sections[2], Eigen::fix<3>)) =
              master_im(pos_idx + j, Eigen::seqN(2, mrope_sections[2], Eigen::fix<3>));
          ++curr_token_idx;
        }
      }
    } else {
      size_t temporal_offset = 0;
      size_t height_offset = mrope_sections[0];
      size_t width_offset = height_offset + mrope_sections[1];
      size_t curr_token_idx = token_idx;
      for (size_t i = 0; i < num_grids_h; ++i) {
        for (size_t j = 0; j < num_grids_w; ++j) {
          // Temporal.
          updated_re.block(curr_token_idx, temporal_offset, 1, mrope_sections[0]) =
              (master_re.block(pos_idx, temporal_offset, 1, mrope_sections[0]));
          // Height.
          updated_re.block(curr_token_idx, height_offset, 1, mrope_sections[1]) =
              (master_re.block(pos_idx + i, height_offset, 1, mrope_sections[1]));
          // Width.
          updated_re.block(curr_token_idx, width_offset, 1, mrope_sections[2]) =
              (master_re.block(pos_idx + j, width_offset, 1, mrope_sections[2]));
          ++curr_token_idx;
        }
      }
      curr_token_idx = token_idx;
      for (size_t i = 0; i < num_grids_h; ++i) {
        for (size_t j = 0; j < num_grids_w; ++j) {
          // Temporal.
          updated_im.block(curr_token_idx, temporal_offset, 1, mrope_sections[0]) =
              (master_im.block(pos_idx, temporal_offset, 1, mrope_sections[0]));
          // Height.
          updated_im.block(curr_token_idx, height_offset, 1, mrope_sections[1]) =
              (master_im.block(pos_idx + i, height_offset, 1, mrope_sections[1]));
          // Width.
          updated_im.block(curr_token_idx, width_offset, 1, mrope_sections[2]) =
              (master_im.block(pos_idx + j, width_offset, 1, mrope_sections[2]));
          ++curr_token_idx;
        }
      }
    }
    ++image_idx;
    pos_idx += max_num_grids_hw;
    token_idx += mm_tokens_per_image;
  }
  return updated_rope_table;
}

} // namespace llima
} // namespace simaai
