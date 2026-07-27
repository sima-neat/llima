#ifndef _SIMA_LLIMA_IMAGE_PROCESSOR_
#define _SIMA_LLIMA_IMAGE_PROCESSOR_

#include <filesystem>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "utils.hpp"
#include "vlm_config.hpp"

namespace simaai {
namespace llima {

class ImageProcessor {
public:
  ImageProcessor(const VlmConfig& vlm_cfg, bool do_pad_to_square, bool do_center_crop,
                 cv::InterpolationFlags interpolation, double rescale_factor,
                 std::vector<double> image_mean, std::vector<double> image_std);
  ~ImageProcessor() {}

  std::vector<Eigen::bfloat16> preprocess(const std::filesystem::path image_path);
  std::vector<Eigen::bfloat16> preprocess(const std::vector<uint8_t>& image_path);
  std::vector<Eigen::bfloat16> preprocess(const cv::Mat& rgb_image);

  static void read_env_vars() {
    // Set the debug info from env variables.
    _do_save_preprocessed_images =
        get_env_var("SIMA_LLIMA_RUN_SAVE_PREPROC_IMAGES", _do_save_preprocessed_images);
    if (_do_save_preprocessed_images)
      _save_image_dir = get_env_var("SIMA_LLIMA_RUN_SAVE_IMAGE_DIR", _save_image_dir);
  }

private:
  std::vector<Eigen::bfloat16> _preprocess(cv::Mat image);
  cv::Mat _convert_to_rgb(cv::Mat image);
  cv::Mat _pad_to_square(cv::Mat image);
  std::vector<Eigen::bfloat16> _cast_and_patchify(cv::Mat image);
  std::vector<Eigen::bfloat16> _cast_and_patchify_lfm2_gemma4(cv::Mat image);
  std::vector<Eigen::bfloat16> _cast_and_patchify_qwen(cv::Mat image);

  const VlmConfig& _vlm_cfg;
  size_t _target_height;
  size_t _target_width;
  bool _do_pad_to_square;
  bool _do_center_crop;
  cv::InterpolationFlags _interpolation;
  double _rescale_factor;
  std::vector<double> _image_mean;
  std::vector<double> _image_std;

  inline static bool _do_save_preprocessed_images = false;
  inline static std::filesystem::path _save_image_dir = "debug/preproc_images";
};

} // namespace llima
} // namespace simaai

#endif
