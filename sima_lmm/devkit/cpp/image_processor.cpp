#include <algorithm>
#include <bit>
#include <fstream>
#include <stdexcept>

#include "eigen_types.hpp"
#include "image_processor.hpp"

namespace simaai {
namespace llima {


uint16_t _convert_to_bfloat16(float data) {
    uint32_t u = std::bit_cast<uint32_t>(data);
    uint32_t bias = 0x7FFF + ((u >> 16) & 1);
    u += bias;
    return static_cast<uint16_t>(u >> 16);
}


ImageProcessor::ImageProcessor(
    const VlmConfig& vlm_cfg,
    bool do_pad_to_square,
    bool do_center_crop,
    cv::InterpolationFlags interpolation,
    double rescale_factor,
    std::vector<double> image_mean,
    std::vector<double> image_std
) : _vlm_cfg(vlm_cfg),
    _target_height(_vlm_cfg.vm_cfg.value().image_sizes[0]),
    _target_width(_vlm_cfg.vm_cfg.value().image_sizes[1]),
    _do_pad_to_square(do_pad_to_square),
    _do_center_crop(do_center_crop),
    _interpolation(interpolation),
    _rescale_factor(rescale_factor),
    _image_mean(std::move(image_mean)),
    _image_std(std::move(image_std))
{
    if (_do_pad_to_square)
        assert(_target_height == _target_width);
    if (_do_center_crop)
        assert(_target_height == _target_width);
}


std::vector<Eigen::bfloat16> ImageProcessor::preprocess(const std::filesystem::path image_path) {
    if (image_path.extension() == ".bin") {
        // Image is already preprocessed. Implicit assumption from the .bin extension.
        auto file_size = std::filesystem::file_size(image_path);
        std::vector<Eigen::bfloat16> image(file_size / 2);
        std::ifstream ifs(image_path, std::ios::binary);
        ifs.read(reinterpret_cast<char*>(image.data()), file_size);
        ifs.close();
        return image;
    } else {
        // Read image from file.
        auto image = cv::imread(image_path, cv::IMREAD_COLOR);

        // Convert from BGR (default) to RGB.
        image = _convert_to_rgb(std::move(image));

        // Preprocess.
        return _preprocess(std::move(image));
    }
}


std::vector<Eigen::bfloat16> ImageProcessor::preprocess(const std::vector<uint8_t>& image_bytes) {
    // Decode the image from bytes.
    auto image = cv::imdecode(image_bytes, cv::IMREAD_COLOR);

    // Convert from BGR (default) to RGB.
    image = _convert_to_rgb(std::move(image));

    // Preprocess.
    return _preprocess(std::move(image));
}


std::vector<Eigen::bfloat16> ImageProcessor::preprocess(const cv::Mat& rgb_image) {
    if (rgb_image.type() != CV_8UC3) {
        throw std::runtime_error("ImageProcessor::preprocess requires CV_8UC3 RGB input");
    }

    return _preprocess(rgb_image.clone());
}


std::vector<Eigen::bfloat16> ImageProcessor::_preprocess(cv::Mat image) {
    if (_do_pad_to_square)
        image = _pad_to_square(image);

    int resize_height, resize_width;
    if (_do_center_crop) {
        auto orig_height = image.rows;
        auto orig_width = image.cols;
        if (orig_height < orig_width) {
            resize_height = _target_height;
            resize_width = static_cast<int>(
                static_cast<double>(orig_width * _target_height) / orig_height
            );
        } else {
            resize_width = _target_width;
            resize_height = static_cast<int>(
                static_cast<double>(orig_height * _target_width) / orig_width
            );
        }
    } else {
        resize_height = _target_height;
        resize_width = _target_width;
    }

    // Note that the opencv resize is not using the same algorithm as PIL's resize used in the
    // transformers python library. It is expected that the outputs are not matching.
    cv::resize(image, image, cv::Size(resize_width, resize_height), 0, 0, _interpolation);

    if (_do_center_crop) {
        int top = (image.rows - _target_height) / 2;
        int left = (image.cols - _target_width) / 2;
        image = image(cv::Rect(left, top, _target_width, _target_height));
    }

    // Rescale and normalize.
    image.convertTo(image, CV_32FC3, _rescale_factor);
    cv::Mat transform_mat = (cv::Mat_<float>(3, 4) << 
        1.0 / _image_std[0], 0, 0, -_image_mean[0] / _image_std[0],
        0, 1.0 / _image_std[1], 0, -_image_mean[1] / _image_std[1],
        0, 0, 1.0 / _image_std[2], -_image_mean[2] / _image_std[2]
    );
    cv::transform(image, image, transform_mat);

    // Cast to bfloat16 and patchify.
    auto cast_image = _cast_and_patchify(std::move(image));

    // Save the preprocessed images.
    if (_do_save_preprocessed_images) {
        std::filesystem::create_directories(_save_image_dir);
        auto num_files = count_regular_files(_save_image_dir);
        auto file_name = _save_image_dir / fmt::format("image{}.bin", num_files);
        std::ofstream ofs(file_name, std::ios::binary);
        ofs.write(reinterpret_cast<char*>(cast_image.data()), cast_image.size() * 2);
        ofs.close();
    }
    return cast_image;
}


cv::Mat ImageProcessor::_convert_to_rgb(cv::Mat image) {
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);
    } else {
        assert(image.channels() == 3);
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    }
    return image;
}


cv::Mat ImageProcessor::_pad_to_square(cv::Mat image) {
    auto orig_height = image.rows;
    auto orig_width = image.cols;

    if (orig_height == orig_width)
        return image;

    cv::Scalar background_color(
        static_cast<uint8_t>(_image_mean[0] * 255),
        static_cast<uint8_t>(_image_mean[1] * 255),
        static_cast<uint8_t>(_image_mean[2] * 255)
    );
    auto max_dim = std::max(orig_height, orig_width);
    cv::Mat result(max_dim, max_dim, image.type(), background_color);
    if (orig_width > orig_height) {
        image.copyTo(result(cv::Rect(0, (max_dim - orig_height) / 2, orig_width, orig_height)));
    } else {
        image.copyTo(result(cv::Rect((max_dim - orig_width) / 2, 0, orig_width, orig_height)));
    }
    return result;
}


std::vector<Eigen::bfloat16> ImageProcessor::_cast_and_patchify(cv::Mat image) {
    if (
        _vlm_cfg.model_type.starts_with("vlm-lfm2")
        || _vlm_cfg.model_type.starts_with("vlm-gemma4")
    ) {
        return _cast_and_patchify_lfm2_gemma4(std::move(image));
    } else if (_vlm_cfg.model_type.starts_with("vlm-qwen")) {
        return _cast_and_patchify_qwen(std::move(image));
    } else {
        std::vector<Eigen::bfloat16> cast_tensor(image.total() * image.channels());
        Eigen::Map<ArrayXf> image_tensor_map(
            reinterpret_cast<float*>(image.data), cast_tensor.size()
        );
        Eigen::Map<ArrayXbf> cast_tensor_map(cast_tensor.data(), cast_tensor.size());
        cast_tensor_map = image_tensor_map.cast<Eigen::bfloat16>();
        return cast_tensor;
    }
}


std::vector<Eigen::bfloat16> ImageProcessor::_cast_and_patchify_lfm2_gemma4(cv::Mat image) {
    // Prepare parameters.
    const auto& vm_cfg = _vlm_cfg.vm_cfg.value();
    const Eigen::Index channel = 3;
    const Eigen::Index spatial_patch_size = vm_cfg.spatial_patch_size;
    const Eigen::Index num_patches_height = vm_cfg.num_spatial_patches[0];
    const Eigen::Index num_patches_width = vm_cfg.num_spatial_patches[1];

    // Create the output tensor.
    std::vector<Eigen::bfloat16> patched_tensor(image.total() * image.channels());

    // Map the tensors to Eigen types.
    Eigen::TensorMap<Tensor3Df> image_tensor_map(
        reinterpret_cast<float*>(image.data), _target_height, _target_width, channel
    );
    Eigen::TensorMap<Tensor5Dbf> patched_tensor_map(
        patched_tensor.data(), num_patches_height, num_patches_width, spatial_patch_size,
        spatial_patch_size, channel
    );

    // Reshape dimensions.
    Eigen::array<Eigen::Index, 5> reshape_dims = {
        num_patches_height, spatial_patch_size, num_patches_width, spatial_patch_size, channel
    };

    // Transpose axes.
    Eigen::array<Eigen::Index, 5> transpose_axes = {0, 2, 1, 3, 4};

    // Reshape and transpose.
    patched_tensor_map.device(get_eigen_device()) = (
        image_tensor_map.cast<Eigen::bfloat16>().reshape(reshape_dims).shuffle(transpose_axes)
    );
    return patched_tensor;
}


std::vector<Eigen::bfloat16> ImageProcessor::_cast_and_patchify_qwen(cv::Mat image) {
    // Prepare parameters.
    const auto& vm_cfg = _vlm_cfg.vm_cfg.value();
    const Eigen::Index channel = 3;
    const Eigen::Index temporal_patch_size = vm_cfg.temporal_patch_size;
    const Eigen::Index spatial_patch_size = vm_cfg.spatial_patch_size;
    const Eigen::Index spatial_merge_size = vm_cfg.spatial_merge_size;
    const Eigen::Index num_grids_t = 1;
    const Eigen::Index num_grids_h = vm_cfg.num_spatial_patches[0] / vm_cfg.spatial_merge_size;
    const Eigen::Index num_grids_w = vm_cfg.num_spatial_patches[1] / vm_cfg.spatial_merge_size;

    // Create the output tensor.
    std::vector<Eigen::bfloat16> patched_tensor(
        image.total() * image.channels() * temporal_patch_size
    );

    // Map the tensors to Eigen types.
    Eigen::TensorMap<Tensor3Df> image_tensor_map(
        reinterpret_cast<float*>(image.data), _target_height, _target_width, channel
    );
    Eigen::TensorMap<Tensor9Dbf> patched_tensor_map(
        patched_tensor.data(), num_grids_t, num_grids_h, num_grids_w, spatial_merge_size,
        spatial_merge_size, channel, temporal_patch_size, spatial_patch_size, spatial_patch_size
    );

    // Reshape dimensions.
    Eigen::array<Eigen::Index, 9> reshape_dims = {
        1, 1, num_grids_h, spatial_merge_size, spatial_patch_size, num_grids_w,
        spatial_merge_size, spatial_patch_size, channel
    };

    // Transpose axes.
    Eigen::array<Eigen::Index, 9> transpose_axes = {0, 2, 5, 3, 6, 8, 1, 4, 7};

    // Broadcast dimensions.
    Eigen::array<Eigen::Index, 9> broadcast_dims = {1, 1, 1, 1, 1, 1, temporal_patch_size, 1, 1};

    // Reshape and transpose.
    patched_tensor_map.device(get_eigen_device()) = (
        image_tensor_map.cast<Eigen::bfloat16>()
            .reshape(reshape_dims).shuffle(transpose_axes).eval().broadcast(broadcast_dims)
    );
    return patched_tensor;
}


}
}
