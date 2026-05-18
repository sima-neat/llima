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


#include "vision_model.hpp"


namespace simaai {
namespace llima {

VisionModel::VisionModel(
    std::filesystem::path model_path
) : BaseModel(model_path),
    _vm_cfg(_cfg.vm_cfg.value()),
    _mm_cfg(_cfg.mm_cfg.value())
{
    _initialize();
}


std::vector<Eigen::bfloat16> VisionModel::run_model(
    const std::vector<Eigen::bfloat16>& ifm_tensor
) {
    // Upload the ifm.
    get_buffer("vision_ifm").upload(ifm_tensor.data());

    // Run the models.
    _model_ptr->run();

    // Download the ofm.
    auto& ofm_buf = get_buffer("vision_ofm");
    std::vector<Eigen::bfloat16> ofm_tensor(ofm_buf.get_num_elems());
    ofm_buf.download(ofm_tensor.data());
    return ofm_tensor;
}


void VisionModel::run_model(
    const std::vector<Eigen::bfloat16>& ifm_tensor, std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    // Upload the ifm.
    get_buffer("vision_ifm").upload(ifm_tensor.data());

    // Run the models.
    _model_ptr->run(nullptr, ofm_map_ptr);
}


void VisionModel::_initialize() {
    _logger->info("Vision model initialize starting ...");
    BaseModel::_initialize();
    _define_models();
    MLAModelWithBuffer::load_all_models(false, _elf_dir / _cfg.vision_model_name);
    _logger->info("Vision model initialize completed");
}


void VisionModel::_finalize() {
    _logger->info("Vision model finalize starting ...");
    MLAModelWithBuffer::free_all_models(_elf_dir / _cfg.vision_model_name);
    BaseModel::_finalize();
    _logger->info("Vision model finalize completed");
}


void VisionModel::_define_buffers() {
    if (
        _cfg.model_type.starts_with("vlm-lfm2")
        || _cfg.model_type.starts_with("vlm-qwen")
        || _cfg.model_type.starts_with("vlm-gemma4")
    ) {
        // In Huggingface, the patchify step is implemented in the image preprocessor instead of the
        // model.
        size_t num_channels = 3;
        uint32_t patch_feature_size = (
            _vm_cfg.temporal_patch_size * _vm_cfg.spatial_patch_size * _vm_cfg.spatial_patch_size
            * num_channels
        );
        uint32_t seq_len = (
            _vm_cfg.num_spatial_patches[0] * _vm_cfg.num_spatial_patches[1] + _vm_cfg.cls_embed
        );
        define_buffer(
            "vision_ifm",
            {seq_len, patch_feature_size},
            "bfloat16",
            false
        );
    } else {
        define_buffer(
            "vision_ifm",
            {_vm_cfg.image_sizes[0], _vm_cfg.image_sizes[1], 3},
            "bfloat16",
            false
        );
    }

    define_buffer("vision_ofm", {_mm_cfg.mm_tokens_per_image, _cfg.lm_cfg.hidden_size});
    for (size_t i = 0; i < _vm_cfg.deepstack_visual_indexes.size(); ++i) {
        define_buffer(
            fmt::format("deepstack_feature_l{}", i),
            {_mm_cfg.mm_tokens_per_image, _cfg.lm_cfg.hidden_size}
        );
    }
}


void VisionModel::_define_models() {
    auto elf_file_name = fmt::format("{}_stage1_mla.elf", _cfg.vision_model_name);
    std::vector<MLABufferSlice> ofms{&get_buffer("vision_ofm")};
    for (size_t i = 0; i < _vm_cfg.deepstack_visual_indexes.size(); ++i) {
        ofms.emplace_back(&get_buffer(fmt::format("deepstack_feature_l{}", i)));
    }
    _model_ptr = std::make_unique<MLAModelWithBuffer>(
        _elf_dir / elf_file_name,
        std::vector<MLABufferSlice>{MLABufferSlice{&get_buffer("vision_ifm")}},
        std::move(ofms)
    );
}


}
}
