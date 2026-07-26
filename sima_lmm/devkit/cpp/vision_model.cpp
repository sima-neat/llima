
#include "vision_model.hpp"


namespace simaai {
namespace llima {

VisionModel::VisionModel(
    std::filesystem::path model_path
) : VisionModel(current_mla_execution_session(), std::move(model_path)) {}

VisionModel::VisionModel(
    std::shared_ptr<MlaExecutionSession> session,
    std::filesystem::path model_path
) : BaseModel(model_path, std::move(session)),
    _vm_cfg(_cfg.vm_cfg.value()),
    _mm_cfg(_cfg.mm_cfg.value())
{
    _initialize();
}


std::vector<Eigen::bfloat16> VisionModel::run_model(
    const std::vector<Eigen::bfloat16>& ifm_tensor
) {
    require_healthy_mla_session();
    MlaExecutionSegment segment(_mla_session);
    // Upload the ifm.
    get_buffer("vision_ifm").upload(ifm_tensor.data());

    // Run the models.
    _model_ptr->add_to_segment(segment);
    segment.commit();

    // Download the ofm.
    auto& ofm_buf = get_buffer("vision_ofm");
    std::vector<Eigen::bfloat16> ofm_tensor(ofm_buf.get_num_elems());
    ofm_buf.download(ofm_tensor.data());
    return ofm_tensor;
}


void VisionModel::run_model(
    const std::vector<Eigen::bfloat16>& ifm_tensor, std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    require_healthy_mla_session();
    MlaExecutionSegment segment(_mla_session);
    // Upload the ifm.
    get_buffer("vision_ifm").upload(ifm_tensor.data());

    // Run the models.
    _model_ptr->add_to_segment(segment, nullptr, ofm_map_ptr);
    segment.commit();
}


void VisionModel::_initialize() {
    _logger->info("Vision model initialize starting ...");
    BaseModel::_initialize();
    _define_models();
    _model_ptr->load_related_models(_elf_dir / _cfg.vision_model_name);
    _logger->info("Vision model initialize completed");
}


void VisionModel::_finalize() {
    _logger->info("Vision model finalize starting ...");
    _model_ptr->free_related_models(_elf_dir / _cfg.vision_model_name);
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
        _mla_session,
        _elf_dir / elf_file_name,
        std::vector<MLABufferSlice>{MLABufferSlice{&get_buffer("vision_ifm")}},
        std::move(ofms)
    );
}


}
}
