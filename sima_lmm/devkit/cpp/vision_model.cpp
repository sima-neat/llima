
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


void VisionModel::run_model(
    const std::vector<Eigen::bfloat16>& ifm_tensor, std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
) {
    // Upload the ifm.
    get_buffer("vision_ifm").upload(ifm_tensor.data());

    // Run the models.
    for (size_t i = 0; i + 1 < _model_ptrs.size(); ++i) {
        _model_ptrs[i]->add_to_queue();
    }
    _model_ptrs.back()->add_to_queue(nullptr, ofm_map_ptr);
    MLAModelWithBuffer::run_queue();
}


void VisionModel::_initialize() {
    _logger->info("Vision model initialize starting ...");
    BaseModel::_initialize();
    _define_models();
    for (const auto& model_name: _cfg.vision_model_name) {
        MLAModelWithBuffer::load_all_models(_elf_dir / model_name);
    }
    _logger->info("Vision model initialize completed");
}


void VisionModel::_finalize() {
    _logger->info("Vision model finalize starting ...");
    for (const auto& model_name: _cfg.vision_model_name) {
        MLAModelWithBuffer::free_all_models(_elf_dir / model_name);
    }
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

    define_buffer(
        "vision_ofm",
        {_mm_cfg.mm_tokens_per_image, _cfg.lm_cfg.hidden_size},
        _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16"
    );
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        define_buffer("vision_ofm_scale", {_mm_cfg.mm_tokens_per_image, 1});
    }
    if (_cfg.vision_model_name.size() > 1) {
        uint32_t seq_len = (
            _vm_cfg.num_spatial_patches[0] * _vm_cfg.num_spatial_patches[1] + _vm_cfg.cls_embed
        );
        // Split ELFs are not compiled for in-place execution, so alternate between two buffers.
        define_buffer("vision_hidden_0", {seq_len, _vm_cfg.hidden_size});
        define_buffer("vision_hidden_1", {seq_len, _vm_cfg.hidden_size});
    }
    for (size_t i = 0; i < _vm_cfg.deepstack_visual_indexes.size(); ++i) {
        define_buffer(
            fmt::format("deepstack_feature_l{}", i),
            {_mm_cfg.mm_tokens_per_image, _cfg.lm_cfg.hidden_size}
        );
    }
}


void VisionModel::_define_models() {
    if (_cfg.vision_model_name.empty()) {
        throw std::runtime_error("vision_model_name must contain at least one model");
    }

    std::vector<MLABufferSlice> final_ofms{&get_buffer("vision_ofm")};
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        final_ofms.emplace_back(&get_buffer("vision_ofm_scale"));
    }
    for (size_t i = 0; i < _vm_cfg.deepstack_visual_indexes.size(); ++i) {
        final_ofms.emplace_back(&get_buffer(fmt::format("deepstack_feature_l{}", i)));
    }

    for (size_t i = 0; i < _cfg.vision_model_name.size(); ++i) {
        const bool is_first = i == 0;
        const bool is_last = i + 1 == _cfg.vision_model_name.size();
        auto& ifm = get_buffer(
            is_first ? "vision_ifm" : fmt::format("vision_hidden_{}", (i - 1) % 2)
        );
        std::vector<MLABufferSlice> ofms = is_last
            ? final_ofms
            : std::vector<MLABufferSlice>{
                &get_buffer(fmt::format("vision_hidden_{}", i % 2))
            };
        auto elf_file_name = fmt::format("{}_stage1_mla.elf", _cfg.vision_model_name[i]);
        _model_ptrs.emplace_back(std::make_unique<MLAModelWithBuffer>(
            _elf_dir / elf_file_name,
            std::vector<MLABufferSlice>{&ifm},
            std::move(ofms)
        ));
    }
}


}
}
