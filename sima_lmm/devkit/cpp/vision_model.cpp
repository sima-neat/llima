
#include <fstream>

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

    // Each layer is a separate dispatcher job so other MLA workloads can run
    // between dependent vision layers.
    for (size_t i = 0; i + 1 < _model_ptrs.size(); ++i) {
        _model_ptrs[i]->run();
    }
    _model_ptrs.back()->run(nullptr, ofm_map_ptr);
}


void VisionModel::_initialize() {
    _logger->info("Vision model initialize starting ...");
    _validate_model_names();
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
    for (size_t i = 0; i < _cfg.vision_model_name.size(); ++i) {
        const bool is_first = i == 0;
        const bool is_last = i + 1 == _cfg.vision_model_name.size();
        auto& ifm = get_buffer(
            is_first ? "vision_ifm" : fmt::format("vision_hidden_{}", (i - 1) % 2)
        );
        std::vector<MLABufferSlice> ofms;
        if (is_last) {
            ofms.emplace_back(&get_buffer("vision_ofm"));
            if (_cfg.pipeline_cfg.quantize_embeddings) {
                ofms.emplace_back(&get_buffer("vision_ofm_scale"));
            }
        } else {
            ofms.emplace_back(&get_buffer(fmt::format("vision_hidden_{}", i % 2)));
        }
        for (size_t deepstack_idx = 0;
             deepstack_idx < _vm_cfg.deepstack_visual_indexes.size();
             ++deepstack_idx) {
            if (_cfg.vision_model_name.size() == 1
                || _vm_cfg.deepstack_visual_indexes[deepstack_idx] == i) {
                ofms.emplace_back(
                    &get_buffer(fmt::format("deepstack_feature_l{}", deepstack_idx))
                );
                if (_cfg.vision_model_name.size() > 1) {
                    break;
                }
            }
        }
        auto elf_file_name = fmt::format("{}_stage1_mla.elf", _cfg.vision_model_name[i]);
        _model_ptrs.emplace_back(std::make_unique<MLAModelWithBuffer>(
            _elf_dir / elf_file_name,
            std::vector<MLABufferSlice>{&ifm},
            std::move(ofms)
        ));
    }
}


void VisionModel::_validate_model_names() const {
    if (_cfg.vision_model_name.empty()) {
        throw std::runtime_error("vision_model_name must contain at least one model");
    }

    const auto config_path = _devkit_dir / "vlm_config.json";
    const auto config_json = nlohmann::json::parse(std::ifstream(config_path));
    if (!config_json.contains("vm_cfg")
        || !config_json["vm_cfg"].contains("num_hidden_layers")) {
        throw std::runtime_error(fmt::format(
            "Cannot find vm_cfg.num_hidden_layers in {}", config_path.string()
        ));
    }
    size_t expected_layers = config_json["vm_cfg"]["num_hidden_layers"].get<size_t>();
    if (_cfg.model_type == "vlm-llava") {
        if (expected_layers == 0) {
            throw std::runtime_error("LLaVA vision model must contain at least one layer");
        }
        --expected_layers;
    }
    if (_cfg.vision_model_name.size() == 1) {
        // Preserve compatibility with existing monolithic vision repositories.
        return;
    }
    if (_cfg.vision_model_name.size() != expected_layers) {
        throw std::runtime_error(fmt::format(
            "Layered vision model requires {} ordered artifacts, but vision_model_name "
            "contains {}",
            expected_layers,
            _cfg.vision_model_name.size()
        ));
    }

    const std::string first_suffix = "_layer0";
    const auto& first_name = _cfg.vision_model_name.front();
    if (!first_name.ends_with(first_suffix)) {
        throw std::runtime_error(fmt::format(
            "Layered vision model must start with an artifact named '*{}', got '{}'",
            first_suffix,
            first_name
        ));
    }
    const auto base_name = first_name.substr(0, first_name.size() - first_suffix.size());
    for (size_t layer_idx = 0; layer_idx < expected_layers; ++layer_idx) {
        const auto expected_name = fmt::format("{}_layer{}", base_name, layer_idx);
        if (_cfg.vision_model_name[layer_idx] != expected_name) {
            throw std::runtime_error(fmt::format(
                "Layered vision artifact {} must be '{}', got '{}'",
                layer_idx,
                expected_name,
                _cfg.vision_model_name[layer_idx]
            ));
        }
    }
}


}
}
