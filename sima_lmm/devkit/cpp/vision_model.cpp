
#include <utility>

#include "vision_model.hpp"


namespace simaai {
namespace llima {

VisionModel::VisionModel(std::filesystem::path model_path)
    : VisionModel(std::move(model_path), nullptr) {}

VisionModel::VisionModel(
    std::filesystem::path model_path,
    std::shared_ptr<FileProvider> file_provider
) : BaseModel(model_path, std::move(file_provider)),
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

    if (_model_ptrs.size() == 1) {
        _model_ptrs.front()->run(nullptr, ofm_map_ptr);
        return;
    }

    // Each layer is a separate MLA job so other MLA workloads can run
    // between dependent vision layers.
    const uint8_t final_ofm_count = _cfg.pipeline_cfg.quantize_embeddings ? 2 : 1;
    for (size_t layer_idx = 0; layer_idx < _model_ptrs.size(); ++layer_idx) {
        const bool is_last = layer_idx + 1 == _model_ptrs.size();
        std::map<uint8_t, MLABufferSlice> layer_ofm_map;

        if (ofm_map_ptr != nullptr) {
            if (is_last) {
                for (uint8_t ofm_idx = 0; ofm_idx < final_ofm_count; ++ofm_idx) {
                    if (const auto it = ofm_map_ptr->find(ofm_idx); it != ofm_map_ptr->end()) {
                        layer_ofm_map.emplace(ofm_idx, it->second);
                    }
                }
            }

            for (size_t deepstack_idx = 0;
                 deepstack_idx < _vm_cfg.deepstack_visual_indexes.size();
                 ++deepstack_idx) {
                if (_vm_cfg.deepstack_visual_indexes[deepstack_idx] == layer_idx) {
                    const uint8_t layer_ofm_idx = is_last ? final_ofm_count : 1;
                    const uint8_t request_ofm_idx = final_ofm_count + deepstack_idx;
                    if (const auto it = ofm_map_ptr->find(request_ofm_idx);
                        it != ofm_map_ptr->end()) {
                        layer_ofm_map.emplace(layer_ofm_idx, it->second);
                    }
                    break;
                }
            }
        }

        _model_ptrs[layer_idx]->run(
            nullptr, layer_ofm_map.empty() ? nullptr : &layer_ofm_map
        );
    }
}


void VisionModel::_initialize() {
    _logger->info("Vision model initialize starting ...");
    _validate_model_names();
    BaseModel::_initialize();
    if (_cfg.model_type == "vlm-gemma4_unified") {
        _unified_position_table.resize(static_cast<size_t>(_vm_cfg.num_position_embeddings) * 2 * _vm_cfg.hidden_size);
        auto stream = _files->open_stream("devkit/vision_positions.bin");
        stream->read(reinterpret_cast<char*>(_unified_position_table.data()), _unified_position_table.size() * sizeof(Eigen::bfloat16));
        if (!*stream) throw std::runtime_error("Invalid Unified vision position table");
    }
    _define_models();
    for (const auto& model_name: _cfg.vision_model_name) {
        MLAModelWithBuffer::load_all_models(_elf_dir / model_name, _files.get());
    }
    if (_cfg.support_audio()) {
        std::vector<MLABufferSlice> outputs{&get_buffer("vision_ofm")};
        if (_cfg.pipeline_cfg.quantize_embeddings) outputs.emplace_back(&get_buffer("vision_ofm_scale"));
        _audio_model = std::make_unique<MLAModelWithBuffer>(
            _elf_dir / (_cfg.audio_model_name + "_stage1_mla.elf"),
            std::vector<MLABufferSlice>{&get_buffer("audio_ifm")},
            std::move(outputs));
        MLAModelWithBuffer::load_all_models(_elf_dir / _cfg.audio_model_name, _files.get());
    }
    _logger->info("Vision model initialize completed");
}


void VisionModel::_finalize() {
    _logger->info("Vision model finalize starting ...");
    for (const auto& model_name: _cfg.vision_model_name) {
        MLAModelWithBuffer::free_all_models(_elf_dir / model_name);
    }
    if (_cfg.support_audio()) MLAModelWithBuffer::free_all_models(_elf_dir / _cfg.audio_model_name);
    BaseModel::_finalize();
    _logger->info("Vision model finalize completed");
}


void VisionModel::_define_buffers() {
    if (_cfg.support_audio()) {
        define_buffer("audio_ifm", {_cfg.pipeline_cfg.input_token_group_size,
                      _cfg.audio_cfg->samples_per_token}, "bfloat16", false);
    }
    if (_cfg.model_type == "vlm-gemma4_unified") {
        const uint32_t side = _vm_cfg.spatial_patch_size * _vm_cfg.spatial_merge_size;
        define_buffer("vision_ifm", {_cfg.pipeline_cfg.input_token_group_size, side * side * 3}, "bfloat16", false);
        define_buffer("vision_position_ifm", {_cfg.pipeline_cfg.input_token_group_size, _vm_cfg.hidden_size});
    } else if (
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
        {_cfg.model_type == "vlm-gemma4_unified" ? _cfg.pipeline_cfg.input_token_group_size : _mm_cfg.mm_tokens_per_image, _cfg.lm_cfg.hidden_size},
        _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16"
    );
    if (_cfg.pipeline_cfg.quantize_embeddings) {
        define_buffer("vision_ofm_scale",
            {_cfg.model_type == "vlm-gemma4_unified" ? _cfg.pipeline_cfg.input_token_group_size : _mm_cfg.mm_tokens_per_image, 1});
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
        std::vector<MLABufferSlice> ifms{&ifm};
        if (_cfg.model_type == "vlm-gemma4_unified") ifms.emplace_back(&get_buffer("vision_position_ifm"));
        _model_ptrs.emplace_back(std::make_unique<MLAModelWithBuffer>(
            _elf_dir / elf_file_name,
            std::move(ifms),
            std::move(ofms)
        ));
    }
}

void VisionModel::run_unified_model(std::span<const Eigen::bfloat16> patches,
                                  std::span<const uint16_t> positions,
                                  const MLABufferSlice& destination,
                                  const MLABufferSlice* scale_destination) {
    const size_t tokens = positions.size() / 2;
    const size_t group = _cfg.pipeline_cfg.input_token_group_size;
    const size_t side = _vm_cfg.spatial_patch_size * _vm_cfg.spatial_merge_size;
    const size_t patch_dim = side * side * 3;
    const size_t hidden = _cfg.lm_cfg.hidden_size;
    if (tokens == 0 || positions.size() % 2 || tokens > _vm_cfg.max_soft_tokens || patches.size() != tokens * patch_dim) {
        throw std::invalid_argument("Unified image patches/positions do not match the token budget");
    }
    if (destination.get_buf_shapes()->front() != tokens) throw std::invalid_argument("Image placeholder count mismatch");
    std::vector<Eigen::bfloat16> patch_chunk(group * patch_dim);
    std::vector<Eigen::bfloat16> position_chunk(group * hidden);
    for (size_t begin = 0; begin < tokens; begin += group) {
        const size_t valid = std::min(group, tokens - begin);
        std::fill(patch_chunk.begin(), patch_chunk.end(), Eigen::bfloat16{0.0f});
        std::fill(position_chunk.begin(), position_chunk.end(), Eigen::bfloat16{0.0f});
        std::copy_n(patches.begin() + begin * patch_dim, valid * patch_dim, patch_chunk.begin());
        for (size_t row = 0; row < valid; ++row) {
            const auto x = positions[(begin + row) * 2];
            const auto y = positions[(begin + row) * 2 + 1];
            if (x >= _vm_cfg.num_position_embeddings || y >= _vm_cfg.num_position_embeddings) {
                throw std::invalid_argument("Image position exceeds the factorized table");
            }
            for (size_t c = 0; c < hidden; ++c) {
                position_chunk[row * hidden + c] = Eigen::bfloat16{
                    static_cast<float>(_unified_position_table[(x * 2) * hidden + c])
                    + static_cast<float>(_unified_position_table[(y * 2 + 1) * hidden + c])};
            }
        }
        get_buffer("vision_ifm").upload(patch_chunk.data());
        get_buffer("vision_position_ifm").upload(position_chunk.data());
        _model_ptrs.front()->run();
        _copy_media_output(destination, scale_destination, begin, valid);
    }
}


void VisionModel::run_unified_audio(std::span<const Eigen::bfloat16> frames,
                                  const MLABufferSlice& destination,
                                  const MLABufferSlice* scale_destination) {
    const auto& audio = _cfg.audio_cfg.value();
    const size_t tokens = frames.size() / audio.samples_per_token;
    const size_t group = _cfg.pipeline_cfg.input_token_group_size;
    if (tokens == 0 || frames.size() % audio.samples_per_token || tokens > audio.max_tokens
        || destination.get_buf_shapes()->front() != tokens) {
        throw std::invalid_argument("Audio frames/placeholder count do not match the token budget");
    }
    std::vector<Eigen::bfloat16> chunk(group * audio.samples_per_token);
    for (size_t begin = 0; begin < tokens; begin += group) {
        const size_t valid = std::min(group, tokens - begin);
        std::fill(chunk.begin(), chunk.end(), Eigen::bfloat16{0.0f});
        std::copy_n(frames.begin() + begin * audio.samples_per_token, valid * audio.samples_per_token, chunk.begin());
        get_buffer("audio_ifm").upload(chunk.data());
        _audio_model->run();
        _copy_media_output(destination, scale_destination, begin, valid);
    }
}

void VisionModel::_copy_media_output(const MLABufferSlice& destination,
                                    const MLABufferSlice* scale_destination, size_t begin, size_t valid) {
    auto copy = [&](MLABuffer& output, const MLABufferSlice& target) {
        // Both buffers use MLA row padding, including the one-channel scale rows.
        const size_t row_bytes = round_up_to(output.get_shape().back() * output.get_elem_size(), MLA_ROW_SIZE);
        output.invalidate_cache(0, valid * row_bytes);
        target.get_buf_ptr()->upload_raw(output.get_virtual_addr(),
            target.get_byte_offset() + begin * row_bytes, valid * row_bytes);
    };
    if (_cfg.pipeline_cfg.quantize_embeddings && scale_destination == nullptr) {
        throw std::invalid_argument("Quantized media output requires an embedding scale destination");
    }
    copy(get_buffer("vision_ofm"), destination);
    if (_cfg.pipeline_cfg.quantize_embeddings) copy(get_buffer("vision_ofm_scale"), *scale_destination);
}

void VisionModel::_validate_model_names() const {
    if (_cfg.vision_model_name.empty()) {
        throw std::runtime_error("vision_model_name must contain at least one model");
    }

    const char* config_name = "devkit/vlm_config.json";
    const auto config_json = nlohmann::json::parse(*_files->open_stream(config_name));
    if (!config_json.contains("vm_cfg")
        || !config_json["vm_cfg"].contains("num_hidden_layers")) {
        throw std::runtime_error(fmt::format(
            "Cannot find vm_cfg.num_hidden_layers in {}", config_name
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
