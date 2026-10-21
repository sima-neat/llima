#include "audio_model.hpp"

namespace simaai::llima {

AudioModel::AudioModel(std::filesystem::path path, std::shared_ptr<FileProvider> files)
    : BaseModel(std::move(path), std::move(files)) {
    if (!_cfg.support_audio() || _cfg.audio_model_name.size() != _cfg.audio_cfg->num_hidden_layers)
        throw std::invalid_argument("Incomplete Gemma audio encoder configuration");
    BaseModel::_initialize();
    try {
        for (size_t layer = 0; layer < _cfg.audio_model_name.size(); ++layer) {
            const bool last = layer + 1 == _cfg.audio_model_name.size();
            auto& input = get_buffer(layer == 0 ? "audio_features" : fmt::format("audio_hidden_{}", (layer - 1) % 2));
            // The final layer writes directly into the caller's prompt buffers.
            std::vector<MLABufferSlice> outputs(last && _cfg.pipeline_cfg.quantize_embeddings ? 2 : 1);
            if (!last) outputs[0] = MLABufferSlice(&get_buffer(fmt::format("audio_hidden_{}", layer % 2)));
            _models.emplace_back(std::make_unique<MLAModelWithBuffer>(
                _elf_dir / (_cfg.audio_model_name[layer] + "_stage1_mla.elf"),
                std::vector<MLABufferSlice>{&input, &get_buffer("audio_mask")}, std::move(outputs)));
            MLAModelWithBuffer::load_all_models(_elf_dir / _cfg.audio_model_name[layer], _files.get());
        }
    } catch (...) {
        _finalize();
        throw;
    }
}

void AudioModel::_define_buffers() {
    const auto& audio = *_cfg.audio_cfg;
    const auto frontend = nlohmann::json::parse(*_files->open_stream("devkit/audio_preprocessor_config.json"));
    const auto mel_bins = frontend.value("num_mel_bins", 128U);
    define_buffer("audio_features", {audio.num_frames(), mel_bins});
    define_buffer("audio_mask", {audio.num_frames(), 1});
    define_buffer("audio_hidden_0", {audio.padded_tokens(), audio.hidden_size});
    define_buffer("audio_hidden_1", {audio.padded_tokens(), audio.hidden_size});
}

void AudioModel::_finalize() {
    for (const auto& name : _cfg.audio_model_name) MLAModelWithBuffer::free_all_models(_elf_dir / name);
    BaseModel::_finalize();
}

void AudioModel::run_model(const AudioInput& audio, std::map<uint8_t, MLABufferSlice>* ofm_map_ptr) {
    const auto& cfg = *_cfg.audio_cfg;
    if (audio.features.size() != get_buffer("audio_features").get_num_elems()
        || audio.mask.size() != cfg.num_frames() || !audio.num_tokens || audio.num_tokens > cfg.max_tokens)
        throw std::invalid_argument("Audio features do not match the compiled encoder");
    get_buffer("audio_features").upload(audio.features.data());
    get_buffer("audio_mask").upload(audio.mask.data());
    for (size_t layer = 0; layer < _models.size(); ++layer)
        _models[layer]->add_to_queue(nullptr, layer + 1 == _models.size() ? ofm_map_ptr : nullptr);
    MLAModelWithBuffer::run_queue();
}

}
