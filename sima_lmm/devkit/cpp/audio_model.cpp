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
            auto& output = get_buffer(last ? "audio_output" : fmt::format("audio_hidden_{}", layer % 2));
            std::vector<MLABufferSlice> outputs{&output};
            if (last && _cfg.pipeline_cfg.quantize_embeddings) outputs.emplace_back(&get_buffer("audio_scales"));
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
    define_buffer("audio_features", {audio.num_frames(), audio.num_mel_bins});
    define_buffer("audio_mask", {audio.num_frames(), 1});
    define_buffer("audio_hidden_0", {audio.padded_tokens(), audio.hidden_size});
    define_buffer("audio_hidden_1", {audio.padded_tokens(), audio.hidden_size});
    define_buffer("audio_output", {audio.max_tokens, _cfg.lm_cfg.hidden_size},
                  _cfg.pipeline_cfg.quantize_embeddings ? "int8" : "bfloat16");
    if (_cfg.pipeline_cfg.quantize_embeddings) define_buffer("audio_scales", {audio.max_tokens, 1});
}

void AudioModel::_finalize() {
    for (const auto& name : _cfg.audio_model_name) MLAModelWithBuffer::free_all_models(_elf_dir / name);
    BaseModel::_finalize();
}

void AudioModel::run_model(const AudioInput& audio) {
    const auto& cfg = *_cfg.audio_cfg;
    if (audio.features.size() != static_cast<size_t>(cfg.num_frames()) * cfg.num_mel_bins
        || audio.mask.size() != cfg.num_frames() || !audio.num_tokens || audio.num_tokens > cfg.max_tokens)
        throw std::invalid_argument("Audio features do not match the compiled encoder");
    get_buffer("audio_features").upload(audio.features.data());
    get_buffer("audio_mask").upload(audio.mask.data());
    for (auto& model : _models) model->add_to_queue();
    MLAModelWithBuffer::run_queue();
}

void AudioModel::copy_output(uint32_t begin, uint32_t rows, const MLABufferSlice& destination,
                             const MLABufferSlice* scales) {
    if (!rows || begin >= _cfg.audio_cfg->max_tokens || rows > _cfg.audio_cfg->max_tokens - begin
        || (_cfg.pipeline_cfg.quantize_embeddings != (scales != nullptr)))
        throw std::invalid_argument("Invalid audio output span or missing embedding scales");
    auto copy = [&](MLABuffer& source, const MLABufferSlice& target) {
        const size_t stride = source.get_buf_len(std::vector<uint32_t>{1, static_cast<uint32_t>(source.get_shape().back())});
        source.invalidate_cache(begin * stride, rows * stride);
        target.get_buf_ptr()->upload_raw(static_cast<const uint8_t*>(source.get_virtual_addr()) + begin * stride,
                                        target.get_byte_offset(), rows * stride);
    };
    copy(get_buffer("audio_output"), destination);
    if (scales) copy(get_buffer("audio_scales"), *scales);
}

}
