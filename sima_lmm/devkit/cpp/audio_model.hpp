#pragma once

#include "audio_preprocessing.hpp"
#include "base_model.hpp"

namespace simaai::llima {

class AudioModel : public BaseModel<VlmConfig> {
    public:
        AudioModel(std::filesystem::path path, std::shared_ptr<FileProvider> files);
        ~AudioModel() override { _finalize(); }
        void run_model(const AudioInput& audio, std::map<uint8_t, MLABufferSlice>* ofm_map_ptr);
    private:
        void _define_buffers() override;
        void _finalize() override;
        std::vector<std::unique_ptr<MLAModelWithBuffer>> _models;
};

}
