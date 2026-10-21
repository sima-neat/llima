#pragma once

#include "audio_preprocessing.hpp"
#include "base_model.hpp"

namespace simaai::llima {

class AudioModel : public BaseModel<VlmConfig> {
    public:
        AudioModel(std::filesystem::path path, std::shared_ptr<FileProvider> files);
        ~AudioModel() override { _finalize(); }
        void run_model(const AudioInput& audio);
        void copy_output(uint32_t begin, uint32_t rows, const MLABufferSlice& destination,
                         const MLABufferSlice* scales);
    private:
        void _define_buffers() override;
        void _finalize() override;
        std::vector<std::unique_ptr<MLAModelWithBuffer>> _models;
};

}
