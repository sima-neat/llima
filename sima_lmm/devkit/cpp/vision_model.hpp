#ifndef _SIMA_LLIMA_VISION_MODEL_
#define _SIMA_LLIMA_VISION_MODEL_

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "base_model.hpp"
#include "file_provider.hpp"
#include "vlm_config.hpp"


namespace simaai {
namespace llima {

class VisionModel : public BaseModel<VlmConfig> {
    public:
        // Existing disk-loading overload; preserves source compatibility.
        VisionModel(std::filesystem::path model_path);
        // Reads the model files through file_provider (nullptr = from disk).
        VisionModel(
            std::filesystem::path model_path,
            std::shared_ptr<FileProvider> file_provider
        );
        virtual ~VisionModel() { _finalize(); };

        void run_model(
            const std::vector<Eigen::bfloat16>& ifm_tensor,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        );
        void run_unified_model(const std::vector<Eigen::bfloat16>& patches,
                               const std::vector<uint16_t>& positions,
                               const MLABufferSlice& destination);
        void run_unified_audio(const std::vector<Eigen::bfloat16>& frames,
                               const MLABufferSlice& destination);

    private:
        virtual void _initialize() override;
        virtual void _finalize() override;

        virtual void _define_buffers() override;
        void _define_models();
        void _validate_model_names() const;
        void _copy_media_output(const MLABufferSlice& destination, size_t begin, size_t valid);

        const VisionModelConfig& _vm_cfg;
        const MMConnectionConfig& _mm_cfg;
        std::vector<std::unique_ptr<MLAModelWithBuffer>> _model_ptrs;
        std::vector<Eigen::bfloat16> _unified_position_table;
        std::unique_ptr<MLAModelWithBuffer> _audio_model;
};

}
}

#endif
