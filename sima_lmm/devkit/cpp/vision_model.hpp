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
        // Reads the model files from disk (the signature from before the
        // FileProvider seam, kept so older builds still link).
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

    private:
        virtual void _initialize() override;
        virtual void _finalize() override;

        virtual void _define_buffers() override;
        void _define_models();
        void _validate_model_names() const;

        const VisionModelConfig& _vm_cfg;
        const MMConnectionConfig& _mm_cfg;
        std::vector<std::unique_ptr<MLAModelWithBuffer>> _model_ptrs;
};

}
}

#endif
