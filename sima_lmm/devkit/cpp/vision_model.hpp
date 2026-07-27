#ifndef _SIMA_LLIMA_VISION_MODEL_
#define _SIMA_LLIMA_VISION_MODEL_

#include <vector>

#include <Eigen/Dense>

#include "base_model.hpp"
#include "vlm_config.hpp"

namespace simaai {
namespace llima {

class VisionModel : public BaseModel<VlmConfig> {
public:
  VisionModel(std::filesystem::path model_path);
  virtual ~VisionModel() {
    _finalize();
  };

  std::vector<Eigen::bfloat16> run_model(const std::vector<Eigen::bfloat16>& ifm_tensor);
  void run_model(const std::vector<Eigen::bfloat16>& ifm_tensor,
                 std::map<uint8_t, MLABufferSlice>* ofm_map_ptr);

private:
  virtual void _initialize() override;
  virtual void _finalize() override;

  virtual void _define_buffers() override;
  void _define_models();

  const VisionModelConfig& _vm_cfg;
  const MMConnectionConfig& _mm_cfg;
  std::unique_ptr<MLAModelWithBuffer> _model_ptr;
};

} // namespace llima
} // namespace simaai

#endif
