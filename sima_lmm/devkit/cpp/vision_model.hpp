//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

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
        virtual ~VisionModel() { _finalize(); };

        std::vector<Eigen::bfloat16> run_model(const std::vector<Eigen::bfloat16>& ifm_tensor);
        void run_model(
            const std::vector<Eigen::bfloat16>& ifm_tensor,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        );

    private:
        virtual void _initialize() override;
        virtual void _finalize() override;

        virtual void _define_buffers() override;
        void _define_models();

        const VisionModelConfig& _vm_cfg;
        const MMConnectionConfig& _mm_cfg;
        std::unique_ptr<MLAModelWithBuffer> _model_ptr;
};

}
}

#endif
