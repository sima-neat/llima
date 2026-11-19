#ifndef _SIMA_LLIMA_VISION_MODEL_
#define _SIMA_LLIMA_VISION_MODEL_

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <spdlog/logger.h>
#include "file_provider.hpp"
#include "mla_buffer.hpp"
#include "vlm_config.hpp"


namespace simaai {
namespace llima {

template<class Config> struct ModelResources;
struct VisionWorkspace;

class VisionModel {
    public:
        // Existing disk-loading overload; preserves source compatibility.
        VisionModel(std::filesystem::path model_path);
        // Reads the model files through file_provider (nullptr = from disk).
        VisionModel(
            std::filesystem::path model_path,
            std::shared_ptr<FileProvider> file_provider
        );
        virtual ~VisionModel();

        void run_model(
            const std::vector<Eigen::bfloat16>& ifm_tensor,
            std::map<uint8_t, MLABufferSlice>* ofm_map_ptr
        );

    private:
        friend class VisionLanguageModel;
        std::shared_ptr<ModelResources<VlmConfig>> _resources;
        const VlmConfig& _cfg;
        const std::filesystem::path& _elf_dir;
        const std::filesystem::path& _devkit_dir;
        const std::shared_ptr<FileProvider>& _files;
        std::shared_ptr<spdlog::logger> _logger;
        std::unique_ptr<VisionWorkspace> _workspace;
        void define_buffer(const std::string& name, const std::vector<size_t>& shape,
                           const std::string& dtype = "bfloat16", bool align_last_dim = true);
        MLABuffer& get_buffer(const std::string& name);
        void _initialize();
        void _finalize();

        void _define_buffers();
        void _define_models();
        void _validate_model_names() const;

        const VisionModelConfig& _vm_cfg;
        const MMConnectionConfig& _mm_cfg;
};

}
}

#endif
