#ifndef _SIMA_LLIMA_BASE_MODEL_
#define _SIMA_LLIMA_BASE_MODEL_

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "file_provider.hpp"
#include "mla_buffer.hpp"
#include "mla_model.hpp"
#include "vlm_config.hpp"
#include "whisper_config.hpp"


namespace simaai {
namespace llima {

template <typename T>
class BaseModel {
    protected:
        BaseModel(
            std::filesystem::path model_path,
            std::shared_ptr<FileProvider> file_provider = nullptr
        ) requires std::is_same_v<T, VlmConfig>
          : _elf_dir(model_path / "elf_files"), _devkit_dir(model_path / "devkit"),
            _files(file_provider
                ? std::move(file_provider)
                : std::make_shared<DiskFileProvider>(model_path))
        {
            auto llima_logger = spdlog::get("llima");
            _logger = llima_logger? llima_logger->clone("VLM") : spdlog::default_logger();

            const char* config_file_name = "devkit/vlm_config.json";
            try {
                _cfg = nlohmann::json::parse(*_files->open_stream(config_file_name)).get<VlmConfig>();
            } catch (const std::exception& e) {
                std::cerr << "Failed to load vlm config: " << config_file_name << ", "
                    << e.what() << std::endl;
                throw;
            }
        }

        BaseModel(
            std::filesystem::path model_path,
            std::shared_ptr<FileProvider> file_provider = nullptr
        ) requires std::is_same_v<T, WhisperConfig>
          : _elf_dir(model_path / "elf_files"), _devkit_dir(model_path / "devkit"),
            _files(file_provider
                ? std::move(file_provider)
                : std::make_shared<DiskFileProvider>(model_path))
        {
            auto llima_logger = spdlog::get("llima");
            _logger = llima_logger? llima_logger->clone("Whisper") : spdlog::default_logger();

            std::filesystem::path config_file_name = _devkit_dir / "whisper_config.json";
            try {
                _cfg = nlohmann::json::parse(std::ifstream(config_file_name)).get<WhisperConfig>();
            } catch (const std::exception& e) {
                std::cerr << "Failed to load whisper config: " << config_file_name << ", "
                    << e.what() << std::endl;
                throw;
            }
        }

        virtual ~BaseModel() { if (!_buf_map.empty()) _finalize(); }
        void define_buffer(
            const std::string& name,
            const std::vector<size_t>& shape,
            const std::string& dtype = "bfloat16",
            bool align_last_dim = true
        ) {
            if (_buf_map.contains(name)) {
                _buf_map.erase(name);
            }
            _buf_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(name),
                std::forward_as_tuple(name, shape, dtype, align_last_dim)
            );
        }
        MLABuffer& get_buffer(const std::string& name) { return _buf_map.at(name); }
        bool has_buffer(const std::string& name) { return _buf_map.contains(name); }

        virtual void _define_buffers() {}
        virtual void _initialize() {
            _logger->info("BASE initialize starting ...");
            _define_buffers();
            for (auto& [name, buf]: _buf_map) {
                buf.allocate();
            }
            _logger->info("BASE initialize completed ...");
        }
        virtual void _finalize() {
            _logger->info("BASE finalize starting ...");
            // Free the buffers.
            for (auto& [name, buf]: _buf_map)
                buf.free();
            _buf_map.clear();
            _logger->info("BASE finalize completed");
        }


        T _cfg;
        std::filesystem::path _elf_dir;
        std::filesystem::path _devkit_dir;
        // The seam between LLiMa and the files it loads (Approach B). Every
        // text-path read routes through this. The default is a DiskFileProvider
        // rooted at model_path, so today's behaviour is byte-for-byte the same.
        // A PCIe provider can be injected via the constructor instead.
        std::shared_ptr<FileProvider> _files;
        std::map<std::string, MLABuffer> _buf_map;

        // Logging.
        std::shared_ptr<spdlog::logger> _logger;
};


}
}

#endif
