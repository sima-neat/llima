#pragma once

// Private package assets; model instances keep their bindings and scratch
// separately.
#include "file_provider.hpp"
#include "mla_buffer.hpp"
#include "vlm_config.hpp"
#include "whisper_config.hpp"
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <type_traits>

namespace simaai::llima {

// All package assets share the same disk/deferred-provider identity policy.
template <class Resource>
std::shared_ptr<Resource> acquire_package_resource(
    const std::filesystem::path &path, std::shared_ptr<FileProvider> provider) {
  const auto root = std::filesystem::absolute(path).lexically_normal();
  if (!provider)
    provider = std::make_shared<DiskFileProvider>(path);
  if (std::filesystem::absolute(provider->reserve("elf_files"))
          .lexically_normal() != root / "elf_files")
    throw std::invalid_argument(
        "FileProvider ELF location does not match model_path; "
        "pass the provider's staging root as model_path");
  const auto identity = dynamic_cast<DiskFileProvider *>(provider.get())
                            ? nullptr
                            : provider.get();
  static std::mutex mutex;
  static std::map<std::pair<std::filesystem::path, FileProvider *>,
                  std::weak_ptr<Resource>> cache;
  std::lock_guard lock(mutex);
  auto &weak = cache[{root, identity}];
  if (auto existing = weak.lock())
    return existing;
  auto created = std::make_shared<Resource>(path, std::move(provider));
  weak = created;
  return created;
}

template <class Config> struct ModelResources {
  static std::shared_ptr<ModelResources>
  acquire(const std::filesystem::path &path,
          std::shared_ptr<FileProvider> provider) {
    return acquire_package_resource<ModelResources>(path, std::move(provider));
  }

  ModelResources(const std::filesystem::path &path,
                 std::shared_ptr<FileProvider> provider)
      : elf_dir(path / "elf_files"), devkit_dir(path / "devkit"),
        files(std::move(provider)) {
    constexpr bool language = std::is_same_v<Config, VlmConfig>;
    const char *name =
        language ? "devkit/vlm_config.json" : "devkit/whisper_config.json";
    cfg =
        nlohmann::json::parse(*files->open_stream(name)).template get<Config>();
    auto runtime_logger = spdlog::get("llima");
    logger = runtime_logger
                 ? runtime_logger->clone(language ? "VLM" : "Whisper")
                 : spdlog::default_logger();
    if constexpr (language) {
      // Normalize legacy metadata once, before sharing the configuration.
      if (cfg.lm_cfg.rope_cfg.rope_dimension_count == 0)
        cfg.lm_cfg.rope_cfg.rope_dimension_count = cfg.lm_cfg.attn_cfg.head_dim;
      if (cfg.lm_cfg.layer_types.empty()) {
        for (uint8_t layer = 0; layer < cfg.lm_cfg.num_hidden_layers; ++layer)
          cfg.lm_cfg.layer_types.emplace_back(
              cfg.lm_cfg.attn_cfg.swa_enable &&
                      (layer + 1) % (cfg.lm_cfg.attn_cfg.swa_ratio + 1)
                  ? "sliding_attention"
                  : "full_attention");
      }
    }
  }

  Config cfg;
  std::filesystem::path elf_dir;
  std::filesystem::path devkit_dir;
  std::shared_ptr<FileProvider> files;
  std::shared_ptr<spdlog::logger> logger;
  std::map<std::string, MLABuffer> buffers;
  bool ready = false;
};

} // namespace simaai::llima
