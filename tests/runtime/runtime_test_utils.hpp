#pragma once

#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace simaai::llima::test {

inline constexpr const char* kLlimaModelsPathEnv = "LLIMA_MODELS_PATH";
inline constexpr const char* kDefaultLlimaModelsPath = "/media/nvme/llima/models";
inline constexpr const char* kDefaultTextModelName = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4";
inline constexpr const char* kDefaultVlmModelName = "LFM2.5-VL-450M-a16w4";
inline constexpr const char* kDefaultAsrModelName = "whisper-small-a16w8";
inline constexpr const char* kDefaultAssetsPath = "/usr/share/sima_lmm/assets";

inline std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

inline std::string env_value(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : trim(value);
}

inline std::filesystem::path resolve_model_dir(
    const char* model_env,
    const char* default_model_name,
    const char* model_kind,
    const char* expected_config
) {
    std::string model_name = env_value(model_env);
    if (model_name.empty()) {
        model_name = default_model_name;
    }
    if (
        model_name.empty() || model_name.front() == '/'
        || model_name.find('/') != std::string::npos
        || model_name.find("..") != std::string::npos
    ) {
        throw std::runtime_error(
            std::string(model_env)
            + " must be a model directory name under LLIMA_MODELS_PATH: " + model_name
        );
    }

    std::string model_root = env_value(kLlimaModelsPathEnv);
    if (model_root.empty()) {
        model_root = kDefaultLlimaModelsPath;
    }
    const std::filesystem::path model_dir =
        std::filesystem::path(model_root) / model_name;
    if (!std::filesystem::is_regular_file(model_dir / expected_config)) {
        throw std::runtime_error(
            std::string(model_env) + " resolves to a missing or invalid "
            + model_kind + " model directory: " + model_dir.string()
        );
    }
    return model_dir;
}

inline std::filesystem::path resolve_asset(const char* file_name) {
    std::string assets_root = env_value("SIMA_LMM_TEST_ASSETS_PATH");
    if (assets_root.empty()) {
        assets_root = kDefaultAssetsPath;
    }
    const std::filesystem::path asset = std::filesystem::path(assets_root) / file_name;
    if (!std::filesystem::is_regular_file(asset)) {
        throw std::runtime_error("Missing installed runtime test asset: " + asset.string());
    }
    return asset;
}

}  // namespace simaai::llima::test
