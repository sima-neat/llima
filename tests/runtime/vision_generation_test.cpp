#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <spdlog/common.h>

#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_VLM_MODEL";

}  // namespace

int main() {
    bool connected = false;
    try {
        const std::filesystem::path model_dir =
            simaai::llima::test::resolve_model_dir(
                kModelEnv,
                simaai::llima::test::kDefaultVlmModelName,
                "LLiMa VLM",
                "devkit/vlm_config.json"
            );
        const std::filesystem::path image =
            simaai::llima::test::resolve_asset("sjc.jpg");
        std::cout << "LLIMA_VLM model_dir=" << model_dir << '\n';
        std::cout << "LLIMA_VLM image=" << image << '\n';

        simaai::llima::connect(
            {},
            "/tmp/sima_lmm_vision_generation_test.log",
            spdlog::level::info
        );
        connected = true;

        std::string response;
        {
            simaai::llima::VisionLanguageModel model(model_dir);
            if (!model.support_image()) {
                throw std::runtime_error("VLM model does not support images");
            }

            auto chat = model.create_chat();
            chat.add_image(image);
            chat.add_query("Describe this image in a short phrase.");
            auto result = model.run_model(chat, 48);
            if (!result.has_value()) {
                throw std::runtime_error("Vision generation was interrupted");
            }
            response = simaai::llima::test::trim(std::move(*result));
            if (response.empty()) {
                throw std::runtime_error("Vision generation returned empty text");
            }
        }

        simaai::llima::disconnect();
        connected = false;
        std::cout << "LLIMA_VLM text=" << response << '\n';
    } catch (const std::exception& error) {
        if (connected) {
            try {
                simaai::llima::disconnect();
            } catch (...) {
            }
        }
        std::cerr << "Vision generation test failed: " << error.what() << '\n';
        return 1;
    }

    std::cout << "Vision generation test passed\n";
    return 0;
}
