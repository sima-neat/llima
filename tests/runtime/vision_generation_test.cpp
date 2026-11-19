#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <spdlog/common.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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
            cv::Mat rgb;
            cv::cvtColor(cv::imread(image.string()), rgb, cv::COLOR_BGR2RGB);
            std::vector<cv::Mat> images{rgb};
            simaai::llima::GenerationOptions options;
            options.max_new_tokens = 48;
            options.images = images;
            model.reset_session(0);
            const auto from_memory = model.run_model(chat, options);
            if (from_memory != result) throw std::runtime_error("In-memory image differs from file input");
            auto encoded = model.encode_images(images);
            options.images = {};
            options.encoded_images = encoded;
            model.reset_session(0);
            if (model.run_model(chat, options) != result)
                throw std::runtime_error("Reusable encoded image differs from fresh encoding");

            const auto alternate = model.create_session();
            images[0] = cv::Mat(rgb.size(), CV_8UC3, cv::Scalar(255, 0, 0));
            options.encoded_images = {};
            options.images = images;
            const auto reference_red = model.run_model(alternate, chat, options);
            if (model.run_model(0, chat, options) != reference_red)
                throw std::runtime_error("Same placeholder tokens reused another image's cache");
            options.images = {};
            options.encoded_images = encoded;
            if (model.run_model(alternate, chat, options) != result)
                throw std::runtime_error("Alternating images/sessions corrupted encoded-image reuse");
            model.release_session(alternate);

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
