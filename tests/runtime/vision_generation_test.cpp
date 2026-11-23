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
            auto result = model.generate(chat, {.max_new_tokens = 48});
            if (result.finish == simaai::llima::ChatResult::Finish::Cancelled) {
                throw std::runtime_error("Vision generation was interrupted");
            }
            cv::Mat rgb;
            cv::cvtColor(cv::imread(image.string()), rgb, cv::COLOR_BGR2RGB);
            std::vector<cv::Mat> images{rgb};
            simaai::llima::GenerationOptions options;
            options.max_new_tokens = 48;
            options.images = images;
            model.reset_session(simaai::llima::kDefaultSession);
            const auto from_memory = model.generate(chat, options).token_ids;
            if (from_memory != result.token_ids) throw std::runtime_error("In-memory image differs from file input");
            auto encoded = model.encode_images(images);
            options.images = {};
            options.encoded_images = encoded;
            model.reset_session(simaai::llima::kDefaultSession);
            if (model.generate(chat, options).token_ids != result.token_ids)
                throw std::runtime_error("Reusable encoded image differs from fresh encoding");

            auto pair_chat = model.create_chat();
            pair_chat.add_image(image);
            pair_chat.add_image(image);
            pair_chat.add_query("Describe these images in a short phrase.");
            cv::Mat red(rgb.size(), CV_8UC3, cv::Scalar(255, 0, 0));
            const std::vector<cv::Mat> originals{rgb, red};
            const auto workspace_bytes = model.workspace_memory_bytes();
            const auto pair = model.encode_images(originals);
            if (pair.size() != originals.size() || model.workspace_memory_bytes() != workspace_bytes)
                throw std::runtime_error("Encoding images changed the language workspace");
            simaai::llima::GenerationOptions pair_options;
            pair_options.max_new_tokens = 8;
            pair_options.encoded_images = pair;
            const auto pair_reference = model.generate(pair_chat, pair_options).token_ids;
            model.reset_session(simaai::llima::kDefaultSession);
            pair_options.images = images;
            pair_options.encoded_images = std::span(pair).last(1);
            if (pair_reference.empty() || model.generate(pair_chat, pair_options).token_ids != pair_reference)
                throw std::runtime_error("Mixed fresh and encoded image rows differ from the encoded reference");

            const auto alternate = model.create_session();
            images[0] = red;
            options.encoded_images = {};
            options.images = images;
            options.session = alternate;
            const auto reference_red = model.generate(chat, options).token_ids;
            options.session = simaai::llima::kDefaultSession;
            if (reference_red.empty() || model.generate(chat, options).token_ids != reference_red)
                throw std::runtime_error("Same placeholder tokens reused another image's cache");
            options.images = {};
            options.encoded_images = encoded;
            options.session = alternate;
            if (model.generate(chat, options).token_ids != result.token_ids)
                throw std::runtime_error("Alternating images/sessions corrupted encoded-image reuse");
            model.release_session(alternate);

            response = simaai::llima::test::trim(std::move(result.text));
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
