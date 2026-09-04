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

            auto count_message_images = [](const auto& messages) {
                size_t count = 0;
                for (const auto& message: messages) {
                    const auto& content = message.at("content");
                    if (!content.is_array())
                        continue;
                    for (const auto& item: content) {
                        if (item.value("type", "") == "image")
                            ++count;
                    }
                }
                return count;
            };

            auto image_state_chat = model.create_chat();
            image_state_chat.add_image(image);
            image_state_chat.add_query("Describe the first image.");
            image_state_chat.add_image(image);
            image_state_chat.clear_images();
            if (
                image_state_chat.get_images().size() != 1
                || count_message_images(image_state_chat.get_messages()) != 1
            ) {
                throw std::runtime_error(
                    "Clearing queued images changed a previously submitted image"
                );
            }

            image_state_chat.add_query("Continue without a new image.");
            image_state_chat.add_image(image);
            image_state_chat.add_query("Describe the second image.");
            if (
                image_state_chat.get_images().size() != 2
                || count_message_images(image_state_chat.get_messages()) != 2
            ) {
                throw std::runtime_error("New image was not associated with its own query");
            }
            image_state_chat.clear_history();
            if (!image_state_chat.get_images().empty()) {
                throw std::runtime_error("Clearing history retained image files");
            }
            for (const auto& message: image_state_chat.get_messages()) {
                if (message.value("role", "") != "system") {
                    throw std::runtime_error("Clearing history retained a conversation turn");
                }
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
