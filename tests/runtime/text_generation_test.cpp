#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <spdlog/common.h>

#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";
constexpr const char* kExpectedText = "The capital of Germany is Berlin.";

} // namespace

int main() {
  bool connected = false;
  try {
    const std::filesystem::path model_dir = simaai::llima::test::resolve_model_dir(
        kModelEnv, simaai::llima::test::kDefaultTextModelName, "LLiMa text",
        "devkit/vlm_config.json");
    std::cout << "LLIMA_LLM model_dir=" << model_dir << '\n';

    simaai::llima::connect({}, "/tmp/sima_lmm_text_generation_test.log", spdlog::level::info);
    connected = true;

    std::string response;
    {
      simaai::llima::VisionLanguageModel model(model_dir);
      if (model.support_image()) {
        throw std::runtime_error("Text-only model unexpectedly supports images");
      }

      auto chat = model.create_chat();
      chat.set_system_prompt("You are concise.");
      chat.add_query("What is the capital of Germany?");
      auto result = model.run_model(chat, 24);
      if (!result.has_value()) {
        throw std::runtime_error("Text generation was interrupted");
      }
      response = simaai::llima::test::trim(std::move(*result));
    }

    simaai::llima::disconnect();
    connected = false;

    std::cout << "LLIMA_LLM text=" << response << '\n';
    if (response != kExpectedText) {
      throw std::runtime_error("Unexpected generated text: " + response);
    }
  } catch (const std::exception& error) {
    if (connected) {
      try {
        simaai::llima::disconnect();
      } catch (...) {
      }
    }
    std::cerr << "Text generation test failed: " << error.what() << '\n';
    return 1;
  }

  std::cout << "Text generation test passed\n";
  return 0;
}
