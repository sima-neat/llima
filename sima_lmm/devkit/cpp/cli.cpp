#include <iostream>
#include <optional>

#include <fmt/format.h>

#include "cli.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {


const std::string CLI::_COMMANDS = R"(
add image <fn>     : add an image.
clear image        : clear all the images.
set system <prompt>: set system prompt.
clear system       : clear system prompt, chat history and images.
clear history      : clear chat history and images.
print history      : print chat history.
set audio <fn>     : set the audio file to be transcribed as query.
set language <lang>: set transcription language; omit <lang> or use auto to detect it.
set lora           : set the model to use LoRA weights from a npy_files folder.
unset lora         : revert LoRA model to baseline model.
quit               : quit.
help               : print this page.
)";


CLI::CLI(
    std::filesystem::path vlm_model_path,
    std::optional<std::filesystem::path> whisper_model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    bool do_parallel_load
) : _vision_language_model_ptr(
        std::make_unique<VisionLanguageModel>(
            vlm_model_path, system_prompt, chat_template, do_parallel_load
        )
    )
{
    if (_singleton_ptr)
        throw std::runtime_error("Only one CLI instance can be created");
    _singleton_ptr = this;

    if (whisper_model_path.has_value()) {
        _whisper_model_ptr = std::make_unique<WhisperModel>(
            whisper_model_path.value(), do_parallel_load
        );
    }

    auto llima_logger = spdlog::get("llima");
    _logger = llima_logger? llima_logger->clone("CLI") : spdlog::default_logger();

    struct sigaction new_sigint_action;
    new_sigint_action.sa_handler = [](int sig) {
        assert(sig == SIGINT);
        if (_singleton_ptr) {
            _singleton_ptr->stop();
        }
    };
    sigemptyset(&new_sigint_action.sa_mask);
    new_sigint_action.sa_flags = 0;
    sigaction(SIGINT, &new_sigint_action, &_old_sigint_action);
}


CLI::~CLI() {
    stop();
    sigaction(SIGINT, &_old_sigint_action, nullptr);
}


void CLI::run() {
    std::string language = "en";
    auto chat = _vision_language_model_ptr->create_chat();
    std::string command;
    ReadlineSupport readline_support;
    std::optional<double> last_ttft;
    std::optional<double> last_tps;
    _vision_language_model_ptr->set_info_callback(
        [&](const std::string& metric_type, double metric_value) {
            if (metric_type == "ttft") {
                last_ttft = metric_value;
            } else if (metric_type == "tps") {
                last_tps = metric_value;
            }
        }
    );
    while (true) {
        auto maybe_command = ReadlineSupport::read_line(">>> ");
        if (!maybe_command) break;
        std::string command = std::move(*maybe_command);
        readline_support.add_to_history(command);
        
        if (command == "quit") {
            break;
        } else if (command == "help") {
            std::cout << _COMMANDS << std::endl;
            continue;
        } else if (command.starts_with("set lora ")) {
            constexpr auto pos = std::string("set lora ").length();
            std::string lora_name = command.substr(pos);
            try {
                _vision_language_model_ptr->set_reloc(lora_name);
                chat.clear_history();
                std::cout << "Set LoRA and cleared chat history" << std::endl;
            } catch (const std::exception& ex) {
                std::cout << "Failed to set LoRA: " << ex.what() << std::endl;
            }
            continue;
        } else if (command == "unset lora") {
            _vision_language_model_ptr->unset_reloc();
            chat.clear_history();
            std::cout << "Un-set LoRA and cleared chat history" << std::endl;
            continue;
        } else if (command == "set language") {
            language.clear();
            continue;
        } else if (command.starts_with("set language ")) {
            constexpr auto pos = std::string("set language ").length();
            language = command.substr(pos);
            continue;
        } else if (command.starts_with("add image ")) {
            constexpr auto pos = std::string("add image ").length();
            std::filesystem::path image_file_name = command.substr(pos);
            if (!_vision_language_model_ptr->support_image()) {
                std::cout << "Inference with image is not supported." << std::endl;
            } else if (!std::filesystem::is_regular_file(image_file_name)) {
                std::cout << "Image file not found: " << image_file_name << std::endl;
            } else {
                chat.add_image(image_file_name);
            }
            continue;
        } else if (command.starts_with("set system ")) {
            constexpr auto pos = std::string("set system ").length();
            chat.set_system_prompt(command.substr(pos));
            std::cout << "Set system message and cleared chat history." << std::endl;
            continue;
        } else if (command == "clear system") {
            chat.clear_system_prompt();
            std::cout << "Cleared system message and chat history." << std::endl;
            continue;
        } else if (command == "clear history") {
            chat.clear_history();
            std::cout << "Cleared chat history." << std::endl;
            continue;
        } else if (command == "print history") {
            chat.print_history();
            continue;
        } else if (command.starts_with("set audio ")) {
            constexpr auto pos = std::string("set audio ").length();
            std::filesystem::path audio_file_name = command.substr(pos);
            if (!_whisper_model_ptr) {
                std::cout << "Audio model not available." << std::endl;
                continue;
            } else if (!std::filesystem::is_regular_file(audio_file_name)) {
                std::cout << "Audio file not found: " << audio_file_name << std::endl;
                continue;
            }
            std::cout << "Transcribed query: " << std::flush;
            _whisper_model_ptr->set_text_callback(
                [](const std::string& text, bool stream_end) {
                    std::cout << text << std::flush;
                    if (stream_end)
                        std::cout << std::endl;
                }
            );
            struct WhisperTextCallbackGuard {
                WhisperModel* model;
                ~WhisperTextCallbackGuard() {
                    model->set_text_callback([](const std::string&, bool) {});
                }
            } callback_guard{_whisper_model_ptr.get()};
            auto result = _whisper_model_ptr->run_model(audio_file_name, language);
            chat.add_query(result.text);
        } else {
            std::cout << "Query: " << command << std::endl;
            chat.add_query(command);
        }
        std::cout << "Assistant: " << std::flush;

        last_ttft.reset();
        last_tps.reset();
        auto response = _vision_language_model_ptr->run_model(chat);
        if (response.has_value()) {
            chat.add_response(trim(std::move(response.value())));
            if (last_ttft.has_value()) {
                std::cout << "TTFT: " << fmt::format("{:.2f}s", *last_ttft) << std::endl;
            }
            if (last_tps.has_value()) {
                std::cout << "TPS: " << fmt::format("{:.2f}", *last_tps) << std::endl;
            }
        } else {
            chat.clear_history();
            std::cout << std::endl
                << "User interrupt received. Cleared chat history." << std::endl
                << "Type quit to quit." << std::endl;
        }
    }
}


void CLI::stop() {
    _logger->info("User interrupt received. Stopping the model.");
    _vision_language_model_ptr->stop_model();
}

}
}
