#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>

#include "cli.hpp"
#include "reasoning_parser.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {


const std::string CLI::_COMMANDS = R"(
add image <fn>       : add an image.
set system <prompt>  : set system prompt for the active cache session.
clear system         : clear system prompt, chat history and images for the active session.
clear history        : clear prompts, responses and images for the active session; keep system prompt.
print history        : print chat history for the active session.
use cache <id>       : select or create a named cache session.
use default cache    : select the legacy default cache session.
remove cache <id>    : remove a named cache session.
clear caches         : remove every KV cache and reset all session histories.
print caches         : print active and allocated cache information.
set audio <fn>       : set the audio file to be transcribed as query.
set language <lang>  : set transcription language; omit <lang> or use auto to detect it.
set lora             : set the model to use LoRA weights from a npy_files folder.
unset lora           : revert LoRA model to baseline model.
enable-thinking      : enable thinking mode and clear active chat history.
disable-thinking     : disable thinking mode and clear active chat history.
quit                 : quit.
help                 : print this page.
)";


CLI::CLI(
    std::filesystem::path vlm_model_path,
    std::optional<std::filesystem::path> whisper_model_path,
    std::optional<std::filesystem::path> draft_model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template
) : CLI(
        std::move(vlm_model_path),
        std::move(whisper_model_path),
        std::move(draft_model_path),
        std::move(system_prompt),
        std::move(chat_template),
        1
    ) {}


CLI::CLI(
    std::filesystem::path vlm_model_path,
    std::optional<std::filesystem::path> whisper_model_path,
    std::optional<std::filesystem::path> draft_model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    size_t max_kv_cache_slots
) : _vision_language_model_ptr(
        std::make_unique<VisionLanguageModel>(
            vlm_model_path, system_prompt, chat_template, max_kv_cache_slots
        )
    ),
    _max_kv_cache_slots(max_kv_cache_slots)
{
    if (_singleton_ptr)
        throw std::runtime_error("Only one CLI instance can be created");
    _singleton_ptr = this;

    if (whisper_model_path.has_value()) {
        _whisper_model_ptr = std::make_unique<WhisperModel>(whisper_model_path.value());
    }

    if (draft_model_path.has_value()) {
        _vision_language_draft_model_ptr = std::make_unique<VisionLanguageModel>(
            draft_model_path.value(), system_prompt, chat_template, max_kv_cache_slots
        );
        // Hand the draft to the target so VLM::run_model dispatches to
        // speculative decoding automatically.
        _vision_language_model_ptr->set_draft_vlm(_vision_language_draft_model_ptr.get());
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
    const Chat initial_chat = _vision_language_model_ptr->create_chat();
    std::map<std::optional<std::string>, std::unique_ptr<Chat>> chats;
    std::optional<std::string> active_cache_id;
    chats.emplace(std::nullopt, std::make_unique<Chat>(initial_chat));

    auto active_chat = [&]() -> Chat& {
        return *chats.at(active_cache_id);
    };
    auto clear_all_histories = [&]() {
        for (auto& [cache_id, chat_ptr] : chats) {
            (void)cache_id;
            chat_ptr->clear_history();
        }
    };
    auto reset_sessions = [&]() {
        chats.clear();
        chats.emplace(std::nullopt, std::make_unique<Chat>(initial_chat));
        active_cache_id.reset();
    };
    auto cache_label = [](const std::optional<std::string>& cache_id) {
        return cache_id.value_or("default");
    };

    const bool highlight_draft = []() {
        const char* value = std::getenv("SIMA_LLIMA_ENABLE_DRAFT_HIGHLIGHT");
        return value != nullptr && (
            std::strcmp(value, "true") == 0 || std::strcmp(value, "1") == 0
        );
    }();
    ReadlineSupport readline_support;
    while (true) {
        auto maybe_command = ReadlineSupport::read_line(">>> ");
        if (!maybe_command) break;
        std::string command = std::move(*maybe_command);
        readline_support.add_to_history(command);

        Chat& chat = active_chat();
        if (command == "quit") {
            break;
        } else if (command == "help") {
            std::cout << _COMMANDS << std::endl;
            continue;
        } else if (command == "use default cache") {
            active_cache_id.reset();
            std::cout << "Using default cache session." << std::endl;
            continue;
        } else if (command.starts_with("use cache ")) {
            constexpr auto pos = std::string("use cache ").length();
            auto cache_id = trim(command.substr(pos));
            if (cache_id.empty()) {
                std::cout << "Cache ID must not be empty." << std::endl;
                continue;
            }
            chats.try_emplace(cache_id, std::make_unique<Chat>(initial_chat));
            active_cache_id = std::move(cache_id);
            std::cout << "Using cache session '" << *active_cache_id << "'." << std::endl;
            continue;
        } else if (command.starts_with("remove cache ")) {
            constexpr auto pos = std::string("remove cache ").length();
            auto cache_id = trim(command.substr(pos));
            if (cache_id.empty()) {
                std::cout << "Cache ID must not be empty." << std::endl;
                continue;
            }
            const bool removed_cache = _vision_language_model_ptr->remove_kv_cache(cache_id);
            const bool removed_chat = chats.erase(cache_id) != 0;
            if (active_cache_id == cache_id) active_cache_id.reset();
            if (removed_cache || removed_chat) {
                std::cout << "Removed cache session '" << cache_id << "'." << std::endl;
            } else {
                std::cout << "Cache session not found: '" << cache_id << "'." << std::endl;
            }
            continue;
        } else if (command == "clear caches") {
            _vision_language_model_ptr->clear_kv_caches();
            reset_sessions();
            std::cout << "Cleared all KV caches and reset session histories." << std::endl;
            continue;
        } else if (command == "print caches") {
            std::cout << "Active cache: " << cache_label(active_cache_id) << std::endl
                      << "Allocated caches: "
                      << _vision_language_model_ptr->kv_cache_count() << "/"
                      << _max_kv_cache_slots << std::endl
                      << "Device bytes per cache slot: "
                      << _vision_language_model_ptr->kv_cache_bytes_per_slot() << std::endl
                      << "Known sessions:";
            for (const auto& [cache_id, chat_ptr] : chats) {
                (void)chat_ptr;
                std::cout << " " << cache_label(cache_id);
                if (cache_id == active_cache_id) std::cout << "*";
            }
            std::cout << std::endl;
            continue;
        } else if (command.starts_with("set lora ")) {
            constexpr auto pos = std::string("set lora ").length();
            std::string lora_name = command.substr(pos);
            try {
                _vision_language_model_ptr->set_reloc(lora_name);
                clear_all_histories();
                std::cout << "Set LoRA and cleared all session histories" << std::endl;
            } catch (const std::exception& ex) {
                std::cout << "Failed to set LoRA: " << ex.what() << std::endl;
            }
            continue;
        } else if (command == "unset lora") {
            _vision_language_model_ptr->unset_reloc();
            clear_all_histories();
            std::cout << "Un-set LoRA and cleared all session histories" << std::endl;
            continue;
        } else if (command == "enable-thinking") {
            if (reasoning_format_for_model(_vision_language_model_ptr->model_type())
                == ReasoningFormat::None) {
                std::cout << "Thinking is not supported for this model." << std::endl;
                continue;
            }
            chat.set_enable_thinking(true);
            chat.clear_history();
            std::cout << "Enabled thinking and cleared active chat history." << std::endl;
            continue;
        } else if (command == "disable-thinking") {
            chat.set_enable_thinking(false);
            chat.clear_history();
            std::cout << "Disabled thinking and cleared active chat history." << std::endl;
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
        } else if (command == "clear image") {
            std::cout << "The clear image command is not supported; use clear history instead."
                      << std::endl;
            continue;
        } else if (command.starts_with("set system ")) {
            constexpr auto pos = std::string("set system ").length();
            chat.set_system_prompt(command.substr(pos));
            std::cout << "Set system message and cleared active chat history." << std::endl;
            continue;
        } else if (command == "clear system") {
            chat.clear_system_prompt();
            std::cout << "Cleared system message and active chat history." << std::endl;
            continue;
        } else if (command == "clear history") {
            chat.clear_history();
            std::cout << "Cleared active chat history." << std::endl;
            continue;
        } else if (command == "print history") {
            chat.print_history();
            continue;
        }

        auto chat_before_query = std::make_unique<Chat>(chat);
        if (command.starts_with("set audio ")) {
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
                [](const std::string& text, bool stream_end, bool) {
                    std::cout << text << std::flush;
                    if (stream_end)
                        std::cout << std::endl;
                }
            );
            struct WhisperTextCallbackGuard {
                WhisperModel* model;
                ~WhisperTextCallbackGuard() {
                    model->set_text_callback([](const std::string&, bool, bool) {});
                }
            } callback_guard{_whisper_model_ptr.get()};
            auto result = _whisper_model_ptr->run_model(audio_file_name, language);
            chat.add_query(result.text);
        } else {
            std::cout << "Query: " << command << " [cache="
                      << cache_label(active_cache_id) << "]" << std::endl;
            chat.add_query(command);
        }
        ReasoningStreamParser reasoning_parser(
            reasoning_format_for_model(_vision_language_model_ptr->model_type()),
            chat.get_enable_thinking()
        );
        std::string final_response;
        bool saw_reasoning = false;
        bool saw_content = false;
        std::optional<uint32_t> cached_prompt_tokens;
        std::optional<bool> cache_created;

        const auto print_text = [&](const std::string& text, bool from_draft) {
            if (from_draft && highlight_draft && !text.empty()) {
                std::cout << "\033[32m" << text << "\033[0m";
            } else {
                std::cout << text;
            }
            std::cout << std::flush;
        };

        _vision_language_model_ptr->set_info_callback(
            [&](const std::string& metric, double value) {
                if (metric == "cached_prompt_tokens") {
                    cached_prompt_tokens = static_cast<uint32_t>(value);
                } else if (metric == "cache_created") {
                    cache_created = value != 0.0;
                }
            }
        );
        _vision_language_model_ptr->set_text_callback(
            [&](const std::string& text, bool stream_end, bool from_draft) {
                for (auto& event : reasoning_parser.add(text, stream_end, from_draft)) {
                    if (event.reasoning) {
                        if (!saw_reasoning) {
                            std::cout << "Thinking:\n";
                            saw_reasoning = true;
                        }
                        print_text(event.text, event.from_draft);
                        continue;
                    }

                    if (!saw_content) {
                        std::cout << (saw_reasoning ? "\nAnswer:\n" : "Assistant: ");
                        saw_content = true;
                    }
                    final_response += event.text;
                    print_text(event.text, event.from_draft);
                }
                if (stream_end) {
                    if (!saw_reasoning && !saw_content) std::cout << "Assistant: ";
                    std::cout << std::endl;
                }
            }
        );
        struct CallbackGuard {
            VisionLanguageModel* model;
            ~CallbackGuard() {
                model->set_info_callback([](const std::string&, double) {});
                model->set_text_callback([](const std::string&, bool, bool) {});
            }
        } callback_guard{_vision_language_model_ptr.get()};

        std::optional<std::string> response;
        try {
            // run_model dispatches to speculative decoding internally when a
            // draft VLM was registered at construction time.
            response = _vision_language_model_ptr->run_model(
                chat, std::nullopt, active_cache_id
            );
        } catch (const KVCacheCapacityError& ex) {
            chats[active_cache_id] = std::move(chat_before_query);
            std::cout << "KV-cache capacity reached: " << ex.what() << std::endl
                      << "Remove a cache session or restart with a larger "
                         "--max-kv-cache-slots value."
                      << std::endl;
            continue;
        }

        if (cached_prompt_tokens.has_value() && cache_created.has_value()) {
            std::cout << "Cache [" << cache_label(active_cache_id) << "]: reused "
                      << *cached_prompt_tokens << " prompt token(s), "
                      << (*cache_created ? "allocated a new slot" : "reused its slot")
                      << "." << std::endl;
        }
        if (response.has_value()) {
            auto answer = trim(std::move(final_response));
            if (answer.empty()) {
                chat.clear_history();
                std::cout << "No final answer generated. Cleared active chat history."
                          << std::endl;
            } else {
                chat.add_response(std::move(answer));
            }
        } else {
            chat.clear_history();
            std::cout << std::endl
                << "User interrupt received. Cleared active chat history." << std::endl
                << "Type quit to quit." << std::endl;
        }
    }
}

void CLI::stop() {
    _vision_language_model_ptr->stop_model();
}

}
}
