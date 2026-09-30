// VlmGenerator: keeps one Chat across prompts (like the devkit CLI),
// runs LLiMa, and waits for the streamer to finish before it returns. So
// genai.final is always sent after the last token.
#include "vlm_generator.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>

#include "chat.hpp"
#include "pcie_file_provider.hpp"
#include "reasoning_parser.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {

namespace {
constexpr const char* kNoThinking = "Thinking is not supported for this model.";
}  // namespace

VlmGenerator::VlmGenerator(VisionLanguageModel& vlm, std::filesystem::path recv_root,
                           std::string image_serve_root)
    : _vlm(vlm), _recv_root(std::move(recv_root)),
      _image_serve_root(std::move(image_serve_root)), _chat(_vlm.create_chat()) {
    // The model's default system prompt, if any, is the chat's first message.
    const auto& messages = _chat.get_messages();
    if (messages.is_array() && !messages.empty() && messages[0].is_object() &&
        messages[0].value("role", "") == "system" && messages[0].contains("content") &&
        messages[0]["content"].is_string()) {
        _default_system_prompt = messages[0]["content"].get<std::string>();
    }
    _system_prompt = _default_system_prompt;
}

bool VlmGenerator::_supports_thinking() const {
    return reasoning_format_for_model(_vlm.model_type()) != ReasoningFormat::None;
}

void VlmGenerator::_clear_history() {
    _chat.clear_history();
    _kept_images.clear();
    _kept_image_leaves.clear();
}

void VlmGenerator::_apply_settings(const std::string& system_prompt, bool enable_thinking) {
    _chat.set_system_prompt(system_prompt);
    _chat.set_enable_thinking(enable_thinking);
    _system_prompt = system_prompt;
    _clear_history();
    // Chat::set_system_prompt("") ends in clear_messages(), which keeps a
    // leading system message, so the old prompt would stay (the devkit has
    // the same bug). Drop it here; the images are already cleared above.
    if (system_prompt.empty()) {
        const auto& messages = _chat.get_messages();
        if (messages.is_array() && !messages.empty() && messages[0].is_object() &&
            messages[0].value("role", "") == "system") {
            _chat.set_messages(nlohmann::ordered_json::array());
        }
    }
}

RunResult VlmGenerator::run(const PromptRequest& request, EventBridge& bridge) {
    // An image for a text-only model is a user mistake: refuse it before
    // anything is added, so the conversation is kept.
    if (!request.images.empty() && !_vlm.support_image()) {
        throw GenerationError("this model does not accept images", /*history_cleared=*/false);
    }
    // Thinking on a model without it: refuse before the settings change below
    // clears the conversation (same check as reset()).
    if (request.enable_thinking && !_supports_thinking()) {
        throw GenerationError(kNoThinking, /*history_cleared=*/false);
    }
    // A different system prompt or thinking mode starts a new conversation.
    const std::string system_prompt =
        effective_system_prompt(request.system_prompt, _default_system_prompt);
    const bool new_settings =
        system_prompt != _system_prompt || request.enable_thinking != _chat.get_enable_thinking();
    // Two images with the same file name would overwrite each other on the card.
    // Refuse before anything is added, so the conversation is kept. With new
    // settings the old images are about to be cleared, so only this request counts.
    try {
        check_image_leaves(request.images,
                           new_settings ? std::set<std::string>{} : _kept_image_leaves);
    } catch (const std::invalid_argument& e) {
        throw GenerationError(e.what(), /*history_cleared=*/false);
    }

    // Only the final answer (no thinking) goes into the history, like
    // cli.cpp:202-237; hidden LFM2 reasoning is not sent (see StreamSplit).
    StreamSplit split(reasoning_format_for_model(_vlm.model_type()), request.enable_thinking);
    std::string answer;

    // Detach the callbacks on every exit path: the bridge must not hear from
    // LLiMa once this run is over (same guard idea as cli.cpp:244-249).
    // Declared after `split` / `answer`, so it is destroyed first.
    struct CallbackGuard {
        VisionLanguageModel& vlm;
        ~CallbackGuard() {
            vlm.set_text_callback([](const std::string&, bool, bool) {});
            vlm.set_info_callback([](const std::string&, double) {});
        }
    } guard{_vlm};
    _vlm.set_text_callback([&bridge, &split, &answer](const std::string& text, bool stream_end,
                                                      bool from_draft) {
        const StreamSplit::Out out = split.add(text, stream_end, from_draft);
        bridge.on_text(out.to_host, stream_end, from_draft);
        answer += out.to_answer;
    });
    _vlm.set_info_callback([&bridge](const std::string& metric, double value) {
        bridge.on_info(metric, value);
    });

    try {
        if (new_settings) {
            _apply_settings(system_prompt, request.enable_thinking);
        }

        // Pull each new image over PCIe and keep it until the history is cleared.
        for (const std::string& name : request.images) {
            const ImageTarget target = image_target_for(name);
            auto provider = std::make_shared<PcieFileProvider>(
                _recv_root, _image_serve_root, target.subfolder);
            const std::filesystem::path pulled = provider->get_path(target.filename);
            _kept_images.add([provider, pulled] { provider->evict(pulled); });
            _kept_image_leaves.insert(target.filename);
            _chat.add_image(pulled);
        }
        _chat.add_query(request.prompt);

        const std::optional<std::string> response = _vlm.run_model(
            _chat, cap_max_new_tokens(request.max_new_tokens,
                                      _vlm._language_model_ptr->get_max_num_tokens()));
        // run_model can return before LLiMa's streamer thread has fired the last
        // text/info callbacks. Wait for it, so genai.final goes out after the last token.
        _vlm.wait_for_streamer_completion();

        // A cancel that raced the natural end still counts as a cancel.
        const bool completed = response.has_value() && !bridge.cancel_requested();
        switch (after_run(completed, answer)) {
            case AfterRun::KeepAnswer:
                _chat.add_response(trim_answer(answer));
                return RunResult{true, false};
            case AfterRun::ClearInterrupted:
                _clear_history();
                return RunResult{false, true};
            case AfterRun::ClearEmpty:
                _clear_history();
                return RunResult{true, true};
        }
    } catch (const std::exception& e) {
        // A failed question must not leave a half-added question or image behind.
        _clear_history();
        throw GenerationError(e.what(), /*history_cleared=*/true);
    }
    return RunResult{true, false};   // not reached: every AfterRun case returns
}

void VlmGenerator::stop() { _vlm.stop_model(); }

ChatReply VlmGenerator::reset(const std::optional<std::string>& system_prompt,
                              bool enable_thinking) {
    if (enable_thinking && !_supports_thinking()) {
        return ChatReply{false, kNoThinking};
    }
    _apply_settings(effective_system_prompt(system_prompt, _default_system_prompt), enable_thinking);
    return ChatReply{true, ""};
}

std::string VlmGenerator::history() {
    return _chat.get_messages().dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
