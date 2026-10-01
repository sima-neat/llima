// VlmGenerator: the real Generator. It runs prompts through LLiMa's
// VisionLanguageModel and sends LLiMa's callbacks to the EventBridge.
// Flow: BackendLoop -> VlmGenerator -> LLiMa -> EventBridge -> host.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_VLM_GENERATOR_
#define _SIMA_LLIMA_PCIE_BACKEND_VLM_GENERATOR_

#include <filesystem>
#include <optional>
#include <set>
#include <string>

#include "backend_loop.hpp"
#include "chat.hpp"
#include "chat_policy.hpp"
#include "vision_language_model.hpp"
#include "vlm_image.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {

// The real Generator: runs prompts through LLiMa's VisionLanguageModel, with
// the streamer callbacks wired to the EventBridge. Keeps ONE Chat for the life
// of the backend, like the devkit CLI: each prompt adds a question
// (and its images) and the answer. Pulled images stay on the card until the
// history is cleared.
class VlmGenerator final : public Generator {
    public:
        // recv_root is where a pulled image lands (same root the model ELF
        // uses); image_serve_root is the host [serve] root the image is pulled
        // from (the CLI's --image-serve-root, default "data").
        VlmGenerator(VisionLanguageModel& vlm, std::filesystem::path recv_root,
                     std::string image_serve_root);
        RunResult run(const PromptRequest& request, EventBridge& bridge) override;
        void stop() override;
        ChatReply reset(const std::optional<std::string>& system_prompt,
                        bool enable_thinking) override;
        std::string history() override;
    private:
        void _apply_settings(const std::string& system_prompt, bool enable_thinking);
        void _clear_history();
        bool _supports_thinking() const;

        VisionLanguageModel& _vlm;
        std::filesystem::path _recv_root;
        std::string _image_serve_root;
        Chat _chat;                           // the kept conversation
        std::string _default_system_prompt;   // what create_chat() started with
        std::string _system_prompt;           // current effective prompt ("" = none)
        KeptFiles _kept_images;               // deletes pulled images on clear
        std::set<std::string> _kept_image_leaves;  // their file names in recv_root
};

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
