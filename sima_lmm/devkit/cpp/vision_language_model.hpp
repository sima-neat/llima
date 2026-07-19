#ifndef _SIMA_LLIMA_VISION_LANGUAGE_MODEL_
#define _SIMA_LLIMA_VISION_LANGUAGE_MODEL_

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <vector>

#include <Eigen/Dense>

#include "base_model.hpp"
#include "chat.hpp"
#include "language_model.hpp"
#include "tool_call_parser.hpp"
#include "vision_model.hpp"
#include "vlm_helper.hpp"

namespace simaai {
namespace llima {

class VisionLanguageModel : public BaseModel<VlmConfig> {
    public:
        VisionLanguageModel(
            std::filesystem::path model_path,
            std::optional<std::string> system_prompt = std::nullopt,
            std::optional<std::string> chat_template = std::nullopt
        );
        ~VisionLanguageModel() {}

        std::optional<std::string> run_model(
            const Chat& chat,
            std::optional<uint16_t> max_new_tokens = std::nullopt
        );
        std::vector<uint32_t> run_model(
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt
        );
        std::vector<Eigen::bfloat16> run_model_for_logits(
            std::span<const uint32_t> input_token_ids
        );
        std::vector<double> run_model_for_ttnt(
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt
        );
        void stop_model();

        // Configure a non-owning draft VLM for speculative decoding. When set,
        // run_model dispatches through the spec path automatically.
        void set_draft_vlm(VisionLanguageModel* draft_vlm) { _draft_vlm_ptr = draft_vlm; }

        Chat create_chat() { return Chat(_vlm_helper); }
        bool support_image() const { return _cfg.support_image(); }
        ToolCallFormat tool_call_format() const { return _tool_call_format; }

        void set_info_callback(TextStreamer::InfoCallback callback) {
            _text_streamer.set_info_callback(callback);
        }

        void set_text_callback(TextStreamer::TextCallback callback) {
            _text_streamer.set_text_callback(callback);
        }

        void wait_for_streamer_completion() {
            _text_streamer.wait_streaming();
        }

        void set_reloc(const std::string& reloc_name) {
             _language_model_ptr->set_reloc(reloc_name);
        }
        void unset_reloc() { _language_model_ptr->unset_reloc(); }

    private:
        VlmHelper _vlm_helper;
        TextStreamer _text_streamer;
        ToolCallFormat _tool_call_format = ToolCallFormat::GenericJson;
        std::unique_ptr<VisionModel> _vision_model_ptr;
        std::unique_ptr<LanguageModel> _language_model_ptr;
        std::mutex _run_mutex;  // Protects run_model from concurrent access
        // Non-owning pointer to the draft VLM for speculative decoding;
        // nullptr in non-spec mode. The owner (CLI) outlives this target VLM.
        VisionLanguageModel* _draft_vlm_ptr = nullptr;
};

}
}

#endif
