#ifndef _SIMA_LLIMA_VISION_LANGUAGE_MODEL_
#define _SIMA_LLIMA_VISION_LANGUAGE_MODEL_

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string_view>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/mat.hpp>

#include "chat.hpp"
#include "file_provider.hpp"
#include "generation_types.hpp"
#include "text_streamer.hpp"
#include "tool_call_parser.hpp"

namespace simaai {
namespace llima {

class CLI;
class WEB;
struct LanguageRunnerState;
class VisionLanguageRequest;
struct VlmConfig;
class VisionModel;

struct EncodedImageData;
class EncodedImage {
public:
    EncodedImage() = default;
private:
    std::shared_ptr<const EncodedImageData> _data;
    friend class VisionLanguageModel;
};

struct GenerationOptions {
    std::optional<uint16_t> max_new_tokens;
    std::optional<std::set<uint32_t>> stop_token_ids;
    std::optional<TextStreamer::InfoCallback> info_callback;
    std::optional<TextStreamer::TextCallback> text_callback;
    std::span<const cv::Mat> images;
    std::span<const EncodedImage> encoded_images;
};

class VisionLanguageModel {
    public:
        // Existing disk-loading overload; preserves source compatibility.
        VisionLanguageModel(
            std::filesystem::path model_path,
            std::optional<std::string> system_prompt = std::nullopt,
            std::optional<std::string> chat_template = std::nullopt
        );
        // Reads the model files through file_provider (nullptr = from disk).
        VisionLanguageModel(
            std::filesystem::path model_path,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template,
            std::shared_ptr<FileProvider> file_provider
        );
        ~VisionLanguageModel();
        VisionLanguageModel(const VisionLanguageModel&) = delete;
        VisionLanguageModel& operator=(const VisionLanguageModel&) = delete;

        std::optional<std::string> run_model(
            const Chat& chat,
            std::optional<uint16_t> max_new_tokens = std::nullopt
        );
        std::optional<std::string> run_model(const Chat& chat, const GenerationOptions& options);
        std::optional<std::string> run_model(SessionId session, const Chat& chat,
                                            const GenerationOptions& options);
        std::vector<uint32_t> run_model(
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt
        );
        std::vector<Eigen::bfloat16> run_model_for_logits(
            std::span<const uint32_t> input_token_ids
        );
        LogLikelihoodResult run_model_for_loglikelihood(
            std::span<const uint32_t> input_token_ids,
            size_t continuation_start,
            std::span<const uint32_t> continuation_token_ids,
            bool use_group_prefill = true
        );
        GenerationPerformanceResult run_model_for_ttnt(
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt
        );
        std::vector<EncodedImage> encode_images(std::span<const cv::Mat> images);
        SessionId create_session();
        void reset_session(SessionId id);
        void release_session(SessionId id);
        void set_session_limit(size_t limit);
        std::optional<std::string> run_model(SessionId session, const Chat& chat,
                                            std::optional<uint16_t> max_new_tokens = std::nullopt);
        std::vector<uint32_t> run_model(SessionId session, std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt);
        std::vector<Eigen::bfloat16> run_model_for_logits(SessionId session,
            std::span<const uint32_t> input_token_ids);
        LogLikelihoodResult run_model_for_loglikelihood(SessionId session,
            std::span<const uint32_t> input_token_ids, size_t continuation_start,
            std::span<const uint32_t> continuation_token_ids, bool use_group_prefill = true);
        GenerationPerformanceResult run_model_for_ttnt(SessionId session,
            std::span<const uint32_t> input_token_ids,
            std::optional<uint16_t> override_max_num_tokens = std::nullopt,
            std::optional<std::set<uint32_t>> override_stop_token_ids = std::nullopt);
        size_t session_memory_bytes(SessionId id) const;
        size_t shared_resource_bytes() const;
        size_t workspace_memory_bytes() const;

        void stop_model();

        // Retain the draft's runtime state; run_model dispatches automatically.
        // The caller may release the draft wrapper after attachment.
        void set_draft_vlm(VisionLanguageModel* draft_vlm);

        Chat create_chat();
        bool support_image() const;
        ToolCallFormat tool_call_format() const;

        void set_info_callback(TextStreamer::InfoCallback callback);

        void set_text_callback(TextStreamer::TextCallback callback);

        void wait_for_streamer_completion();

        void set_reloc(const std::string& reloc_name);
        void unset_reloc();
        void set_reloc(SessionId session, const std::string& reloc_name);
        void unset_reloc(SessionId session);
        void set_enable_thinking(bool enable_thinking);
        bool get_enable_thinking() const;

    private:
        std::shared_ptr<LanguageRunnerState> _state;
        std::unique_ptr<VisionModel> _vision_model;
        friend class CLI;
        friend class WEB;
        friend class ZMQServer;

        struct ChatBenchmark {
            uint16_t max_context;
            GenerationPerformanceResult* performance;
            size_t input_tokens = 0;
            std::string formatted_prompt;
        };
        const VlmConfig& config() const;
        std::optional<std::vector<uint32_t>> benchmark_chat(
            const Chat& chat, const GenerationOptions& options, ChatBenchmark& benchmark);
        std::optional<std::vector<uint32_t>> generate_chat_tokens(
            VisionLanguageRequest& request, const Chat& chat, const GenerationOptions& options,
            ChatBenchmark* benchmark = nullptr);
        std::string_view model_type() const;



};

}
}

#endif
