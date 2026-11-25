#ifndef _SIMA_LLIMA_VISION_LANGUAGE_MODEL_
#define _SIMA_LLIMA_VISION_LANGUAGE_MODEL_

#include <concepts>
#include <filesystem>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
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

struct LanguageRuntime;
class RequestScope;
struct PreprocessedChat;
struct VlmConfig;
class VisionModel;

struct EncodedImageData;
// Shared handle to reusable vision-encoder output.
// It can be supplied only to a model with the matching vision package.
class EncodedImage {
public:
    EncodedImage() = default;
private:
    std::shared_ptr<const EncodedImageData> _data;
    friend class VisionLanguageModel;
};

// Limits, callbacks, cancellation and optional images for one generation request.
// The caller must keep the supplied image storage alive until the call returns.
struct GenerationOptions {
    std::optional<uint16_t> max_new_tokens; // Output budget, excluding the input tokens.
    std::optional<std::set<uint32_t>> stop_token_ids;
    std::stop_token stop; // The caller owns cancellation for this request.
    StreamCallbacks callbacks;
    // Raw images fill the first chat placeholders, followed by pre-encoded images.
    std::span<const cv::Mat> images;
    std::span<const EncodedImage> encoded_images;
    bool performance_summary = true;
    bool preserve_structural_tokens = true; // Keep active tool/reasoning markers in text and chunks.
    bool reject_oversized_prompt = false;   // Throw when the prompt cannot fit a generated token.
    std::optional<std::string> cache_id; // Omitted selects the legacy context, which consumes one slot.
};

// Each slot owns full model-specific KV/recurrent storage. Weights and workspace are shared.
struct ModelOptions {
    size_t max_kv_cache_slots = 1;
};

// Public entry point for text and image/text generation, scoring and image encoding.
// Coordinates the language runtime and optional vision encoder. Runs one inference call at a time.
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
        // Constrain the options overload so legacy calls with an empty braced
        // system prompt still resolve to the disk-loading overload above.
        template<class Options> requires std::same_as<Options, ModelOptions>
        VisionLanguageModel(std::filesystem::path model_path, Options options)
            : VisionLanguageModel(std::move(model_path), std::nullopt, std::nullopt,
                                  nullptr, options) {}
        VisionLanguageModel(
            std::filesystem::path model_path,
            std::optional<std::string> system_prompt,
            std::optional<std::string> chat_template,
            std::shared_ptr<FileProvider> file_provider,
            ModelOptions options
        );
        ~VisionLanguageModel();
        VisionLanguageModel(const VisionLanguageModel&) = delete;
        VisionLanguageModel& operator=(const VisionLanguageModel&) = delete;

        // Returns generated token IDs, decoded text, metrics and the finish reason.
        ChatResult generate(const Chat& chat, const GenerationOptions& options = {});
        std::vector<uint32_t> run_model(std::span<const uint32_t> input_token_ids,
                                        const GenerationOptions& options = {});
        std::vector<Eigen::bfloat16> run_model_for_logits(std::span<const uint32_t> input_token_ids,
                                                          const GenerationOptions& options = {});
        LogLikelihoodResult run_model_for_loglikelihood(
            std::span<const uint32_t> input_token_ids, size_t continuation_start,
            std::span<const uint32_t> continuation_token_ids, const GenerationOptions& options = {},
            bool use_group_prefill = true);
        GenerationPerformanceResult run_model_for_ttnt(std::span<const uint32_t> input_token_ids,
                                                       const GenerationOptions& options = {});

        std::vector<EncodedImage> encode_images(std::span<const cv::Mat> images);
        size_t shared_resource_bytes() const;
        size_t workspace_memory_bytes() const;
        size_t kv_cache_count() const;
        size_t kv_cache_bytes_per_slot() const;
        bool remove_kv_cache(const std::string& cache_id);
        void clear_kv_caches();

        void stop_model();

        // Retain the draft's runtime state; run_model dispatches automatically.
        // The caller may release the draft wrapper after attachment.
        void set_draft_vlm(VisionLanguageModel* draft_vlm);

        Chat create_chat();
        bool support_image() const;
        std::string_view model_type() const;
        ToolCallFormat tool_call_format() const;

        void set_reloc(const std::string& reloc_name);
        void unset_reloc();

    private:
        std::shared_ptr<LanguageRuntime> _state;
        std::unique_ptr<VisionModel> _vision_model;
        // Internal ZMQ benchmarking also uses config() and benchmark_chat().
        friend class ZMQServer;

        // Settings and prompt details for an internal benchmark that starts without cached tokens.
        struct ChatBenchmark {
            uint16_t max_context;
            GenerationPerformanceResult* performance;
            size_t input_tokens = 0;
            std::string formatted_prompt;
        };
        const VlmConfig& config() const;
        std::optional<std::vector<uint32_t>> benchmark_chat(
            const Chat& chat, const GenerationOptions& options, ChatBenchmark& benchmark);
        PreprocessedChat prepare_chat_inputs(
            RequestScope& request, const Chat& chat, const GenerationOptions& options,
            ChatBenchmark* benchmark = nullptr);
};

}
}

#endif
