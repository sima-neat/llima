#pragma once

// Private layer binding and scheduling helpers shared by the decoding algorithms.
#include "language_model_state.hpp"
#include "runtime_context.hpp"

namespace simaai::llima {

class LayerExecutor {
  public:
    using Overrides = std::map<uint8_t, MLABufferSlice>;
    struct EmbeddingInput {
        MLABuffer* values;
        MLABuffer* scales;
    };
    enum class LayerOutput { AllRows, CacheOnly, LogitsRow };
    struct LogitsWindow {
        uint16_t first;
        uint16_t width;
    };
    static LogitsWindow logits_window(const LanguageModelConfig& cfg, uint16_t width, uint16_t row);
    struct AttentionPrograms {
        MLAModelWithBuffer& pre;
        MLAModelWithBuffer& cache;
        MLAModelWithBuffer& post;
    };
    struct LayerBindings {
        Overrides* input = nullptr;
        Overrides* cache = nullptr;
        Overrides* post_input = nullptr;
        Overrides* output = nullptr;
    };
    static EmbeddingInput stage_tokens(LanguageModel& model, LanguageModel& source,
                                       std::span<const uint32_t> ids, uint16_t width,
                                       MLABuffer* floating_input = nullptr);
    static Overrides per_layer_inputs(LanguageModel& model, MLABuffer* values, MLABuffer* scales,
                                      uint32_t row, uint16_t rows,
                                      std::optional<uint32_t> token = std::nullopt);
    static void bind_attention_inputs(LanguageModel& model, MLAModelWithBuffer& pre,
                                      MLAModelWithBuffer& post, uint8_t layer, MLABuffer* scales,
                                      uint32_t scale_row, uint16_t rope_row);
    static void bind_kv_outputs(LanguageModel& model, MLAModelWithBuffer& pre, uint8_t layer,
                                uint16_t position);
    static void queue_attention(AttentionPrograms models,
                                LayerBindings bindings);
    static void queue_layer(LanguageModel& model, uint16_t width, uint16_t position, uint8_t layer,
                            MLABuffer* scales, uint32_t scale_row, LayerBindings bindings,
                            LayerOutput output = LayerOutput::AllRows, uint16_t logit_row = 0);

    static std::unique_lock<std::recursive_mutex> lock_execution();

  private:
    static Overrides post_rows(LanguageModel& model, uint16_t width, uint8_t layer,
                               uint16_t position, uint32_t first, uint16_t rows,
                               MLABuffer* hidden = nullptr);
    static AttentionPrograms bind_attention(LanguageModel& model, uint16_t width, uint16_t position,
                                            uint8_t layer, MLABuffer* scales, uint32_t scale_row);
};

} // namespace simaai::llima
