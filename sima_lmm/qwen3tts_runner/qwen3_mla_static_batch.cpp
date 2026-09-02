#include "qwen3_mla_static_batch.hpp"

#include <stdexcept>

namespace simaai::llima::qwen3tts {

Qwen3MlaStaticBatch::Qwen3MlaStaticBatch(std::vector<MLAModelWithBuffer*> models)
    : models_(std::move(models)) {
    if (models_.empty()) throw std::runtime_error("A static MLA batch cannot be empty");
    for (const auto* model : models_) {
        if (!model) throw std::runtime_error("A static MLA batch contains a null model");
    }
}

Qwen3MlaStaticBatch::Qwen3MlaStaticBatch(std::initializer_list<MLAModelWithBuffer*> models)
    : Qwen3MlaStaticBatch(std::vector<MLAModelWithBuffer*>(models)) {}

void Qwen3MlaStaticBatch::run() const {
    for (auto* model : models_) model->add_to_queue();
    MLAModelWithBuffer::run_queue();
}

} // namespace simaai::llima::qwen3tts
