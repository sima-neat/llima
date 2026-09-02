#pragma once

#include <initializer_list>
#include <vector>

#include "mla_model.hpp"

namespace simaai::llima::qwen3tts {

// A persistent, default-bound batch. Model I/O slices are fixed at construction.
class Qwen3MlaStaticBatch {
  public:
    explicit Qwen3MlaStaticBatch(std::vector<MLAModelWithBuffer*> models);
    Qwen3MlaStaticBatch(std::initializer_list<MLAModelWithBuffer*> models);

    void run() const;
    [[nodiscard]] size_t size() const { return models_.size(); }

  private:
    std::vector<MLAModelWithBuffer*> models_;
};

} // namespace simaai::llima::qwen3tts
