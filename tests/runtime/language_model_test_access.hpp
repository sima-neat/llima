#pragma once

#include "language_model.hpp"

namespace simaai::llima {

struct LanguageModelTestAccess {
    static void reset_cache(LanguageModel& model) { model.invalidate_prompt_reuse(); }
    static void truncate_kv(LanguageModel& model, uint16_t length) { model.set_kv_cache_len(length); }
};

} // namespace simaai::llima
