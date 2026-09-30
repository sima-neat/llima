// Pure chat rules for VlmGenerator. See chat_policy.hpp.
#include "chat_policy.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

namespace simaai {
namespace llima {
namespace pcie_backend {

std::string effective_system_prompt(const std::optional<std::string>& requested,
                                    const std::string& model_default) {
    return requested.has_value() ? *requested : model_default;
}

std::string trim_answer(std::string_view text) {
    const auto ws = " \t\n\r";
    const auto b = text.find_first_not_of(ws);
    if (b == std::string_view::npos) return {};
    const auto e = text.find_last_not_of(ws);
    return std::string(text.substr(b, e - b + 1));
}

AfterRun after_run(bool completed, const std::string& answer) {
    if (!completed) return AfterRun::ClearInterrupted;
    return trim_answer(answer).empty() ? AfterRun::ClearEmpty : AfterRun::KeepAnswer;
}

std::optional<uint16_t> cap_max_new_tokens(std::optional<uint16_t> requested,
                                           uint16_t context_tokens) {
    if (!requested.has_value()) return std::nullopt;
    const uint16_t room = static_cast<uint16_t>(UINT16_MAX - context_tokens);
    return std::min({*requested, context_tokens, room});
}

KeptFiles::~KeptFiles() { clear(); }

void KeptFiles::add(std::function<void()> remove) { _removers.push_back(std::move(remove)); }

void KeptFiles::clear() {
    std::vector<std::function<void()>> removers;
    removers.swap(_removers);
    for (auto& remove : removers) {
        try {
            remove();
        } catch (...) {
            // A file we cannot delete is not worth failing over; the start-up
            // recv sweep removes leftovers.
        }
    }
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
