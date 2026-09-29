// Pure rules for the conversation VlmGenerator keeps, copied from
// the devkit CLI (sima_lmm/devkit/cpp/cli.cpp). No LLiMa, no PCIe, so they
// are unit-tested without a card.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_CHAT_POLICY_
#define _SIMA_LLIMA_PCIE_BACKEND_CHAT_POLICY_

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai {
namespace llima {
namespace pcie_backend {

// The system prompt a request asks for. nullopt = the model's default (what
// create_chat() starts with); "" = no system prompt; else that text.
std::string effective_system_prompt(const std::optional<std::string>& requested,
                                    const std::string& model_default);

// The answer as the devkit stores it in the history: trimmed.
std::string trim_answer(std::string_view text);

// What to do with the chat after a run (devkit cli.cpp:254-267).
enum class AfterRun {
    KeepAnswer,        // add the answer to the history
    ClearInterrupted,  // stopped (Ctrl-C): clear the history
    ClearEmpty,        // no final answer: clear the history
};
AfterRun after_run(bool completed, const std::string& answer);

// Delete actions for the image files a chat keeps on the card. clear() runs
// them all, in order, and forgets them; one that throws does not stop the
// rest. The destructor calls clear().
class KeptFiles {
    public:
        KeptFiles() = default;
        ~KeptFiles();
        KeptFiles(const KeptFiles&) = delete;
        KeptFiles& operator=(const KeptFiles&) = delete;

        void add(std::function<void()> remove);
        void clear();
        std::size_t size() const { return _removers.size(); }

    private:
        std::vector<std::function<void()>> _removers;
};

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
