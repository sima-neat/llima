// Pure rules for the conversation VlmGenerator keeps, copied from
// the devkit CLI (sima_lmm/devkit/cpp/cli.cpp). No LLiMa, no PCIe, so they
// are unit-tested without a card.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_CHAT_POLICY_
#define _SIMA_LLIMA_PCIE_BACKEND_CHAT_POLICY_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "reasoning_parser.hpp"

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

// The max_new_tokens to give run_model. LLiMa adds the prompt length to it in
// a uint16_t, so a very large value wraps around and the answer comes back
// empty. An answer can never be longer than the model's context, so cap it
// there; also keep context + cap <= 65535, since the prompt is at most the
// context. nullopt (= the model default) stays nullopt.
std::optional<uint16_t> cap_max_new_tokens(std::optional<uint16_t> requested,
                                           uint16_t context_tokens);

// Splits LLiMa's streamed text into what the host sees (genai.token) and the
// answer the history keeps (no thinking, like cli.cpp:202-237).
// To the host: the raw text, so a model's <think> markers stay visible when
// thinking is on. Only LFM2 with thinking off differs: it still reasons, and
// the devkit CLI hides that reasoning, so the host gets only the answer.
class StreamSplit {
    public:
        StreamSplit(ReasoningFormat format, bool enable_thinking);

        struct Out {
            std::string to_host;
            std::string to_answer;
        };
        Out add(const std::string& text, bool stream_end, bool from_draft);

    private:
        ReasoningStreamParser _parser;
        bool _hide_reasoning;
};

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
