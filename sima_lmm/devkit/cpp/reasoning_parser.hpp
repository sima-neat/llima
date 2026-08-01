#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace simaai {
namespace llima {

enum class ReasoningFormat {
    None,
    Qwen,
    Gemma4,
};

ReasoningFormat reasoning_format_for_model(std::string_view model_type);
std::array<std::string_view, 2> reasoning_special_tokens(ReasoningFormat format);

class ReasoningStreamParser {
    public:
        struct Event {
            std::string text;
            bool reasoning = false;
            bool from_draft = false;
        };

        ReasoningStreamParser(
            ReasoningFormat format,
            bool enabled,
            bool prompt_opens_reasoning = false
        );

        std::vector<Event> add(
            std::string_view text,
            bool done = false,
            bool from_draft = false
        );

    private:
        enum class Mode {
            AwaitingStart,
            Reasoning,
            Content,
            Done,
        };

        void emit(
            std::vector<Event>& events,
            std::string text,
            bool reasoning,
            bool from_draft
        );
        size_t partial_marker_size(std::string_view marker) const;

        std::string_view _start_marker;
        std::string_view _end_marker;
        Mode _mode = Mode::Content;
        bool _optional_start = false;
        std::string _pending;
        bool _pending_from_draft = false;
};

} // namespace llima
} // namespace simaai
