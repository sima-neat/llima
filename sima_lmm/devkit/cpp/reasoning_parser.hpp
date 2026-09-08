#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace simaai {
namespace llima {

enum class ReasoningFormat {
    None,
    Qwen,
    Gemma4,
    Lfm2,
    GptOss,
};

ReasoningFormat reasoning_format_for_model(std::string_view model_type);
std::vector<std::string_view> reasoning_special_tokens(ReasoningFormat format);

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

        // True while tokens are being consumed without being emitted, i.e. the
        // caller sees no output even though the model is generating.
        bool in_hidden_reasoning() const {
            if (_mode == Mode::HiddenReasoning) return true;
            return _mode == Mode::ChannelScan || _mode == Mode::ChannelHeader ||
                (_mode == Mode::ChannelBody && !_body_visible);
        }

    private:
        enum class Mode {
            AwaitingStart,
            AwaitingHiddenStart,
            Reasoning,
            HiddenReasoning,
            Content,
            // Harmony channel machine, used by gpt-oss: scan for the next
            // <|channel|> header, read the channel name up to <|message|>, then
            // route the message body by channel.
            ChannelScan,
            ChannelHeader,
            ChannelBody,
            Done,
        };

        void emit(
            std::vector<Event>& events,
            std::string text,
            bool reasoning,
            bool from_draft
        );
        size_t partial_marker_size(std::string_view marker) const;
        std::vector<Event> add_channels(
            std::string_view text,
            bool done,
            bool from_draft
        );
        void route_channel(std::string_view header);

        std::string_view _start_marker;
        std::string_view _end_marker;
        Mode _mode = Mode::Content;
        bool _optional_start = false;
        bool _show_reasoning = false;
        bool _body_visible = false;
        bool _body_reasoning = false;
        std::string _pending;
        bool _pending_from_draft = false;
};

} // namespace llima
} // namespace simaai
