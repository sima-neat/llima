#include "reasoning_parser.hpp"

#include <algorithm>

namespace simaai {
namespace llima {

namespace {

constexpr std::string_view think_open = "<think>";
constexpr std::string_view think_close = "</think>";
constexpr std::string_view gemma_reasoning_open = "<|channel>thought\n";
constexpr std::string_view gemma_reasoning_close = "<channel|>";
constexpr std::string_view harmony_channel = "<|channel|>";
constexpr std::string_view harmony_message = "<|message|>";
constexpr std::string_view harmony_end = "<|end|>";

// A message body ends at <|end|>; <|channel|> also ends it so a dropped <|end|>
// cannot merge the next message into the current one.
constexpr std::array<std::string_view, 2> harmony_body_enders = {
    harmony_end, harmony_channel
};

struct MarkerMatch {
    size_t pos = std::string::npos;
    std::string_view marker;
};

MarkerMatch find_first_marker(const std::string& text) {
    MarkerMatch match;
    for (const auto marker : harmony_body_enders) {
        const auto pos = text.find(marker);
        if (pos < match.pos) match = {pos, marker};
    }
    return match;
}

} // namespace

ReasoningFormat reasoning_format_for_model(std::string_view model_type) {
    if (model_type.starts_with("llm-qwen3") || model_type.starts_with("vlm-qwen3")) {
        return ReasoningFormat::Qwen;
    }
    if (model_type == "vlm-gemma4") {
        return ReasoningFormat::Gemma4;
    }
    if (model_type == "llm-lfm2") {
        return ReasoningFormat::Lfm2;
    }
    if (model_type == "llm-gpt_oss") {
        return ReasoningFormat::GptOss;
    }
    return ReasoningFormat::None;
}

std::vector<std::string_view> reasoning_special_tokens(ReasoningFormat format) {
    switch (format) {
        case ReasoningFormat::Qwen:
        case ReasoningFormat::Lfm2:
            return {think_open, think_close};
        case ReasoningFormat::Gemma4:
            return {"<|channel>", gemma_reasoning_close};
        case ReasoningFormat::GptOss:
            // <|end|> is preserved too: without it a channel body would run
            // into the role name of the next message.
            return {harmony_channel, harmony_message, harmony_end};
        case ReasoningFormat::None:
            return {};
    }
    return {};
}

ReasoningStreamParser::ReasoningStreamParser(
    ReasoningFormat format,
    bool enabled,
    bool prompt_opens_reasoning
) {
    if (format == ReasoningFormat::Lfm2) {
        // LFM2 Thinking always reasons; disabling only hides its reasoning output.
        _start_marker = think_open;
        _end_marker = think_close;
        _mode = enabled ? Mode::AwaitingStart : Mode::AwaitingHiddenStart;
    } else if (format == ReasoningFormat::GptOss) {
        // Harmony wraps every message in a channel header, and the model may
        // switch channels several times in one response, so follow the headers
        // instead of matching a single open/close pair. Text outside a known
        // channel body is dropped rather than shown.
        _show_reasoning = enabled;
        _mode = Mode::ChannelScan;
    } else if (!enabled || format == ReasoningFormat::None) {
        _mode = Mode::Content;
    } else if (format == ReasoningFormat::Qwen) {
        _start_marker = think_open;
        _end_marker = think_close;
        _mode = Mode::Reasoning;
        _optional_start = true;
    } else {
        _start_marker = gemma_reasoning_open;
        _end_marker = gemma_reasoning_close;
        _mode = prompt_opens_reasoning ? Mode::Reasoning : Mode::AwaitingStart;
    }
}

std::vector<ReasoningStreamParser::Event> ReasoningStreamParser::add(
    std::string_view text,
    bool done,
    bool from_draft
) {
    if (
        _mode == Mode::ChannelScan || _mode == Mode::ChannelHeader ||
        _mode == Mode::ChannelBody
    ) {
        return add_channels(text, done, from_draft);
    }

    std::vector<Event> events;
    if (_mode == Mode::Done) return events;

    if (_pending.empty()) _pending_from_draft = from_draft;
    _pending.append(text);

    while (_mode != Mode::Done) {
        if (
            _mode == Mode::AwaitingStart || _mode == Mode::AwaitingHiddenStart ||
            (_mode == Mode::Reasoning && _optional_start)
        ) {
            const auto first_non_whitespace = _pending.find_first_not_of(" \t\r\n");
            _pending.erase(0, first_non_whitespace);
        }

        if (_mode == Mode::Content) {
            emit(events, std::move(_pending), false, _pending_from_draft);
            _pending.clear();
            if (done) _mode = Mode::Done;
            break;
        }

        if (_mode == Mode::AwaitingStart || _mode == Mode::AwaitingHiddenStart) {
            const bool hide_reasoning = _mode == Mode::AwaitingHiddenStart;
            if (_pending.starts_with(_start_marker)) {
                _pending.erase(0, _start_marker.size());
                _pending_from_draft = from_draft;
                _mode = hide_reasoning ? Mode::HiddenReasoning : Mode::Reasoning;
                continue;
            }
            if (_start_marker.starts_with(_pending)) {
                if (done) {
                    _pending.clear();
                    _mode = Mode::Done;
                }
                break;
            }
            _mode = Mode::Content;
            continue;
        }

        if (_mode == Mode::Reasoning && _optional_start) {
            if (_pending.starts_with(_start_marker)) {
                _pending.erase(0, _start_marker.size());
                _pending_from_draft = from_draft;
                _optional_start = false;
            } else if (_start_marker.starts_with(_pending)) {
                if (done) {
                    _pending.clear();
                    _mode = Mode::Done;
                }
                break;
            } else {
                _optional_start = false;
            }
        }

        const auto close_pos = _pending.find(_end_marker);
        if (close_pos != std::string::npos) {
            if (_mode == Mode::Reasoning) {
                emit(events, _pending.substr(0, close_pos), true, _pending_from_draft);
            }
            _pending.erase(0, close_pos + _end_marker.size());
            _pending_from_draft = from_draft;
            _mode = Mode::Content;
            continue;
        }

        const size_t retained = partial_marker_size(_end_marker);
        if (_mode == Mode::Reasoning) {
            emit(
                events,
                _pending.substr(0, _pending.size() - retained),
                true,
                _pending_from_draft
            );
        }
        _pending.erase(0, _pending.size() - retained);
        if (!_pending.empty()) _pending_from_draft = from_draft;
        if (done) {
            _pending.clear();
            _mode = Mode::Done;
        }
        break;
    }

    return events;
}

void ReasoningStreamParser::route_channel(std::string_view header) {
    // The header is the text between <|channel|> and <|message|>, e.g. "final",
    // "analysis", or "commentary to=functions.get_weather <|constrain|>json".
    const bool has_recipient = header.find(" to=") != std::string_view::npos;
    if (header.starts_with("final")) {
        _body_reasoning = false;
        _body_visible = true;
    } else if (header.starts_with("commentary") && !has_recipient) {
        // Commentary without a recipient is addressed to the user; models
        // answer there outright when the developer message nudges them to.
        _body_reasoning = false;
        _body_visible = true;
    } else {
        // Analysis, tool calls and anything unrecognised are reasoning.
        _body_reasoning = true;
        _body_visible = _show_reasoning && !has_recipient;
    }
}

std::vector<ReasoningStreamParser::Event> ReasoningStreamParser::add_channels(
    std::string_view text,
    bool done,
    bool from_draft
) {
    std::vector<Event> events;
    if (_pending.empty()) _pending_from_draft = from_draft;
    _pending.append(text);

    while (true) {
        if (_mode == Mode::ChannelScan) {
            // Role names and message delimiters live between channels; none of
            // it is meant for the user, so discard up to the next header.
            const auto pos = _pending.find(harmony_channel);
            if (pos == std::string::npos) {
                _pending.erase(0, _pending.size() - partial_marker_size(harmony_channel));
                break;
            }
            _pending.erase(0, pos + harmony_channel.size());
            _mode = Mode::ChannelHeader;
            continue;
        }

        if (_mode == Mode::ChannelHeader) {
            const auto pos = _pending.find(harmony_message);
            if (pos == std::string::npos) break;
            route_channel(std::string_view(_pending).substr(0, pos));
            _pending.erase(0, pos + harmony_message.size());
            _pending_from_draft = from_draft;
            _mode = Mode::ChannelBody;
            continue;
        }

        const auto ender = find_first_marker(_pending);
        if (ender.pos != std::string::npos) {
            if (_body_visible) {
                emit(events, _pending.substr(0, ender.pos), _body_reasoning, _pending_from_draft);
            }
            _pending.erase(0, ender.pos + ender.marker.size());
            _pending_from_draft = from_draft;
            _mode = ender.marker == harmony_channel ? Mode::ChannelHeader : Mode::ChannelScan;
            continue;
        }

        // Hold back any suffix that could be the head of a terminator.
        size_t retained = 0;
        for (const auto marker : harmony_body_enders) {
            retained = std::max(retained, partial_marker_size(marker));
        }
        if (_body_visible) {
            emit(
                events,
                _pending.substr(0, _pending.size() - retained),
                _body_reasoning,
                _pending_from_draft
            );
        }
        _pending.erase(0, _pending.size() - retained);
        if (!_pending.empty()) _pending_from_draft = from_draft;
        break;
    }

    if (done) {
        _pending.clear();
        _mode = Mode::Done;
    }
    return events;
}

void ReasoningStreamParser::emit(
    std::vector<Event>& events,
    std::string text,
    bool reasoning,
    bool from_draft
) {
    if (text.empty()) return;
    events.push_back({std::move(text), reasoning, from_draft});
}

size_t ReasoningStreamParser::partial_marker_size(std::string_view marker) const {
    const size_t max_size = std::min(
        _pending.size(), marker.empty() ? size_t{0} : marker.size() - 1
    );
    for (size_t size = max_size; size > 0; --size) {
        if (_pending.compare(_pending.size() - size, size, marker, 0, size) == 0) {
            return size;
        }
    }
    return 0;
}

} // namespace llima
} // namespace simaai
