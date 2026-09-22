#include "reasoning_parser.hpp"

#include <algorithm>

namespace simaai {
namespace llima {

namespace {

constexpr std::string_view think_open = "<think>";
constexpr std::string_view think_close = "</think>";
constexpr std::string_view gemma_reasoning_open = "<|channel>thought\n";
constexpr std::string_view gemma_reasoning_close = "<channel|>";
constexpr std::string_view gptoss_reasoning_open = "<|channel|>analysis<|message|>";
constexpr std::string_view gptoss_channel = "<|channel|>";
constexpr std::string_view gptoss_message = "<|message|>";

// Content of a gpt-oss channel is shown when the channel is "final", or when it is
// a commentary message carrying a tool call. A plain commentary message is the
// model talking to itself, so it stays hidden like analysis.
bool gptoss_channel_is_visible(std::string_view header) {
    if (header.find("final") != std::string_view::npos) return true;
    if (header.find("commentary") == std::string_view::npos) return false;
    return header.find("to=") != std::string_view::npos
        || header.find("json") != std::string_view::npos;
}

// A channel header is preceded by "<|start|>assistant". Both control tokens decode
// to nothing, so the role name arrives as ordinary text and would otherwise be
// emitted as the tail of the previous message.
size_t gptoss_role_suffix_size(std::string_view text) {
    constexpr std::string_view role = "assistant";
    return text.ends_with(role) ? role.size() : 0;
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

std::array<std::string_view, 2> reasoning_special_tokens(ReasoningFormat format) {
    switch (format) {
        case ReasoningFormat::Qwen:
        case ReasoningFormat::Lfm2:
            return {think_open, think_close};
        case ReasoningFormat::Gemma4:
            return {"<|channel>", gemma_reasoning_close};
        case ReasoningFormat::GptOss:
            return {"<|channel|>", "<|message|>"};
        case ReasoningFormat::None:
            return {"", ""};
    }
    return {"", ""};
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
        // Harmony always emits channels; analysis = reasoning, final = answer.
        // Parse even when thinking is off so only the final channel shows.
        _start_marker = gptoss_reasoning_open;
        _end_marker = gptoss_channel;
        _channel_headers = true;
        _mode = enabled ? Mode::AwaitingStart : Mode::AwaitingHiddenStart;
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
        if (close_pos != std::string::npos && _channel_headers) {
            // The header runs to the message marker; wait for the rest of it.
            const auto header_pos = close_pos + _end_marker.size();
            const auto message_pos = _pending.find(gptoss_message, header_pos);
            if (message_pos == std::string::npos) {
                if (_mode == Mode::Reasoning) {
                    const std::string_view before(_pending.data(), close_pos);
                    emit(
                        events,
                        _pending.substr(0, close_pos - gptoss_role_suffix_size(before)),
                        true,
                        _pending_from_draft
                    );
                }
                _pending.erase(0, close_pos);
                if (done) {
                    _pending.clear();
                    _mode = Mode::Done;
                }
                break;
            }
            const std::string_view header(
                _pending.data() + header_pos, message_pos - header_pos
            );
            if (_mode == Mode::Reasoning) {
                const std::string_view before(_pending.data(), close_pos);
                emit(
                    events,
                    _pending.substr(0, close_pos - gptoss_role_suffix_size(before)),
                    true,
                    _pending_from_draft
                );
            }
            _pending.erase(0, message_pos + gptoss_message.size());
            _pending_from_draft = from_draft;
            // A hidden channel leaves the mode alone, so its content stays hidden.
            if (gptoss_channel_is_visible(header)) _mode = Mode::Content;
            continue;
        }
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
