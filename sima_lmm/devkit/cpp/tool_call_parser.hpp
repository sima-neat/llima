#pragma once

#include <nlohmann/json.hpp>

#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace simaai {
namespace llima {

// Runtime helper for parsing model-emitted tool calls after tokenizer special
// tokens may already have been stripped by the streamer.
nlohmann::json try_parse_tool_calls(std::string_view text);

class ToolCallStreamParser {
    public:
        struct Content {
            std::string text;
        };
        struct ToolCalls {
            nlohmann::json calls;
        };
        using Event = std::variant<Content, ToolCalls>;

        std::vector<Event> add(std::string_view text, bool done = false);

    private:
        enum class Mode {
            Undecided,
            Content,
            Gemma,
            Qwen,
            Json,
            Done,
        };

        Mode decide(bool done) const;

        Mode _mode = Mode::Undecided;
        std::string _buffer;
};

} // namespace llima
} // namespace simaai
