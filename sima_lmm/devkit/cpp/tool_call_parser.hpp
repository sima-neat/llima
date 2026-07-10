#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace simaai {
namespace llima {

// Runtime helper for parsing model-emitted tool calls after tokenizer special
// tokens may already have been stripped by the streamer.
nlohmann::json try_parse_tool_calls(std::string_view text);
nlohmann::json try_parse_tool_calls(
    std::string_view text,
    const std::vector<std::string>& allowed_tool_names
);

class ToolCallStreamParser {
    public:
        struct Content {
            std::string text;
        };
        struct ToolCalls {
            nlohmann::json calls;
        };
        using Event = std::variant<Content, ToolCalls>;

        ToolCallStreamParser() = default;
        explicit ToolCallStreamParser(std::vector<std::string> allowed_tool_names);

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
        std::optional<std::vector<std::string>> _allowed_tool_names;
};

} // namespace llima
} // namespace simaai
