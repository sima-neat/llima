#pragma once

#include "generation_types.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace simaai {
namespace llima {

class Tokenizer;

enum class ToolCallFormat {
    GenericJson,
    Lfm,
    Gemma,
    Mistral,
    Qwen,
    Llama,
    Qwen35,
    GptOss,
};

ToolCallFormat tool_call_format_for_model(std::string_view model_type);
std::vector<std::string> tool_call_special_tokens(ToolCallFormat format);
PreservedTokens resolve_tool_call_special_tokens(
    ToolCallFormat format,
    const Tokenizer& tokenizer
);
std::string decode_tool_call_output(
    const Tokenizer& tokenizer,
    const std::vector<uint32_t>& token_ids,
    const PreservedTokens& preserved_tokens
);

// Parses a model-specific tool-call representation into OpenAI-style calls.
nlohmann::json try_parse_tool_calls(
    ToolCallFormat format,
    std::string_view text,
    const std::vector<std::string>& allowed_tool_names
);

// Buffers streamed output until it can separate ordinary text from model-specific
// tool calls, then emits text chunks or parsed call lists.
class ToolCallStreamParser {
    public:
        // Ordinary text with a flag indicating whether it came from the draft model.
        struct Content {
            std::string text;
            bool from_draft = false;
        };
        // Parsed OpenAI-style tool calls emitted as one stream event.
        struct ToolCalls {
            nlohmann::json calls;
        };
        using Event = std::variant<Content, ToolCalls>;

        explicit ToolCallStreamParser(ToolCallFormat format = ToolCallFormat::GenericJson);
        ToolCallStreamParser(
            ToolCallFormat format,
            std::vector<std::string> allowed_tool_names
        );

        std::vector<Event> add(
            std::string_view text,
            bool done = false,
            bool from_draft = false
        );

    private:
        enum class Mode {
            Undecided,
            Content,
            ToolCall,
            Done,
        };

        Mode decide(bool done) const;
        void buffer_content(std::string_view text, bool from_draft);
        std::vector<Event> take_buffered_content();

        Mode _mode = Mode::Undecided;
        ToolCallFormat _format = ToolCallFormat::GenericJson;
        std::string _buffer;
        std::vector<Content> _content_buffer;
        std::optional<std::vector<std::string>> _allowed_tool_names;
};

} // namespace llima
} // namespace simaai
