#include "tool_call_parser.hpp"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <regex>

namespace simaai {
namespace llima {
namespace {

std::string_view trim_left_view(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) {
        text.remove_prefix(1);
    }
    return text;
}

std::string_view trim_view(std::string_view text) {
    text = trim_left_view(text);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) {
        text.remove_suffix(1);
    }
    return text;
}

bool is_prefix_of(std::string_view prefix, std::string_view text) {
    return prefix.size() <= text.size() && text.substr(0, prefix.size()) == prefix;
}

nlohmann::json build_tool_call_entry(
    const nlohmann::json& parsed,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    if (!parsed.is_object() || !parsed.contains("name") || !parsed["name"].is_string())
        return nullptr;

    const auto name = parsed["name"].get<std::string>();
    if (name.empty()) return nullptr;
    if (allowed_tool_names != nullptr &&
        std::find(allowed_tool_names->begin(), allowed_tool_names->end(), name) ==
            allowed_tool_names->end()) {
        return nullptr;
    }

    const nlohmann::json* raw_args = nullptr;
    if (parsed.contains("arguments")) {
        raw_args = &parsed["arguments"];
    } else if (parsed.contains("parameters")) {
        raw_args = &parsed["parameters"];
    }
    if (raw_args == nullptr) return nullptr;

    nlohmann::json args;
    if (raw_args->is_object()) {
        args = *raw_args;
    } else if (raw_args->is_string()) {
        args = nlohmann::json::parse(raw_args->get<std::string>());
        if (!args.is_object()) return nullptr;
    } else {
        return nullptr;
    }

    std::string id = parsed.contains("id") && parsed["id"].is_string()
        ? parsed["id"].get<std::string>()
        : std::to_string(std::time(nullptr)).substr(2) + std::to_string(id_counter++ % 10);

    return {
        {"id", id},
        {"type", "function"},
        {"function", {{"name", name}, {"arguments", args.dump()}}}
    };
}

size_t find_matching_brace(std::string_view text, size_t start) {
    if (start >= text.size() || text[start] != '{') return std::string_view::npos;

    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t idx = start; idx < text.size(); ++idx) {
        const char current = text[idx];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (current == '\\') {
                escaped = true;
            } else if (current == '"') {
                in_string = false;
            }
            continue;
        }

        if (current == '"') {
            in_string = true;
        } else if (current == '{') {
            ++depth;
        } else if (current == '}') {
            --depth;
            if (depth == 0) return idx;
            if (depth < 0) return std::string_view::npos;
        }
    }
    return std::string_view::npos;
}

std::string gemma4_bare_to_json(const std::string& text) {
    static const std::regex quote_marker(R"(<\|"\|>)");
    static const std::regex unquoted_key(R"((\w+)\s*:)");
    static const std::regex unquoted_val(R"(:\s*([^{}\[\]",\s][^{}\[\]",]*))");
    std::string normalized_quotes = std::regex_replace(text, quote_marker, "\"");
    std::string with_quoted_keys = std::regex_replace(normalized_quotes, unquoted_key, "\"$1\":");
    return std::regex_replace(with_quoted_keys, unquoted_val, ":\"$1\"");
}

nlohmann::json parse_json_array_tool_calls(
    const nlohmann::json& parsed,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    if (!parsed.is_array() || parsed.empty()) return nullptr;

    nlohmann::json result = nlohmann::json::array();
    for (const auto& item : parsed) {
        auto entry = build_tool_call_entry(item, id_counter, allowed_tool_names);
        if (entry.is_null()) return nullptr;
        result.push_back(std::move(entry));
    }
    return result;
}

nlohmann::json parse_plain_json_tool_calls(
    std::string_view text,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    nlohmann::json result = nlohmann::json::array();
    size_t pos = 0;
    bool parsed_any = false;
    while (pos < text.size()) {
        while (pos < text.size() &&
               std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
            ++pos;
        }
        if (pos == text.size()) break;
        if (text[pos] == ';' || text[pos] == ',') {
            if (!parsed_any) return nullptr;
            ++pos;
            while (pos < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
                ++pos;
            }
            if (pos == text.size()) return nullptr;
        }
        if (text[pos] != '{') return nullptr;
        auto close = find_matching_brace(text, pos);
        if (close == std::string_view::npos) return nullptr;
        auto parsed = nlohmann::json::parse(std::string(text.substr(pos, close - pos + 1)));
        auto entry = build_tool_call_entry(parsed, id_counter, allowed_tool_names);
        if (entry.is_null()) return nullptr;
        result.push_back(std::move(entry));
        parsed_any = true;
        pos = close + 1;
    }
    return result.empty() ? nullptr : result;
}

nlohmann::json try_parse_tool_calls_impl(
    std::string_view text,
    const std::vector<std::string>* allowed_tool_names
) {
    text = trim_view(text);
    if (text.empty()) return nullptr;

    int id_counter = 0;
    try {
        if (text.starts_with("call:")) {
            nlohmann::json result = nlohmann::json::array();
            size_t pos = 0;
            while (pos < text.size()) {
                while (pos < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
                    ++pos;
                }
                if (pos == text.size()) break;
                if (!text.substr(pos).starts_with("call:")) return nullptr;

                pos += 5;
                auto brace = text.find('{', pos);
                if (brace == std::string_view::npos) return nullptr;
                const auto name = trim_view(text.substr(pos, brace - pos));
                auto close = find_matching_brace(text, brace);
                if (close == std::string_view::npos) return nullptr;
                std::string args =
                    gemma4_bare_to_json(std::string(text.substr(brace, close - brace + 1)));
                auto entry = build_tool_call_entry(
                    {{"name", std::string(name)}, {"arguments", args}},
                    id_counter,
                    allowed_tool_names
                );
                if (entry.is_null()) return nullptr;
                result.push_back(std::move(entry));
                pos = close + 1;
            }
            return result.empty() ? nullptr : result;
        }

        constexpr std::string_view mistral_prefix = "[TOOL_CALLS]";
        if (text.starts_with(mistral_prefix)) {
            text.remove_prefix(mistral_prefix.size());
            text = trim_left_view(text);
            if (text.empty()) return nullptr;
            auto parsed = nlohmann::json::parse(std::string(text));
            return parse_json_array_tool_calls(parsed, id_counter, allowed_tool_names);
        }

        if (text.starts_with('[')) {
            auto parsed = nlohmann::json::parse(std::string(text));
            return parse_json_array_tool_calls(parsed, id_counter, allowed_tool_names);
        }

        constexpr std::string_view open_tag = "<tool_call>";
        constexpr std::string_view close_tag = "</tool_call>";
        if (text.starts_with(open_tag)) {
            nlohmann::json result = nlohmann::json::array();
            std::size_t pos = 0;
            while (pos < text.size()) {
                while (pos < text.size() &&
                       std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
                    ++pos;
                }
                if (pos == text.size()) break;
                if (!text.substr(pos).starts_with(open_tag)) return nullptr;

                const auto content_start = pos + open_tag.size();
                const auto tag_end = text.find(close_tag, content_start);
                if (tag_end == std::string_view::npos) return nullptr;
                auto parsed = nlohmann::json::parse(
                    std::string(text.substr(content_start, tag_end - content_start))
                );
                auto entry = build_tool_call_entry(parsed, id_counter, allowed_tool_names);
                if (entry.is_null()) return nullptr;
                result.push_back(std::move(entry));
                pos = tag_end + close_tag.size();
            }
            return result.empty() ? nullptr : result;
        }

        return parse_plain_json_tool_calls(text, id_counter, allowed_tool_names);
    } catch (const nlohmann::json::exception&) {
        return nullptr;
    }
}

} // namespace

nlohmann::json try_parse_tool_calls(std::string_view text) {
    return try_parse_tool_calls_impl(text, nullptr);
}

nlohmann::json try_parse_tool_calls(
    std::string_view text,
    const std::vector<std::string>& allowed_tool_names
) {
    return try_parse_tool_calls_impl(text, &allowed_tool_names);
}

ToolCallStreamParser::ToolCallStreamParser(std::vector<std::string> allowed_tool_names)
    : _allowed_tool_names(std::move(allowed_tool_names)) {
}

std::vector<ToolCallStreamParser::Event> ToolCallStreamParser::add(
    std::string_view text,
    bool done
) {
    if (_mode == Mode::Content) {
        if (text.empty()) return {};
        return {Content{std::string(text)}};
    }
    if (_mode == Mode::Done) {
        return {};
    }

    _buffer.append(text);

    if (_mode == Mode::Undecided) {
        auto decision = decide(done);
        if (decision == Mode::Undecided) {
            return {};
        }
        if (decision == Mode::Content) {
            _mode = Mode::Content;
            std::string content = std::move(_buffer);
            _buffer.clear();
            if (content.empty()) return {};
            return {Content{std::move(content)}};
        }
        _mode = decision;
    }

    if (!done) return {};

    auto parsed = _allowed_tool_names.has_value()
        ? try_parse_tool_calls(_buffer, *_allowed_tool_names)
        : try_parse_tool_calls(_buffer);
    if (!parsed.is_null()) {
        _mode = Mode::Done;
        _buffer.clear();
        return {ToolCalls{std::move(parsed)}};
    }

    _mode = Mode::Content;
    std::string content = std::move(_buffer);
    _buffer.clear();
    if (content.empty()) return {};
    return {Content{std::move(content)}};
}

ToolCallStreamParser::Mode ToolCallStreamParser::decide(bool done) const {
    const auto stripped = trim_left_view(_buffer);
    if (stripped.empty()) {
        return done ? Mode::Content : Mode::Undecided;
    }

    constexpr std::string_view gemma_marker = "call:";
    constexpr std::string_view qwen_marker = "<tool_call>";

    if (is_prefix_of(stripped, gemma_marker)) {
        return stripped.size() == gemma_marker.size() ? Mode::Gemma : Mode::Undecided;
    }
    if (is_prefix_of(gemma_marker, stripped)) {
        return Mode::Gemma;
    }

    if (is_prefix_of(stripped, qwen_marker)) {
        return stripped.size() == qwen_marker.size() ? Mode::Qwen : Mode::Undecided;
    }
    if (is_prefix_of(qwen_marker, stripped)) {
        return Mode::Qwen;
    }

    if (stripped.front() == '{' || stripped.front() == '[') {
        return Mode::Json;
    }

    return Mode::Content;
}

} // namespace llima
} // namespace simaai
