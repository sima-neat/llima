#include "tool_call_parser.hpp"

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

bool is_prefix_of(std::string_view prefix, std::string_view text) {
    return prefix.size() <= text.size() && text.substr(0, prefix.size()) == prefix;
}

nlohmann::json build_tool_call_entry(const nlohmann::json& parsed, int& id_counter) {
    if (!parsed.contains("name") || !parsed["name"].is_string())
        return nullptr;

    auto extract_args = [](const nlohmann::json& val) -> nlohmann::json {
        if (val.is_object()) return val;
        if (val.is_string()) {
            try { return nlohmann::json::parse(val.get<std::string>()); } catch (...) {}
        }
        return nlohmann::json::object();
    };
    nlohmann::json args = nlohmann::json::object();
    if (parsed.contains("arguments"))
        args = extract_args(parsed["arguments"]);
    else if (parsed.contains("parameters"))
        args = extract_args(parsed["parameters"]);

    std::string id = parsed.contains("id") && parsed["id"].is_string()
        ? parsed["id"].get<std::string>()
        : std::to_string(std::time(nullptr)).substr(2) + std::to_string(id_counter++ % 10);

    return {
        {"id", id},
        {"type", "function"},
        {"function", {{"name", parsed["name"]}, {"arguments", args.dump()}}}
    };
}

size_t find_matching_brace(std::string_view text, size_t start) {
    int depth = 0;
    for (size_t idx = start; idx < text.size(); ++idx) {
        if (text[idx] == '{') {
            ++depth;
        } else if (text[idx] == '}' && --depth == 0) {
            return idx;
        }
    }
    return std::string_view::npos;
}

std::string gemma4_bare_to_json(const std::string& text) {
    static const std::regex unquoted_key(R"((\w+)\s*:)");
    static const std::regex unquoted_val(R"(:\s*([^{}\[\]",\s][^{}\[\]",]*))");
    std::string normalized_quotes = std::regex_replace(text, std::regex(R"(<\|"\|>)"), "\"");
    std::string with_quoted_keys = std::regex_replace(normalized_quotes, unquoted_key, "\"$1\":");
    return std::regex_replace(with_quoted_keys, unquoted_val, ":\"$1\"");
}

nlohmann::json parse_plain_json_tool_calls(std::string_view text, int& id_counter) {
    nlohmann::json result = nlohmann::json::array();
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() &&
               (std::isspace(static_cast<unsigned char>(text[pos])) != 0 || text[pos] == ';')) {
            ++pos;
        }
        if (pos == text.size()) break;
        if (text[pos] != '{') return nullptr;
        auto close = find_matching_brace(text, pos);
        if (close == std::string_view::npos) return nullptr;
        auto parsed = nlohmann::json::parse(std::string(text.substr(pos, close - pos + 1)));
        auto entry = build_tool_call_entry(parsed, id_counter);
        if (entry.is_null()) return nullptr;
        result.push_back(entry);
        pos = close + 1;
    }
    return result.empty() ? nullptr : result;
}

} // namespace

nlohmann::json try_parse_tool_calls(std::string_view text) {
    text = trim_left_view(text);
    int id_counter = 0;
    try {
        if (text.starts_with("call:")) {
            nlohmann::json result = nlohmann::json::array();
            size_t pos = 0;
            while (pos < text.size() && text.substr(pos).starts_with("call:")) {
                pos += 5;
                auto brace = text.find('{', pos);
                if (brace == std::string_view::npos) return nullptr;
                std::string_view name = text.substr(pos, brace - pos);
                auto close = find_matching_brace(text, brace);
                if (close == std::string_view::npos) return nullptr;
                std::string args =
                    gemma4_bare_to_json(std::string(text.substr(brace, close - brace + 1)));
                auto entry = build_tool_call_entry(
                    {{"name", std::string(name)}, {"arguments", args}}, id_counter
                );
                if (entry.is_null()) return nullptr;
                result.push_back(entry);
                pos = close + 1;
            }
            return result;
        }

        const std::string mistral_prefix = "[TOOL_CALLS] ";
        auto mistral_pos = text.find(mistral_prefix);
        if (mistral_pos != std::string::npos) {
            auto array_start = mistral_pos + mistral_prefix.size();
            auto parsed = nlohmann::json::parse(std::string(text.substr(array_start)));
            if (!parsed.is_array()) return nullptr;
            nlohmann::json result = nlohmann::json::array();
            for (const auto& item : parsed) {
                auto entry = build_tool_call_entry(item, id_counter);
                if (entry.is_null()) return nullptr;
                result.push_back(entry);
            }
            return result;
        }

        if (text.starts_with('[')) {
            auto parsed = nlohmann::json::parse(std::string(text));
            if (parsed.is_array()) {
                nlohmann::json result = nlohmann::json::array();
                for (const auto& item : parsed) {
                    auto entry = build_tool_call_entry(item, id_counter);
                    if (entry.is_null()) return nullptr;
                    result.push_back(entry);
                }
                return result;
            }
        }

        if (text.find("<tool_call>") != std::string_view::npos) {
            nlohmann::json result = nlohmann::json::array();
            std::size_t search_pos = 0;
            while (true) {
                auto tag_start = text.find("<tool_call>", search_pos);
                auto tag_end = text.find("</tool_call>", search_pos);
                if (tag_start == std::string_view::npos || tag_end == std::string_view::npos)
                    break;
                constexpr std::size_t open_tag_len = 11;  // "<tool_call>"
                constexpr std::size_t close_tag_len = 12; // "</tool_call>"
                tag_start += open_tag_len;
                auto parsed = nlohmann::json::parse(
                    std::string(text.substr(tag_start, tag_end - tag_start))
                );
                auto entry = build_tool_call_entry(parsed, id_counter);
                if (entry.is_null()) return nullptr;
                result.push_back(entry);
                search_pos = tag_end + close_tag_len;
            }
            if (!result.empty()) return result;
        }

        return parse_plain_json_tool_calls(text, id_counter);
    } catch (...) {
        return nullptr;
    }
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

    auto parsed = try_parse_tool_calls(_buffer);
    if (!parsed.is_null()) {
        _mode = Mode::Done;
        _buffer.clear();
        return {ToolCalls{std::move(parsed)}};
    }

    if (done) {
        _mode = Mode::Content;
        std::string content = std::move(_buffer);
        _buffer.clear();
        if (content.empty()) return {};
        return {Content{std::move(content)}};
    }

    return {};
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
