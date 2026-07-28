#include "tool_call_parser.hpp"

#include "tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <ctime>
#include <regex>
#include <stdexcept>

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

constexpr std::string_view lfm_open = "<|tool_call_start|>";
constexpr std::string_view lfm_close = "<|tool_call_end|>";
constexpr std::string_view gemma_open = "<|tool_call>";
constexpr std::string_view gemma_close = "<tool_call|>";
constexpr std::string_view gemma_call = "call:";
constexpr std::string_view gemma_quote = R"(<|"|>)";
constexpr std::string_view mistral_prefix = "[TOOL_CALLS]";
constexpr std::string_view qwen_open = "<tool_call>";
constexpr std::string_view qwen_close = "</tool_call>";

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

size_t find_matching_brace(
    std::string_view text,
    size_t start,
    std::string_view quote_marker = {}
) {
    if (start >= text.size() || text[start] != '{') return std::string_view::npos;

    int depth = 0;
    bool in_string = false;
    bool in_marker_string = false;
    bool escaped = false;
    for (size_t idx = start; idx < text.size(); ++idx) {
        if (!in_string && !quote_marker.empty() && text.substr(idx).starts_with(quote_marker)) {
            in_marker_string = !in_marker_string;
            idx += quote_marker.size() - 1;
            continue;
        }
        if (in_marker_string) continue;

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
    static const std::regex unquoted_key(
        R"(([{,]\s*)([^{}\[\],:"]*?[^{}\[\],:\s"])\s*:)"
    );
    static const std::regex unquoted_val(R"(:\s*([^{}\[\]",\s][^{}\[\]",]*))");
    std::string protected_text;
    bool in_marker_string = false;
    for (size_t pos = 0; pos < text.size();) {
        if (text.substr(pos).starts_with(gemma_quote)) {
            in_marker_string = !in_marker_string;
            protected_text.append(gemma_quote.data(), gemma_quote.size());
            pos += gemma_quote.size();
        } else {
            protected_text += in_marker_string && text[pos] == ':' ? "\\u003a" :
                std::string(1, text[pos]);
            ++pos;
        }
    }
    std::string normalized_quotes = std::regex_replace(protected_text, quote_marker, "\"");
    std::string with_quoted_keys = std::regex_replace(
        normalized_quotes, unquoted_key, "$1\"$2\":"
    );

    std::string result;
    size_t previous = 0;
    for (std::sregex_iterator it(with_quoted_keys.begin(), with_quoted_keys.end(), unquoted_val), end;
         it != end; ++it) {
        const auto& match = *it;
        const auto match_pos = static_cast<size_t>(match.position());
        const auto value_pos = static_cast<size_t>(match.position(1) - match.position());
        const auto raw_value = match.str(1);
        const auto value = trim_view(raw_value);

        result.append(with_quoted_keys, previous, match_pos - previous);
        result.append(match.str(), 0, value_pos);
        const auto parsed = nlohmann::json::parse(std::string(value), nullptr, false);
        result += parsed.is_discarded()
            ? nlohmann::json(std::string(value)).dump()
            : std::string(value);
        result += raw_value.substr(value.size());
        previous = match_pos + static_cast<size_t>(match.length());
    }
    result.append(with_quoted_keys, previous, std::string::npos);
    return result;
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
    const std::vector<std::string>* allowed_tool_names,
    bool allow_function_wrapper
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
        if (allow_function_wrapper && parsed.is_object() && parsed.size() == 1 &&
            parsed.contains("function") && parsed["function"].is_object()) {
            parsed = parsed["function"];
        }
        auto entry = build_tool_call_entry(parsed, id_counter, allowed_tool_names);
        if (entry.is_null()) return nullptr;
        result.push_back(std::move(entry));
        parsed_any = true;
        pos = close + 1;
    }
    return result.empty() ? nullptr : result;
}

std::vector<std::string_view> split_top_level(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    size_t part_start = 0;
    int paren_depth = 0;
    int square_depth = 0;
    int curly_depth = 0;
    char quote = '\0';
    bool escaped = false;
    for (size_t idx = 0; idx < text.size(); ++idx) {
        const char current = text[idx];
        if (quote != '\0') {
            if (escaped) {
                escaped = false;
            } else if (current == '\\') {
                escaped = true;
            } else if (current == quote) {
                quote = '\0';
            }
            continue;
        }
        if (current == '\'' || current == '"') {
            quote = current;
        } else if (current == '(') {
            ++paren_depth;
        } else if (current == ')') {
            --paren_depth;
        } else if (current == '[') {
            ++square_depth;
        } else if (current == ']') {
            --square_depth;
        } else if (current == '{') {
            ++curly_depth;
        } else if (current == '}') {
            --curly_depth;
        } else if (current == separator && paren_depth == 0 && square_depth == 0 &&
                   curly_depth == 0) {
            parts.push_back(text.substr(part_start, idx - part_start));
            part_start = idx + 1;
        }
        if (paren_depth < 0 || square_depth < 0 || curly_depth < 0) return {};
    }
    if (quote != '\0' || paren_depth != 0 || square_depth != 0 || curly_depth != 0) {
        return {};
    }
    parts.push_back(text.substr(part_start));
    return parts;
}

std::optional<nlohmann::json> parse_python_value(std::string_view value) {
    value = trim_view(value);
    if (value.size() >= 2 && ((value.front() == '\'' && value.back() == '\'') ||
                              (value.front() == '"' && value.back() == '"'))) {
        const char quote = value.front();
        std::string result;
        for (size_t idx = 1; idx + 1 < value.size(); ++idx) {
            if (value[idx] == '\\' && idx + 2 < value.size()) {
                const char escaped = value[++idx];
                switch (escaped) {
                    case '\\': case '\'': case '"': result += escaped; break;
                    case 'n': result += '\n'; break;
                    case 'r': result += '\r'; break;
                    case 't': result += '\t'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'u':
                        if (idx + 4 < value.size() &&
                            std::all_of(value.begin() + idx + 1, value.begin() + idx + 5,
                                        [](char c) { return std::isxdigit(
                                            static_cast<unsigned char>(c)) != 0; })) {
                            const auto decoded = nlohmann::json::parse(
                                "\"\\u" + std::string(value.substr(idx + 1, 4)) + "\"",
                                nullptr, false);
                            if (decoded.is_string()) {
                                result += decoded.get<std::string>();
                                idx += 4;
                                break;
                            }
                        }
                        [[fallthrough]];
                    default:
                        result += '\\';
                        result += escaped;
                        break;
                }
                continue;
            }
            if (value[idx] == quote) return std::nullopt;
            result += value[idx];
        }
        return result;
    }
    if (value == "True") return true;
    if (value == "False") return false;
    if (value == "None") return nlohmann::json(nullptr);
    try {
        return nlohmann::json::parse(std::string(value));
    } catch (const nlohmann::json::exception&) {
        return std::nullopt;
    }
}

nlohmann::json parse_lfm_tool_calls(
    std::string_view text,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    text = trim_view(text);
    if (text.starts_with('[') && text.ends_with(']')) {
        text.remove_prefix(1);
        text.remove_suffix(1);
    }
    const auto calls = split_top_level(text, ',');
    if (calls.empty()) return nullptr;

    nlohmann::json result = nlohmann::json::array();
    for (auto call : calls) {
        call = trim_view(call);
        const auto open = call.find('(');
        if (open == std::string_view::npos || call.empty() || call.back() != ')') {
            return nullptr;
        }
        const auto name = trim_view(call.substr(0, open));
        if (name.empty()) return nullptr;

        nlohmann::json arguments = nlohmann::json::object();
        const auto raw_args = call.substr(open + 1, call.size() - open - 2);
        if (!trim_view(raw_args).empty()) {
            const auto entries = split_top_level(raw_args, ',');
            if (entries.empty()) return nullptr;
            for (auto entry : entries) {
                const auto equal = entry.find('=');
                if (equal == std::string_view::npos) return nullptr;
                const auto key = trim_view(entry.substr(0, equal));
                const auto value = parse_python_value(entry.substr(equal + 1));
                if (key.empty() || !value.has_value()) return nullptr;
                arguments[std::string(key)] = *value;
            }
        }
        auto parsed = build_tool_call_entry(
            {{"name", std::string(name)}, {"arguments", arguments}}, id_counter,
            allowed_tool_names
        );
        if (parsed.is_null()) return nullptr;
        result.push_back(std::move(parsed));
    }
    return result.empty() ? nullptr : result;
}

nlohmann::json parse_gemma_tool_calls(
    std::string_view text,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    if (text.starts_with(gemma_open)) {
        if (!text.ends_with(gemma_close)) return nullptr;
        text.remove_prefix(gemma_open.size());
        text.remove_suffix(gemma_close.size());
        text = trim_view(text);
    }
    if (!text.starts_with(gemma_call)) return nullptr;

    nlohmann::json result = nlohmann::json::array();
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
            ++pos;
        }
        if (pos == text.size()) break;
        if (!text.substr(pos).starts_with(gemma_call)) return nullptr;

        pos += gemma_call.size();
        const auto brace = text.find('{', pos);
        if (brace == std::string_view::npos) return nullptr;
        const auto name = trim_view(text.substr(pos, brace - pos));
        const auto close = find_matching_brace(text, brace, gemma_quote);
        if (close == std::string_view::npos) return nullptr;
        const auto arguments = gemma4_bare_to_json(
            std::string(text.substr(brace, close - brace + 1)));
        auto entry = build_tool_call_entry(
            {{"name", std::string(name)}, {"arguments", arguments}}, id_counter,
            allowed_tool_names);
        if (entry.is_null()) return nullptr;
        result.push_back(std::move(entry));
        pos = close + 1;

        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
            ++pos;
        }
        if (text.substr(pos).starts_with(gemma_close)) {
            pos += gemma_close.size();
            while (pos < text.size() &&
                   std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
                ++pos;
            }
            if (!text.substr(pos).starts_with(gemma_open)) return nullptr;
            pos += gemma_open.size();
        }
    }
    return result.empty() ? nullptr : result;
}

nlohmann::json parse_mistral_tool_calls(
    std::string_view text,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    if (!text.starts_with(mistral_prefix)) return nullptr;
    text = trim_left_view(text.substr(mistral_prefix.size()));
    if (text.empty()) return nullptr;
    return parse_json_array_tool_calls(
        nlohmann::json::parse(std::string(text)), id_counter, allowed_tool_names);
}

nlohmann::json parse_qwen_tool_calls(
    std::string_view text,
    int& id_counter,
    const std::vector<std::string>* allowed_tool_names
) {
    if (!text.starts_with(qwen_open)) return nullptr;

    nlohmann::json result = nlohmann::json::array();
    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) {
            ++pos;
        }
        if (pos == text.size()) break;
        if (!text.substr(pos).starts_with(qwen_open)) return nullptr;

        const auto content_start = pos + qwen_open.size();
        const auto tag_end = text.find(qwen_close, content_start);
        if (tag_end == std::string_view::npos) return nullptr;
        auto entry = build_tool_call_entry(
            nlohmann::json::parse(std::string(text.substr(content_start, tag_end - content_start))),
            id_counter, allowed_tool_names);
        if (entry.is_null()) return nullptr;
        result.push_back(std::move(entry));
        pos = tag_end + qwen_close.size();
    }
    return result.empty() ? nullptr : result;
}

nlohmann::json try_parse_tool_calls_impl(
    ToolCallFormat format,
    std::string_view text,
    const std::vector<std::string>* allowed_tool_names
) {
    text = trim_view(text);
    if (text.empty()) return nullptr;

    int id_counter = 0;
    try {
        switch (format) {
            case ToolCallFormat::Lfm:
                if (!text.starts_with(lfm_open) || !text.ends_with(lfm_close)) return nullptr;
                return parse_lfm_tool_calls(
                    text.substr(lfm_open.size(), text.size() - lfm_open.size() - lfm_close.size()),
                    id_counter, allowed_tool_names);
            case ToolCallFormat::Gemma:
                return parse_gemma_tool_calls(text, id_counter, allowed_tool_names);
            case ToolCallFormat::Mistral:
                return parse_mistral_tool_calls(text, id_counter, allowed_tool_names);
            case ToolCallFormat::Qwen:
                return parse_qwen_tool_calls(text, id_counter, allowed_tool_names);
            case ToolCallFormat::Llama:
                return parse_plain_json_tool_calls(text, id_counter, allowed_tool_names, true);
            case ToolCallFormat::GenericJson:
                if (text.starts_with('[')) {
                    return parse_json_array_tool_calls(
                        nlohmann::json::parse(std::string(text)), id_counter, allowed_tool_names);
                }
                return parse_plain_json_tool_calls(text, id_counter, allowed_tool_names, false);
        }
    } catch (const nlohmann::json::exception&) {
        return nullptr;
    }
    return nullptr;
}

} // namespace

nlohmann::json try_parse_tool_calls(
    ToolCallFormat format,
    std::string_view text,
    const std::vector<std::string>& allowed_tool_names
) {
    return try_parse_tool_calls_impl(format, text, &allowed_tool_names);
}

ToolCallFormat tool_call_format_for_model(std::string_view model_type) {
    if (model_type == "llm-lfm2" || model_type == "vlm-lfm2_vl") {
        return ToolCallFormat::Lfm;
    }
    if (model_type == "vlm-gemma4") {
        return ToolCallFormat::Gemma;
    }
    if (model_type == "llm-mistral") {
        return ToolCallFormat::Mistral;
    }
    if (model_type == "llm-qwen2" || model_type == "llm-qwen3" ||
        model_type == "vlm-qwen2_5_vl" || model_type == "vlm-qwen3_vl") {
        return ToolCallFormat::Qwen;
    }
    if (model_type == "llm-llama") {
        return ToolCallFormat::Llama;
    }
    return ToolCallFormat::GenericJson;
}

std::vector<std::string> tool_call_special_tokens(ToolCallFormat format) {
    switch (format) {
        case ToolCallFormat::Lfm:
            return {std::string(lfm_open), std::string(lfm_close)};
        case ToolCallFormat::Gemma:
            return {std::string(gemma_open), std::string(gemma_close), R"(<|"|>)"};
        case ToolCallFormat::Mistral:
            return {std::string(mistral_prefix)};
        default:
            return {};
    }
}

PreservedToolCallTokens resolve_tool_call_special_tokens(
    ToolCallFormat format,
    const Tokenizer& tokenizer
) {
    PreservedToolCallTokens resolved;
    for (const auto& token : tool_call_special_tokens(format)) {
        try {
            resolved.emplace_back(tokenizer.token_to_id(token), token);
        } catch (const std::exception&) {
            throw std::runtime_error(
                "Tool calling protocol token '" + token + "' is missing from the loaded tokenizer"
            );
        }
    }
    return resolved;
}

std::string decode_tool_call_output(
    const Tokenizer& tokenizer,
    const std::vector<uint32_t>& token_ids,
    const PreservedToolCallTokens& preserved_tokens
) {
    std::string result;
    std::vector<uint32_t> normal_tokens;
    const auto flush_normal = [&]() {
        if (!normal_tokens.empty()) {
            result += tokenizer.decode(normal_tokens, true);
            normal_tokens.clear();
        }
    };

    for (const auto token_id : token_ids) {
        const auto marker = std::find_if(
            preserved_tokens.begin(), preserved_tokens.end(),
            [token_id](const auto& entry) { return entry.first == token_id; }
        );
        if (marker == preserved_tokens.end()) {
            normal_tokens.push_back(token_id);
            continue;
        }
        flush_normal();
        result += marker->second;
    }
    flush_normal();
    return result;
}

ToolCallStreamParser::ToolCallStreamParser(ToolCallFormat format) : _format(format) {
}

ToolCallStreamParser::ToolCallStreamParser(
    ToolCallFormat format,
    std::vector<std::string> allowed_tool_names
) : _format(format), _allowed_tool_names(std::move(allowed_tool_names)) {
}

std::vector<ToolCallStreamParser::Event> ToolCallStreamParser::add(
    std::string_view text,
    bool done,
    bool from_draft
) {
    if (_mode == Mode::Content) {
        if (text.empty()) return {};
        return {Content{std::string(text), from_draft}};
    }
    if (_mode == Mode::Done) {
        return {};
    }

    _buffer.append(text);
    buffer_content(text, from_draft);

    if (_mode == Mode::Undecided) {
        auto decision = decide(done);
        if (decision == Mode::Undecided) {
            return {};
        }
        if (decision == Mode::Content) {
            _mode = Mode::Content;
            _buffer.clear();
            return take_buffered_content();
        }
        _mode = decision;
    }

    if (!done) return {};

    const auto parsed = try_parse_tool_calls_impl(
        _format, _buffer, _allowed_tool_names.has_value() ? &*_allowed_tool_names : nullptr);
    if (!parsed.is_null()) {
        _mode = Mode::Done;
        _buffer.clear();
        _content_buffer.clear();
        return {ToolCalls{std::move(parsed)}};
    }

    _mode = Mode::Content;
    _buffer.clear();
    return take_buffered_content();
}

void ToolCallStreamParser::buffer_content(std::string_view text, bool from_draft) {
    if (text.empty()) return;
    if (!_content_buffer.empty() && _content_buffer.back().from_draft == from_draft) {
        _content_buffer.back().text.append(text);
        return;
    }
    _content_buffer.push_back(Content{std::string(text), from_draft});
}

std::vector<ToolCallStreamParser::Event> ToolCallStreamParser::take_buffered_content() {
    std::vector<Event> events;
    events.reserve(_content_buffer.size());
    for (auto& content : _content_buffer) {
        events.emplace_back(std::move(content));
    }
    _content_buffer.clear();
    return events;
}

ToolCallStreamParser::Mode ToolCallStreamParser::decide(bool done) const {
    const auto stripped = trim_left_view(_buffer);
    if (stripped.empty()) {
        return done ? Mode::Content : Mode::Undecided;
    }

    const auto marker_mode = [&](std::string_view marker) {
        if (is_prefix_of(stripped, marker)) {
            return stripped.size() == marker.size() ? Mode::ToolCall
                : done ? Mode::Content : Mode::Undecided;
        }
        return is_prefix_of(marker, stripped) ? Mode::ToolCall : Mode::Content;
    };

    switch (_format) {
        case ToolCallFormat::Lfm:
            return marker_mode(lfm_open);
        case ToolCallFormat::Gemma: {
            const auto wrapper_mode = marker_mode(gemma_open);
            return wrapper_mode == Mode::Content ? marker_mode(gemma_call) : wrapper_mode;
        }
        case ToolCallFormat::Mistral:
            return marker_mode(mistral_prefix);
        case ToolCallFormat::Qwen:
            return marker_mode(qwen_open);
        case ToolCallFormat::Llama:
            return stripped.front() == '{' ? Mode::ToolCall : Mode::Content;
        case ToolCallFormat::GenericJson:
            return stripped.front() == '{' || stripped.front() == '[' ? Mode::ToolCall
                                                                       : Mode::Content;
    }
    return Mode::Content;
}

} // namespace llima
} // namespace simaai
