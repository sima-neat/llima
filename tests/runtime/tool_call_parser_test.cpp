#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "tool_call_parser.hpp"

namespace {

using simaai::llima::ToolCallFormat;
using simaai::llima::ToolCallStreamParser;
using simaai::llima::try_parse_tool_calls;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

nlohmann::json arguments_from(const nlohmann::json& calls) {
    return nlohmann::json::parse(
        calls.at(0).at("function").at("arguments").get<std::string>()
    );
}

void test_rejects_premature_lfm_string_close() {
    const std::vector<std::string> allowed_tools = {"send"};
    const auto malformed = try_parse_tool_calls(
        ToolCallFormat::Lfm,
        "<|tool_call_start|>[send(message='a' + 'b')]<|tool_call_end|>",
        allowed_tools
    );
    expect(
        malformed.is_null(),
        "LFM values with an unescaped closing quote must fail closed"
    );

    const auto escaped = try_parse_tool_calls(
        ToolCallFormat::Lfm,
        R"(<|tool_call_start|>[send(message='a\'b')]<|tool_call_end|>)",
        allowed_tools
    );
    expect(!escaped.is_null(), "escaped LFM quotes must remain valid");
    if (!escaped.is_null()) {
        expect(
            arguments_from(escaped).at("message") == "a'b",
            "escaped LFM quotes must be decoded"
        );
    }
}

void test_quotes_complete_gemma_keys() {
    const auto calls = try_parse_tool_calls(
        ToolCallFormat::Gemma,
        R"(<|tool_call>call:configure{user-id:<|"|>123<|"|>,endpoint-url:<|"|>https://example.com/a:b<|"|>,options:{retry-count:7}}<tool_call|>)",
        {"configure"}
    );
    expect(!calls.is_null(), "Gemma keys containing hyphens must parse");
    if (calls.is_null()) return;

    const auto arguments = arguments_from(calls);
    expect(arguments.at("user-id") == "123", "the complete Gemma key must be preserved");
    expect(
        arguments.at("endpoint-url") == "https://example.com/a:b",
        "Gemma marker strings and complete keys must both be preserved"
    );
    expect(
        arguments.at("options").at("retry-count") == 7,
        "nested Gemma keys containing hyphens must be preserved"
    );
}

void test_preserves_stream_content_provenance() {
    ToolCallStreamParser parser(ToolCallFormat::Lfm, {"send"});

    const auto undecided = parser.add("<", false, true);
    expect(undecided.empty(), "a partial LFM marker must remain buffered");

    const auto content = parser.add("ordinary", false, false);
    expect(content.size() == 2, "buffered draft and target chunks must remain separate");
    if (content.size() == 2) {
        const auto* draft = std::get_if<ToolCallStreamParser::Content>(&content[0]);
        const auto* target = std::get_if<ToolCallStreamParser::Content>(&content[1]);
        expect(
            draft != nullptr && draft->text == "<" && draft->from_draft,
            "the buffered draft chunk must retain its provenance"
        );
        expect(
            target != nullptr && target->text == "ordinary" && !target->from_draft,
            "the target chunk must retain its provenance"
        );
    }

    const auto later_draft = parser.add(" draft", false, true);
    expect(later_draft.size() == 1, "content mode must emit subsequent chunks");
    if (later_draft.size() == 1) {
        const auto* event = std::get_if<ToolCallStreamParser::Content>(&later_draft[0]);
        expect(
            event != nullptr && event->text == " draft" && event->from_draft,
            "content mode must propagate draft provenance"
        );
    }
}

} // namespace

int main() {
    test_rejects_premature_lfm_string_close();
    test_quotes_complete_gemma_keys();
    test_preserves_stream_content_provenance();

    if (failures != 0) {
        std::cerr << failures << " tool-call parser assertion(s) failed\n";
        return 1;
    }
    std::cout << "Tool-call parser tests passed\n";
    return 0;
}
