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


void test_parses_gemma_json_tool_call_envelope() {
    const auto calls = try_parse_tool_calls(
        ToolCallFormat::Gemma,
        R"({"tool_calls":[{"name":"set_fan_speed","arguments":{"level":3}},{"name":"set_ac","arguments":{"enabled":false}}],"content":""})",
        {"set_fan_speed", "set_ac"}
    );
    expect(!calls.is_null(), "Gemma JSON tool-call envelopes must parse");
    if (calls.is_null()) return;

    expect(calls.size() == 2, "all Gemma JSON envelope calls must be preserved");
    expect(
        calls.at(0).at("function").at("name") == "set_fan_speed" &&
            arguments_from(calls).at("level") == 3,
        "the first Gemma JSON envelope call must preserve its arguments"
    );
    expect(
        calls.at(1).at("function").at("name") == "set_ac" &&
            nlohmann::json::parse(
                calls.at(1).at("function").at("arguments").get<std::string>()
            ).at("enabled") == false,
        "the second Gemma JSON envelope call must preserve its arguments"
    );

    const auto wrapped_calls = try_parse_tool_calls(
        ToolCallFormat::Gemma,
        R"(<|tool_call>{"tool_calls":[{"name":"set_ac","arguments":{"enabled":false}}],"content":""}<tool_call|>)",
        {"set_ac"}
    );
    expect(
        !wrapped_calls.is_null() && wrapped_calls.size() == 1 &&
            wrapped_calls.at(0).at("function").at("name") == "set_ac",
        "wrapped Gemma JSON tool-call envelopes must parse"
    );
}

void test_rejects_unsafe_gemma_json_tool_call_envelopes() {
    const auto mixed_content = try_parse_tool_calls(
        ToolCallFormat::Gemma,
        R"({"tool_calls":[{"name":"set_ac","arguments":{"enabled":false}}],"content":"I also answered in prose"})",
        {"set_ac"}
    );
    expect(mixed_content.is_null(), "Gemma JSON envelopes with prose must fail closed");

    const auto unknown_tool = try_parse_tool_calls(
        ToolCallFormat::Gemma,
        R"({"tool_calls":[{"name":"delete_everything","arguments":{}}],"content":""})",
        {"set_ac"}
    );
    expect(unknown_tool.is_null(), "Gemma JSON envelopes must enforce the tool allowlist");
}

void test_streams_gemma_json_tool_call_envelope() {
    ToolCallStreamParser parser(ToolCallFormat::Gemma, {"set_ac"});
    expect(
        parser.add(R"({"tool_calls":)", false).empty(),
        "a partial Gemma JSON envelope must remain buffered"
    );
    const auto events = parser.add(
        R"([{"name":"set_ac","arguments":{"enabled":false}}],"content":""})",
        true
    );
    expect(events.size() == 1, "a complete Gemma JSON envelope must emit one event");
    if (events.size() != 1) return;
    const auto* calls = std::get_if<ToolCallStreamParser::ToolCalls>(&events[0]);
    expect(calls != nullptr, "the Gemma JSON envelope event must be structured tool calls");
    if (calls != nullptr) {
        expect(calls->calls.size() == 1, "the streamed Gemma envelope call must be preserved");
    }

    ToolCallStreamParser wrapped_parser(ToolCallFormat::Gemma, {"set_ac"});
    expect(
        wrapped_parser.add(R"(<|tool_call>{"tool_calls":)", false).empty(),
        "a partial wrapped Gemma JSON envelope must remain buffered"
    );
    const auto wrapped_events = wrapped_parser.add(
        R"([{"name":"set_ac","arguments":{"enabled":false}}],"content":""}<tool_call|>)",
        true
    );
    expect(
        wrapped_events.size() == 1,
        "a complete wrapped Gemma JSON envelope must emit one event"
    );
    if (wrapped_events.size() != 1) return;
    const auto* wrapped_calls =
        std::get_if<ToolCallStreamParser::ToolCalls>(&wrapped_events[0]);
    expect(
        wrapped_calls != nullptr && wrapped_calls->calls.size() == 1,
        "the streamed wrapped Gemma envelope must be structured tool calls"
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
    test_parses_gemma_json_tool_call_envelope();
    test_rejects_unsafe_gemma_json_tool_call_envelopes();
    test_streams_gemma_json_tool_call_envelope();
    test_preserves_stream_content_provenance();

    if (failures != 0) {
        std::cerr << failures << " tool-call parser assertion(s) failed\n";
        return 1;
    }
    std::cout << "Tool-call parser tests passed\n";
    return 0;
}
