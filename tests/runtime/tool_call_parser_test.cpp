#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "tool_call_parser.hpp"

namespace {

using simaai::llima::ToolCallFormat;
using simaai::llima::ToolCallStreamParser;
using simaai::llima::tool_call_format_for_model;
using simaai::llima::tool_call_special_tokens;
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
        parser.add(R"({"tool_)", false).empty(),
        "a partial Gemma JSON envelope must remain buffered"
    );
    const auto events = parser.add(
        R"(calls":[{"name":"set_ac","arguments":{"enabled":false}}],"content":""})",
        true
    );
    expect(events.size() == 1, "a complete Gemma JSON envelope must emit one event");
    if (events.size() != 1) return;
    const auto* calls = std::get_if<ToolCallStreamParser::ToolCalls>(&events[0]);
    expect(calls != nullptr, "the Gemma JSON envelope event must be structured tool calls");
    if (calls != nullptr) {
        expect(calls->calls.size() == 1, "the streamed Gemma envelope call must be preserved");
    }

    for (const auto content_value :
         {std::string_view(R"("")"), std::string_view("null")}) {
        ToolCallStreamParser content_first_parser(ToolCallFormat::Gemma, {"set_ac"});
        const std::string prefix =
            std::string(R"({"content":)") + std::string(content_value) + R"(,"tool_)";
        expect(
            content_first_parser.add(prefix, false).empty(),
            "a content-first Gemma envelope prefix must remain buffered"
        );
        const auto content_first_events = content_first_parser.add(
            R"(calls":[{"name":"set_ac","arguments":{"enabled":false}}]})",
            true
        );
        expect(
            content_first_events.size() == 1,
            "a content-first Gemma envelope must emit one event"
        );
        if (content_first_events.size() == 1) {
            const auto* content_first_calls =
                std::get_if<ToolCallStreamParser::ToolCalls>(&content_first_events[0]);
            expect(
                content_first_calls != nullptr && content_first_calls->calls.size() == 1,
                "empty and null content-first envelopes must emit structured tool calls"
            );
        }
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

void test_streams_non_tool_gemma_json_without_waiting_for_end() {
    ToolCallStreamParser parser(ToolCallFormat::Gemma, {"set_ac"});
    expect(
        parser.add("{", false).empty(),
        "an opening JSON brace must remain undecided"
    );

    const auto events = parser.add(R"("answer":42})", false);
    expect(
        events.size() == 1,
        "a non-tool Gemma JSON key must release buffered content before the stream ends"
    );
    if (events.size() == 1) {
        const auto* content = std::get_if<ToolCallStreamParser::Content>(&events[0]);
        expect(
            content != nullptr && content->text == R"({"answer":42})",
            "the released Gemma JSON content must preserve all buffered bytes"
        );
    }

    ToolCallStreamParser content_parser(ToolCallFormat::Gemma, {"set_ac"});
    expect(
        content_parser.add(R"({"content":")", false).empty(),
        "a content value that may still be empty must remain undecided"
    );
    const auto content_events = content_parser.add(R"(hello"})", false);
    expect(
        content_events.size() == 1,
        "non-empty content must be released before the stream ends"
    );
    if (content_events.size() == 1) {
        const auto* content =
            std::get_if<ToolCallStreamParser::Content>(&content_events[0]);
        expect(
            content != nullptr && content->text == R"({"content":"hello"})",
            "non-empty content-first JSON must preserve all buffered bytes"
        );
    }
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

void test_parses_qwen35_xml_tool_calls() {
    const auto calls = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call>
<function=set_ac_temperature>
<parameter=temperature_celsius>
30
</parameter>
</function>
</tool_call>
<tool_call>
<function=set_ac>
<parameter=enabled>
true
</parameter>
</function>
</tool_call>)",
        {"set_ac_temperature", "set_ac"}
    );
    expect(!calls.is_null(), "Qwen3.5 function/parameter XML must parse");
    if (calls.is_null()) return;

    expect(calls.size() == 2, "all Qwen3.5 XML calls must be preserved");
    expect(
        calls.at(0).at("function").at("name") == "set_ac_temperature" &&
            arguments_from(calls).at("temperature_celsius") == 30,
        "Qwen3.5 numeric parameters must preserve their JSON type"
    );
    expect(
        calls.at(1).at("function").at("name") == "set_ac" &&
            nlohmann::json::parse(
                calls.at(1).at("function").at("arguments").get<std::string>()
            ).at("enabled") == true,
        "Qwen3.5 boolean parameters must preserve their JSON type"
    );

    const auto string_call = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call>
<function=send_message>
<parameter=message>
line one
line two
</parameter>
</function>
</tool_call>)",
        {"send_message"}
    );
    expect(!string_call.is_null(), "Qwen3.5 multiline string parameters must parse");
    if (!string_call.is_null()) {
        expect(
            arguments_from(string_call).at("message") == "line one\nline two",
            "Qwen3.5 multiline string parameters must preserve internal newlines"
        );
    }

    const auto no_argument_call = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call><function=get_hvac_status></function></tool_call>)",
        {"get_hvac_status"}
    );
    expect(
        !no_argument_call.is_null() && arguments_from(no_argument_call).empty(),
        "Qwen3.5 functions without parameters must produce empty arguments"
    );
}

void test_rejects_invalid_qwen35_xml_tool_calls() {
    const auto unknown_tool = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call><function=delete_everything></function></tool_call>)",
        {"set_ac"}
    );
    expect(unknown_tool.is_null(), "Qwen3.5 XML must enforce the tool allowlist");

    const auto missing_parameter_close = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call><function=set_ac><parameter=enabled>true</function></tool_call>)",
        {"set_ac"}
    );
    expect(
        missing_parameter_close.is_null(),
        "malformed Qwen3.5 parameter XML must fail closed"
    );

    const auto duplicate_parameter = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call><function=set_ac><parameter=enabled>true</parameter><parameter=enabled>false</parameter></function></tool_call>)",
        {"set_ac"}
    );
    expect(
        duplicate_parameter.is_null(),
        "Qwen3.5 XML with duplicate parameters must fail closed"
    );

    const auto trailing_prose = try_parse_tool_calls(
        ToolCallFormat::Qwen35,
        R"(<tool_call><function=set_ac><parameter=enabled>true</parameter></function></tool_call> done)",
        {"set_ac"}
    );
    expect(trailing_prose.is_null(), "Qwen3.5 XML with trailing prose must fail closed");
}

void test_streams_qwen35_xml_tool_calls() {
    ToolCallStreamParser parser(ToolCallFormat::Qwen35, {"set_fan_speed"});
    expect(
        parser.add("<tool_", false).empty(),
        "a partial Qwen3.5 tool marker must remain buffered"
    );
    const auto events = parser.add(
        R"(call>
<function=set_fan_speed>
<parameter=level>
3
</parameter>
</function>
</tool_call>)",
        true
    );
    expect(events.size() == 1, "a complete Qwen3.5 XML call must emit one event");
    if (events.size() == 1) {
        const auto* calls = std::get_if<ToolCallStreamParser::ToolCalls>(&events[0]);
        expect(
            calls != nullptr && calls->calls.size() == 1 &&
                arguments_from(calls->calls).at("level") == 3,
            "the streamed Qwen3.5 XML event must contain structured tool calls"
        );
    }
}

void test_maps_qwen35_model_type() {
    expect(
        tool_call_format_for_model("vlm-qwen3_5") == ToolCallFormat::Qwen35,
        "vlm-qwen3_5 must use the Qwen3.5 XML parser"
    );
    expect(
        tool_call_format_for_model("vlm-qwen3_vl") == ToolCallFormat::Qwen,
        "existing Qwen model types must retain the JSON parser"
    );
}

void test_preserves_qwen35_tool_call_tokens() {
    const auto tokens = tool_call_special_tokens(ToolCallFormat::Qwen35);
    expect(
        tokens == std::vector<std::string>{"<tool_call>", "</tool_call>"},
        "Qwen3.5 opening and closing tool-call tokens must survive streaming decode"
    );
    expect(
        tool_call_special_tokens(ToolCallFormat::Qwen).empty(),
        "legacy Qwen token preservation must remain unchanged"
    );
}

} // namespace

int main() {
    test_rejects_premature_lfm_string_close();
    test_quotes_complete_gemma_keys();
    test_parses_gemma_json_tool_call_envelope();
    test_rejects_unsafe_gemma_json_tool_call_envelopes();
    test_streams_gemma_json_tool_call_envelope();
    test_streams_non_tool_gemma_json_without_waiting_for_end();
    test_preserves_stream_content_provenance();
    test_parses_qwen35_xml_tool_calls();
    test_rejects_invalid_qwen35_xml_tool_calls();
    test_streams_qwen35_xml_tool_calls();
    test_maps_qwen35_model_type();
    test_preserves_qwen35_tool_call_tokens();

    if (failures != 0) {
        std::cerr << failures << " tool-call parser assertion(s) failed\n";
        return 1;
    }
    std::cout << "Tool-call parser tests passed\n";
    return 0;
}
