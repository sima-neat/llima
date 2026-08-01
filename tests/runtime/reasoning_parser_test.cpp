#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include "reasoning_parser.hpp"

using simaai::llima::ReasoningFormat;
using simaai::llima::ReasoningStreamParser;
using simaai::llima::reasoning_format_for_model;

namespace {

struct ParsedText {
    std::string reasoning;
    std::string content;
    std::vector<bool> reasoning_provenance;
    std::vector<bool> content_provenance;
};

void append_events(
    ParsedText& parsed,
    std::vector<ReasoningStreamParser::Event> events
) {
    for (auto& event : events) {
        if (event.reasoning) {
            parsed.reasoning += event.text;
            parsed.reasoning_provenance.push_back(event.from_draft);
        } else {
            parsed.content += event.text;
            parsed.content_provenance.push_back(event.from_draft);
        }
    }
}

void assert_no_markers(const ParsedText& parsed) {
    const auto combined = parsed.reasoning + parsed.content;
    assert(combined.find("<think>") == std::string::npos);
    assert(combined.find("</think>") == std::string::npos);
    assert(combined.find("<|channel>") == std::string::npos);
    assert(combined.find("<channel|>") == std::string::npos);
}

void test_model_mapping() {
    assert(reasoning_format_for_model("llm-qwen3") == ReasoningFormat::Qwen);
    assert(reasoning_format_for_model("vlm-qwen3_vl") == ReasoningFormat::Qwen);
    assert(reasoning_format_for_model("llm-qwen3_5") == ReasoningFormat::Qwen);
    assert(reasoning_format_for_model("vlm-qwen3_5_vl") == ReasoningFormat::Qwen);
    assert(reasoning_format_for_model("vlm-gemma4") == ReasoningFormat::Gemma4);
    assert(reasoning_format_for_model("llm-qwen2") == ReasoningFormat::None);
}

void test_qwen_generated_open_and_split_close() {
    ReasoningStreamParser parser(ReasoningFormat::Qwen, true);
    ParsedText parsed;
    append_events(parsed, parser.add("<thi"));
    append_events(parsed, parser.add("nk>reasoning </thi", false, true));
    append_events(parsed, parser.add("nk>final", true));

    assert(parsed.reasoning == "reasoning ");
    assert(parsed.content == "final");
    assert(parsed.reasoning_provenance == std::vector<bool>({true}));
    assert(parsed.content_provenance == std::vector<bool>({false}));
    assert_no_markers(parsed);
}

void test_qwen_prompt_opened_and_truncated() {
    ReasoningStreamParser complete(ReasoningFormat::Qwen, true);
    ParsedText parsed;
    append_events(parsed, complete.add("reasoning", false, true));
    append_events(parsed, complete.add("</think>answer", true, false));
    assert(parsed.reasoning == "reasoning");
    assert(parsed.content == "answer");
    assert(parsed.reasoning_provenance == std::vector<bool>({true}));
    assert(parsed.content_provenance == std::vector<bool>({false}));

    ReasoningStreamParser truncated(ReasoningFormat::Qwen, true);
    ParsedText truncated_text;
    append_events(truncated_text, truncated.add("unfinished</thi", true));
    assert(truncated_text.reasoning == "unfinished");
    assert(truncated_text.content.empty());
    assert_no_markers(truncated_text);
}

void test_gemma_generated_and_prompt_opened_boundaries() {
    ReasoningStreamParser generated(ReasoningFormat::Gemma4, true);
    ParsedText parsed;
    append_events(parsed, generated.add("<|channel>tho"));
    append_events(parsed, generated.add("ught\nreason", false, true));
    append_events(parsed, generated.add("ing<channel"));
    append_events(parsed, generated.add("|>answer", true));
    assert(parsed.reasoning == "reasoning");
    assert(parsed.content == "answer");
    assert(parsed.reasoning_provenance == std::vector<bool>({true, false}));
    assert_no_markers(parsed);

    ReasoningStreamParser prompt_opened(ReasoningFormat::Gemma4, true, true);
    ParsedText continued;
    append_events(continued, prompt_opened.add("tool result reasoning<channel|>done", true));
    assert(continued.reasoning == "tool result reasoning");
    assert(continued.content == "done");
    assert_no_markers(continued);
}

void test_disabled_and_unsupported_pass_through() {
    for (auto parser : {
             ReasoningStreamParser(ReasoningFormat::Qwen, false),
             ReasoningStreamParser(ReasoningFormat::None, true),
         }) {
        ParsedText parsed;
        append_events(parsed, parser.add("ordinary answer", true, true));
        assert(parsed.reasoning.empty());
        assert(parsed.content == "ordinary answer");
        assert(parsed.content_provenance == std::vector<bool>({true}));
    }
}

} // namespace

int main() {
    test_model_mapping();
    test_qwen_generated_open_and_split_close();
    test_qwen_prompt_opened_and_truncated();
    test_gemma_generated_and_prompt_opened_boundaries();
    test_disabled_and_unsupported_pass_through();
    std::cout << "reasoning parser tests passed\n";
    return 0;
}
