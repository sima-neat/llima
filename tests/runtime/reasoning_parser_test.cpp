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
    assert(combined.find("<|channel|>") == std::string::npos);
    assert(combined.find("<|message|>") == std::string::npos);
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
    append_events(parsed, parser.add("\n<thi"));
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
    append_events(parsed, generated.add("\n<|channel>tho"));
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

// Gemma 4 can emit thought channels after tool results even when thinking is disabled.
void test_gemma_hides_its_own_thought_when_thinking_is_off() {
    ReasoningStreamParser after_tools(ReasoningFormat::Gemma4, false);
    ParsedText parsed;
    append_events(parsed, after_tools.add("<|chan"));
    append_events(parsed, after_tools.add("nel>thought\n"));
    append_events(parsed, after_tools.add("<channel|>The file is saved.", true));
    assert(parsed.reasoning.empty());
    assert(parsed.content == "The file is saved.");
    assert_no_markers(parsed);

    ReasoningStreamParser with_text(ReasoningFormat::Gemma4, false);
    ParsedText hidden;
    append_events(hidden, with_text.add("<|channel>thought\nlet me check<channel|>Done.", true));
    assert(hidden.reasoning.empty());
    assert(hidden.content == "Done.");

    ReasoningStreamParser call(ReasoningFormat::Gemma4, false);
    ParsedText calling;
    append_events(calling, call.add("<|tool_call>call:x{}<tool_call|>", true));
    assert(calling.content == "<|tool_call>call:x{}<tool_call|>");

    // Ordinary output is preserved exactly, however it is chunked.
    const auto run = [](const std::vector<std::string>& chunks) {
        ReasoningStreamParser parser(ReasoningFormat::Gemma4, false);
        ParsedText out;
        for (size_t i = 0; i < chunks.size(); ++i) {
            append_events(out, parser.add(chunks[i], i + 1 == chunks.size()));
        }
        assert(out.reasoning.empty());
        return out.content;
    };
    assert(run({"\n    return 42;"}) == "\n    return 42;");
    assert(run({"\n  ", "return 42;"}) == "\n  return 42;");
    assert(run({" \n\t"}) == " \n\t");
    assert(run({"<"}) == "<");
    assert(run({"\n<|chan"}) == "\n<|chan");
    assert(run({"<|tool_call>call:x{}<tool_call|>"}) == "<|tool_call>call:x{}<tool_call|>");
    // A thought channel preceded by whitespace is still hidden.
    assert(run({"\n ", "<|channel>thought\n<channel|>Done."}) == "Done.");
    assert(run({"\n<|channel>thought\nhmm<channel|>Done."}) == "Done.");

    // Buffered whitespace keeps the origin of the chunk it came from.
    ReasoningStreamParser mixed(ReasoningFormat::Gemma4, false);
    ParsedText origins;
    append_events(origins, mixed.add("\n  ", false, false));
    append_events(origins, mixed.add("answer", true, true));
    assert(origins.content == "\n  answer");
    assert(origins.content_provenance == std::vector<bool>({false, true}));

    ReasoningStreamParser reversed(ReasoningFormat::Gemma4, false);
    ParsedText reversed_origins;
    append_events(reversed_origins, reversed.add("\n", false, true));
    append_events(reversed_origins, reversed.add("  ", false, false));
    append_events(reversed_origins, reversed.add("<", true, true));
    assert(reversed_origins.content == "\n  <");
    assert(reversed_origins.content_provenance == std::vector<bool>({true, false, true}));
}

// gpt-oss tags every message with a channel header. Reasoning ends at any header,
// and the role name that precedes it must not reach the output.
void test_gptoss_analysis_to_final() {
    ReasoningStreamParser parser(ReasoningFormat::GptOss, true);
    ParsedText parsed;
    append_events(parsed, parser.add("<|channel|>analysis<|message|>Need answer."));
    // The role arrives before the header it belongs to, in its own chunk.
    append_events(parsed, parser.add("assistant"));
    append_events(parsed, parser.add("<|channel|>final<|message|>Paris is the capital of France.", true));

    assert(parsed.reasoning == "Need answer.");
    assert(parsed.content == "Paris is the capital of France.");
    assert_no_markers(parsed);
}

void test_gptoss_commentary_is_hidden_but_final_survives() {
    ReasoningStreamParser parser(ReasoningFormat::GptOss, true);
    ParsedText parsed;
    append_events(parsed, parser.add("<|channel|>analysis<|message|>thinking."));
    append_events(parsed, parser.add("<|channel|>commentary<|message|>aside."));
    append_events(parsed, parser.add("<|channel|>final<|message|>answer.", true));

    // A plain commentary message is the model talking to itself.
    assert(parsed.reasoning == "thinking.aside.");
    assert(parsed.content == "answer.");
    assert_no_markers(parsed);
}

void test_gptoss_tool_call_commentary_is_shown() {
    ReasoningStreamParser parser(ReasoningFormat::GptOss, true);
    ParsedText parsed;
    append_events(parsed, parser.add("<|channel|>analysis<|message|>pick a tool."));
    append_events(
        parsed,
        parser.add("<|start|>assistant to=functions.get<|channel|>commentary json<|message|>{\"a\":1}", true)
    );

    assert(parsed.reasoning == "pick a tool.");
    assert(parsed.content == "<|start|>assistant to=functions.get<|channel|>commentary json<|message|>{\"a\":1}");
}

void test_gptoss_hidden_reasoning_still_yields_final() {
    ReasoningStreamParser parser(ReasoningFormat::GptOss, false);
    ParsedText parsed;
    append_events(parsed, parser.add("<|channel|>analysis<|message|>thinking."));
    append_events(parsed, parser.add("<|channel|>commentary<|message|>aside."));
    append_events(parsed, parser.add("<|cha"));
    append_events(parsed, parser.add("nnel|>final<|mess"));
    append_events(parsed, parser.add("age|>The final answer must remain visible.", true));

    assert(parsed.reasoning.empty());
    assert(parsed.content == "The final answer must remain visible.");
    assert_no_markers(parsed);
}

// A response that ends on commentary never reaches a final channel, so the answer
// has to come out of the commentary message or it is lost.
void test_gptoss_tool_call_commentary_without_final() {
    ReasoningStreamParser parser(ReasoningFormat::GptOss, false);
    ParsedText parsed;
    append_events(parsed, parser.add("<|channel|>analysis<|message|>pick a tool."));
    append_events(
        parsed,
        parser.add("<|start|>assistant to=functions.get<|channel|>commentary json<|message|>{\"a\":1}", true)
    );

    assert(parsed.reasoning.empty());
    assert(parsed.content == "<|start|>assistant to=functions.get<|channel|>commentary json<|message|>{\"a\":1}");
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
    test_gemma_hides_its_own_thought_when_thinking_is_off();
    test_gptoss_analysis_to_final();
    test_gptoss_commentary_is_hidden_but_final_survives();
    test_gptoss_tool_call_commentary_is_shown();
    test_gptoss_hidden_reasoning_still_yields_final();
    test_gptoss_tool_call_commentary_without_final();
    test_disabled_and_unsupported_pass_through();
    std::cout << "reasoning parser tests passed\n";
    return 0;
}
