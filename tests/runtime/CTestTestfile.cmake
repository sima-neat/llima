add_test(
    runtime.dispatcher_lifecycle
    "./sima_lmm_dispatcher_lifecycle_test"
)
set_tests_properties(
    runtime.dispatcher_lifecycle
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;smoke"
        RESOURCE_LOCK dispatcher
        TIMEOUT 30
)

add_test(
    runtime.text_generation
    "./sima_lmm_text_generation_test"
)
set_tests_properties(
    runtime.text_generation
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;genai;llm;long"
        RESOURCE_LOCK dispatcher
        TIMEOUT 900
)

add_test(
    runtime.vision_generation
    "./sima_lmm_vision_generation_test"
)
set_tests_properties(
    runtime.vision_generation
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;genai;vlm;long"
        RESOURCE_LOCK dispatcher
        TIMEOUT 900
)

add_test(
    runtime.asr_transcription
    "./sima_lmm_asr_transcription_test"
)
set_tests_properties(
    runtime.asr_transcription
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;genai;asr;long"
        RESOURCE_LOCK dispatcher
        TIMEOUT 900
)

add_test(
    runtime.tool_call_parser
    "./sima_lmm_tool_call_parser_test"
)
set_tests_properties(
    runtime.tool_call_parser
    PROPERTIES
        LABELS "devkit;runtime;unit"
        TIMEOUT 30
)

add_test(
    runtime.reasoning_parser
    "./sima_lmm_reasoning_parser_test"
)
set_tests_properties(
    runtime.reasoning_parser
    PROPERTIES
        LABELS "devkit;runtime;unit"
        TIMEOUT 30
)
