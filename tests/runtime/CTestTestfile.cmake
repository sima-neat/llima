add_test(runtime.embedding_offload "./sima_lmm_embedding_offload_test")
set_tests_properties(runtime.embedding_offload PROPERTIES LABELS "runtime;unit" TIMEOUT 30)
add_test(runtime.embedding_offload_generation "./sima_lmm_embedding_offload_generation_test")
set_tests_properties(runtime.embedding_offload_generation PROPERTIES LABELS "devkit;runtime;genai;long" RESOURCE_LOCK mla TIMEOUT 900)

add_test(
    runtime.mla_rt_lifecycle
    "./sima_lmm_mla_rt_lifecycle_test"
)
set_tests_properties(
    runtime.mla_rt_lifecycle
    PROPERTIES
        LABELS "devkit;runtime;mla;smoke"
        RESOURCE_LOCK mla
        TIMEOUT 30
)

add_test(
    runtime.text_generation
    "./sima_lmm_text_generation_test"
)
set_tests_properties(
    runtime.text_generation
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;llm;long"
        RESOURCE_LOCK mla
        TIMEOUT 900
)

add_test(
    runtime.vision_generation
    "./sima_lmm_vision_generation_test"
)
set_tests_properties(
    runtime.vision_generation
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;vlm;long"
        RESOURCE_LOCK mla
        TIMEOUT 900
)

add_test(
    runtime.asr_transcription
    "./sima_lmm_asr_transcription_test"
)
set_tests_properties(
    runtime.asr_transcription
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;asr;long"
        RESOURCE_LOCK mla
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

add_test(
    runtime.gemma4_mtp_helpers
    "./sima_lmm_gemma4_mtp_helpers_test"
)
set_tests_properties(
    runtime.gemma4_mtp_helpers
    PROPERTIES
        LABELS "devkit;runtime;unit;gemma4;speculative"
        TIMEOUT 30
)
add_test(runtime.file_provider "./sima_lmm_file_provider_test")
set_tests_properties(runtime.file_provider PROPERTIES LABELS "devkit;runtime;provider" TIMEOUT 30)
add_test(runtime.file_provider_routing "./sima_lmm_file_provider_routing_test")
set_tests_properties(runtime.file_provider_routing PROPERTIES LABELS "devkit;runtime;long" RESOURCE_LOCK mla TIMEOUT 900)

add_test(runtime.request_lifecycle "./sima_lmm_request_lifecycle_test")
set_tests_properties(runtime.request_lifecycle PROPERTIES LABELS "devkit;runtime;unit" TIMEOUT 30)
add_test(runtime.session_generation "./sima_lmm_session_generation_test")
set_tests_properties(runtime.session_generation PROPERTIES LABELS "devkit;runtime;genai;long" RESOURCE_LOCK mla TIMEOUT 900)
foreach(method IN ITEMS eagle3 dflash gemma4_mtp)
    add_test(runtime.speculative_session_generation.${method} "/usr/bin/python3" "./run_speculative_session_test.py"
             "./sima_lmm_session_generation_test" "${method}")
    set_tests_properties(runtime.speculative_session_generation.${method} PROPERTIES
        LABELS "devkit;runtime;genai;speculative;long" RESOURCE_LOCK mla TIMEOUT 900 SKIP_RETURN_CODE 77)
endforeach()
add_test(runtime.lora_session "./sima_lmm_lora_session_test")
set_tests_properties(runtime.lora_session PROPERTIES LABELS "devkit;runtime;genai;lora" RESOURCE_LOCK mla TIMEOUT 600)
add_test(runtime.program_lifetime "./sima_lmm_runtime_program_lifetime_test")
set_tests_properties(runtime.program_lifetime PROPERTIES LABELS "devkit;runtime" RESOURCE_LOCK mla TIMEOUT 120)
