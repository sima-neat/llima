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

# FileProvider seam and PCIe GenAI card backend: pure unit tests, no MLA, no
# PCIe hardware and no model needed.
foreach(_name IN ITEMS
        file_provider
        pcie_genai_protocol
        pcie_genai_vlm_image
        pcie_genai_chat_policy
        pcie_genai_event_bridge
        pcie_genai_backend_loop
        pcie_genai_lifecycle
        pcie_genai_recv_sweep)
    add_test(runtime.${_name} "./sima_lmm_${_name}_test")
    set_tests_properties(runtime.${_name} PROPERTIES LABELS "devkit;runtime;unit" TIMEOUT 30)
endforeach()
# Installed only when the build found simaai_svc.h.
foreach(_name IN ITEMS pcie_file_provider pcie_genai_dl_svc_client)
    # ctest reads this file from inside the test folder, so a relative path works
    # here (CMAKE_CURRENT_LIST_DIR is not set in ctest).
    if(EXISTS "sima_lmm_${_name}_test")
        add_test(runtime.${_name} "./sima_lmm_${_name}_test")
        set_tests_properties(runtime.${_name} PROPERTIES LABELS "devkit;runtime;unit" TIMEOUT 30)
    endif()
endforeach()
