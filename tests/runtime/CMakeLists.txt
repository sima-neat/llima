add_executable(
    sima_lmm_dispatcher_lifecycle_test
    dispatcher_lifecycle_test.cpp
)
add_executable(
    sima_lmm_text_generation_test
    text_generation_test.cpp
)
add_executable(
    sima_lmm_vision_generation_test
    vision_generation_test.cpp
)
add_executable(
    sima_lmm_asr_transcription_test
    asr_transcription_test.cpp
)
add_executable(
    sima_lmm_tool_call_parser_test
    tool_call_parser_test.cpp
)
add_executable(
    sima_lmm_reasoning_parser_test
    reasoning_parser_test.cpp
)
target_link_libraries(
    sima_lmm_dispatcher_lifecycle_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_text_generation_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_vision_generation_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_asr_transcription_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_tool_call_parser_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_reasoning_parser_test
    PRIVATE
        sima_lmm_runtime
)
set_target_properties(
    sima_lmm_dispatcher_lifecycle_test
    sima_lmm_text_generation_test
    sima_lmm_vision_generation_test
    sima_lmm_asr_transcription_test
    sima_lmm_tool_call_parser_test
    sima_lmm_reasoning_parser_test
    PROPERTIES
        INSTALL_RPATH
            "$ORIGIN/../..;${CMAKE_INSTALL_FULL_LIBDIR};${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
)

add_test(
    NAME runtime.dispatcher_lifecycle
    COMMAND "$<TARGET_FILE:sima_lmm_dispatcher_lifecycle_test>"
)
add_test(
    NAME runtime.text_generation
    COMMAND "$<TARGET_FILE:sima_lmm_text_generation_test>"
)
add_test(
    NAME runtime.vision_generation
    COMMAND "$<TARGET_FILE:sima_lmm_vision_generation_test>"
)
add_test(
    NAME runtime.asr_transcription
    COMMAND "$<TARGET_FILE:sima_lmm_asr_transcription_test>"
)
add_test(
    NAME runtime.tool_call_parser
    COMMAND
        "$<TARGET_FILE:sima_lmm_tool_call_parser_test>"
)
add_test(
    NAME runtime.reasoning_parser
    COMMAND
        "$<TARGET_FILE:sima_lmm_reasoning_parser_test>"
)
set_tests_properties(
    runtime.dispatcher_lifecycle
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;smoke"
        RESOURCE_LOCK dispatcher
        TIMEOUT 30
)
set_tests_properties(
    runtime.text_generation
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;genai;llm;long"
        RESOURCE_LOCK dispatcher
        TIMEOUT 900
)
set_tests_properties(
    runtime.vision_generation
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;genai;vlm;long"
        RESOURCE_LOCK dispatcher
        TIMEOUT 900
)
set_tests_properties(
    runtime.asr_transcription
    PROPERTIES
        LABELS "devkit;runtime;dispatcher;genai;asr;long"
        RESOURCE_LOCK dispatcher
        TIMEOUT 900
)
set_tests_properties(
    runtime.tool_call_parser
    PROPERTIES
        LABELS "devkit;runtime;unit"
        TIMEOUT 30
)
set_tests_properties(
    runtime.reasoning_parser
    PROPERTIES
        LABELS "devkit;runtime;unit"
        TIMEOUT 30
)

install(
    TARGETS
        sima_lmm_dispatcher_lifecycle_test
        sima_lmm_text_generation_test
        sima_lmm_vision_generation_test
        sima_lmm_asr_transcription_test
        sima_lmm_tool_call_parser_test
        sima_lmm_reasoning_parser_test
    RUNTIME DESTINATION lib/sima-lmm/tests
    COMPONENT extras
)

install(
    FILES CTestTestfile.cmake
    DESTINATION lib/sima-lmm/tests
    COMPONENT extras
)

install(
    FILES
        CMakeLists.txt
        asr_transcription_test.cpp
        dispatcher_lifecycle_test.cpp
        pytest.ini
        runtime_test_utils.hpp
        test_cli_black_box.py
        test_http_black_box.py
        test_mla_memory_cleanup.py
        test_model_manager.py
        test_python_runtime_lifecycle.py
        test_zmq_black_box.py
        text_generation_test.cpp
        "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/reasoning_parser.hpp"
        reasoning_parser_test.cpp
        tool_call_parser_test.cpp
        vision_generation_test.cpp
    DESTINATION share/sima-lmm/tests/runtime
    COMPONENT extras
)

install(
    PROGRAMS
        prepare_genai_models.sh
        run_with_cancellation_cleanup.sh
    DESTINATION share/sima-lmm/tests/runtime
    COMPONENT extras
)
