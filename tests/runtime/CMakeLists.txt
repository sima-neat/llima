add_executable(sima_lmm_embedding_offload_generation_test embedding_offload_generation_test.cpp)
target_link_libraries(sima_lmm_embedding_offload_generation_test PRIVATE sima_lmm_runtime)
add_executable(sima_lmm_embedding_offload_test embedding_offload_test.cpp
    "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/embedding_offload.cpp")
target_include_directories(sima_lmm_embedding_offload_test PRIVATE "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp")
target_link_libraries(sima_lmm_embedding_offload_test PRIVATE OpenMP::OpenMP_CXX)
add_test(NAME runtime.embedding_offload COMMAND sima_lmm_embedding_offload_test)
set_tests_properties(runtime.embedding_offload PROPERTIES LABELS "runtime;unit" TIMEOUT 30)
add_test(NAME runtime.embedding_offload_generation COMMAND sima_lmm_embedding_offload_generation_test)
set_tests_properties(runtime.embedding_offload_generation PROPERTIES LABELS "devkit;runtime;genai;long" RESOURCE_LOCK mla TIMEOUT 900)

# Unit test for the file seam. Compiles file_provider.cpp directly (no MLA
# runtime needed), like the embedding_offload unit test above.
add_executable(sima_lmm_file_provider_test file_provider_test.cpp
    "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/file_provider.cpp")
target_include_directories(sima_lmm_file_provider_test PRIVATE "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp")
target_link_libraries(sima_lmm_file_provider_test PRIVATE fmt::fmt)
add_test(NAME runtime.file_provider COMMAND sima_lmm_file_provider_test)
set_tests_properties(runtime.file_provider PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

# PcieFileProvider pulls in simaai_svc.h (EP-only simaai_svc_xfer_opts) for its
# types, but resolves the svc functions at runtime with dlopen, so it needs only
# the HEADER to build, not the svc library. Guard on the include dir alone so
# this unit test builds on any sysroot that ships the header. The injected fetch
# fn means the test never dlopen's libsimaaipep.so, so no PCIe stack is required.
if(SIMAAI_SVC_INCLUDE_DIR)
    add_executable(sima_lmm_pcie_file_provider_test pcie_file_provider_test.cpp
        "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/pcie_file_provider.cpp")
    target_include_directories(sima_lmm_pcie_file_provider_test PRIVATE
        "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp"
        "${SIMAAI_SVC_INCLUDE_DIR}")
    target_link_libraries(sima_lmm_pcie_file_provider_test PRIVATE
        ${CMAKE_DL_LIBS} fmt::fmt spdlog::spdlog)
    add_test(NAME runtime.pcie_file_provider COMMAND sima_lmm_pcie_file_provider_test)
    set_tests_properties(runtime.pcie_file_provider PROPERTIES LABELS "runtime;unit" TIMEOUT 30)
endif()

# pcie-genai-backend logic. Its sources live in
# sima_lmm/devkit/pcie_backend/, outside the core runtime GLOB. These tests
# compile only the pieces they check and use a fake svc client: no daemon, no
# MLA, no svc library.
set(PCIE_BACKEND_DIR "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/pcie_backend")
add_executable(sima_lmm_pcie_genai_protocol_test pcie_genai_protocol_test.cpp
    "${PCIE_BACKEND_DIR}/genai_protocol.cpp")
target_include_directories(sima_lmm_pcie_genai_protocol_test PRIVATE "${PCIE_BACKEND_DIR}")
target_link_libraries(sima_lmm_pcie_genai_protocol_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME runtime.pcie_genai_protocol COMMAND sima_lmm_pcie_genai_protocol_test)
set_tests_properties(runtime.pcie_genai_protocol PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

# The image-name split is pure std::filesystem (vlm_image.cpp), so it compiles
# without the LLiMa runtime; the full pull + add_image path is hardware-tested.
add_executable(sima_lmm_pcie_genai_vlm_image_test pcie_genai_vlm_image_test.cpp
    "${PCIE_BACKEND_DIR}/vlm_image.cpp")
target_include_directories(sima_lmm_pcie_genai_vlm_image_test PRIVATE "${PCIE_BACKEND_DIR}")
add_test(NAME runtime.pcie_genai_vlm_image COMMAND sima_lmm_pcie_genai_vlm_image_test)
set_tests_properties(runtime.pcie_genai_vlm_image PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

# The pure chat rules VlmGenerator uses (no LLiMa runtime needed).
add_executable(sima_lmm_pcie_genai_chat_policy_test pcie_genai_chat_policy_test.cpp
    "${PCIE_BACKEND_DIR}/chat_policy.cpp"
    "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/reasoning_parser.cpp")
target_include_directories(sima_lmm_pcie_genai_chat_policy_test PRIVATE "${PCIE_BACKEND_DIR}"
    "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp")
add_test(NAME runtime.pcie_genai_chat_policy COMMAND sima_lmm_pcie_genai_chat_policy_test)
set_tests_properties(runtime.pcie_genai_chat_policy PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

add_executable(sima_lmm_pcie_genai_event_bridge_test pcie_genai_event_bridge_test.cpp
    "${PCIE_BACKEND_DIR}/event_bridge.cpp"
    "${PCIE_BACKEND_DIR}/genai_protocol.cpp")
target_include_directories(sima_lmm_pcie_genai_event_bridge_test PRIVATE "${PCIE_BACKEND_DIR}")
target_link_libraries(sima_lmm_pcie_genai_event_bridge_test PRIVATE
    nlohmann_json::nlohmann_json fmt::fmt spdlog::spdlog)
add_test(NAME runtime.pcie_genai_event_bridge COMMAND sima_lmm_pcie_genai_event_bridge_test)
set_tests_properties(runtime.pcie_genai_event_bridge PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

find_package(Threads REQUIRED)
add_executable(sima_lmm_pcie_genai_backend_loop_test pcie_genai_backend_loop_test.cpp
    "${PCIE_BACKEND_DIR}/backend_loop.cpp"
    "${PCIE_BACKEND_DIR}/event_bridge.cpp"
    "${PCIE_BACKEND_DIR}/genai_protocol.cpp")
target_include_directories(sima_lmm_pcie_genai_backend_loop_test PRIVATE "${PCIE_BACKEND_DIR}")
target_link_libraries(sima_lmm_pcie_genai_backend_loop_test PRIVATE
    nlohmann_json::nlohmann_json fmt::fmt spdlog::spdlog Threads::Threads)
add_test(NAME runtime.pcie_genai_backend_loop COMMAND sima_lmm_pcie_genai_backend_loop_test)
set_tests_properties(runtime.pcie_genai_backend_loop PROPERTIES LABELS "runtime;unit" TIMEOUT 60)

add_executable(sima_lmm_pcie_genai_lifecycle_test pcie_genai_lifecycle_test.cpp
    "${PCIE_BACKEND_DIR}/lifecycle.cpp")
target_include_directories(sima_lmm_pcie_genai_lifecycle_test PRIVATE "${PCIE_BACKEND_DIR}")
target_link_libraries(sima_lmm_pcie_genai_lifecycle_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME runtime.pcie_genai_lifecycle COMMAND sima_lmm_pcie_genai_lifecycle_test)
set_tests_properties(runtime.pcie_genai_lifecycle PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

add_executable(sima_lmm_pcie_genai_recv_sweep_test pcie_genai_recv_sweep_test.cpp
    "${PCIE_BACKEND_DIR}/recv_sweep.cpp")
target_include_directories(sima_lmm_pcie_genai_recv_sweep_test PRIVATE "${PCIE_BACKEND_DIR}")
add_test(NAME runtime.pcie_genai_recv_sweep COMMAND sima_lmm_pcie_genai_recv_sweep_test)
set_tests_properties(runtime.pcie_genai_recv_sweep PROPERTIES LABELS "runtime;unit" TIMEOUT 30)

if(SIMAAI_SVC_INCLUDE_DIR)
    add_executable(sima_lmm_pcie_genai_dl_svc_client_test pcie_genai_dl_svc_client_test.cpp
        "${PCIE_BACKEND_DIR}/dl_svc_client.cpp")
    target_include_directories(sima_lmm_pcie_genai_dl_svc_client_test PRIVATE
        "${PCIE_BACKEND_DIR}" "${SIMAAI_SVC_INCLUDE_DIR}")
    target_link_libraries(sima_lmm_pcie_genai_dl_svc_client_test PRIVATE
        ${CMAKE_DL_LIBS} fmt::fmt spdlog::spdlog)
    add_test(NAME runtime.pcie_genai_dl_svc_client COMMAND sima_lmm_pcie_genai_dl_svc_client_test)
    set_tests_properties(runtime.pcie_genai_dl_svc_client PROPERTIES LABELS "runtime;unit" TIMEOUT 30)
endif()

add_executable(
    sima_lmm_mla_rt_lifecycle_test
    mla_rt_lifecycle_test.cpp
)
add_executable(
    sima_lmm_text_generation_test
    text_generation_test.cpp
)
add_executable(
    sima_lmm_file_provider_routing_test
    file_provider_routing_test.cpp
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
    sima_lmm_mla_rt_lifecycle_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_text_generation_test
    PRIVATE
        sima_lmm_runtime
)
target_link_libraries(
    sima_lmm_file_provider_routing_test
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
    sima_lmm_embedding_offload_generation_test
    sima_lmm_embedding_offload_test
    sima_lmm_mla_rt_lifecycle_test
    sima_lmm_text_generation_test
    sima_lmm_file_provider_routing_test
    sima_lmm_vision_generation_test
    sima_lmm_asr_transcription_test
    sima_lmm_tool_call_parser_test
    sima_lmm_reasoning_parser_test
    PROPERTIES
        INSTALL_RPATH
            "$ORIGIN/../..;${CMAKE_INSTALL_FULL_LIBDIR}"
)

add_test(
    NAME runtime.mla_rt_lifecycle
    COMMAND "$<TARGET_FILE:sima_lmm_mla_rt_lifecycle_test>"
)
add_test(
    NAME runtime.text_generation
    COMMAND "$<TARGET_FILE:sima_lmm_text_generation_test>"
)
add_test(
    NAME runtime.file_provider_routing
    COMMAND "$<TARGET_FILE:sima_lmm_file_provider_routing_test>"
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
    runtime.mla_rt_lifecycle
    PROPERTIES
        LABELS "devkit;runtime;mla;smoke"
        RESOURCE_LOCK mla
        TIMEOUT 30
)
set_tests_properties(
    runtime.text_generation
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;llm;long"
        RESOURCE_LOCK mla
        TIMEOUT 900
)
set_tests_properties(
    runtime.file_provider_routing
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;llm;long"
        RESOURCE_LOCK mla
        TIMEOUT 900
)
set_tests_properties(
    runtime.vision_generation
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;vlm;long"
        RESOURCE_LOCK mla
        TIMEOUT 900
)
set_tests_properties(
    runtime.asr_transcription
    PROPERTIES
        LABELS "devkit;runtime;mla;genai;asr;long"
        RESOURCE_LOCK mla
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
        sima_lmm_embedding_offload_generation_test
        sima_lmm_embedding_offload_test
        sima_lmm_mla_rt_lifecycle_test
        sima_lmm_text_generation_test
        sima_lmm_file_provider_routing_test
        sima_lmm_vision_generation_test
        sima_lmm_asr_transcription_test
        sima_lmm_tool_call_parser_test
        sima_lmm_reasoning_parser_test
        sima_lmm_file_provider_test
        sima_lmm_pcie_genai_protocol_test
        sima_lmm_pcie_genai_vlm_image_test
        sima_lmm_pcie_genai_chat_policy_test
        sima_lmm_pcie_genai_event_bridge_test
        sima_lmm_pcie_genai_backend_loop_test
        sima_lmm_pcie_genai_lifecycle_test
        sima_lmm_pcie_genai_recv_sweep_test
    RUNTIME DESTINATION lib/sima-lmm/tests
    COMPONENT extras
)
# Built only when simaai_svc.h is found; the packaged CTestTestfile.cmake
# registers them only if the executable is there.
if(SIMAAI_SVC_INCLUDE_DIR)
    install(
        TARGETS
            sima_lmm_pcie_file_provider_test
            sima_lmm_pcie_genai_dl_svc_client_test
        RUNTIME DESTINATION lib/sima-lmm/tests
        COMPONENT extras
    )
endif()

install(
    FILES CTestTestfile.cmake
    DESTINATION lib/sima-lmm/tests
    COMPONENT extras
)

install(
    FILES
        CMakeLists.txt
        embedding_offload_generation_test.cpp
        embedding_offload_test.cpp
        "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/embedding_offload.hpp"
        "${PROJECT_SOURCE_DIR}/sima_lmm/devkit/cpp/embedding_offload.cpp"
        asr_transcription_test.cpp
        mla_rt_lifecycle_test.cpp
        pytest.ini
        runtime_test_utils.hpp
        test_cli_black_box.py
        test_http_black_box.py
        test_model_manager.py
        test_python_runtime_lifecycle.py
        test_zmq_black_box.py
        text_generation_test.cpp
        file_provider_test.cpp
        file_provider_routing_test.cpp
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
