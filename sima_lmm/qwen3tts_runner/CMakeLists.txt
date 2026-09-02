add_library(qwen3tts_static_batches STATIC qwen3_mla_static_batch.cpp)
target_compile_features(qwen3tts_static_batches PRIVATE cxx_std_20)
target_include_directories(qwen3tts_static_batches PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sima_lmm/devkit/cpp
)
target_link_libraries(qwen3tts_static_batches PRIVATE ${TARGET_RUNTIME})

add_executable(qwen3tts
    numpy_pcg64.cpp
    qwen3tts_runner.cpp
    qwen3_tts_main.cpp
)
target_compile_features(qwen3tts PRIVATE cxx_std_20)
target_compile_definitions(qwen3tts PRIVATE
    FMT_SHARED
    SPDLOG_COMPILED_LIB
    SPDLOG_FMT_EXTERNAL
    SPDLOG_SHARED_LIB
    QWEN3_CP_SPLIT_HEADS=1
)
target_include_directories(qwen3tts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sima_lmm/devkit/cpp
)
target_link_libraries(qwen3tts PRIVATE
    qwen3tts_static_batches
    ${TARGET_RUNTIME}
    OpenSSL::Crypto
    Eigen3::Eigen
)
target_link_options(qwen3tts PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(qwen3tts PROPERTIES
    BUILD_RPATH "${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
    INSTALL_RPATH "${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
)
install(
    TARGETS qwen3tts
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    COMPONENT cli
)
