# Qwen3-TTS only: this target is deliberately opt-in so standard LLiMa
# builds and other model runtimes neither require libtorch nor change outputs.
# libtorch is staged outside the SiMa sysroot for this cross-build.  Allow
# only this lookup to use its explicit target-library paths; restore the
# toolchain's strict sysroot policy immediately afterwards.
set(SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR "" CACHE PATH
    "Directory containing the target libtorch shared-library closure")
if(SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR STREQUAL "")
    get_filename_component(_qwen3tts_torch_prefix "${Torch_DIR}/../../.." ABSOLUTE)
    set(SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR "${_qwen3tts_torch_prefix}/lib" CACHE PATH
        "Directory containing the target libtorch shared-library closure" FORCE)
endif()
if(NOT EXISTS "${SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR}/libtorch.so")
    message(FATAL_ERROR
        "SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR must contain target libtorch.so; got "
        "${SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR}")
endif()

# Torch's CMake configuration calls FindProtobuf. Pin it to the conda-forge
# ABI bundled alongside libtorch, rather than the SDK sysroot's incompatible
# protobuf target.
set(_qwen3tts_protobuf_library "${SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR}/libprotobuf.so")
set(_qwen3tts_protobuf_include "${SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR}/../include")
foreach(_qwen3tts_protobuf_var
    Protobuf_LIBRARY Protobuf_LIBRARY_RELEASE Protobuf_LIBRARY_DEBUG
    PROTOBUF_LIBRARY PROTOBUF_LIBRARY_RELEASE PROTOBUF_LIBRARY_DEBUG)
    set(${_qwen3tts_protobuf_var} "${_qwen3tts_protobuf_library}" CACHE FILEPATH
        "Bundled Qwen3-TTS protobuf library" FORCE)
endforeach()
foreach(_qwen3tts_protobuf_var Protobuf_INCLUDE_DIR PROTOBUF_INCLUDE_DIR)
    set(${_qwen3tts_protobuf_var} "${_qwen3tts_protobuf_include}" CACHE PATH
        "Bundled Qwen3-TTS protobuf include directory" FORCE)
endforeach()
unset(_qwen3tts_protobuf_var)

set(_qwen3tts_saved_find_root_library_mode "${CMAKE_FIND_ROOT_PATH_MODE_LIBRARY}")
set(_qwen3tts_saved_find_root_include_mode "${CMAKE_FIND_ROOT_PATH_MODE_INCLUDE}")
set(_qwen3tts_saved_find_root_package_mode "${CMAKE_FIND_ROOT_PATH_MODE_PACKAGE}")
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)
find_package(Torch REQUIRED)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY "${_qwen3tts_saved_find_root_library_mode}")
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE "${_qwen3tts_saved_find_root_include_mode}")
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE "${_qwen3tts_saved_find_root_package_mode}")
unset(_qwen3tts_saved_find_root_library_mode)
unset(_qwen3tts_saved_find_root_include_mode)
unset(_qwen3tts_saved_find_root_package_mode)

set(SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_LIBRARY "" CACHE FILEPATH
    "NumPy ILP64 OpenBLAS library used by the exact Qwen3-TTS text projection")
if(NOT EXISTS "${SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_LIBRARY}")
    message(FATAL_ERROR
        "Set SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_LIBRARY to NumPy's libopenblas64_p-r0 shared library.")
endif()
get_filename_component(SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_DIR
    "${SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_LIBRARY}" DIRECTORY)
get_filename_component(SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_NAME
    "${SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_LIBRARY}" NAME)

set(SIMA_LMM_QWEN3TTS_RUNTIME_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/sima-lmm/qwen3tts")
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
    QWEN3_ENDPOINT_INCREMENTAL=1
    QWEN3_NUMPY_OPENBLAS_NAME="${SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_NAME}"
)
target_include_directories(qwen3tts PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/sima_lmm/devkit/cpp
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(qwen3tts PRIVATE
    qwen3tts_static_batches
    ${TARGET_RUNTIME}
    ${TORCH_LIBRARIES}
    ${CMAKE_DL_LIBS}
    OpenSSL::Crypto
)
target_link_options(qwen3tts PRIVATE "-Wl,--disable-new-dtags")
set_target_properties(qwen3tts PROPERTIES
    BUILD_RPATH "$ORIGIN/../lib;$ORIGIN/../lib/torch/lib;$ORIGIN/../lib/numpy.libs;${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN/../lib/torch/lib;$ORIGIN/../lib/numpy.libs;${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
)
install(
    TARGETS qwen3tts
    RUNTIME DESTINATION "${SIMA_LMM_QWEN3TTS_RUNTIME_INSTALL_DIR}/bin"
    COMPONENT qwen3tts
)
install(
    DIRECTORY "${SIMA_LMM_QWEN3TTS_TORCH_LIBRARY_DIR}/"
    DESTINATION "${SIMA_LMM_QWEN3TTS_RUNTIME_INSTALL_DIR}/lib/torch/lib"
    COMPONENT qwen3tts
    FILES_MATCHING PATTERN "*.so*"
)
install(
    DIRECTORY "${SIMA_LMM_QWEN3TTS_NUMPY_OPENBLAS_DIR}/"
    DESTINATION "${SIMA_LMM_QWEN3TTS_RUNTIME_INSTALL_DIR}/lib/numpy.libs"
    COMPONENT qwen3tts
    FILES_MATCHING PATTERN "*.so*"
)
