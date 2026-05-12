cmake_minimum_required(VERSION 3.24)

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# Target names.
set(TARGET_BIND cpp_ext)
set(TARGET_RUNTIME sima_lmm_runtime)

# Set the build flags.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()

# Find all the dependencies.
find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)

# Use the pypi version of nanobind instead of the official repo since it is a "pre-built" library.
# The nanobind is installed when python/pip runs the pyproject.toml.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED NO_CMAKE_FIND_ROOT_PATH)

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenSSL REQUIRED)

# FFTW3 does not provide cmake file so find_package cannot find it.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)
pkg_check_modules(CPPZMQ REQUIRED cppzmq)

include(FetchContent)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "Force offline build")

macro(Safe_FetchContent_MakeAvailable NAME LICENSE_NAME)
    FetchContent_GetProperties(${NAME})
    if(NOT ${NAME}_POPULATED)
        # If not found on system, populate the metadata to set the SOURCE_DIR and BINARY_DIR
        # variables.
        FetchContent_Populate(${NAME})
        message(STATUS "Using local source for ${NAME}: ${${NAME}_SOURCE_DIR}")

        # Add the directory manually with EXCLUDE_FROM_ALL.
        add_subdirectory("${${NAME}_SOURCE_DIR}" "${${NAME}_BINARY_DIR}" EXCLUDE_FROM_ALL)

        # Install the license file.
        message(STATUS "Install license file: ${${NAME}_SOURCE_DIR}/${LICENSE_NAME}")
        install(
            FILES "${${NAME}_SOURCE_DIR}/${LICENSE_NAME}"
            DESTINATION "docs/licenses"
            RENAME "LICENSE.${NAME}"
        )
    else()
        message(STATUS "Using system-provided package for ${NAME}")
    endif()
endmacro()

# For other libraries that do not have pypi package, debian package or the packages with required
# versions, use git submodule to download source first and build it to avoid downloading dynamically
# at build time.
set(JSON_Diagnostics ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    json
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/json"
    FIND_PACKAGE_ARGS 3.12.1 NAMES nlohmann_json
)
Safe_FetchContent_MakeAvailable(json LICENSE.MIT)

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    fmt
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fmt"
    FIND_PACKAGE_ARGS 12.1.0 NAMES fmt
)
Safe_FetchContent_MakeAvailable(fmt LICENSE)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    spdlog
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/spdlog"
    FIND_PACKAGE_ARGS 1.17.0 NAMES spdlog
)
Safe_FetchContent_MakeAvailable(spdlog LICENSE)

set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    minja
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/minja"
)
Safe_FetchContent_MakeAvailable(minja LICENSE)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp"
)
Safe_FetchContent_MakeAvailable(tokenizers_cpp LICENSE)

FetchContent_Declare(
    cnpy
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy"
)
Safe_FetchContent_MakeAvailable(cnpy LICENSE)
if(TARGET example1)
    set_target_properties(example1 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()
set_target_properties(
    cnpy PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    eigen
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen"
    FIND_PACKAGE_ARGS 5.0.1 NAMES Eigen3
)
Safe_FetchContent_MakeAvailable(eigen LICENSE)

# https://github.com/yhirose/cpp-httplib/issues/2313
find_package(zstd CONFIG REQUIRED)
if (NOT TARGET zstd::libzstd)
    add_library(zstd::libzstd INTERFACE IMPORTED)
    set_property(TARGET zstd::libzstd PROPERTY INTERFACE_LINK_LIBRARIES zstd)
endif()
FetchContent_Declare(
    httplib
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cpp-httplib"
    FIND_PACKAGE_ARGS 0.30.1 NAMES httplib
)
Safe_FetchContent_MakeAvailable(httplib LICENSE)

set(LLAMA_HTTPLIB OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    llama
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
)
Safe_FetchContent_MakeAvailable(llama LICENSE)

if(NOT TARGET msgpack-cxx)
    set(MSGPACK_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(MSGPACK_CXX20 ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        msgpack-cxx
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/msgpack-c"
        FIND_PACKAGE_ARGS 7.0.0 NAMES msgpack-cxx
    )
    Safe_FetchContent_MakeAvailable(msgpack-cxx LICENSE_1_0.txt)
endif()

set(
    EXTERNAL_LIBS
    fmt spdlog minja tokenizers_cpp cnpy httplib llama ggml ggml-base ggml-cpu msgpack-cxx
)

foreach(lib_target ${EXTERNAL_LIBS})
    if(TARGET ${lib_target})
        set_target_properties(${lib_target} PROPERTIES INSTALL_RPATH "\$ORIGIN")
    endif()
endforeach()


find_library(
    SIMAAI_DISPATCHER_LIB
    NAMES neatdispatchercore simaaidispatcher
    PATHS
        "${SYSROOT}/usr/lib/aarch64-linux-gnu/neat/runtime"
        "${SYSROOT}/usr/lib/aarch64-linux-gnu"
        "${SYSROOT}/usr/lib"
    REQUIRED
)


# Build the core library.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "sima_lmm/devkit/cpp/*.cpp"
)
list(FILTER SOURCES EXCLUDE REGEX "sima_lmm/devkit/cpp/binding.cpp")

add_library(${TARGET_RUNTIME} SHARED ${SOURCES})
set_target_properties(${TARGET_RUNTIME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
message(STATUS "Using MLA dispatcher library: ${SIMAAI_DISPATCHER_LIB}")
set_target_properties(${TARGET_RUNTIME} PROPERTIES INSTALL_RPATH "\$ORIGIN")
target_compile_features(${TARGET_RUNTIME} PUBLIC cxx_std_20)

target_compile_options(
    ${TARGET_RUNTIME}
    PUBLIC
    $<$<STREQUAL:"${CMAKE_SYSTEM_PROCESSOR}","aarch64">:-march=armv8.2-a+crypto>
    $<$<STREQUAL:"${CMAKE_SYSTEM_PROCESSOR}","aarch64">:-mtune=cortex-a65>
    "--param" "destructive-interference-size=64"
    -fno-stack-protector
    -ffunction-sections
    -fdata-sections
)

target_include_directories(
    ${TARGET_RUNTIME}
    PUBLIC
    "sima_lmm/devkit/cpp"
    # cnpy include directory is not automatically added when library is linked.
    ${cnpy_SOURCE_DIR}
    # tokenizers_cpp include directory is not added to tokenizers_c target.
    ${tokenizers_cpp_SOURCE_DIR}/include
)

target_link_libraries(
    ${TARGET_RUNTIME}
    PUBLIC
    OpenMP::OpenMP_CXX
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    OpenSSL::Crypto
    PkgConfig::FFTW3
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    ${CPPZMQ_LIBRARIES}
    msgpack-cxx
    fmt::fmt
    spdlog::spdlog
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    minja
    cnpy
    tokenizers_c
    httplib::httplib
    llama
    simaaimem
    ${SIMAAI_DISPATCHER_LIB}
)
# For simaai headers. Currently this is for simaai-memory-lib. Add it to the end so that it can
# search other paths first.
target_include_directories(${TARGET_RUNTIME} SYSTEM AFTER PUBLIC "${SYSROOT}/usr/include/simaai")
target_include_directories(${TARGET_RUNTIME} SYSTEM AFTER PUBLIC "${SYSROOT}/usr/include")

nanobind_add_module(
    ${TARGET_BIND}
    NB_SHARED
    NOMINSIZE
    LTO
    sima_lmm/devkit/cpp/binding.cpp
)

target_link_libraries(
    ${TARGET_BIND}
    PRIVATE
    ${TARGET_RUNTIME}
)
set_target_properties(${TARGET_BIND} PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")

install(
    TARGETS
    nanobind fmt spdlog msgpack-cxx llama ggml ggml-base ggml-cpu cnpy minja
    LIBRARY DESTINATION lib
)
install(TARGETS ${TARGET_BIND} LIBRARY DESTINATION devkit)
install(TARGETS ${TARGET_RUNTIME})
