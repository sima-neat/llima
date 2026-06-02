cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME sima_lmm)
endif()

set(SIMA_LMM_PLATFORM_VERSION "")
execute_process(
    COMMAND python3 -c "import json; print(json.load(open('deps/manifest.json', encoding='utf-8'))['platform-version'])"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE PLATFORM_VERSION_RESULT
    OUTPUT_VARIABLE SIMA_LMM_PLATFORM_VERSION
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT PLATFORM_VERSION_RESULT EQUAL 0 OR SIMA_LMM_PLATFORM_VERSION STREQUAL "")
    message(FATAL_ERROR "Failed to read platform-version from deps/manifest.json")
endif()

if(NOT DEFINED SKBUILD_PROJECT_VERSION)
    set(SKBUILD_PROJECT_VERSION "${SIMA_LMM_PLATFORM_VERSION}")
endif()

project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Target names.
set(TARGET_BIND cpp_ext)
set(TARGET_RUNTIME sima_lmm_runtime)
option(SIMA_LMM_BUILD_PYTHON "Build the Python nanobind extension." ON)
set(SYSROOT "" CACHE PATH "Optional target sysroot prefix")
set(
    SIMA_LMM_PYTHON_EXTENSION_INSTALL_DIR
    "devkit"
    CACHE STRING
    "Install destination for the cpp_ext Python extension. Wheels install relative to sima_lmm."
)
option(SIMA_LMM_INSTALL_PYTHON_PACKAGE "Install the lean sima_lmm Python runtime package." OFF)
set(
    SIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR
    "lib/python3/dist-packages/sima_lmm"
    CACHE STRING
    "Install destination for the lean sima_lmm Python runtime package."
)
# Set the build flags.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
    set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
endif()

# Find Python binding dependencies only when building the extension. The C++ runtime package can
# be configured and installed without Python/nanobind.
if(SIMA_LMM_BUILD_PYTHON)
    find_package(Python 3.11 COMPONENTS Interpreter Development REQUIRED)

    # Use the pypi version of nanobind instead of the official repo since it is a "pre-built"
    # library. nanobind is installed when python/pip runs the pyproject.toml.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT
    )
    find_package(nanobind CONFIG REQUIRED NO_CMAKE_FIND_ROOT_PATH)
endif()

find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(OpenSSL REQUIRED)
find_package(NeatInternals REQUIRED)
find_package(nlohmann_json 3.11.2 REQUIRED)

# FFTW3 does not provide cmake file so find_package cannot find it.
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(SWRESAMPLE REQUIRED libswresample)
pkg_check_modules(CPPZMQ REQUIRED cppzmq)
pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)
pkg_check_modules(Readline REQUIRED IMPORTED_TARGET readline)

include(FetchContent)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "Force offline build")

set(SIMA_LMM_CORE_DOC_DIR "${CMAKE_INSTALL_DATADIR}/doc/sima-lmm-core")
set(SIMA_LMM_DEV_DOC_DIR "${CMAKE_INSTALL_DATADIR}/doc/sima-lmm-dev")
set(SIMA_LMM_CLI_DOC_DIR "${CMAKE_INSTALL_DATADIR}/doc/sima-lmm-cli")

function(install_sima_lmm_project_copyright COMPONENT DOC_DIR)
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
        DESTINATION "${DOC_DIR}"
        RENAME "copyright"
        COMPONENT ${COMPONENT}
    )
endfunction()

function(install_sima_lmm_third_party_license LICENSE_FILE LICENSE_NAME)
    if(NOT EXISTS "${LICENSE_FILE}")
        message(WARNING "Third-party license file not found: ${LICENSE_FILE}")
        return()
    endif()

    foreach(_sima_lmm_component IN ITEMS core dev)
        if(_sima_lmm_component STREQUAL "core")
            set(_sima_lmm_doc_dir "${SIMA_LMM_CORE_DOC_DIR}")
        else()
            set(_sima_lmm_doc_dir "${SIMA_LMM_DEV_DOC_DIR}")
        endif()
        install(
            FILES "${LICENSE_FILE}"
            DESTINATION "${_sima_lmm_doc_dir}/third_party"
            RENAME "${LICENSE_NAME}"
            COMPONENT ${_sima_lmm_component}
        )
    endforeach()
endfunction()

install_sima_lmm_project_copyright(core "${SIMA_LMM_CORE_DOC_DIR}")
install_sima_lmm_project_copyright(dev "${SIMA_LMM_DEV_DOC_DIR}")
install_sima_lmm_project_copyright(cli "${SIMA_LMM_CLI_DOC_DIR}")

macro(Safe_FetchContent_MakeAvailable NAME LICENSE_NAME)
    FetchContent_GetProperties(${NAME})
    if(NOT ${NAME}_POPULATED)
        # If not found on system, populate the metadata to set the SOURCE_DIR and BINARY_DIR
        # variables.
        FetchContent_Populate(${NAME})
        message(STATUS "Using local source for ${NAME}: ${${NAME}_SOURCE_DIR}")

        # Add the directory manually with EXCLUDE_FROM_ALL.
        add_subdirectory("${${NAME}_SOURCE_DIR}" "${${NAME}_BINARY_DIR}" EXCLUDE_FROM_ALL)

        install_sima_lmm_third_party_license(
            "${${NAME}_SOURCE_DIR}/${LICENSE_NAME}"
            "LICENSE.${NAME}"
        )
    else()
        message(STATUS "Using system-provided package for ${NAME}")
    endif()
endmacro()

find_package(fmt 9.1 CONFIG REQUIRED)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
find_package(spdlog 1.10 CONFIG REQUIRED)

set(MINJA_TEST_ENABLED OFF CACHE BOOL "" FORCE)
set(MINJA_EXAMPLE_ENABLED OFF CACHE BOOL "" FORCE)
set(minja_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/minja")
add_library(minja INTERFACE)
target_include_directories(
    minja
    INTERFACE
    $<BUILD_INTERFACE:${minja_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(
    minja
    INTERFACE
    nlohmann_json::nlohmann_json
)
install_sima_lmm_third_party_license(
    "${minja_SOURCE_DIR}/LICENSE"
    "LICENSE.minja"
)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
set(MSGPACK_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(MSGPACK_USE_BOOST OFF CACHE BOOL "" FORCE)
set(MSGPACK_CXX20 ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    tokenizers_cpp
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp"
)
Safe_FetchContent_MakeAvailable(tokenizers_cpp LICENSE)

set(ENABLE_STATIC ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    cnpy
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cnpy"
)
Safe_FetchContent_MakeAvailable(cnpy LICENSE)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

pkg_check_modules(HTTPLIB REQUIRED cpp-httplib)
find_library(HTTPLIB_LIBRARY
    NAMES cpp-httplib
    PATHS ${HTTPLIB_LIBRARY_DIRS}
    NO_DEFAULT_PATH
)

set(LLAMA_TOOLS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_TESTS_INSTALL OFF CACHE BOOL "" FORCE)
set(LLAMA_HTTPLIB OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    llama
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
)
Safe_FetchContent_MakeAvailable(llama LICENSE)

set(SIMA_LMM_MSGPACK_LICENSE_INSTALLED OFF)
if(NOT TARGET msgpack-cxx)
    set(MSGPACK_CXX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/msgpack-c")
    if(NOT EXISTS "${MSGPACK_CXX_SOURCE_DIR}/CMakeLists.txt")
        set(MSGPACK_CXX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/msgpack")
    endif()
    FetchContent_Declare(
        msgpack-cxx
        SOURCE_DIR "${MSGPACK_CXX_SOURCE_DIR}"
        FIND_PACKAGE_ARGS 7.0.0 NAMES msgpack-cxx
    )
    Safe_FetchContent_MakeAvailable(msgpack-cxx LICENSE_1_0.txt)
    set(SIMA_LMM_MSGPACK_LICENSE_INSTALLED ON)
endif()

if(NOT DEFINED MSGPACK_CXX_SOURCE_DIR)
    set(MSGPACK_CXX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/msgpack-c")
endif()
if(NOT SIMA_LMM_MSGPACK_LICENSE_INSTALLED)
    install_sima_lmm_third_party_license(
        "${MSGPACK_CXX_SOURCE_DIR}/LICENSE_1_0.txt"
        "LICENSE.msgpack-cxx"
    )
endif()
install_sima_lmm_third_party_license(
    "${MSGPACK_CXX_SOURCE_DIR}/NOTICE"
    "NOTICE.msgpack-cxx"
)
set(SIMA_LMM_EXTRA_THIRD_PARTY_LICENSE_PATHS
    "third_party/tokenizers-cpp/sentencepiece/LICENSE"
    "third_party/llama.cpp/licenses/LICENSE-jsonhpp"
)
set(SIMA_LMM_EXTRA_THIRD_PARTY_LICENSE_NAMES
    "LICENSE.sentencepiece"
    "LICENSE.llama.cpp-jsonhpp"
)
list(LENGTH SIMA_LMM_EXTRA_THIRD_PARTY_LICENSE_PATHS _sima_lmm_extra_license_count)
math(EXPR _sima_lmm_extra_license_last "${_sima_lmm_extra_license_count} - 1")
foreach(_sima_lmm_license_index RANGE ${_sima_lmm_extra_license_last})
    list(GET SIMA_LMM_EXTRA_THIRD_PARTY_LICENSE_PATHS ${_sima_lmm_license_index} _sima_lmm_license_path)
    list(GET SIMA_LMM_EXTRA_THIRD_PARTY_LICENSE_NAMES ${_sima_lmm_license_index} _sima_lmm_license_name)
    install_sima_lmm_third_party_license(
        "${CMAKE_CURRENT_SOURCE_DIR}/${_sima_lmm_license_path}"
        "${_sima_lmm_license_name}"
    )
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
set(SIMAAI_DISPATCHER_INSTALL_LIB_DIR "${CMAKE_INSTALL_FULL_LIBDIR}/neat/runtime")

# Build the core library.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "sima_lmm/devkit/cpp/*.cpp"
)
list(FILTER SOURCES EXCLUDE REGEX "sima_lmm/devkit/cpp/binding.cpp")

add_library(${TARGET_RUNTIME} SHARED ${SOURCES})
set_target_properties(
    ${TARGET_RUNTIME}
    PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION TRUE
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)
message(STATUS "Using MLA dispatcher library: ${SIMAAI_DISPATCHER_LIB}")
set_target_properties(
    ${TARGET_RUNTIME} PROPERTIES
    INSTALL_RPATH "\$ORIGIN;${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
)
target_compile_features(${TARGET_RUNTIME} PUBLIC cxx_std_20)
target_compile_definitions(
    ${TARGET_RUNTIME}
    PRIVATE
        SIMA_LMM_DEFAULT_ASSETS_DIR="${CMAKE_INSTALL_FULL_DATADIR}/sima_lmm/assets"
)

target_compile_options(
    ${TARGET_RUNTIME}
    PRIVATE
    $<$<STREQUAL:"${CMAKE_SYSTEM_PROCESSOR}","aarch64">:-march=armv8.2-a+crypto>
    $<$<STREQUAL:"${CMAKE_SYSTEM_PROCESSOR}","aarch64">:-mtune=cortex-a65>
    "--param" "destructive-interference-size=64"
    -fno-stack-protector
    -ffunction-sections
    -fdata-sections
)
target_compile_options(${TARGET_RUNTIME} PUBLIC ${HTTPLIB_CFLAGS_OTHER})

target_include_directories(
    ${TARGET_RUNTIME}
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sima_lmm/devkit/cpp>
    $<BUILD_INTERFACE:${MSGPACK_CXX_SOURCE_DIR}/include>
    # cnpy include directory is not automatically added when library is linked.
    $<BUILD_INTERFACE:${cnpy_SOURCE_DIR}>
    $<BUILD_INTERFACE:${llama_SOURCE_DIR}/ggml/include>
    # tokenizers_cpp include directory is not added to tokenizers_c target.
    $<BUILD_INTERFACE:${tokenizers_cpp_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(${TARGET_RUNTIME} SYSTEM PUBLIC ${HTTPLIB_INCLUDE_DIRS})

target_link_libraries(
    ${TARGET_RUNTIME}
    PRIVATE
        OpenMP::OpenMP_CXX
        OpenSSL::Crypto
        PkgConfig::FFTW3
        cnpy-static
        simaaimem
        ${SIMAAI_DISPATCHER_LIB}
    PUBLIC
    NeatInternals::Headers
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    ${AVCODEC_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVUTIL_LIBRARIES}
    ${SWRESAMPLE_LIBRARIES}
    ${CPPZMQ_LIBRARIES}
    PkgConfig::JSONCPP
    msgpack-cxx
    fmt::fmt
    spdlog::spdlog
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    minja
    tokenizers_c
    ${HTTPLIB_LIBRARY}
    llama
    PkgConfig::Readline
)
# For simaai headers. Currently this is for simaai-memory-lib. Add it to the end so that it can
# search other paths first.
target_include_directories(${TARGET_RUNTIME} SYSTEM AFTER PRIVATE
    $<BUILD_INTERFACE:${SYSROOT}/usr/include/simaai>
    $<BUILD_INTERFACE:${SYSROOT}/usr/include>
)

if(SIMA_LMM_BUILD_PYTHON)
    nanobind_add_module(
        ${TARGET_BIND}
        NOMINSIZE
        LTO
        sima_lmm/devkit/cpp/binding.cpp
    )

    target_link_libraries(
        ${TARGET_BIND}
        PRIVATE
        ${TARGET_RUNTIME}
    )
    set_target_properties(
        ${TARGET_BIND} PROPERTIES
        INSTALL_RPATH "\$ORIGIN;${CMAKE_INSTALL_FULL_LIBDIR};${SIMAAI_DISPATCHER_INSTALL_LIB_DIR}"
    )
endif()

set(SIMA_LMM_PUBLIC_HEADERS
    sima_lmm/devkit/cpp/base_model.hpp
    sima_lmm/devkit/cpp/chat.hpp
    sima_lmm/devkit/cpp/eigen_types.hpp
    sima_lmm/devkit/cpp/image_processor.hpp
    sima_lmm/devkit/cpp/language_model.hpp
    sima_lmm/devkit/cpp/mla_buffer.hpp
    sima_lmm/devkit/cpp/mla_model.hpp
    sima_lmm/devkit/cpp/nlohmann_optional.hpp
    sima_lmm/devkit/cpp/rope_utils.hpp
    sima_lmm/devkit/cpp/setup.hpp
    sima_lmm/devkit/cpp/text_streamer.hpp
    sima_lmm/devkit/cpp/tokenizer.hpp
    sima_lmm/devkit/cpp/utils.hpp
    sima_lmm/devkit/cpp/vision_language_model.hpp
    sima_lmm/devkit/cpp/vision_model.hpp
    sima_lmm/devkit/cpp/vlm_config.hpp
    sima_lmm/devkit/cpp/vlm_helper.hpp
    sima_lmm/devkit/cpp/whisper_config.hpp
    sima_lmm/devkit/cpp/whisper_model.hpp
)

if(SIMA_LMM_BUILD_PYTHON)
    install(
        TARGETS ${TARGET_BIND}
        LIBRARY DESTINATION ${SIMA_LMM_PYTHON_EXTENSION_INSTALL_DIR}
        COMPONENT cli
    )
endif()
install(
    TARGETS ${TARGET_RUNTIME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT core
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT core
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT core
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
    DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sima_lmm/assets"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/sima_lmm
    COMPONENT core
    PATTERN "__pycache__" EXCLUDE
)
install(
    FILES ${SIMA_LMM_PUBLIC_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sima_lmm
    COMPONENT dev
)

# Header-only/public-header dependencies used by the installed LLiMa headers.
# Do not install headers that are owned by Debian -dev packages into /usr/include:
# doing so conflicts with normal package ownership (for example nlohmann-json3-dev).
# Dependencies without a reliable SDK package are still vendored for now.
install(DIRECTORY "${minja_SOURCE_DIR}/include/minja" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev)
install(DIRECTORY "${tokenizers_cpp_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev)
install(DIRECTORY "${llama_SOURCE_DIR}/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev)
install(DIRECTORY "${llama_SOURCE_DIR}/ggml/include/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} COMPONENT dev)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimaLMMConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SimaLMMConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SimaLMM
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/SimaLMMTargets.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/SimaLMMTargets.cmake"
    @ONLY
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SimaLMMConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/SimaLMMConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SimaLMMConfigVersion.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SimaLMMTargets.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SimaLMM
    COMPONENT dev
)

if(SIMA_LMM_INSTALL_PYTHON_PACKAGE)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/sima_lmm_VERSION"
"major: ${PROJECT_VERSION_MAJOR}
minor: ${PROJECT_VERSION_MINOR}
patch: ${PROJECT_VERSION_PATCH}
")

    install(
        FILES
            "${CMAKE_CURRENT_SOURCE_DIR}/sima_lmm/__init__.py"
        DESTINATION ${SIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR}
        COMPONENT cli
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/sima_lmm_VERSION"
        DESTINATION ${SIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR}
        RENAME VERSION
        COMPONENT cli
    )
    install(
        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/sima_lmm/devkit"
        DESTINATION ${SIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR}
        COMPONENT cli
        PATTERN "cpp" EXCLUDE
        PATTERN "__pycache__" EXCLUDE
        PATTERN "*.pyc" EXCLUDE
    )

    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/llima"
"#!/usr/bin/python3
from sima_lmm.devkit.devkit_demo import main
if __name__ == \"__main__\":
    main()
")
    install(
        PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/llima"
        DESTINATION ${CMAKE_INSTALL_BINDIR}
        COMPONENT cli
    )
endif()

# ------------------------------------------------------------------------------
# CPack (DEB)
# ------------------------------------------------------------------------------

set(CPACK_PACKAGE_NAME "sima-lmm")
set(CPACK_PACKAGE_VENDOR "SiMa.ai")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "SiMa LMM runtime")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "SiMa.ai Build Team <build@sima.ai>")
set(CPACK_GENERATOR "DEB")
set(CPACK_COMPONENTS_ALL core dev cli)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "SiMa.ai Build Team <build@sima.ai>")
set(CPACK_DEBIAN_PACKAGE_SECTION "libs")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "arm64")

set(CPACK_COMPONENT_CORE_DESCRIPTION
    "SiMa LMM shared C++ runtime library")
set(CPACK_COMPONENT_DEV_DESCRIPTION
    "SiMa LMM C++ development headers and CMake package")
set(CPACK_COMPONENT_CLI_DESCRIPTION
    "SiMa LMM Python runtime CLI")

set(CPACK_DEBIAN_CORE_PACKAGE_NAME "sima-lmm-core")
set(CPACK_DEBIAN_DEV_PACKAGE_NAME "sima-lmm-dev")
set(CPACK_DEBIAN_CLI_PACKAGE_NAME "sima-lmm-cli")

string(CONCAT CPACK_DEBIAN_CORE_PACKAGE_DEPENDS
    "neat-runtime, simaai-memory-lib (= 2.0.0), "
    "libcpp-httplib0.11, libfftw3-double3, "
    "libopencv-core406, libopencv-imgproc406, libopencv-imgcodecs406, "
    "libavcodec59, libavformat59, libavutil57, libswresample4, "
    "libzmq5, libssl3, libreadline8, libgomp1, "
    "libfmt9, libspdlog1.10")
string(CONCAT CPACK_DEBIAN_DEV_PACKAGE_DEPENDS
    "sima-lmm-core (= ${CPACK_PACKAGE_VERSION}), "
    "neat-internals-dev, "
    "nlohmann-json3-dev (>= 3.11.2), libeigen3-dev, libfmt-dev, libspdlog-dev, "
    "libcpp-httplib-dev, libjsoncpp-dev, libopencv-dev, libfftw3-dev, "
    "libavcodec-dev, libavformat-dev, libavutil-dev, libswresample-dev, "
    "libssl-dev, cppzmq-dev, libzmq3-dev")
set(CPACK_DEBIAN_CLI_PACKAGE_DEPENDS
    "sima-lmm-core (= ${CPACK_PACKAGE_VERSION}), simaai-utils, python3")

set(CPACK_INSTALL_CMAKE_PROJECTS
    "${CMAKE_BINARY_DIR};${PROJECT_NAME};ALL;/"
)

include(CPack)
