cmake_minimum_required(VERSION 3.20)
project(llima_runtime_host_tests LANGUAGES CXX)
enable_testing()

find_package(Eigen3 3.4 REQUIRED CONFIG)
find_package(nlohmann_json 3.11 REQUIRED CONFIG)
find_path(FMT_INCLUDE_DIR fmt/format.h REQUIRED)
find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h REQUIRED)

set(runtime_dir "${CMAKE_CURRENT_LIST_DIR}/../../../runtime")
add_executable(sima_lmm_attention_cache_test attention_cache_test.cpp
    "${runtime_dir}/src/language/speculative/gemma4_mtp_helpers.cpp")
target_compile_features(sima_lmm_attention_cache_test PRIVATE cxx_std_20)
target_include_directories(sima_lmm_attention_cache_test PRIVATE
    "${runtime_dir}/src" "${runtime_dir}/include/sima_lmm" "${FMT_INCLUDE_DIR}" "${SPDLOG_INCLUDE_DIR}")
target_link_libraries(sima_lmm_attention_cache_test PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
target_compile_definitions(sima_lmm_attention_cache_test PRIVATE FMT_HEADER_ONLY SPDLOG_FMT_EXTERNAL)
add_test(NAME runtime.attention_cache COMMAND sima_lmm_attention_cache_test)
set_tests_properties(runtime.attention_cache PROPERTIES LABELS "devkit;runtime;unit" TIMEOUT 30)

add_executable(sima_lmm_cache_slots_test cache_slots_test.cpp)
target_compile_features(sima_lmm_cache_slots_test PRIVATE cxx_std_20)
target_include_directories(sima_lmm_cache_slots_test PRIVATE
    "${runtime_dir}/src" "${runtime_dir}/include/sima_lmm")
add_test(NAME runtime.cache_slots COMMAND sima_lmm_cache_slots_test)
set_tests_properties(runtime.cache_slots PROPERTIES LABELS "devkit;runtime;unit" TIMEOUT 30)
