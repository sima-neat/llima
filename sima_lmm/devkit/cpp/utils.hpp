//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************


#ifndef _SIMA_UTILS_
#define _SIMA_UTILS_

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <optional>
#include <ranges>
#include <ratio>
#include <string_view>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>


// Performance profiling code insertion with switch to turn on/off.
#ifndef SIMA_PERF_PROFILE
#define SIMA_PERF_PROFILE 0 
#endif

// Make the symbol to be visible by other users of the .so file.
#define EXPORT __attribute__((visibility("default")))


namespace simaai {
namespace llima {

class ChronoTimer {
    public:
        ChronoTimer(bool start_now = false) {
            if (start_now) start();
            else _begin = {};
        };
        void start() { _begin = std::chrono::high_resolution_clock::now(); };

        template <typename T = std::ratio<1, 1>>
        double stop(bool restart = false) {
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, T>(end - _begin).count();
            if (SIMA_PERF_PROFILE)
                if constexpr (std::is_same_v<T, std::ratio<1, 1>>)
                    spdlog::info("[PERF PROFILE] Duration: {:.5f}s", duration);
            if (restart)
                _begin = end;
            return duration;
        };

    private:
        std::chrono::high_resolution_clock::time_point _begin;
};


// Useful size/dimension calculation functions.
#define MLA_ROW_SIZE 16

template <typename T>
concept C = std::is_integral_v<T>;

inline auto ceil_div(C auto x, C auto y) { return (x + y - 1) / y; }
inline auto round_up_to(C auto x, C auto y) { return ceil_div(x, y) * y; }
inline auto round_up_to_row(C auto x) { return round_up_to(x, MLA_ROW_SIZE); }


inline std::string trim(std::string in) {
    auto is_space = [](unsigned char c) {
        return std::isspace(c);
    };

    auto view = in
        | std::views::drop_while(is_space)
        | std::views::reverse
        | std::views::drop_while(is_space)
        | std::views::reverse;
    return {view.begin(), view.end()};
}


std::vector<uint8_t> base64_decode(std::string_view base64_data);


inline size_t count_regular_files(const std::filesystem::path& dir_path) {
    auto it = std::filesystem::directory_iterator(dir_path);
    return std::count_if(std::filesystem::begin(it), std::filesystem::end(it), [](const auto& entry) {
        return entry.is_regular_file();
    });
}


inline bool get_env_var(const std::string& name, bool default_value = false) {
    auto var_ptr = std::getenv(name.c_str());
    if (var_ptr == nullptr)
        return default_value;
    auto var = std::strcmp(var_ptr, "true") == 0 || std::strcmp(var_ptr, "1") == 0;

    if (var) {
        auto msg = fmt::format("{}: {}", name, var);
        spdlog::info(msg);
        std::cout << msg << std::endl << std::flush;
    }
    return var;
}


inline std::string get_env_var(const std::string& name, std::string default_value = "") {
    auto var_ptr = std::getenv(name.c_str());
    if (var_ptr == nullptr)
        return default_value;
    std::string var(var_ptr);
    if (!var.empty()) {
        auto msg = fmt::format("{}: {}", name, var);
        spdlog::info(msg);
        std::cout << msg << std::endl << std::flush;
    }
    return var;
}


extern std::optional<std::filesystem::path> sample_image_file_name;
extern std::optional<std::filesystem::path> sample_audio_file_name;

void set_sample_image_file_name(std::filesystem::path file_name);
void set_sample_audio_file_name(std::filesystem::path file_name);


}
}


#endif
