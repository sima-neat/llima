#ifndef _SIMA_LLIMA_SETUP_
#define _SIMA_LLIMA_SETUP_

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <spdlog/spdlog.h>

#include "utils.hpp"

namespace simaai {
namespace llima {

EXPORT void set_log_level(spdlog::level::level_enum log_level);

EXPORT void initialize_default_sample_files();

EXPORT void connect(
    const std::vector<std::string>& mla_rt_args,
    const std::filesystem::path& log_file_name = "run.log",
    const spdlog::level::level_enum log_level = spdlog::level::warn
);


EXPORT void disconnect();


}
}

#endif
