#include <chrono>

#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/list.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <spdlog/spdlog.h>

#include "cli.hpp"
#include "setup.hpp"
#include "web.hpp"
#include "zmq_server.hpp"


namespace nb = nanobind;

NB_MODULE(cpp_ext, m) {
    m.doc() = "LLIMA Python Interface";
    using namespace simaai::llima;

    m.def(
        "connect_cpp",
        [](
            const std::vector<std::string>& mla_rt_args,
            const std::string& log_file_name,
            int log_level
        ) {
            // Convert python logging level to spdlog logging level.
            // logging.CRITICAL	50	spdlog::level::critical
            spdlog::level::level_enum spdlog_log_level;
            switch (log_level) {
                case 0:
                    // logging.NOTSET
                    spdlog_log_level = spdlog::level::off;
                    break;
                case 10:
                    // logging.DEBUG
                    spdlog_log_level = spdlog::level::debug;
                    break;
                case 20:
                    // logging.INFO
                    spdlog_log_level = spdlog::level::info;
                    break;
                case 30:
                    // logging.WARNING
                    spdlog_log_level = spdlog::level::warn;
                    break;
                case 40:
                    // logging.ERROR
                    spdlog_log_level = spdlog::level::err;
                    break;
                case 50:
                    // logging.CRITICAL
                    spdlog_log_level = spdlog::level::critical;
                    break;
                default:
                    std::runtime_error(
                        "Unable to convert python logging level to spdlog logging level: "
                        + std::to_string(log_level)
                    );
            }
            connect(mla_rt_args, log_file_name, spdlog_log_level);
        },
        nb::arg("mla_rt_args"),
        nb::arg("log_file_name"),
        nb::arg("log_level") = 0
    );
    m.def("disconnect_cpp", &disconnect);

    nb::class_<CLI>(m, "CLI")
        .def(
            nb::init<
                std::filesystem::path,
                std::optional<std::filesystem::path>,
                std::optional<std::filesystem::path>,
                std::optional<std::string>,
                std::optional<std::string>
            >(),
            nb::arg("model_path"),
            nb::arg("whisper_model_path") = nb::none(),
            nb::arg("draft_model_path") = nb::none(),
            nb::arg("system_prompt") = nb::none(),
            nb::arg("chat_template") = nb::none()
        )
        .def("run", &CLI::run)
    ;

    nb::class_<WEB>(m, "WEB")
        .def(
            nb::init<
                std::filesystem::path,
                std::optional<std::filesystem::path>,
                std::optional<std::filesystem::path>,
                std::optional<std::string>,
                std::optional<std::string>
            >(),
            nb::arg("model_path"),
            nb::arg("whisper_model_path") = nb::none(),
            nb::arg("draft_model_path") = nb::none(),
            nb::arg("system_prompt") = nb::none(),
            nb::arg("chat_template") = nb::none()
        )
        .def("run", &WEB::run)
    ;

    nb::class_<ZMQServer>(m, "ZMQServer")
        .def(nb::init<const std::filesystem::path&, uint32_t>())
        .def("run", &ZMQServer::run)
        .def("stop", &ZMQServer::stop)
    ;
}
