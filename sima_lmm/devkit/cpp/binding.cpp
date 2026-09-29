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
#include "file_provider.hpp"
#include "pcie_file_provider.hpp"
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
            "__init__",
            [](
                CLI* self,
                std::filesystem::path model_path,
                std::optional<std::filesystem::path> whisper_model_path,
                std::optional<std::filesystem::path> draft_model_path,
                std::optional<std::string> system_prompt,
                std::optional<std::string> chat_template,
                std::optional<std::string> pcie_serve_root,
                std::optional<std::string> pcie_subfolder,
                std::optional<std::filesystem::path> pcie_recv_root
            ) {
                // This is the only place we pick which provider to use.
                // --pcie sends all three pcie_* values, so we make a
                // PcieFileProvider (pulls files over PCIe). Disk mode sends
                // none, so we pass nullptr and the CLI uses the default
                // DiskFileProvider (reads local files).
                std::shared_ptr<FileProvider> file_provider =
                    (pcie_serve_root && pcie_subfolder && pcie_recv_root)
                        ? std::make_shared<PcieFileProvider>(
                              *pcie_recv_root, *pcie_serve_root, *pcie_subfolder)
                        : nullptr;
                new (self) CLI(
                    std::move(model_path), std::move(whisper_model_path),
                    std::move(draft_model_path), std::move(system_prompt),
                    std::move(chat_template), std::move(file_provider));
            },
            nb::arg("model_path"),
            nb::arg("whisper_model_path") = nb::none(),
            nb::arg("draft_model_path") = nb::none(),
            nb::arg("system_prompt") = nb::none(),
            nb::arg("chat_template") = nb::none(),
            nb::arg("pcie_serve_root") = nb::none(),
            nb::arg("pcie_subfolder") = nb::none(),
            nb::arg("pcie_recv_root") = nb::none()
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
                std::optional<std::string>,
                bool
            >(),
            nb::arg("model_path"),
            nb::arg("whisper_model_path") = nb::none(),
            nb::arg("draft_model_path") = nb::none(),
            nb::arg("system_prompt") = nb::none(),
            nb::arg("chat_template") = nb::none(),
            nb::arg("enable_thinking") = false
        )
        .def("run", &WEB::run)
    ;

    nb::class_<ZMQServer>(m, "ZMQServer")
        .def(
            nb::init<
                const std::filesystem::path&,
                uint32_t,
                std::optional<std::filesystem::path>
            >(),
            nb::arg("model_path"),
            nb::arg("port"),
            nb::arg("draft_model_path") = nb::none()
        )
        .def("run", &ZMQServer::run)
        .def("stop", &ZMQServer::stop)
    ;
}
