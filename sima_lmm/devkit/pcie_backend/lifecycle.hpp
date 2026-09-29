// Lifecycle helpers for pcie-genai-backend: the queue pid file, the status
// file, the log file and the pep recv root. The host (RemoteRuntime) starts
// us over SSH and polls the status file until state is "ready".
// Same file layout as pcie_pipeline_builder, so the host treats both the same.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_LIFECYCLE_
#define _SIMA_LLIMA_PCIE_BACKEND_LIFECYCLE_

#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

namespace simaai {
namespace llima {
namespace pcie_backend {

// The card-side status record the host polls over SSH
// (neat/core/pcie_host/src/RemoteRuntime.cpp read_status / wait_ready).
// Same fields as pcie_pipeline_builder.cpp writes, so both programs look the
// same to RemoteRuntime.
struct BackendStatus {
    int schema = 1;
    std::string state;     // starting / ready / stopping / exited / failed
    int queue = -1;
    long long pid = -1;
    std::string mode = "genai";
    std::string model;
    std::string started_at;
    std::string updated_at;
    std::string message;
    std::optional<std::string> error_code;
};

std::string now_utc_iso8601();
std::string status_to_json(const BackendStatus& status);

// Writes the status file atomically (temp file + rename), so the host never
// reads half a file.
class StatusWriter {
    public:
        explicit StatusWriter(std::filesystem::path path);
        void write(BackendStatus status) const;
    private:
        std::filesystem::path _path;
};

class QueueBusyError : public std::runtime_error {
    public:
        using std::runtime_error::runtime_error;
};

// Claims a queue with an O_EXCL pid file. A pid file left by a dead process,
// or by a live process that is not `program_name`, is stale and taken over.
class QueueOwnership {
    public:
        QueueOwnership(std::filesystem::path pid_path, std::filesystem::path status_path,
                       std::string program_name);
        ~QueueOwnership();
        QueueOwnership(const QueueOwnership&) = delete;
        QueueOwnership& operator=(const QueueOwnership&) = delete;

        void acquire();           // throws QueueBusyError or std::runtime_error
        void release() noexcept;  // removes the pid file only if it is still ours
    private:
        std::filesystem::path _pid_path;
        std::filesystem::path _status_path;
        std::string _program_name;
        bool _owned = false;
};

void require_directory(const std::filesystem::path& path);
// Send stdout and stderr to the queue log (the host launches us with nohup).
void redirect_to_log(const std::filesystem::path& path);

// The pep daemon's default receive directory: `default-recv = <name>` at top
// level, resolved through the `[recv]` section (e.g. recv5g -> /tmp/pcie-recv).
// Port of sima_lmm/devkit/pcie_config.py recv_root_from_pep_conf().
std::optional<std::filesystem::path> recv_root_from_pep_conf(
    const std::filesystem::path& conf = "/etc/simaai/simaai-pep-daemon.conf");

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
