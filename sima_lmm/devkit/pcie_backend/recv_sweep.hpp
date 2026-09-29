#ifndef _SIMA_LLIMA_PCIE_BACKEND_RECV_SWEEP_
#define _SIMA_LLIMA_PCIE_BACKEND_RECV_SWEEP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace simaai {
namespace llima {
namespace pcie_backend {

// Clean-up of the pep daemon's recv root (/tmp/pcie-recv on mla-hl83, inside the /tmp tmpfs, so
// every file in it uses RAM). The only use of that folder is: the daemon
// writes "<name>.part", renames it to "<name>", and the app that asked for it
// opens it at once. So a file that nobody has open and that became ready more
// than N seconds ago was either consumed or abandoned, and can go. On Linux a
// process that already has a file open or mapped keeps its data after unlink.

// Every regular file under `root` that some process has open or mapped,
// found by scanning <proc>/<pid>/fd and <proc>/<pid>/maps (like fuser).
// Directory handles are not files: the pep daemon keeps one on the root all
// the time. `unreadable` counts processes that could not be inspected (for a
// non-root caller: every root process, including the daemon).
struct OpenFiles {
    std::set<std::filesystem::path> paths;
    std::size_t unreadable = 0;
};
OpenFiles open_files_under(const std::filesystem::path& root,
                           const std::filesystem::path& proc = "/proc");

struct SweepEntry {
    std::filesystem::path path;
    std::uintmax_t bytes = 0;
    double age_s = 0.0;   // seconds since ctime
    std::string reason;   // for kept files: "in use", "young", "delete failed: ..."
};
struct SweepResult {
    std::vector<SweepEntry> deleted;
    std::vector<SweepEntry> kept;
    std::uintmax_t bytes_deleted = 0;
};

// Delete each regular file under `root` that is not in `in_use` and whose
// ctime is more than `max_age` before `now`. ctime, not mtime: the rename
// sets ctime to the moment the file became ready and nothing can set it back,
// while mtime can be anything (touch -d, or svc XF_PRESERVE copies the
// source's). Never deletes directories or symlinks, never follows symlinks,
// never throws.
SweepResult sweep_recv_root(const std::filesystem::path& root,
                            const std::set<std::filesystem::path>& in_use,
                            std::chrono::seconds max_age,
                            std::chrono::system_clock::time_point now =
                                std::chrono::system_clock::now());

// MemAvailable from /proc/meminfo, in MiB; nullopt if it cannot be read.
std::optional<long long> mem_available_mib(
    const std::filesystem::path& meminfo = "/proc/meminfo");

// "30" -> 30. Empty, negative, out of range, or trailing text -> nullopt.
std::optional<long long> parse_non_negative(std::string_view text);

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
