#include "recv_sweep.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <fstream>
#include <system_error>

#include <sys/stat.h>

namespace simaai {
namespace llima {
namespace pcie_backend {

namespace fs = std::filesystem;

namespace {

// The root in the form both functions compare against, plus "/" so that
// "/tmp/pcie-recv" itself (a directory handle) and "/tmp/pcie-recvX" never match.
std::string root_prefix(const fs::path& root) {
    std::error_code ec;
    const fs::path base = fs::weakly_canonical(root, ec);
    return (ec ? root.lexically_normal() : base).string() + "/";
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

bool is_pid(const std::string& name) {
    return !name.empty() && std::all_of(name.begin(), name.end(),
                                        [](unsigned char c) { return std::isdigit(c) != 0; });
}

std::chrono::system_clock::time_point to_time_point(const struct timespec& ts) {
    return std::chrono::system_clock::time_point(
        std::chrono::duration_cast<std::chrono::system_clock::duration>(
            std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec)));
}

}  // namespace

OpenFiles open_files_under(const fs::path& root, const fs::path& proc) {
    OpenFiles out;
    const std::string prefix = root_prefix(root);
    const std::string deleted_suffix = " (deleted)";
    std::error_code ec;
    fs::directory_iterator pids(proc, ec);
    for (const fs::directory_iterator end; !ec && pids != end; pids.increment(ec)) {
        const fs::path pid_dir = pids->path();
        if (!is_pid(pid_dir.filename().string())) continue;

        std::error_code fd_ec;
        fs::directory_iterator fds(pid_dir / "fd", fd_ec);
        if (fd_ec) {
            // ENOENT: the process just exited. Anything else (EACCES for
            // another user's process): we cannot know what it holds.
            if (fd_ec != std::errc::no_such_file_or_directory) ++out.unreadable;
            continue;
        }
        for (const fs::directory_iterator end; !fd_ec && fds != end; fds.increment(fd_ec)) {
            std::error_code link_ec;
            const std::string target = fs::read_symlink(fds->path(), link_ec).string();
            if (!link_ec && starts_with(target, prefix)) {
                out.paths.insert(fs::path(target).lexically_normal());
            }
        }

        // A mapped file needs no open fd (mmap, then close). Line format:
        // "addr perms offset dev inode   /path", the path may hold spaces.
        std::ifstream maps(pid_dir / "maps");
        std::string line;
        while (std::getline(maps, line)) {
            const auto at = line.find(prefix);
            if (at == std::string::npos) continue;
            std::string path = line.substr(at);
            if (path.size() >= deleted_suffix.size() &&
                path.compare(path.size() - deleted_suffix.size(), deleted_suffix.size(),
                             deleted_suffix) == 0) {
                continue;   // already unlinked: nothing to delete
            }
            out.paths.insert(fs::path(path).lexically_normal());
        }
    }
    return out;
}

SweepResult sweep_recv_root(const fs::path& root, const std::set<fs::path>& in_use,
                            const std::chrono::seconds max_age,
                            const std::chrono::system_clock::time_point now) {
    SweepResult result;
    std::error_code ec;
    const fs::path base = fs::weakly_canonical(root, ec);
    if (ec || !fs::is_directory(base, ec)) return result;

    // Default options: directory symlinks are not followed.
    fs::recursive_directory_iterator it(base, fs::directory_options::skip_permission_denied, ec);
    for (const fs::recursive_directory_iterator end; !ec && it != end; it.increment(ec)) {
        std::error_code st_ec;
        const fs::file_status st = it->symlink_status(st_ec);
        if (st_ec || !fs::is_regular_file(st)) continue;   // dirs, symlinks, sockets: never touched

        const fs::path path = it->path().lexically_normal();
        struct stat sb{};
        if (::lstat(path.c_str(), &sb) != 0) continue;   // gone meanwhile (its consumer deleted it)
        const auto ctime = to_time_point(sb.st_ctim);

        SweepEntry entry;
        entry.path = path;
        entry.bytes = static_cast<std::uintmax_t>(sb.st_size);
        entry.age_s = std::chrono::duration<double>(now - ctime).count();

        if (in_use.count(path) != 0) {
            entry.reason = "in use";
            result.kept.push_back(std::move(entry));
            continue;
        }
        if (now - ctime <= max_age) {
            entry.reason = "young";
            result.kept.push_back(std::move(entry));
            continue;
        }
        std::error_code rm_ec;
        if (fs::remove(path, rm_ec)) {
            result.bytes_deleted += entry.bytes;
            result.deleted.push_back(std::move(entry));
        } else if (rm_ec) {
            entry.reason = "delete failed: " + rm_ec.message();
            result.kept.push_back(std::move(entry));
        }
    }
    return result;
}

std::optional<long long> mem_available_mib(const fs::path& meminfo) {
    std::ifstream in(meminfo);
    std::string line;
    constexpr std::string_view key = "MemAvailable:";
    while (std::getline(in, line)) {
        if (line.compare(0, key.size(), key) != 0) continue;
        std::size_t i = key.size();
        while (i < line.size() && line[i] == ' ') ++i;
        long long kb = 0;
        const auto [ptr, err] = std::from_chars(line.data() + i, line.data() + line.size(), kb);
        if (err != std::errc() || ptr == line.data() + i) return std::nullopt;
        return kb / 1024;
    }
    return std::nullopt;
}

std::optional<long long> parse_non_negative(std::string_view text) {
    long long value = 0;
    const auto [ptr, err] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || err != std::errc() || ptr != text.data() + text.size() || value < 0) {
        return std::nullopt;
    }
    return value;
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
