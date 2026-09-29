// Lifecycle helpers: claim a queue with a pid file, write the status file
// safely, send output to the queue log, and find the pep recv root.
// The host reads these files over SSH, so they must never be half-written.
#include "lifecycle.hpp"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace simaai {
namespace llima {
namespace pcie_backend {

namespace {

// kill(pid, 0) sends nothing; it only checks the pid. EPERM means the process
// exists but belongs to another user, so it is still alive.
bool pid_is_live(pid_t pid) {
    if (pid <= 0) return false;
    if (::kill(pid, 0) == 0) return true;
    return errno == EPERM;
}

std::optional<pid_t> read_pid_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    long long value = -1;
    in >> value;
    if (!in || value <= 0 || value > std::numeric_limits<pid_t>::max()) return std::nullopt;
    return static_cast<pid_t>(value);
}

bool proc_cmdline_contains(pid_t pid, const std::string& needle) {
    std::ifstream in("/proc/" + std::to_string(static_cast<long long>(pid)) + "/cmdline",
                     std::ios::in | std::ios::binary);
    if (!in) return false;
    std::string cmd((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    for (char& ch : cmd) {
        if (ch == '\0') ch = ' ';
    }
    return cmd.find(needle) != std::string::npos;
}

std::string trim(const std::string& s) {
    const auto begin = s.find_first_not_of(" \t\r");
    if (begin == std::string::npos) return "";
    const auto end = s.find_last_not_of(" \t\r");
    return s.substr(begin, end - begin + 1);
}

}  // namespace

std::string now_utc_iso8601() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    char buf[32]{};
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

std::string status_to_json(const BackendStatus& s) {
    nlohmann::ordered_json out{
        {"schema", s.schema},
        {"state", s.state},
        {"queue", s.queue},
        {"pid", s.pid},
        {"mode", s.mode},
        {"model", s.model},
        {"started_at", s.started_at},
        {"updated_at", s.updated_at},
        {"message", s.message},
        {"error_code", s.error_code.has_value() ? nlohmann::ordered_json(*s.error_code)
                                                : nlohmann::ordered_json(nullptr)},
    };
    return out.dump(2, ' ', false, nlohmann::json::error_handler_t::replace) + "\n";
}

StatusWriter::StatusWriter(std::filesystem::path path) : _path(std::move(path)) {}

// The host reads this file over SSH at any time (wait_ready polls it).
// Write a temp file first, then rename() it over the real one: rename is
// atomic, so the host sees the old file or the new one, never half a file.
// The host waits for state "ready" with pid == the pid it launched.
void StatusWriter::write(BackendStatus status) const {
    status.updated_at = now_utc_iso8601();
    const std::filesystem::path tmp =
        _path.string() + ".tmp." + std::to_string(static_cast<long long>(::getpid()));
    {
        std::ofstream out(tmp, std::ios::out | std::ios::trunc);
        if (!out) throw std::runtime_error("failed to open status temp file " + tmp.string());
        out << status_to_json(status);
        out.flush();
        if (!out) throw std::runtime_error("failed to write status temp file " + tmp.string());
    }
    std::error_code ec;
    std::filesystem::rename(tmp, _path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        throw std::runtime_error("failed to replace status file " + _path.string());
    }
}

QueueOwnership::QueueOwnership(std::filesystem::path pid_path, std::filesystem::path status_path,
                               std::string program_name)
  : _pid_path(std::move(pid_path)), _status_path(std::move(status_path)),
    _program_name(std::move(program_name)) {}

QueueOwnership::~QueueOwnership() { release(); }

// A pid file can be left behind after a crash, and the pid can later be reused
// by another program. So the old file is a real owner only if its pid is alive
// AND /proc/<pid>/cmdline shows our program. Otherwise it is stale: remove it
// (and its old status file, so the host cannot read a stale "ready").
void QueueOwnership::acquire() {
    if (std::filesystem::exists(_pid_path)) {
        const std::optional<pid_t> old_pid = read_pid_file(_pid_path);
        if (old_pid.has_value() && pid_is_live(*old_pid) &&
            proc_cmdline_contains(*old_pid, _program_name)) {
            throw QueueBusyError("queue already owned by live " + _program_name + " pid " +
                                 std::to_string(static_cast<long long>(*old_pid)));
        }
        std::error_code ec;
        std::filesystem::remove(_pid_path, ec);
        std::filesystem::remove(_status_path, ec);
    }
    // O_EXCL: if two backends start at the same time, only one can create the
    // file. The other gets EEXIST and reports the queue as busy.
    const int fd = ::open(_pid_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                          S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
        if (errno == EEXIST) throw QueueBusyError("queue pid file already exists: " + _pid_path.string());
        throw std::runtime_error("failed to create pid file " + _pid_path.string() + ": " +
                                 std::strerror(errno));
    }
    const std::string text = std::to_string(static_cast<long long>(::getpid())) + "\n";
    const ssize_t wrote = ::write(fd, text.data(), text.size());
    const int close_rc = ::close(fd);
    if (wrote != static_cast<ssize_t>(text.size()) || close_rc != 0) {
        std::error_code ec;
        std::filesystem::remove(_pid_path, ec);
        throw std::runtime_error("failed to write pid file " + _pid_path.string());
    }
    _owned = true;
}

void QueueOwnership::release() noexcept {
    if (!_owned) return;
    // Remove the file only if it still has our pid: a newer owner may have
    // taken over the queue, and its file must stay.
    const std::optional<pid_t> current = read_pid_file(_pid_path);
    if (current.has_value() && *current == ::getpid()) {
        std::error_code ec;
        std::filesystem::remove(_pid_path, ec);
    }
    _owned = false;
}

void require_directory(const std::filesystem::path& path) {
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) return;
    throw std::runtime_error("required lifecycle directory is missing: " + path.string());
}

void redirect_to_log(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC,
                          S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd < 0) {
        throw std::runtime_error("failed to open log file " + path.string() + ": " +
                                 std::strerror(errno));
    }
    if (::dup2(fd, STDOUT_FILENO) < 0 || ::dup2(fd, STDERR_FILENO) < 0) {
        const int saved = errno;
        ::close(fd);
        throw std::runtime_error(std::string("failed to redirect output to the log: ") +
                                 std::strerror(saved));
    }
    ::close(fd);
}

std::optional<std::filesystem::path> recv_root_from_pep_conf(const std::filesystem::path& conf) {
    std::ifstream in(conf);
    if (!in) return std::nullopt;
    std::optional<std::string> default_recv;
    std::map<std::string, std::string> recv_roots;
    std::optional<std::string> section;
    std::string raw;
    while (std::getline(in, raw)) {
        // Comments are whole-line or trailing '#'; paths never contain '#'.
        const std::string line = trim(raw.substr(0, raw.find('#')));
        if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') {
            section = trim(line.substr(1, line.size() - 2));
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (!section.has_value() && key == "default-recv") {
            default_recv = value;
        } else if (section == "recv") {
            recv_roots[key] = value;
        }
    }
    if (!default_recv.has_value()) return std::nullopt;
    const auto it = recv_roots.find(*default_recv);
    if (it == recv_roots.end()) return std::nullopt;
    return std::filesystem::path(it->second);
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
