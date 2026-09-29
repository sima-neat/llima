// Unit test for the pcie-genai-backend lifecycle helpers: the status file the
// host polls for READY, the queue pid file, and the pep-daemon recv root.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "lifecycle.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

std::filesystem::path make_temp_dir() {
    std::string tmpl = (std::filesystem::temp_directory_path() / "llima_p4_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end()); buf.push_back('\0');
    char* made = ::mkdtemp(buf.data());
    if (!made) throw std::runtime_error("mkdtemp failed");
    return std::filesystem::path(made);
}

std::string read_file(const std::filesystem::path& p) {
    std::ifstream in(p);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void test_status_file_matches_what_the_host_reads() {
    const auto dir = make_temp_dir();
    StatusWriter writer(dir / "q3.status");
    BackendStatus s;
    s.state = "ready";
    s.queue = 3;
    s.pid = ::getpid();
    s.model = "Llama-3.2-3B-Instruct-a16w4";
    writer.write(s);
    const auto j = nlohmann::json::parse(read_file(dir / "q3.status"));
    // RemoteRuntime::read_status reads state, queue, pid, message, error_code.
    expect(j["state"] == "ready", "state");
    expect(j["queue"] == 3, "queue");
    expect(j["pid"] == static_cast<long long>(::getpid()), "pid");
    expect(j["error_code"].is_null(), "error_code is null when unset");
    expect(!j["updated_at"].get<std::string>().empty(), "updated_at is stamped");
    for (const auto& e : std::filesystem::directory_iterator(dir))
        expect(e.path().filename() == "q3.status", "no temp file is left behind");
}

void test_queue_ownership() {
    const auto dir = make_temp_dir();
    const auto pid_path = dir / "q3.pid";
    const auto status_path = dir / "q3.status";
    // The program name is matched against /proc/<pid>/cmdline; this test
    // binary's own name stands in for "pcie-genai-backend".
    QueueOwnership first(pid_path, status_path, "pcie_genai_lifecycle_test");
    first.acquire();
    expect(std::stoll(read_file(pid_path)) == ::getpid(), "pid file holds our pid");
    QueueOwnership second(pid_path, status_path, "pcie_genai_lifecycle_test");
    bool busy = false;
    try { second.acquire(); } catch (const QueueBusyError&) { busy = true; }
    expect(busy, "a live owner with our program name makes the queue busy");
    first.release();
    expect(!std::filesystem::exists(pid_path), "release removes the pid file");
}

void test_stale_pid_file_is_taken_over() {
    const auto dir = make_temp_dir();
    const auto pid_path = dir / "q3.pid";
    const auto status_path = dir / "q3.status";
    // pid 1 is alive but is not our program: the file is stale.
    std::ofstream(pid_path) << "1\n";
    std::ofstream(status_path) << "{}";
    QueueOwnership owner(pid_path, status_path, "no-such-program-name");
    owner.acquire();
    expect(std::stoll(read_file(pid_path)) == ::getpid(), "a stale pid file is replaced");
    expect(!std::filesystem::exists(status_path), "the stale status file is removed");
}

void test_recv_root_from_pep_conf() {
    const auto dir = make_temp_dir();
    const auto conf = dir / "simaai-pep-daemon.conf";
    std::ofstream(conf) << "# pep daemon\n"
                           "default-recv  = recv5g\n"
                           "\n"
                           "[recv]\n"
                           "recv5g = /tmp/pcie-recv   # inside a 5 GB /tmp\n"
                           "tmp = /tmp\n";
    const auto root = recv_root_from_pep_conf(conf);
    expect(root.has_value() && *root == "/tmp/pcie-recv", "default-recv name resolves via [recv]");
    expect(!recv_root_from_pep_conf(dir / "missing.conf").has_value(), "missing file -> nullopt");
    const auto bad = dir / "bad.conf";
    std::ofstream(bad) << "default-recv = nope\n[recv]\nrecv5g = /tmp/pcie-recv\n";
    expect(!recv_root_from_pep_conf(bad).has_value(), "unknown name -> nullopt");
}

void test_require_directory() {
    const auto dir = make_temp_dir();
    bool threw = false;
    try { require_directory(dir / "nope"); } catch (const std::runtime_error&) { threw = true; }
    expect(threw, "a missing lifecycle directory is an error");
    require_directory(dir);
}
}  // namespace

int main() {
    test_status_file_matches_what_the_host_reads();
    test_queue_ownership();
    test_stale_pid_file_is_taken_over();
    test_recv_root_from_pep_conf();
    test_require_directory();
    if (failures == 0) std::cout << "pcie_genai_lifecycle_test passed\n";
    return failures == 0 ? 0 : 1;
}
