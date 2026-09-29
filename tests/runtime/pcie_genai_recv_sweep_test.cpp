// Unit test for the recv-root sweep: delete closed files older than N,
// keep open/mapped/young files, never touch directories or symlinks.
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "recv_sweep.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
namespace fs = std::filesystem;
using namespace std::chrono_literals;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

fs::path make_temp_dir() {
    std::string tmpl = (fs::temp_directory_path() / "llima_sweep_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end()); buf.push_back('\0');
    char* made = ::mkdtemp(buf.data());
    if (!made) throw std::runtime_error("mkdtemp failed");
    return fs::weakly_canonical(fs::path(made));   // same form the sweep uses
}

void write_file(const fs::path& p, std::size_t bytes) {
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << std::string(bytes, 'x');
}

std::chrono::system_clock::time_point ctime_of(const fs::path& p) {
    struct stat sb{};
    ::lstat(p.c_str(), &sb);
    return std::chrono::system_clock::time_point(std::chrono::duration_cast<std::chrono::system_clock::duration>(
        std::chrono::seconds(sb.st_ctim.tv_sec) + std::chrono::nanoseconds(sb.st_ctim.tv_nsec)));
}

bool has(const std::vector<SweepEntry>& v, const fs::path& p, const std::string& reason = "") {
    for (const auto& e : v) if (e.path == p && (reason.empty() || e.reason == reason)) return true;
    return false;
}

void test_old_unused_file_is_deleted() {
    const auto root = make_temp_dir();
    const auto f = root / "elf_files" / "a.elf";
    write_file(f, 1000);
    const auto r = sweep_recv_root(root, {}, 30s, ctime_of(f) + 31s);
    expect(!fs::exists(f), "an unused file older than N must be deleted");
    expect(has(r.deleted, f) && r.bytes_deleted == 1000, "the deletion is reported with its size");
    expect(fs::is_directory(root / "elf_files"), "directories must stay");
}

void test_age_boundary() {
    const auto root = make_temp_dir();
    const auto f = root / "b.bin";
    write_file(f, 10);
    auto r = sweep_recv_root(root, {}, 30s, ctime_of(f) + 30s);
    expect(fs::exists(f) && has(r.kept, f, "young"), "age == N is kept (only older than N is deleted)");
    r = sweep_recv_root(root, {}, 30s, ctime_of(f) + 31s);
    expect(!fs::exists(f), "age N+1 s is deleted");
}

void test_young_file_is_kept() {
    const auto root = make_temp_dir();
    const auto f = root / "fresh.bin";
    write_file(f, 10);
    const auto r = sweep_recv_root(root, {}, 30s);   // real now: ~0 s old
    expect(fs::exists(f) && has(r.kept, f, "young"), "a file just renamed into place must be kept");
}

void test_in_use_file_is_kept_even_if_old() {
    const auto root = make_temp_dir();
    const auto f = root / "held.bin";
    write_file(f, 10);
    const auto r = sweep_recv_root(root, {f}, 30s, ctime_of(f) + 3600s);
    expect(fs::exists(f) && has(r.kept, f, "in use"), "an open or mapped file is never deleted");
}

// Review focus 3.
void test_symlinks_are_not_followed_or_deleted() {
    const auto root = make_temp_dir();
    const auto outside = make_temp_dir() / "target.bin";
    write_file(outside, 10);
    fs::create_symlink(outside, root / "link.bin");
    fs::create_directory_symlink(outside.parent_path(), root / "linkdir");
    const auto r = sweep_recv_root(root, {}, 30s, ctime_of(outside) + 3600s);
    expect(fs::is_symlink(root / "link.bin"), "a symlink is not deleted");
    expect(fs::exists(outside), "a symlink target outside the root is not touched");
    expect(r.deleted.empty(), "nothing is deleted through symlinks");
}

// Review focus 4.
void test_missing_root() {
    const auto r = sweep_recv_root("/nonexistent/llima_sweep_root", {}, 30s);
    expect(r.deleted.empty() && r.kept.empty(), "a missing root is a no-op, not an error");
}

void test_open_file_is_seen() {
    const auto root = make_temp_dir();
    const auto f = root / "devkit" / "open.bin";
    write_file(f, 10);
    {
        std::ifstream held(f, std::ios::binary);
        expect(open_files_under(root).paths.count(f) == 1, "a file this process has open is in use");
    }
    expect(open_files_under(root).paths.count(f) == 0, "once closed it is no longer in use");
}

// Review focus 2 (name with a space).
void test_mapped_file_is_seen_after_close() {
    const auto root = make_temp_dir();
    const auto f = root / "mapped file.gguf";
    write_file(f, 4096);
    const int fd = ::open(f.c_str(), O_RDONLY);
    void* map = ::mmap(nullptr, 4096, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);   // only the mapping keeps it in use now
    expect(map != MAP_FAILED && open_files_under(root).paths.count(f) == 1,
           "a mapped file (fd closed) is in use, even with a space in its name");
    ::munmap(map, 4096);
    expect(open_files_under(root).paths.count(f) == 0, "after munmap it is no longer in use");
}

void test_directory_handles_are_ignored() {
    const auto root = make_temp_dir();
    const int dfd = ::open(root.c_str(), O_RDONLY | O_DIRECTORY);   // like the pep daemon's fd 6 / 9
    const auto open = open_files_under(root);
    expect(open.paths.count(root) == 0 && open.paths.empty(), "a directory handle on the root is not a file in use");
    ::close(dfd);
}

// Review focus 1 and 2: a fake /proc.
void test_fake_proc() {
    const auto root = make_temp_dir();
    const auto proc = make_temp_dir();
    // pid 100: an fd link into the root, a mapping of a deleted file, a mapping outside.
    fs::create_directories(proc / "100" / "fd");
    fs::create_symlink(root / "x.bin", proc / "100" / "fd" / "3");
    fs::create_symlink("/etc/hostname", proc / "100" / "fd" / "4");
    std::ofstream(proc / "100" / "maps")
        << "ffff0000-ffff1000 r--p 00000000 00:1a 42 " << (root / "y.bin").string() << '\n'
        << "ffff1000-ffff2000 r--p 00000000 00:1a 43 " << (root / "z.bin").string() << " (deleted)\n"
        << "ffff2000-ffff3000 r--p 00000000 00:1a 44 /usr/lib/libc.so.6\n";
    // pid 200: cannot be inspected ("fd" is not a directory, like EACCES for a non-root caller).
    fs::create_directories(proc / "200");
    std::ofstream(proc / "200" / "fd") << "";
    // not a pid: ignored.
    fs::create_directories(proc / "self_like" / "fd");
    const auto open = open_files_under(root, proc);
    expect(open.paths.count(root / "x.bin") == 1, "an fd link into the root is in use");
    expect(open.paths.count(root / "y.bin") == 1, "a mapping into the root is in use");
    expect(open.paths.count(root / "z.bin") == 0, "a (deleted) mapping is ignored");
    expect(open.paths.size() == 2, "nothing outside the root is reported");
    expect(open.unreadable == 1, "a process that cannot be inspected is counted");
}

void test_mem_available() {
    const auto dir = make_temp_dir();
    std::ofstream(dir / "meminfo") << "MemTotal:        6049792 kB\nMemFree:          100000 kB\n"
                                      "MemAvailable:    5756556 kB\nBuffers: 0 kB\n";
    expect(mem_available_mib(dir / "meminfo") == 5621, "MemAvailable kB -> MiB");
    std::ofstream(dir / "nokey") << "MemTotal: 1 kB\n";
    expect(!mem_available_mib(dir / "nokey").has_value(), "missing key -> nullopt");
    expect(!mem_available_mib(dir / "missing").has_value(), "missing file -> nullopt");
}

// Review focus 5.
void test_parse_non_negative() {
    expect(parse_non_negative("30") == 30 && parse_non_negative("0") == 0, "plain numbers");
    expect(!parse_non_negative("").has_value(), "empty");
    expect(!parse_non_negative("abc").has_value(), "not a number");
    expect(!parse_non_negative("-5").has_value(), "negative");
    expect(!parse_non_negative("30s").has_value(), "trailing text");
    expect(!parse_non_negative("99999999999999999999").has_value(), "out of range");
}
}  // namespace

int main() {
    test_old_unused_file_is_deleted();
    test_age_boundary();
    test_young_file_is_kept();
    test_in_use_file_is_kept_even_if_old();
    test_symlinks_are_not_followed_or_deleted();
    test_missing_root();
    test_open_file_is_seen();
    test_mapped_file_is_seen_after_close();
    test_directory_handles_are_ignored();
    test_fake_proc();
    test_mem_available();
    test_parse_non_negative();
    if (failures == 0) std::cout << "pcie_genai_recv_sweep_test passed\n";
    return failures == 0 ? 0 : 1;
}
