// Unit test for PcieFileProvider using an injected fake fetch function.
// Proves: reserve does not fetch; fetch materializes the file, maps the remote
// name to "<subfolder>/<relative>", and passes the local dst as a name relative
// to recv_root (never absolute); evict deletes the file; get_path treats a
// missing file as a hard error (throws) while exists() probes softly (false on
// ENOENT). The fakes model the daemon resolving that relative name under recv_root.
#include <cerrno>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "pcie_file_provider.hpp"

namespace {
using simaai::llima::PcieFileProvider;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }

std::filesystem::path make_temp_dir() {
    std::string tmpl = (std::filesystem::temp_directory_path() / "llima_pcie_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end()); buf.push_back('\0');
    char* made = ::mkdtemp(buf.data());
    if (!made) throw std::runtime_error("mkdtemp failed");
    return std::filesystem::path(made);
}

void test_reserve_does_not_fetch() {
    auto root = make_temp_dir();
    bool called = false;
    PcieFileProvider p(root, "models", "llama",
        [&](const std::string&, const std::string&, const std::string&) { called = true; return 0; });
    const auto path = p.reserve("elf_files/x.elf");
    expect(path == root / "elf_files" / "x.elf", "reserve must return recv_root/name");
    expect(!called, "reserve must NOT call the fetch function");
    expect(!std::filesystem::exists(path), "reserve must not create the file");
}

void test_fetch_maps_remote_name_and_writes_file() {
    auto root = make_temp_dir();
    std::string seen_serve, seen_remote, seen_local;
    PcieFileProvider p(root, "models", "llama",
        [&](const std::string& s, const std::string& r, const std::string& l) {
            seen_serve = s; seen_remote = r; seen_local = l;
            // model the daemon resolving the relative dst under recv_root
            const auto dst = root / l;
            std::filesystem::create_directories(dst.parent_path());
            std::ofstream(dst, std::ios::binary) << "ELF";   // fake the pull
            return 0;
        });
    const auto path = p.reserve("elf_files/x.elf");
    p.fetch(path);
    expect(seen_serve == "models", "fetch must pass the serve-root");
    expect(seen_remote == "llama/elf_files/x.elf", "fetch must map path to <subfolder>/<relative>");
    expect(seen_local == "elf_files/x.elf", "fetch local dest must be the name relative to recv_root");
    expect(std::filesystem::is_regular_file(path), "fetch must materialize the file");
}

void test_evict_deletes_file() {
    auto root = make_temp_dir();
    PcieFileProvider p(root, "models", "llama",
        [&](const std::string&, const std::string&, const std::string& l) {
            const auto dst = root / l;
            std::filesystem::create_directories(dst.parent_path());
            std::ofstream(dst, std::ios::binary) << "ELF"; return 0; });
    const auto path = p.reserve("elf_files/x.elf");
    p.fetch(path);
    p.evict(path);
    expect(!std::filesystem::exists(path), "evict must delete the on-disk copy");
}

void test_fetch_throws_on_error() {
    auto root = make_temp_dir();
    PcieFileProvider p(root, "models", "llama",
        [](const std::string&, const std::string&, const std::string&) { return -5; });
    bool threw = false;
    try { p.fetch(p.reserve("elf_files/x.elf")); } catch (const std::exception&) { threw = true; }
    expect(threw, "fetch must throw when the fetch function returns a nonzero error");
}

void test_get_path_not_found_throws() {
    auto root = make_temp_dir();
    PcieFileProvider p(root, "models", "llama",
        [](const std::string&, const std::string&, const std::string&) { return -ENOENT; });
    bool threw = false;
    try { p.get_path("missing.json"); } catch (const std::exception&) { threw = true; }
    expect(threw, "get_path on ENOENT must throw (required file missing / root unconfigured)");
}

void test_exists_false_on_enoent_true_on_hit() {
    auto root = make_temp_dir();
    // ENOENT -> optional file absent -> exists() is false, and must NOT throw.
    PcieFileProvider absent(root, "models", "llama",
        [](const std::string&, const std::string&, const std::string&) { return -ENOENT; });
    bool threw = false;
    bool present = true;
    try { present = absent.exists("devkit/llama_embeddings.bin"); }
    catch (const std::exception&) { threw = true; }
    expect(!threw, "exists() on ENOENT must not throw");
    expect(!present, "exists() on ENOENT must return false");

    // rc==0 -> present.
    PcieFileProvider hit(root, "models", "llama",
        [&](const std::string&, const std::string&, const std::string& l) {
            const auto dst = root / l;
            std::filesystem::create_directories(dst.parent_path());
            std::ofstream(dst, std::ios::binary) << "BIN"; return 0; });
    expect(hit.exists("devkit/llama_embeddings.bin"), "exists() must be true when the pull succeeds");
}

void test_exists_throws_on_other_error() {
    auto root = make_temp_dir();
    PcieFileProvider p(root, "models", "llama",
        [](const std::string&, const std::string&, const std::string&) { return -EIO; });
    bool threw = false;
    try { p.exists("devkit/llama_embeddings.bin"); } catch (const std::exception&) { threw = true; }
    expect(threw, "exists() on a non-ENOENT error must throw");
}

// A fake fetch that writes `content` at the relative dst and counts calls.
PcieFileProvider::FetchFn counting_fetch(const std::filesystem::path& root, int& calls,
                                         const std::string& content = "DATA") {
    return [&root, &calls, content](const std::string&, const std::string&, const std::string& l) {
        ++calls;
        const auto dst = root / l;
        std::filesystem::create_directories(dst.parent_path());
        std::ofstream(dst, std::ios::binary) << content;
        return 0;
    };
}

std::string read_all(std::istream& s) {
    return std::string(std::istreambuf_iterator<char>(s), std::istreambuf_iterator<char>());
}

void test_exists_then_open_stream_pulls_once() {
    auto root = make_temp_dir();
    int calls = 0;
    PcieFileProvider p(root, "models", "llama", counting_fetch(root, calls, "BIN"));
    expect(p.exists("devkit/x.bin"), "exists() must be true after a good pull");
    auto s = p.open_stream("devkit/x.bin");
    expect(calls == 1, "exists() + open_stream() must pull the file only once");
    expect(read_all(*s) == "BIN", "open_stream must return the pulled bytes");
}

void test_get_path_twice_pulls_once() {
    auto root = make_temp_dir();
    int calls = 0;
    PcieFileProvider p(root, "models", "llama", counting_fetch(root, calls));
    p.get_path("devkit/a.npy");
    p.get_path("devkit/a.npy");
    expect(calls == 1, "a second get_path of a file still on disk must not pull again");
}

void test_open_stream_deletes_file_but_stream_still_reads() {
    auto root = make_temp_dir();
    int calls = 0;
    PcieFileProvider p(root, "models", "llama", counting_fetch(root, calls, "CFG"));
    auto s = p.open_stream("devkit/c.json");
    expect(!std::filesystem::exists(root / "devkit" / "c.json"),
           "open_stream must delete the disk copy after opening it");
    expect(read_all(*s) == "CFG", "the open stream must still read all bytes after the delete");
    auto s2 = p.open_stream("devkit/c.json");
    expect(calls == 2, "opening the file again after the delete must pull it again");
    expect(read_all(*s2) == "CFG", "the second stream must read the bytes too");
}

void test_release_deletes_and_next_get_path_pulls() {
    auto root = make_temp_dir();
    int calls = 0;
    PcieFileProvider p(root, "models", "llama", counting_fetch(root, calls));
    const auto path = p.get_path("devkit/d2t.npy");
    p.release("devkit/d2t.npy");
    expect(!std::filesystem::exists(path), "release must delete the pulled copy");
    p.get_path("devkit/d2t.npy");
    expect(calls == 2, "get_path after release must pull again");
}

void test_stale_file_from_earlier_run_is_pulled_fresh() {
    auto root = make_temp_dir();
    std::filesystem::create_directories(root / "devkit");
    std::ofstream(root / "devkit" / "old.json", std::ios::binary) << "STALE";
    int calls = 0;
    PcieFileProvider p(root, "models", "llama", counting_fetch(root, calls, "FRESH"));
    auto s = p.open_stream("devkit/old.json");
    expect(calls == 1, "a file this provider did not pull must be pulled, not reused");
    expect(read_all(*s) == "FRESH", "the stream must read the freshly pulled bytes");
}

void test_pulls_files_is_true() {
    auto root = make_temp_dir();
    int calls = 0;
    PcieFileProvider p(root, "models", "llama", counting_fetch(root, calls));
    expect(p.pulls_files(), "PcieFileProvider must report pulls_files() == true");
}

void test_get_path_other_errors_throw() {
    auto root = make_temp_dir();
    PcieFileProvider p(root, "models", "llama",
        [](const std::string&, const std::string&, const std::string&) { return -EIO; });
    bool threw = false;
    try { p.get_path("file.elf"); } catch (const std::exception&) { threw = true; }
    expect(threw, "get_path on non-ENOENT error must throw");
}
}  // namespace

// llima run's provider choice: all three PCIe values, or none. Some of them
// is an error, never a silent fall back to the local disk.
void test_provider_from_options_all_or_none() {
    using simaai::llima::pcie_provider_from_options;
    expect(pcie_provider_from_options(std::nullopt, std::nullopt, std::nullopt) == nullptr,
           "no PCIe values = disk (nullptr)");
    const auto p = pcie_provider_from_options(std::string("models"), std::string("m"),
                                              std::filesystem::path("/tmp/pcie-recv"));
    expect(p != nullptr && p->pulls_files(), "all three = a PCIe provider");
    std::string what;
    try {
        pcie_provider_from_options(std::string("models"), std::string("m"), std::nullopt);
    } catch (const std::invalid_argument& e) {
        what = e.what();
    }
    expect(what.find("missing: pcie_recv_root") != std::string::npos,
           "a missing recv root is refused and named: " + what);
    what.clear();
    try {
        pcie_provider_from_options(std::nullopt, std::string("m"), std::nullopt);
    } catch (const std::invalid_argument& e) {
        what = e.what();
    }
    expect(what.find("pcie_serve_root, pcie_recv_root") != std::string::npos,
           "every missing value is named: " + what);
}

int main() {
    try {
        test_reserve_does_not_fetch();
        test_fetch_maps_remote_name_and_writes_file();
        test_evict_deletes_file();
        test_fetch_throws_on_error();
        test_get_path_not_found_throws();
        test_exists_false_on_enoent_true_on_hit();
        test_exists_throws_on_other_error();
        test_get_path_other_errors_throw();
        test_exists_then_open_stream_pulls_once();
        test_get_path_twice_pulls_once();
        test_open_stream_deletes_file_but_stream_still_reads();
        test_release_deletes_and_next_get_path_pulls();
        test_stale_file_from_earlier_run_is_pulled_fresh();
        test_pulls_files_is_true();
        test_provider_from_options_all_or_none();
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
    if (failures == 0) { std::cout << "pcie_file_provider_test: all checks passed\n"; return 0; }
    std::cerr << "pcie_file_provider_test: " << failures << " failed\n"; return 1;
}
