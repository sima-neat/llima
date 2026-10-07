// Unit test for the LLiMa file seam: FileProvider + DiskFileProvider.
//
// DiskFileProvider is the default provider. It must behave exactly like
// today's direct disk reads: names are relative to the model root and
// include the sub-folder ("devkit/x.json", "elf_files/y.elf"); get_path
// joins onto the root, open_stream opens a byte stream, and release is a
// no-op (nothing is deleted on the disk path).

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <istream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include "file_provider.hpp"
#include "whisper_model.hpp"

namespace {

using simaai::llima::DiskFileProvider;
using simaai::llima::FileProvider;

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::string read_all(std::istream& in) {
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::filesystem::path make_temp_dir() {
    std::string tmpl =
        (std::filesystem::temp_directory_path() / "llima_fp_XXXXXX").string();
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    char* made = ::mkdtemp(buf.data());
    if (made == nullptr) {
        throw std::runtime_error("mkdtemp failed");
    }
    return std::filesystem::path(made);
}

// A model layout on disk, exactly as LLiMa expects it today.
struct TempModel {
    std::filesystem::path root;
    TempModel() : root(make_temp_dir()) {
        write_file(root / "devkit" / "vlm_config.json", R"({"model_type":"llama"})");
        // A .bin embedding with an embedded NUL byte to prove byte fidelity.
        write_file(root / "devkit" / "embeddings.bin",
                   std::string("\x01\x00\x02\x03", 4));
        write_file(root / "elf_files" / "model.elf", "FAKE-ELF-BYTES");
    }
    ~TempModel() {
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }
};

void test_get_path_joins_root_and_relative_name() {
    TempModel model;
    DiskFileProvider provider(model.root);
    const auto elf = provider.get_path("elf_files/model.elf");
    expect(elf == model.root / "elf_files" / "model.elf",
           "get_path must join the model root with the folder-qualified name");
    expect(std::filesystem::is_regular_file(elf),
           "get_path must point at the real file that exists on disk");
}

void test_open_stream_returns_file_contents() {
    TempModel model;
    DiskFileProvider provider(model.root);
    auto stream = provider.open_stream("devkit/vlm_config.json");
    expect(stream != nullptr, "open_stream must return a stream");
    expect(read_all(*stream) == R"({"model_type":"llama"})",
           "open_stream must yield the exact file contents");
}

void test_open_stream_preserves_binary_bytes() {
    TempModel model;
    DiskFileProvider provider(model.root);
    auto stream = provider.open_stream("devkit/embeddings.bin");
    const std::string expected("\x01\x00\x02\x03", 4);
    expect(read_all(*stream) == expected,
           "open_stream must preserve raw bytes (including NUL) for .bin files");
}

void test_release_is_a_no_op_on_disk() {
    TempModel model;
    DiskFileProvider provider(model.root);
    const auto elf = provider.get_path("elf_files/model.elf");
    provider.release("elf_files/model.elf");
    expect(std::filesystem::is_regular_file(elf),
           "release must not delete files on the disk provider");
}

void test_usable_through_base_interface() {
    TempModel model;
    DiskFileProvider concrete(model.root);
    FileProvider& provider = concrete;
    auto stream = provider.open_stream("devkit/vlm_config.json");
    expect(read_all(*stream) == R"({"model_type":"llama"})",
           "DiskFileProvider must work when called through FileProvider&");
}

void test_reserve_returns_path_without_requiring_file() {
    TempModel model;
    DiskFileProvider provider(model.root);
    // reserve names a file that does NOT exist yet; it must still return the
    // path (define-time reservation must not require the bytes to be present).
    const auto p = provider.reserve("elf_files/not_pulled_yet.elf");
    expect(p == model.root / "elf_files" / "not_pulled_yet.elf",
           "reserve must join root and name, like get_path, without fetching");
    expect(!std::filesystem::exists(p),
           "reserve must not create or require the file on disk");
}

void test_reserve_matches_get_path_on_disk() {
    TempModel model;
    DiskFileProvider provider(model.root);
    expect(provider.reserve("elf_files/model.elf") ==
               provider.get_path("elf_files/model.elf"),
           "on the disk provider reserve and get_path must be identical");
}

void test_fetch_and_evict_are_no_ops_on_disk() {
    TempModel model;
    DiskFileProvider provider(model.root);
    const auto elf = provider.get_path("elf_files/model.elf");
    provider.fetch(elf);   // must not throw, must not change the file
    expect(std::filesystem::is_regular_file(elf),
           "fetch must be a no-op on disk (file already local)");
    provider.evict(elf);   // must NOT delete on the disk provider
    expect(std::filesystem::is_regular_file(elf),
           "evict must not delete files on the disk provider");
}

void test_open_stream_missing_file_throws() {
    TempModel model;
    DiskFileProvider provider(model.root);
    bool threw = false;
    try { provider.open_stream("devkit/tokenizer.json"); }   // not in TempModel
    catch (const std::exception&) { threw = true; }
    expect(threw, "open_stream of a missing file must throw, not return an empty stream");
}

void test_disk_does_not_pull_files() {
    TempModel model;
    DiskFileProvider provider(model.root);
    expect(!provider.pulls_files(),
           "disk provider must report pulls_files() == false (parallel load allowed)");
}

void test_whisper_preprocessor_directory_and_provider() {
    TempModel model;
    const auto custom = model.root / "custom preprocessing directory";
    const std::string config = R"({"mel_filters":[[0.0]]})";
    write_file(custom / "preprocessor_config.json", config);
    write_file(model.root / "devkit/preprocessor_config.json", config);
    // The legacy overload reads the supplied directory, not parent/devkit.
    simaai::llima::WhisperPreprocessor from_directory(custom);
    simaai::llima::WhisperPreprocessor trailing_separator(custom.string() + "/");
    simaai::llima::WhisperPreprocessor from_provider(
        std::make_shared<DiskFileProvider>(model.root)
    );
}

}  // namespace

int main() {
    try {
        test_get_path_joins_root_and_relative_name();
        test_open_stream_returns_file_contents();
        test_open_stream_preserves_binary_bytes();
        test_release_is_a_no_op_on_disk();
        test_usable_through_base_interface();
        test_reserve_returns_path_without_requiring_file();
        test_reserve_matches_get_path_on_disk();
        test_fetch_and_evict_are_no_ops_on_disk();
        test_open_stream_missing_file_throws();
        test_disk_does_not_pull_files();
        test_whisper_preprocessor_directory_and_provider();
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unexpected exception: " << e.what() << '\n';
        return 1;
    }

    if (failures == 0) {
        std::cout << "file_provider_test: all checks passed\n";
        return 0;
    }
    std::cerr << "file_provider_test: " << failures << " check(s) failed\n";
    return 1;
}
