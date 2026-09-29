// Routing test for the LLiMa file seam (PCIe GenAI).
//
// The unit test file_provider_test.cpp proves DiskFileProvider behaves like a
// direct disk read. This test proves the other half: that LLiMa actually goes
// THROUGH a FileProvider for its model files instead of opening paths directly.
//
// It does this with a spy provider: a decorator that wraps a real
// DiskFileProvider, records every name LLiMa asks for, and then delegates the
// real read so the model still loads. Because BaseModel shares one provider
// instance down into LanguageModel, the single spy sees every load-time read.
//
// A missed routing site is invisible on disk today (it just reads the same
// file directly) and only breaks later under a PCIe provider. This test turns
// "we believe all sites route" into "the model could not load its config, ELF,
// tokenizer or embeddings without asking the provider first" -- so a reverted
// routing site fails here, on a normal board, before any PCIe run.
//
// Scope: it asserts the sites that ALWAYS fire for a plain text model
// (vlm_config.json, an ELF, a tokenizer source, an embeddings source) and
// prints the full manifest of requested names for inspection. It deliberately
// does not assert the conditional sites (chat_template.jinja vs .json, EAGLE3
// d2t.npy, per-layer embeddings, relocation .npy, the VLM preprocessor) --
// those do not all exist for one model, so asserting them would fail on
// correct code.

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <istream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <spdlog/common.h>

#include "file_provider.hpp"
#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

namespace {

constexpr const char* kModelEnv = "SIMA_TEST_LLIMA_TEXT_MODEL";

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// Wraps a real DiskFileProvider, records every requested name, then delegates.
// The model loads for real; we inspect the recording afterwards.
class SpyFileProvider : public simaai::llima::FileProvider {
    public:
        explicit SpyFileProvider(std::filesystem::path root)
            : _disk(std::move(root)) {}

        std::filesystem::path get_path(std::string_view name) override {
            record("get_path", name);
            return _disk.get_path(name);
        }

        std::unique_ptr<std::istream> open_stream(std::string_view name) override {
            record("open_stream", name);
            return _disk.open_stream(name);
        }

        void release(std::string_view name) override {
            record("release", name);
            _disk.release(name);
        }

        std::filesystem::path reserve(std::string_view name) override {
            _reserved.emplace_back(name);
            return _disk.reserve(name);
        }

        const std::vector<std::string>& names() const { return _names; }

        const std::vector<std::string>& reserved() const { return _reserved; }

        // True if any requested name contains the given substring.
        bool requested(std::string_view needle) const {
            return std::any_of(
                _names.begin(), _names.end(),
                [&](const std::string& n) {
                    return n.find(needle) != std::string::npos;
                }
            );
        }

        // True if any requested name lies in the given top-level folder.
        bool requested_in_folder(std::string_view folder) const {
            return std::any_of(
                _names.begin(), _names.end(),
                [&](const std::string& n) { return n.rfind(folder, 0) == 0; }
            );
        }

    private:
        simaai::llima::DiskFileProvider _disk;
        std::vector<std::string> _names;
        std::vector<std::string> _reserved;

        void record(const char* how, std::string_view name) {
            _names.emplace_back(name);
            std::cout << "SPY " << how << ' ' << name << '\n';
        }
};

}  // namespace

int main() {
    bool connected = false;
    try {
        const std::filesystem::path model_dir =
            simaai::llima::test::resolve_model_dir(
                kModelEnv,
                simaai::llima::test::kDefaultTextModelName,
                "LLiMa text",
                "devkit/vlm_config.json"
            );
        std::cout << "LLIMA_LLM model_dir=" << model_dir << '\n';

        simaai::llima::connect(
            {},
            "/tmp/sima_lmm_file_provider_routing_test.log",
            spdlog::level::info
        );
        connected = true;

        auto spy = std::make_shared<SpyFileProvider>(model_dir);
        {
            // Constructing the model runs every load-time read site (config,
            // tokenizer, ELF definitions, embeddings, scales) plus a warmup.
            simaai::llima::VisionLanguageModel model(
                model_dir, std::nullopt, std::nullopt, spy
            );
        }

        simaai::llima::disconnect();
        connected = false;

        std::cout << "SPY recorded " << spy->names().size()
                  << " file request(s)\n";

        // Always-present sites for a plain text model. If any of these were
        // reverted to a direct disk read, the spy would not have seen them.
        expect(spy->requested("devkit/vlm_config.json"),
               "vlm_config.json must be read through the provider (site 1)");
        expect(spy->requested_in_folder("elf_files/"),
               "at least one ELF must be resolved through the provider (site 9)");
        expect(spy->requested("tokenizer.json") || spy->requested(".gguf"),
               "the tokenizer source (tokenizer.json or GGUF) must go through "
               "the provider (sites 2/4)");
        expect(spy->requested("_embeddings.bin") || spy->requested("_embeddings.npy"),
               "the token embeddings must be read through the provider "
               "(sites 10/11)");

        const auto& reserved = spy->reserved();
        const bool any_elf_reserved = std::any_of(
            reserved.begin(), reserved.end(),
            [](const std::string& n) { return n.find("elf_files/") != std::string::npos; });
        expect(any_elf_reserved,
               "ELF paths must be RESERVED at define time (deferred pull), not get_path'd");

        // Guard against a silent no-op provider: a real load asks for many
        // files, so an almost-empty manifest means routing collapsed.
        expect(spy->names().size() >= 4,
               "expected several file requests through the provider");

    } catch (const std::exception& error) {
        if (connected) {
            try {
                simaai::llima::disconnect();
            } catch (...) {
            }
        }
        std::cerr << "File provider routing test failed: " << error.what() << '\n';
        return 1;
    }

    if (failures == 0) {
        std::cout << "File provider routing test passed\n";
        return 0;
    }
    std::cerr << "File provider routing test: " << failures
              << " check(s) failed\n";
    return 1;
}
