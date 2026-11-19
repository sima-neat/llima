#include <fstream>
#include <iostream>
#include <thread>
#include <nlohmann/json.hpp>
#include "mla_model.hpp"
#include "runtime_context.hpp"
#include "runtime_test_utils.hpp"
#include "setup.hpp"

using namespace simaai::llima;

int main() {
    bool connected = false;
    try {
        const auto root = test::resolve_model_dir("SIMA_TEST_LLIMA_TEXT_MODEL",
            test::kDefaultTextModelName, "program lifetime", "devkit/vlm_config.json");
        const auto cfg = nlohmann::json::parse(std::ifstream(root / "devkit/vlm_config.json"));
        const auto directory = root / "elf_files";
        const auto prefix = cfg.at("language_model_name").get<std::string>() + "_";
        std::filesystem::path path;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (!entry.is_regular_file() || entry.path().extension() != ".elf"
                || !entry.path().filename().string().starts_with(prefix)) continue;
            if (path.empty() || entry.path() < path) path = entry.path();
        }
        if (path.empty()) throw std::runtime_error("No compiled language program found in " + directory.string());
        connect({}, "/tmp/llima-program-lifetime.log", spdlog::level::warn);
        connected = true;
        auto& context = RuntimeContext::instance();
        auto owner = std::make_unique<MLAModelWithBuffer>(path, std::vector<MLABufferSlice>{},
                                                         std::vector<MLABufferSlice>{});
        owner->load();
        auto old_lease = context.reserve(path);
        const auto index = old_lease->index;
        std::weak_ptr<RuntimeProgram> expired = old_lease;
        old_lease.reset();

        // Force the last destructor to wait while a replacement reserves/loads the same path.
        std::unique_lock lock(context.execution_mutex);
        std::jthread retiring([instance = std::move(owner)]() mutable { instance.reset(); });
        while (!expired.expired()) std::this_thread::yield();
        std::unique_ptr<MLAModelWithBuffer> survivor;
        try {
            survivor = std::make_unique<MLAModelWithBuffer>(path, std::vector<MLABufferSlice>{},
                                                           std::vector<MLABufferSlice>{});
            survivor->load();
        } catch (...) {
            lock.unlock();
            retiring.join();
            throw;
        }
        const auto replacement = context.models.at(index);
        lock.unlock();
        retiring.join();
        if (!replacement || context.models.at(index) != replacement)
            throw std::runtime_error("Retiring lease unloaded the replacement program");
        survivor.reset();
        if (context.models.at(index)) throw std::runtime_error("Last lease did not unload its program");
        disconnect();
        connected = false;
    } catch (const std::exception& error) {
        if (connected) { try { disconnect(); } catch (...) {} }
        std::cerr << "Program lifetime test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Program lifetime test passed\n";
}
