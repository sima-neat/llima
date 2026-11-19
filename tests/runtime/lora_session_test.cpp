#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <system_error>

#include <cnpy.h>

#include "mla_model.hpp"
#include "runtime_context.hpp"
#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

using namespace simaai::llima;
namespace fs = std::filesystem;

namespace {
struct TestPackage {
    fs::path path;
    explicit TestPackage(const fs::path& model) {
        std::string directory = (fs::temp_directory_path() / "llima-lora-XXXXXX").string();
        if (!::mkdtemp(directory.data())) throw std::runtime_error("Could not create test directory");
        path = directory;
        try {
            fs::create_directory(path / "npy_files");
            for (const auto& entry : fs::directory_iterator(fs::absolute(model))) {
                if (entry.path().filename() != "npy_files")
                    fs::create_symlink(entry.path(), path / entry.path().filename());
            }
        } catch (...) {
            fs::remove_all(path);
            throw;
        }
    }
    ~TestPackage() { std::error_code ec; fs::remove_all(path, ec); }
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

fs::path pre_elf(const fs::path& directory) {
    const std::regex pre(R"(_n1_pre_layer\d+_stage1_mla\.elf$)");
    fs::path first;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (std::regex_search(entry.path().filename().string(), pre) &&
            (first.empty() || entry.path() < first))
            first = entry.path();
    }
    if (first.empty()) throw std::runtime_error("No N1 pre ELF in " + directory.string());
    return first;
}

MlaModelIFMOFMLengths port_lengths(const fs::path& elf) {
    MLAModelWithBuffer probe(elf, {}, {});
    probe.load();
    auto& context = RuntimeContext::instance();
    return mla_get_ifm_ofm_len_vector(context.models.at(context.reserve(elf)->index));
}

void malformed(const fs::path& path, const std::string& stem, size_t count) {
    fs::create_directory(path);
    const std::array<uint8_t, 16> bytes{};
    for (size_t i = 0; i < count; ++i)
        cnpy::npy_save((path / (stem + "_" + std::to_string(i) + ".npy")).string(),
                       bytes.data(), {bytes.size()});
}

void reject(VisionLanguageModel& model, const char* adapter, std::string_view diagnostic) {
    try {
        model.set_reloc(adapter);
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()).find(diagnostic) != std::string_view::npos) return;
        throw;
    }
    throw std::runtime_error("Accepted an invalid adapter: " + std::string(adapter));
}

using Buffers = std::vector<std::unique_ptr<MLABuffer>>;

Buffers allocate(const std::vector<int>& lengths, const std::string& name, int pattern) {
    Buffers buffers;
    for (const auto n : lengths) {
        require(n > 0, "Expected nonempty ELF ports");
        const size_t size = static_cast<size_t>(n);
        auto buffer = std::make_unique<MLABuffer>(name, std::vector<size_t>{size}, "int8", false);
        buffer->allocate();
        std::vector<uint8_t> data(size);
        for (size_t i = 0; i < size; ++i)
            data[i] = pattern < 0 ? static_cast<uint8_t>(i * 31u + 7u) : static_cast<uint8_t>(pattern);
        buffer->upload(data.data());
        buffers.push_back(std::move(buffer));
    }
    return buffers;
}

std::vector<MLABufferSlice> views(const Buffers& buffers) {
    std::vector<MLABufferSlice> slices;
    for (auto& buffer : buffers) slices.emplace_back(buffer.get());
    return slices;
}

std::map<std::string, MLABuffer*> reloc_map(const Buffers& buffers) {
    std::map<std::string, MLABuffer*> result;
    for (size_t i = 0; i < buffers.size(); ++i)
        result.emplace(fmt::format("port_{:03d}", i), buffers[i].get());
    return result;
}

std::vector<uint8_t> snapshot(const Buffers& buffers) {
    std::vector<uint8_t> bytes;
    for (auto& buffer : buffers) {
        const auto begin = bytes.size();
        bytes.resize(begin + buffer->get_num_elems());
        buffer->invalidate_cache();
        buffer->download(bytes.data() + begin);
    }
    return bytes;
}

void verify_mixed_queue(const fs::path& elf, const MlaModelIFMOFMLengths& sizes) {
    require(!sizes.reloc_len_array.empty(), "Expected relocatable N1 pre ELF");
    auto inputs = allocate(sizes.ifm_len_array, "input", -1);
    auto out_a = allocate(sizes.ofm_len_array, "out_a", 0);
    auto out_b = allocate(sizes.ofm_len_array, "out_b", 0);
    auto rel_a = allocate(sizes.reloc_len_array, "rel_a", 0x5a);
    auto rel_b = allocate(sizes.reloc_len_array, "rel_b", 0xa5);
    MLAModelWithBuffer a(elf, views(inputs), views(out_a));
    MLAModelWithBuffer b(elf, views(inputs), views(out_b));
    a.update_reloc(reloc_map(rel_a));
    a.run();
    const auto direct_a = snapshot(out_a);
    b.update_reloc(reloc_map(rel_b));
    b.run();
    const auto direct_b = snapshot(out_b);
    require(direct_a != direct_b, "Chosen ELF and input cannot distinguish the two relocations");
    for (auto& buffer : out_a) buffer->clear();
    for (auto& buffer : out_b) buffer->clear();
    a.add_to_queue();
    b.add_to_queue();
    a.reset_reloc();
    b.reset_reloc();
    rel_a.clear();
    rel_b.clear();
    MLAModelWithBuffer::run_queue();
    require(snapshot(out_a) == direct_a && snapshot(out_b) == direct_b,
            "Sharing an ELF changed the queued runs' adapter selection");
}
} // namespace

int main(int argc, char** argv) {
    bool connected = false;
    try {
        if (argc != 1 && argc != 2 && argc != 4)
            throw std::invalid_argument("Usage: lora_session_test [MODEL_DIRECTORY [ADAPTER_A ADAPTER_B]]");
        const fs::path model = argc > 1 ? fs::absolute(argv[1]) :
            test::resolve_model_dir("SIMA_TEST_LLIMA_TEXT_MODEL", test::kDefaultTextModelName,
                                    "LoRA recovery", "devkit/vlm_config.json");
        TestPackage package(model);
        const auto root = package.path / "npy_files";
        if (argc == 4) {
            fs::create_directory_symlink(model / "npy_files" / argv[2], root / "adapter_a");
            fs::create_directory_symlink(model / "npy_files" / argv[3], root / "adapter_b");
        }
        connect({}, "/tmp/llima-lora-session-test.log", spdlog::level::warn);
        connected = true;
        const fs::path elf = pre_elf(package.path / "elf_files");
        const auto sizes = port_lengths(elf);
        malformed(root / "wrong_count", elf.stem().string(), sizes.reloc_len_array.size() + 1);
        malformed(root / "unknown_graph", "model_n1_pre_layer255_stage1_mla", 1);
        {
            VisionLanguageModel target(package.path);
            target.set_text_callback([](const std::string&, bool, bool) {});
            auto chat = target.create_chat();
            chat.clear_system_prompt();
            chat.add_query("Translate: the sky is blue.");
            auto generate = [&] {
                target.reset_session(0);
                auto result = target.run_model(chat, 32);
                require(result && !result->empty(), "Expected generated text");
                return *result;
            };
            const auto base = generate();
            reject(target, "unknown_graph", "graph not present");
            reject(target, "wrong_count", "Relocation count");
            target.unset_reloc();
            require(generate() == base, "Rejected adapter changed base generation");
            if (argc == 4) {
                target.set_reloc("adapter_a");
                const auto a = generate();
                reject(target, "wrong_count", "Relocation count");
                require(generate() == a, "Rejected adapter displaced the selected adapter");
                target.set_reloc("adapter_b");
                require(generate() != a, "Adapters A and B generated identical text");
                target.set_reloc("adapter_a");
                require(generate() == a, "Adapter A was not restored");
                target.unset_reloc();
            }
        }
        if (argc == 4) verify_mixed_queue(elf, sizes);
        disconnect();
    } catch (const std::exception& error) {
        if (connected) { try { disconnect(); } catch (...) {} }
        std::cerr << "LoRA session test failed: " << error.what() << '\n';
        return 1;
    }
    std::cout << "LoRA session test passed\n";
    return 0;
}
