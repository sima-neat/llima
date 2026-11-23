#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <system_error>
#include <thread>

#include <cnpy.h>

#include "mla_model.hpp"
#include "runtime_context.hpp"
#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"
#include "vlm_config.hpp"

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
    VlmConfig disable_group_prefill() {
        const auto devkit = path / "devkit";
        const auto source = fs::read_symlink(devkit);
        fs::remove(devkit);
        fs::create_directory(devkit);
        for (const auto& entry : fs::directory_iterator(source)) {
            if (entry.path().filename() != "vlm_config.json")
                fs::create_symlink(entry.path(), devkit / entry.path().filename());
        }
        auto config = nlohmann::json::parse(std::ifstream(source / "vlm_config.json"));
        config.at("pipeline_cfg")["input_token_group_offsets"] = nullptr;
        std::ofstream(devkit / "vlm_config.json") << config;
        return config.get<VlmConfig>();
    }
    ~TestPackage() { std::error_code ec; fs::remove_all(path, ec); }
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

fs::path pre_elf(const fs::path& directory, unsigned width = 1) {
    const std::regex pre("_n" + std::to_string(width) + R"(_pre_layer\d+_stage1_mla\.elf$)");
    fs::path first;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (std::regex_search(entry.path().filename().string(), pre) &&
            (first.empty() || entry.path() < first))
            first = entry.path();
    }
    if (first.empty()) throw std::runtime_error("No pre ELF for N" + std::to_string(width) + " in " + directory.string());
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

void verify_graph_widths(const fs::path& model) {
    TestPackage package(fs::absolute(model));
    const auto cfg = package.disable_group_prefill();
    const auto width = cfg.lm_cfg.get_single_num_tokens();
    const auto elf = pre_elf(package.path / "elf_files", width);
    const auto root = package.path / "npy_files";
    malformed(root / "unknown_width", "model_n65535_pre_layer0_stage1_mla", 1);
    malformed(root / "wrong_count", elf.stem().string(), port_lengths(elf).reloc_len_array.size() + 1);
    if (cfg.pipeline_cfg.input_token_group_size != width && cfg.pipeline_cfg.input_token_group_size > 1) {
        const auto group = std::regex_replace(elf.filename().string(), std::regex("_n[0-9]+_pre"),
            "_n" + std::to_string(cfg.pipeline_cfg.input_token_group_size) + "_pre");
        if (fs::is_regular_file(elf.parent_path() / group)) {
            const std::array<uint8_t, 1> bytes{};
            const auto section = root / "wrong_count" / (fs::path(group).stem().string() + "_0.npy");
            cnpy::npy_save(section.string(), bytes.data(), {bytes.size()});
        }
    }
    VisionLanguageModel target(package.path);
    reject(target, "unknown_width", "graph not present");
    reject(target, "wrong_count", "Relocation count");
    std::cout << "Validated N" << width << " as the active width with grouped prefill disabled\n";
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

void verify_mixed_queue(const fs::path& elf, const MlaModelIFMOFMLengths& sizes,
                        const fs::path& idle_model = {}) {
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
    auto& context = RuntimeContext::instance();
    const auto program = context.reserve(elf);
    MLAModelWithBuffer::free_all_models(elf);
    require(!context.models.at(program->index), "Explicit unload kept an idle shared program resident");
    for (auto& buffer : out_a) buffer->clear();
    a.run();
    require(snapshot(out_a) == direct_a, "Reloading a program lost the wrapper's adapter");
    b.update_reloc(reloc_map(rel_b));
    b.run();
    const auto direct_b = snapshot(out_b);
    require(direct_a != direct_b, "Chosen ELF and input cannot distinguish the two relocations");
    for (auto& buffer : out_a) buffer->clear();
    for (auto& buffer : out_b) buffer->clear();
    std::unique_ptr<VisionLanguageModel> idle;
    if (!idle_model.empty()) idle = std::make_unique<VisionLanguageModel>(idle_model);
    a.add_to_queue();
    b.add_to_queue();
    idle.reset();
    require(program->queued_runs == 2, "Destroying an idle model consumed another model's queued work");
    std::jthread([&] { MLAModelWithBuffer::free_all_models(elf); }).join();
    require(context.models.at(program->index), "Bulk unload freed another thread's queued program");
    a.reset_reloc();
    b.reset_reloc();
    rel_a.clear();
    rel_b.clear();
    MLAModelWithBuffer::run_queue();
    require(snapshot(out_a) == direct_a && snapshot(out_b) == direct_b,
            "Sharing an ELF changed the queued runs' adapter selection");
    require(!context.models.at(program->index), "Deferred unload kept the program after its final wait");
    a.add_to_queue();
    MLAModelWithBuffer::free_all_models(elf);
    MLAModelWithBuffer::discard_queue();
    require(!context.models.at(program->index), "Discarding the pending run did not finish unloading");
}
} // namespace

int main(int argc, char** argv) {
    bool connected = false;
    try {
        if (argc == 3 && std::string_view(argv[1]) == "--graph-widths") {
            connect({}, "/tmp/llima-lora-width-test.log", spdlog::level::warn);
            connected = true;
            verify_graph_widths(argv[2]);
            disconnect();
            return 0;
        }
        if ((argc == 3 || argc == 4) && std::string_view(argv[1]) == "--queue-elf") {
            connect({}, "/tmp/llima-lora-queue-test.log", spdlog::level::warn);
            connected = true;
            verify_mixed_queue(argv[2], port_lengths(argv[2]), argc == 4 ? argv[3] : "");
            disconnect();
            std::cout << "Mixed-adapter queue matches both direct submissions\n";
            return 0;
        }
        if (argc != 1 && argc != 2 && argc != 4)
            throw std::invalid_argument("Usage: lora_session_test [MODEL_DIRECTORY [ADAPTER_A ADAPTER_B]] | --queue-elf ELF [IDLE_MODEL] | --graph-widths MODEL");
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
        const auto ports = sizes.reloc_len_array.size();
        malformed(root / "wrong_count", elf.stem().string(), ports > 1 ? ports - 1 : ports + 1);
        malformed(root / "unknown_width", "model_n65535_pre_layer0_stage1_mla", 1);
        malformed(root / "unknown_graph", "model_n1_pre_layer255_stage1_mla", 1);
        {
            VisionLanguageModel target(package.path);
            target.set_text_callback([](const std::string&, bool, bool) {});
            auto chat = target.create_chat();
            chat.clear_system_prompt();
            chat.add_query("Translate: the sky is blue.");
            auto generate = [&] {
                target.reset_session(kDefaultSession);
                auto result = target.generate(chat, {.max_new_tokens = 32}).text;
                require(!result.empty(), "Expected generated text");
                return result;
            };
            const auto base = generate();
            reject(target, "unknown_width", "graph not present");
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
