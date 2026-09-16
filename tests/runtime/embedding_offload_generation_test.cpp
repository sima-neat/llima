#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <thread>
#include <nlohmann/json.hpp>

#include "embedding_offload.hpp"
#include "runtime_test_utils.hpp"
#include "setup.hpp"
#include "vision_language_model.hpp"

namespace {
uint64_t allocated_bytes() {
    std::ifstream memory("/dev/simaai-mem");
    std::string line;
    std::getline(memory, line);
    std::smatch match;
    if (!std::regex_search(line, match, std::regex("Total allocated size: (0x[0-9a-fA-F]+)")))
        throw std::runtime_error("Cannot inspect SiMa memory allocation summary");
    return std::stoull(match[1], nullptr, 16);
}
}

int main(int argc, char** argv) {
    using namespace simaai::llima;
    bool connected = false;
    try {
        const auto dir = argc > 1 ? std::filesystem::path(argv[1])
            : test::resolve_model_dir("SIMA_TEST_LLIMA_EMBEDDING_MODEL",
                "Gemma-4-E2B-it-TextOnly-GPTQ-a16w4", "embedding offload", "devkit/vlm_config.json");
        const auto cfg = nlohmann::json::parse(std::ifstream(dir / "devkit/vlm_config.json"));
        uint64_t table_bytes = 0;
        for (const auto& entry : std::filesystem::directory_iterator(dir / "devkit")) {
            if (!entry.path().filename().string().ends_with("_embeddings.bin")) continue;
            if (!embedding_file_on_nvme(entry.path()))
                throw std::runtime_error("Embedding generation test requires raw tables on local NVMe");
            table_bytes += entry.file_size();
        }
        if (!table_bytes) throw std::runtime_error("No raw embedding tables found");
        const auto tokenizer = Tokenizer::from_hf_json(dir / "devkit/tokenizer.json");
        const auto seed = tokenizer->encode("The capital of Germany is Berlin. Explain the answer in simple words. ", true);
        connect({}, "/tmp/llima-embedding-generation.log", spdlog::level::warn);
        connected = true;
        nlohmann::json baseline;
        const auto memory_baseline = allocated_bytes();
        uint64_t resident_bytes = 0;
        for (const char* mode : {"off", "auto"}) {
            setenv("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD", mode, 1);
            const bool resident = std::string_view(mode) == "off";
            auto compare = [&](const std::string& key, const nlohmann::json& value) {
                if (resident) baseline[key] = value;
                else if (baseline.at(key) != value)
                    throw std::runtime_error("Resident/offloaded mismatch: " + key);
            };
            {
                VisionLanguageModel model(dir);
                const auto loaded_bytes = allocated_bytes();
                if (resident) resident_bytes = loaded_bytes;
                else {
                    // Allow allocator accounting granularity, but fail if auto stayed resident.
                    if (loaded_bytes >= resident_bytes || resident_bytes - loaded_bytes < table_bytes / 2)
                        throw std::runtime_error("Automatic offload did not release embedding-table DRAM");
                    std::cout << "Embedding DRAM saved: " << resident_bytes - loaded_bytes << " bytes\n";
                }
                model.set_text_callback([](const std::string&, bool, bool) {});
                if (argc <= 2) {
                    for (const size_t count : {8, 127, 128, 129, 257}) {
                        std::vector<uint32_t> ids(count);
                        for (size_t i = 0; i < count; ++i) ids[i] = seed[i % seed.size()];
                        // Exercise scattered rows as well as repeated-token deduplication.
                        if (count == 257)
                            for (size_t i = 0; i < count; ++i) ids[i] = 1000 + 31 * i;
                        const auto output = model.run_model(ids, count + 8, std::set<uint32_t>{});
                        const auto repeated = model.run_model(ids, count + 8, std::set<uint32_t>{});
                        if (output.empty() || output != repeated)
                            throw std::runtime_error("Repeated request mismatch");
                        compare(std::to_string(count), output);
                    }
                    std::vector<uint32_t> long_prompt(1024, seed.front());
                    std::jthread cancel([&] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        model.stop_model();
                    });
                    const auto interrupted = model.run_model(long_prompt, 2048, std::set<uint32_t>{});
                    cancel.join();
                    if (!interrupted.empty()) throw std::runtime_error("Generation did not honor cancellation");
                    const auto recovered = model.run_model(seed, seed.size() + 8, std::set<uint32_t>{});
                    if (recovered.empty()) throw std::runtime_error("Generation failed after cancellation");
                    compare("after_cancel", recovered);
                    if (model.support_image()) {
                        auto chat = model.create_chat();
                        chat.add_image(test::resolve_asset("sjc.jpg"));
                        chat.add_query("Describe the image briefly.");
                        const auto result = model.run_model(chat, 12);
                        if (!result || result->empty()) throw std::runtime_error("Missing image response");
                        compare("image", *result);
                    }
                    if (cfg["pipeline_cfg"].value("return_logits", false)
                        || cfg["lm_cfg"].value("lm_head_num_splits", 1) > 1) {
                        auto values = model.run_model_for_logits(std::span<const uint32_t>(seed).first(3));
                        std::vector<uint16_t> bits(values.size());
                        std::memcpy(bits.data(), values.data(), bits.size() * sizeof(uint16_t));
                        compare("logits", bits);
                    }
                    if (cfg["pipeline_cfg"].value("return_logits", false)) {
                        for (bool grouped : {false, true}) {
                            const auto score = model.run_model_for_loglikelihood(
                                std::span<const uint32_t>(seed).first(3), 1,
                                std::span<const uint32_t>(seed).subspan(1, 2), grouped);
                            compare(grouped ? "grouped_logprob" : "serial_logprob", score.logprob);
                        }
                    }
                } else {
                    VisionLanguageModel draft(argv[2]);
                    model.set_draft_vlm(&draft);
                    auto chat = model.create_chat();
                    chat.add_query("Explain why Berlin is the capital of Germany.");
                    const auto response = model.run_model(chat, 32);
                    if (!response || response->empty()) throw std::runtime_error("Empty speculative response");
                    compare("speculative", *response);
                    model.set_draft_vlm(nullptr);
                }
            }
            if (allocated_bytes() != memory_baseline)
                throw std::runtime_error("SiMa allocation leak after model teardown");
        }
        disconnect();
        connected = false;
        std::cout << "Embedding offload generation comparison passed\n";
    } catch (const std::exception& error) {
        if (connected) { try { disconnect(); } catch (...) {} }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
