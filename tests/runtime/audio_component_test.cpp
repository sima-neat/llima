#include "mla_buffer.hpp"
#include "mla_model.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>

using namespace simaai::llima;

// Generated compiler fixture: interfaces in ELF persistent-symbol order, raw
// FP32 inputs and output filenames. Symbol order is lexical: output_10 precedes
// output_2. Do not assume tuple order for a fixture with ten or more outputs.
// Keep this runner separate from model-free audio generation unit tests.
int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 3) throw std::invalid_argument("Usage: sima_lmm_audio_component_test FIXTURE_JSON [BENCHMARK_ITERATIONS]");
        size_t iterations = 0;
        if (argc == 3) {
            const std::string_view argument(argv[2]);
            const auto [end, error] = std::from_chars(argument.data(), argument.data() + argument.size(), iterations);
            if (error != std::errc{} || end != argument.data() + argument.size() || !iterations || iterations > 10000) {
                throw std::invalid_argument("Benchmark iterations must be an integer from 1 to 10000");
            }
        }
        const std::filesystem::path path(argv[1]), directory = path.parent_path();
        nlohmann::json fixture;
        std::ifstream(path) >> fixture;
        MLAModelWithBuffer::read_env_vars();
        connect_mla_rt({});
        std::vector<std::unique_ptr<MLABuffer>> inputs, outputs;
        std::vector<MLABufferSlice> ifms, ofms;
        auto allocate = [&](const auto& specs, auto& buffers, auto& slices, bool input) {
            for (const auto& spec : specs) {
                if (spec.at("dtype").template get<std::string>().find("bfloat16") == std::string::npos) {
                    throw std::invalid_argument("Audio fixture interfaces must be BF16");
                }
                const bool aligned = spec.at("dram_layout").template get<std::string>().find("HWC16") != std::string::npos;
                auto buffer = std::make_unique<MLABuffer>(spec.at("name"), spec.at("dram_shape").template get<std::vector<size_t>>(), "bfloat16", aligned);
                buffer->allocate(); buffer->clear();
                if (input) {
                    const auto filename = directory / spec.at("file").template get<std::string>();
                    if (std::filesystem::file_size(filename) != buffer->get_num_elems() * sizeof(float)) throw std::invalid_argument("Input fixture size mismatch");
                    std::vector<float> values(buffer->get_num_elems());
                    std::ifstream stream(filename,std::ios::binary);
                    stream.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(float));
                    if (!stream) throw std::runtime_error("Cannot read audio fixture");
                    std::vector<Eigen::bfloat16> packed(values.size());
                    std::transform(values.begin(),values.end(),packed.begin(),[](float value) {return Eigen::bfloat16(value);});
                    buffer->upload(packed.data());
                }
                slices.emplace_back(buffer.get(),std::vector<uint32_t>(4,0),spec.at("shape").template get<std::vector<uint32_t>>());
                buffers.push_back(std::move(buffer));
            }
        };
        allocate(fixture.at("inputs"), inputs, ifms, true);
        allocate(fixture.at("outputs"), outputs, ofms, false);
        MLAModelWithBuffer model(directory / fixture.at("elf").get<std::string>(),ifms,ofms);
        model.load(); model.run();
        std::vector<double> durations;
        durations.reserve(iterations);
        for (size_t iteration = 0; iteration < iterations; ++iteration) {
            const auto started = std::chrono::steady_clock::now();
            model.run();
            durations.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
        }
        if (!durations.empty()) {
            std::sort(durations.begin(), durations.end());
            const auto middle = durations.size() / 2;
            const double median = durations.size() % 2 ? durations[middle]
                : (durations[middle - 1] + durations[middle]) / 2.;
            std::cout << "benchmark_json=" << nlohmann::json({{"iterations", iterations},
                {"median_ms", median}, {"p95_ms", durations[(durations.size() * 95 - 1) / 100]},
                {"minimum_ms", durations.front()}}).dump() << '\n';
        }
        for (size_t index=0;index<outputs.size();++index) {
            std::vector<Eigen::bfloat16> packed(outputs[index]->get_num_elems());
            outputs[index]->download(packed.data());
            std::vector<float> values(packed.size());
            std::transform(packed.begin(),packed.end(),values.begin(),[](auto value) {return static_cast<float>(value);});
            if (std::any_of(values.begin(),values.end(),[](float value) {return !std::isfinite(value);})) throw std::runtime_error("Non-finite MLA output");
            std::ofstream stream(directory / fixture.at("outputs")[index].at("file").get<std::string>(),std::ios::binary);
            stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
            if (!stream) throw std::runtime_error("Cannot write MLA output");
        }
        model.free();
        std::cout << "Audio component MLA execution passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Audio component failed: " << error.what() << '\n';
        return 1;
    }
}
