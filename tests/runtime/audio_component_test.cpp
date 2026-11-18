#include "mla_buffer.hpp"
#include "mla_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>

using namespace simaai::llima;

// Generated compiler fixture: interfaces in ELF persistent-symbol order, raw
// FP32 inputs and output filenames. Symbol order is lexical: output_10 precedes
// output_2. Do not assume tuple order for a fixture with ten or more outputs.
// Keep this runner separate from model-free audio generation unit tests.
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: sima_lmm_audio_component_test FIXTURE_JSON");
        const std::filesystem::path path(argv[1]), directory = path.parent_path();
        nlohmann::json fixture;
        std::ifstream(path) >> fixture;
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
