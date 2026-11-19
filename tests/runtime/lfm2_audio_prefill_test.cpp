#include "lfm2_audio_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace simaai::llima;

// Save prefix/continuation hidden states for comparison across runtime versions.
// The continuation steps detect incorrect convolution or KV state after padding.
int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: sima_lmm_lfm2_audio_prefill_test BUNDLE OUTPUT_F32");
        Lfm2AudioModel model(argv[1]);
        std::ofstream output(argv[2], std::ios::binary);
        auto save = [&](const std::vector<float>& hidden) {
            if (hidden.size() != model.hidden_size() || std::any_of(hidden.begin(), hidden.end(), [](float x) { return !std::isfinite(x); })) {
                throw std::runtime_error("Invalid prefill hidden state");
            }
            output.write(reinterpret_cast<const char*>(hidden.data()), hidden.size()*sizeof(float));
        };
        for (size_t length : {2, 7, 15, 16, 17, 31, 43}) {
            model.reset();
            std::vector<float> input;
            for (size_t token = 0; token < length; ++token) {
                auto row = model.text_embedding(100 + 37*token);
                input.insert(input.end(), row.begin(), row.end());
            }
            save(model.forward(input));
            for (size_t next = 0; next < 3; ++next) {
                save(model.forward(model.text_embedding(100 + 37*(length + next))));
            }
        }
        output.close();
        if (!output) throw std::runtime_error("Cannot write prefill comparison output");
        std::cout << "Saved 28 prefix/continuation states, width=" << model.hidden_size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
