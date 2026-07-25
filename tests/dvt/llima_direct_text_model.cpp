#include "setup.hpp"
#include "vision_language_model.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <set>
#include <span>
#include <vector>

/*
 * DVT-only, end-to-end text-model qualification for the direct kernel path.
 *
 * Usage:
 *   llima_direct_text_model <compiled-model-directory>
 *
 * Unlike the small ResNet executor smoke, this constructs the production
 * VisionLanguageModel and therefore exercises the complete LLiMa model-family
 * definition, fully-resident package transaction, n128/n1 selection, KV
 * buffers, layer-by-layer two-deep execution segments and greedy token loop.
 * The same prompt is executed twice in one session and the generated token IDs
 * must match exactly.  That is an architecture/determinism gate, not a new
 * accuracy oracle; model accuracy remains owned by the existing product tests.
 *
 * Inspect this executable with `file` and run the AArch64 artifact only on an
 * authorized Modalix DVT.  It must never be executed in the x86 build
 * container.
 */
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0]
                  << " <compiled-model-directory>\n";
        return 64;
    }
    const std::filesystem::path model_path(argv[1]);
    if (!std::filesystem::is_directory(model_path / "devkit") ||
        !std::filesystem::is_directory(model_path / "elf_files")) {
        std::cerr << "model package lacks devkit/ or elf_files/: "
                  << model_path << "\n";
        return 65;
    }

    /*
     * Raw, in-vocabulary token IDs avoid making this kernel-architecture test
     * dependent on a particular chat template.  A 1,025-token prompt crosses
     * the old 256-token component-fixture ceiling and exercises the compiled
     * 0/320/640/960 grouped-prefill offsets before n1 decode.  The requested
     * 1,027-token ceiling remains comfortably inside this package's 2,048
     * token contract while keeping qualification time bounded.
     */
    std::vector<std::uint32_t> prompt(1025, 11U);
    prompt.front() = 9707U;
    constexpr std::uint16_t kTotalTokens = 1027;
    const std::set<std::uint32_t> no_early_stop;

    auto run_independent_session = [&]() {
        simaai::llima::connect(
            {}, "/tmp/llima-direct-text-model.log", spdlog::level::warn
        );
        std::vector<std::uint32_t> result;
        try {
            /*
             * Destroy every model/buffer owner before disconnecting.  Running
             * the two trials in separate kernel contexts also prevents the
             * second result from being satisfied by LLiMa's token cache.
             */
            {
                simaai::llima::VisionLanguageModel model(model_path);
                result = model.run_model(
                    std::span<const std::uint32_t>(prompt), kTotalTokens,
                    no_early_stop
                );
            }
            simaai::llima::disconnect();
        } catch (...) {
            simaai::llima::disconnect();
            throw;
        }
        return result;
    };

    std::vector<std::uint32_t> first;
    std::vector<std::uint32_t> second;
    try {
        first = run_independent_session();
        second = run_independent_session();
    } catch (const std::exception& error) {
        std::cerr << "text model failed: " << error.what() << "\n";
        return 1;
    }

    /*
     * The legacy raw-token API includes its prefill boundary token in the
     * returned tail for some model families, so do not invent a new count
     * contract here.  Require a bounded, nonempty and exactly deterministic
     * result from two genuinely independent hardware sessions.
     */
    if (first.empty() || first.size() > kTotalTokens || second != first) {
        std::cerr << "token determinism mismatch: first=" << first.size()
                  << " second=" << second.size()
                  << " maximum=" << kTotalTokens << "\n";
        return 2;
    }

    std::cout << "LLIMA_DIRECT_TEXT_PASS model=" << model_path.filename()
              << " prompt_tokens=" << prompt.size()
              << " max_total_tokens=" << kTotalTokens
              << " returned_ids=" << first.size()
              << " independent_sessions=2"
              << " greedy_repeat=EXACT token_ids=";
    for (std::size_t index = 0; index < first.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << first[index];
    }
    std::cout << "\n";
    return 0;
}
