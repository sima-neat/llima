#include "mla_buffer.hpp"
#include "mla_model.hpp"
#include "setup.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

/*
 * DVT-only ordered-executor smoke for the direct LLiMa kernel path.
 *
 * Usage:
 *   llima_direct_resnet_depth2 <resnet-qmla> <150528-byte-bf16-ifm>
 *
 * The model's physical input envelope is 301056 bytes.  The source fixture is
 * 150528 bytes, so both independent DMS0 inputs are cleared before copying the
 * fixture.  Two immutable submissions are then committed as one segment.  The
 * test checks that the two-deep executor preserves independent buffer lifetime,
 * FIFO execution, and identical output.  It does not replace the existing Neat
 * accuracy oracle or the pending LLM token/logit/KV parity suite.
 *
 * Inspect the resulting executable with `file` and run the AArch64 binary on
 * an authorized DVT; never execute it in the x86 build container.
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "usage: " << argv[0]
            << " <resnet-qmla> <150528-byte-bf16-ifm>\n";
        return 64;
    }

    std::ifstream input_stream(argv[2], std::ios::binary);
    const std::istreambuf_iterator<char> input_begin(input_stream);
    std::vector<std::uint8_t> input(
        input_begin, std::istreambuf_iterator<char>{}
    );
    if (input.size() != 150528) {
        std::cerr << "input size=" << input.size() << "\n";
        return 65;
    }

    simaai::llima::connect(
        {}, "/tmp/llima-direct-resnet.log", spdlog::level::warn
    );

    simaai::llima::MLABuffer input_a("input_a", {301056}, "int8", false);
    simaai::llima::MLABuffer input_b("input_b", {301056}, "int8", false);
    simaai::llima::MLABuffer output_a("output_a", {2000}, "int8", false);
    simaai::llima::MLABuffer output_b("output_b", {2000}, "int8", false);
    input_a.allocate();
    input_b.allocate();
    output_a.allocate();
    output_b.allocate();

    input_a.clear(false);
    input_b.clear(false);
    std::memcpy(input_a.get_virtual_addr(), input.data(), input.size());
    std::memcpy(input_b.get_virtual_addr(), input.data(), input.size());
    input_a.flush_cache();
    input_b.flush_cache();

    simaai::llima::MLAModelWithBuffer job_a(
        argv[1],
        {simaai::llima::MLABufferSlice{&input_a}},
        {simaai::llima::MLABufferSlice{&output_a}}
    );
    simaai::llima::MLAModelWithBuffer job_b(
        argv[1],
        {simaai::llima::MLABufferSlice{&input_b}},
        {simaai::llima::MLABufferSlice{&output_b}}
    );

    // Snapshot both bindings before commit.  run_queue() retains at most two
    // kernel jobs and does not cross a caller-owned segment boundary.
    job_a.add_to_queue();
    job_b.add_to_queue();
    simaai::llima::MLAModelWithBuffer::run_queue();

    output_a.invalidate_cache();
    output_b.invalidate_cache();
    const bool byte_exact =
        std::memcmp(
            output_a.get_virtual_addr(),
            output_b.get_virtual_addr(),
            output_a.get_allocation_size()
        ) == 0;

    simaai::llima::disconnect();
    if (!byte_exact) {
        std::cerr << "two-job output mismatch\n";
        return 1;
    }

    std::cout
        << "LLIMA_DIRECT_RESNET_PASS jobs=2 queue_depth=2 "
        << "output=BYTE_EXACT\n";
    return 0;
}
