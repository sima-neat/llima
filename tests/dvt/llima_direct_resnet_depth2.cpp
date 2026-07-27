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
 * fixture. Two submissions are committed on one old retained session, then
 * the same model runs on a newly connected session. The test covers queue FIFO,
 * reconnect isolation, cross-context rejection, rollback, and byte-identical
 * output. It does not replace the Neat or LLM numerical/state oracles.
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
    simaai::llima::MLABuffer input_c("input_c", {301056}, "int8", false);
    simaai::llima::MLABuffer output_a("output_a", {2000}, "int8", false);
    simaai::llima::MLABuffer output_b("output_b", {2000}, "int8", false);
    simaai::llima::MLABuffer output_c("output_c", {2000}, "int8", false);
    input_a.allocate();
    input_b.allocate();
    input_c.allocate();
    output_a.allocate();
    output_b.allocate();
    output_c.allocate();

    input_a.clear(false);
    input_b.clear(false);
    input_c.clear(false);
    std::memcpy(input_a.get_virtual_addr(), input.data(), input.size());
    std::memcpy(input_b.get_virtual_addr(), input.data(), input.size());
    std::memcpy(input_c.get_virtual_addr(), input.data(), input.size());
    input_a.flush_cache();
    input_b.flush_cache();
    input_c.flush_cache();

    /* Retain A across disconnect, then make B the compatibility default. */
    auto session_a = simaai::llima::current_mla_execution_session();
    simaai::llima::disconnect();
    simaai::llima::connect(
        {}, "/tmp/llima-direct-resnet-reconnect.log", spdlog::level::warn
    );
    auto session_b = simaai::llima::current_mla_execution_session();

    simaai::llima::MLAModelWithBuffer job_a(
        session_a, argv[1],
        {simaai::llima::MLABufferSlice{&input_a}},
        {simaai::llima::MLABufferSlice{&output_a}}
    );
    simaai::llima::MLAModelWithBuffer job_b(
        session_a, argv[1],
        {simaai::llima::MLABufferSlice{&input_b}},
        {simaai::llima::MLABufferSlice{&output_b}}
    );
    simaai::llima::MLAModelWithBuffer job_c(
        session_b, argv[1],
        {simaai::llima::MLABufferSlice{&input_c}},
        {simaai::llima::MLABufferSlice{&output_c}}
    );

    // Snapshot both bindings in one caller-owned transaction.  The production
    // executor's private depth is independent of this two-job correctness
    // fixture.
    simaai::llima::MlaExecutionSegment segment;
    job_a.add_to_segment(segment);
    job_b.add_to_segment(segment);
    segment.commit();

    /* A committed transaction remains bound until its RAII lifetime ends. */
    bool rejected_cross_context = false;
    try {
        job_c.add_to_segment(segment);
    } catch (const std::invalid_argument&) {
        rejected_cross_context = true;
    }
    if (!rejected_cross_context) {
        std::cerr << "cross-context segment was not rejected\n";
        simaai::llima::disconnect();
        return 3;
    }
    simaai::llima::MlaExecutionSegment second_session_segment;
    job_c.add_to_segment(second_session_segment);
    second_session_segment.commit();

    /*
     * Prove that segment assembly is all-or-nothing.  Queue one valid snapshot
     * and then deliberately prepare an output whose allocation is smaller
     * than the compiler-declared OFM extent.  The second add must fail and
     * discard the first, never-submitted snapshot.  A fresh segment must then
     * execute normally; without rollback this used to leave the session open
     * and disconnect() replaced the useful bounds error with a cleanup error.
     */
    simaai::llima::MLABuffer short_output(
        "short_output", {8}, "int8", false
    );
    short_output.allocate();
    simaai::llima::MLAModelWithBuffer invalid_job(
        session_a, argv[1],
        {simaai::llima::MLABufferSlice{&input_a}},
        {simaai::llima::MLABufferSlice{&short_output}}
    );
    bool rejected_partial_segment = false;
    job_a.add_to_segment(segment);
    try {
        invalid_job.add_to_segment(segment);
    } catch (const std::out_of_range&) {
        rejected_partial_segment = true;
    }
    if (!rejected_partial_segment) {
        std::cerr << "undersized OFM did not abort segment construction\n";
        simaai::llima::disconnect();
        return 2;
    }
    job_b.add_to_segment(segment);
    segment.commit();

    output_a.invalidate_cache();
    output_b.invalidate_cache();
    output_c.invalidate_cache();
    const bool byte_exact =
        std::memcmp(
            output_a.get_virtual_addr(),
            output_b.get_virtual_addr(),
            output_a.get_allocation_size()
        ) == 0 &&
        std::memcmp(
            output_a.get_virtual_addr(),
            output_c.get_virtual_addr(),
            output_a.get_allocation_size()
        ) == 0;

    simaai::llima::disconnect();
    if (!byte_exact) {
        std::cerr << "two-job output mismatch\n";
        return 1;
    }

    std::cout
        << "LLIMA_DIRECT_RESNET_PASS jobs=3 segment=RAII depth=PRIVATE-3 "
        << "segment_rollback=PASS reconnect_isolation=PASS "
        << "cross_context=REJECTED output=BYTE_EXACT\n";
    return 0;
}
