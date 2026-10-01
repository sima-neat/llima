#include <exception>
#include <iostream>
#include <vector>

#include "mla_model.hpp"

template<typename T>
void expect(T condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

// A path that does not exist must be registrable (deferred pull): the ctor
// reserves it; presence is enforced later at load time, not at construction.
void test_ctor_tolerates_absent_path() {
    const std::filesystem::path absent = "/tmp/does_not_exist_llima/elf_files/x.elf";
    bool threw = false;
    try {
        simaai::llima::MLAModelWithBuffer m(absent, {}, {});
    } catch (const std::exception&) {
        threw = true;
    }
    expect(!threw, "MLAModelWithBuffer must register an absent path without throwing");
}

int main() {
    using simaai::llima::connect_mla_rt;
    using simaai::llima::disconnect_mla_rt;

    try {
        connect_mla_rt({});
        connect_mla_rt({});
        disconnect_mla_rt();

        connect_mla_rt({});
        disconnect_mla_rt();

        // Test that the constructor tolerates absent paths (deferred pull).
        test_ctor_tolerates_absent_path();
    } catch (const std::exception& error) {
        std::cerr << "MLA-RT lifecycle test failed: " << error.what() << '\n';
        try {
            disconnect_mla_rt();
        } catch (...) {
        }
        return 1;
    }

    std::cout << "MLA-RT lifecycle test passed\n";
    return 0;
}
