#include <exception>
#include <iostream>
#include <vector>

#include "mla_model.hpp"

int main() {
    using simaai::llima::connect_mla_rt;
    using simaai::llima::disconnect_mla_rt;

    try {
        connect_mla_rt({});
        connect_mla_rt({});
        disconnect_mla_rt();

        connect_mla_rt({});
        disconnect_mla_rt();
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
