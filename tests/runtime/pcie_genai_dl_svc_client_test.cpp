// A board without the PCIe endpoint stack must get a clear error from the
// backend's svc client, not a loader crash (same rule as cpp_ext).
#include <iostream>
#include <stdexcept>
#include <string>

#include "dl_svc_client.hpp"

int main() {
    try {
        simaai::llima::pcie_backend::DlSvcClient client(nullptr, "libsimaaipep-does-not-exist.so");
        std::cerr << "FAIL: a missing svc library must throw\n";
        return 1;
    } catch (const std::runtime_error& e) {
        if (std::string(e.what()).find("PCIe support unavailable") == std::string::npos) {
            std::cerr << "FAIL: unexpected message: " << e.what() << '\n';
            return 1;
        }
    }
    std::cout << "pcie_genai_dl_svc_client_test passed\n";
    return 0;
}
