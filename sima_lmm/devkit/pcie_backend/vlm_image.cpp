// Split a serve-root-relative image name into the (parent, leaf) parts a
// PcieFileProvider needs: it is built for a subfolder and pulls by leaf name.
#include "vlm_image.hpp"

#include <filesystem>

namespace simaai {
namespace llima {
namespace pcie_backend {

ImageTarget image_target_for(const std::string& image_name) {
    const std::filesystem::path p(image_name);
    // generic_string keeps '/' on every platform, matching the serve-root names
    // the host sends. parent_path() of a bare leaf is "", which is the root.
    return {p.parent_path().generic_string(), p.filename().string()};
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
