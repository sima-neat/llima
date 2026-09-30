// The image-name split shared by VlmGenerator and its test. Kept apart from
// vlm_generator.hpp so it can be unit-tested with no LLiMa runtime and no card:
// it is pure std::filesystem, no MLA or PCIe dependency.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_VLM_IMAGE_
#define _SIMA_LLIMA_PCIE_BACKEND_VLM_IMAGE_

#include <set>
#include <string>
#include <vector>

namespace simaai {
namespace llima {
namespace pcie_backend {

// A serve-root-relative image name split into the PcieFileProvider parts.
struct ImageTarget {
    std::string subfolder;  // parent, e.g. "pcie-genai"
    std::string filename;   // leaf, e.g. "h1-1.jpg"
};

// Split "pcie-genai/h1-1.jpg" into {"pcie-genai", "h1-1.jpg"}. The name is
// already validated (relative, no "..") by parse_prompt on the card.
ImageTarget image_target_for(const std::string& image_name);

// The card pulls an image by its leaf name only, into the flat recv root. So
// "a/photo.jpg" and "b/photo.jpg" would both land on <recv>/photo.jpg, and the
// second would replace the first (also one still in the conversation).
// Throws std::invalid_argument if a leaf repeats within image_names or is in
// kept_leaves (the leaves of the images still in the conversation). The host
// names every image uniquely, so this guards only against other senders.
void check_image_leaves(const std::vector<std::string>& image_names,
                        const std::set<std::string>& kept_leaves);

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
