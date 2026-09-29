// Unit test for the pure image-name split used by VlmGenerator. The full
// pull + add_image path needs LLiMa and a card, so it is hardware-tested; this
// pins only the parent/leaf split that decides the PcieFileProvider subfolder.
#include <iostream>
#include <string>

#include "vlm_image.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }
}  // namespace

int main() {
  const ImageTarget t = image_target_for("pcie-genai/h1-1.jpg");
  expect(t.subfolder == "pcie-genai", "subfolder is the parent");
  expect(t.filename == "h1-1.jpg", "filename is the leaf");
  const ImageTarget nested = image_target_for("a/b/c.png");
  expect(nested.subfolder == "a/b", "nested subfolder");
  expect(nested.filename == "c.png", "nested filename");
  if (failures) return 1;
  std::cout << "[PASS] pcie-genai vlm image split\n";
  return 0;
}
