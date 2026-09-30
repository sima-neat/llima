// Unit test for the pure image-name helpers used by VlmGenerator. The full
// pull + add_image path needs LLiMa and a card, so it is hardware-tested; this
// pins the parent/leaf split that decides the PcieFileProvider subfolder, and
// the check that no two images land on the same card file.
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vlm_image.hpp"

namespace {
using namespace simaai::llima::pcie_backend;
int failures = 0;
void expect(bool c, const std::string& m) { if (!c) { std::cerr << "FAIL: " << m << '\n'; ++failures; } }
bool refuses(const std::vector<std::string>& names, const std::set<std::string>& kept) {
  try { check_image_leaves(names, kept); } catch (const std::invalid_argument&) { return true; }
  return false;
}
}  // namespace

int main() {
  const ImageTarget t = image_target_for("pcie-genai/h1-1.jpg");
  expect(t.subfolder == "pcie-genai", "subfolder is the parent");
  expect(t.filename == "h1-1.jpg", "filename is the leaf");
  const ImageTarget nested = image_target_for("a/b/c.png");
  expect(nested.subfolder == "a/b", "nested subfolder");
  expect(nested.filename == "c.png", "nested filename");

  // Two images must never land on the same card file (the pull keeps only the leaf).
  expect(!refuses({"pcie-genai/h1-1-0.jpg", "pcie-genai/h1-1-1.jpg"}, {}),
         "the host's unique names pass");
  expect(refuses({"a/photo.jpg", "b/photo.jpg"}, {}), "same leaf in one request refused");
  expect(refuses({"b/photo.jpg"}, {"photo.jpg"}), "leaf already in the conversation refused");
  expect(!refuses({"b/other.jpg"}, {"photo.jpg"}), "a new leaf next to a kept one passes");
  expect(!refuses({}, {"photo.jpg"}), "no images passes");
  if (failures) return 1;
  std::cout << "[PASS] pcie-genai vlm image split and leaf check\n";
  return 0;
}
