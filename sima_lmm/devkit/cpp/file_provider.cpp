#include "file_provider.hpp"

#include <fstream>
#include <ios>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

namespace simaai {
namespace llima {

DiskFileProvider::DiskFileProvider(std::filesystem::path root)
  : _root(std::move(root)) {}

std::filesystem::path DiskFileProvider::get_path(std::string_view name) {
    return _root / std::filesystem::path(name);
}

std::unique_ptr<std::istream> DiskFileProvider::open_stream(std::string_view name) {
    // Binary so raw bytes (e.g. .bin embeddings) survive unchanged. On the
    // text configs this is identical to a default ifstream on Linux.
    const auto path = get_path(name);
    auto stream = std::make_unique<std::ifstream>(path, std::ios::binary);
    // Fail here with the file name. Otherwise the reader gets an empty stream
    // and fails later with an unclear error (e.g. a JSON or tokenizer error).
    if (!*stream) {
        throw std::runtime_error(fmt::format("Cannot open model file: {}", path.string()));
    }
    return stream;
}

}  // namespace llima
}  // namespace simaai
