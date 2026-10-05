#pragma once

// Private runtime implementation; not part of the installed API.
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <cstdint>

namespace simaai::llima {

enum class EmbeddingOffloadMode { Auto, Off };
EmbeddingOffloadMode embedding_offload_mode();
bool embedding_file_on_nvme(const std::filesystem::path& path);
bool embedding_device_on_nvme(const std::filesystem::path& device);

class DiskEmbeddingTable {
public:
    DiskEmbeddingTable(const std::filesystem::path& path, size_t rows, size_t row_bytes);
    ~DiskEmbeddingTable();
    DiskEmbeddingTable(const DiskEmbeddingTable&) = delete;
    DiskEmbeddingTable& operator=(const DiskEmbeddingTable&) = delete;
    // Copies only requested rows; caller owns MLA cache maintenance. Repeated
    // token IDs within a gather reuse the first destination row.
    void gather(std::span<const uint32_t> ids, void* destination, size_t stride);
private:
    int _fd = -1;
    size_t _rows;
    size_t _row_bytes;
    std::filesystem::path _path;
};

struct EmbeddingOffload {
    std::unique_ptr<DiskEmbeddingTable> normal;
    std::unique_ptr<DiskEmbeddingTable> per_layer;
    std::vector<uint32_t> prompt_ids;
    bool chunk_staging = false;
};

} // namespace simaai::llima
