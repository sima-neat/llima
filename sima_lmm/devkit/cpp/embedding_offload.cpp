#include "embedding_offload.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

namespace simaai::llima {

EmbeddingOffloadMode embedding_offload_mode() {
    const char* value = std::getenv("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD");
    const std::string mode = value ? value : "auto";
    if (mode == "auto") return EmbeddingOffloadMode::Auto;
    if (mode == "off") return EmbeddingOffloadMode::Off;
    throw std::runtime_error("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD must be auto or off");
}

bool embedding_device_on_nvme(const std::filesystem::path& device) {
    std::error_code ec;
    auto disk = std::filesystem::canonical(device, ec);
    if (ec) return false;
    if (std::filesystem::exists(disk / "partition", ec)) disk = disk.parent_path();
    // Check the kernel's device class and transport, not a mount/disk name.
    const auto subsystem = std::filesystem::canonical(disk / "device/subsystem", ec);
    if (ec || subsystem.filename() != "nvme") return false;
    std::string transport;
    std::ifstream(disk / "device/transport") >> transport;
    return transport == "pcie";
}

bool embedding_file_on_nvme(const std::filesystem::path& path) {
    struct stat st {};
    if (stat(path.c_str(), &st) || !S_ISREG(st.st_mode)) return false;
    return embedding_device_on_nvme(
        std::filesystem::path("/sys/dev/block") /
        (std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev)))
    );
}

DiskEmbeddingTable::DiskEmbeddingTable(
    const std::filesystem::path& path, size_t rows, size_t row_bytes
) : _rows(rows), _row_bytes(row_bytes), _path(path) {
    if (!rows || !row_bytes || row_bytes > static_cast<size_t>(std::numeric_limits<off_t>::max()) / rows)
        throw std::runtime_error("Invalid embedding table dimensions");
    _fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (_fd < 0) throw std::system_error(errno, std::generic_category(), "Open embedding " + path.string());
    try {
        struct stat st {};
        if (fstat(_fd, &st)) throw std::system_error(errno, std::generic_category(), "fstat embedding");
        if (!S_ISREG(st.st_mode) || st.st_size < 0 || static_cast<uint64_t>(st.st_size) != rows * row_bytes)
            throw std::runtime_error("Invalid embedding file size: " + path.string());
        posix_fadvise(_fd, 0, 0, POSIX_FADV_RANDOM);
        posix_fadvise(_fd, 0, 0, POSIX_FADV_DONTNEED);
    } catch (...) {
        close(_fd);
        throw;
    }
}

DiskEmbeddingTable::~DiskEmbeddingTable() {
    if (_fd >= 0) close(_fd);
}

void DiskEmbeddingTable::gather(std::span<const uint32_t> ids, void* destination, size_t stride) {
    if (stride < _row_bytes || (!destination && !ids.empty()))
        throw std::runtime_error("Invalid embedding destination");
    for (auto id : ids) if (id >= _rows) throw std::out_of_range("Embedding token ID exceeds vocabulary");
    std::unordered_map<uint32_t, size_t> seen;
    seen.reserve(ids.size());
    std::vector<size_t> unique_rows;
    unique_rows.reserve(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        if (seen.emplace(ids[i], i).second) unique_rows.push_back(i);
    }
    auto* dst = static_cast<uint8_t*>(destination);
    std::exception_ptr error;
    // Reuse the runtime's OpenMP pool. Reads target disjoint staging rows;
    // duplicate copies wait for the implicit barrier. All workers finish before
    // an exception escapes, so buffers cannot be released with I/O in flight.
    #pragma omp parallel for num_threads(8) if(unique_rows.size() >= 8)
    for (size_t task = 0; task < unique_rows.size(); ++task) {
        try {
            const size_t i = unique_rows[task];
            const size_t offset = static_cast<size_t>(ids[i]) * _row_bytes;
            size_t done = 0;
            while (done < _row_bytes) {
                const auto n = pread(_fd, dst + i * stride + done, _row_bytes - done, offset + done);
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) throw std::system_error(errno, std::generic_category(), "Read embedding " + _path.string());
                if (n == 0) throw std::runtime_error("Short embedding read: " + _path.string());
                done += static_cast<size_t>(n);
            }
        } catch (...) {
            #pragma omp critical(llima_embedding_read_error)
            { if (!error) error = std::current_exception(); }
        }
    }
    if (error) std::rethrow_exception(error);
    for (size_t i = 0; i < ids.size(); ++i) {
        const size_t source = seen.at(ids[i]);
        if (source != i) std::memcpy(dst + i * stride, dst + source * stride, _row_bytes);
    }
    // Avoid retaining a second table in Linux's page cache. Advisories are best
    // effort (e.g. another process may hold these pages); no global cache drop.
    const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    for (const auto& [id, row] : seen) {
        const size_t offset = static_cast<size_t>(id) * _row_bytes;
        const size_t aligned = offset / page * page;
        const size_t count = (offset - aligned + _row_bytes + page - 1) / page * page;
        posix_fadvise(_fd, aligned, count, POSIX_FADV_DONTNEED);
    }
}

} // namespace simaai::llima
