#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

#include "embedding_offload.hpp"

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejects(F function) {
    bool rejected = false;
    try { function(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Expected invalid input to fail");
}
}

int main(int argc, char** argv) {
    using namespace simaai::llima;
    char temp[] = "/tmp/llima-embedding-unit-XXXXXX";
    const auto name = mkdtemp(temp);
    if (!name) return 1;
    const std::filesystem::path root(name);
    try {
        for (const auto bytes : {3u, 1536u, 2560u, 8960u, 10752u}) {
            constexpr size_t rows = 257;
            std::vector<uint8_t> input(rows * bytes);
            for (size_t i = 0; i < input.size(); ++i) input[i] = (i * 31 + i / bytes) % 251;
            const auto file = root / "rows.bin";
            { std::ofstream out(file, std::ios::binary); out.write(reinterpret_cast<char*>(input.data()), input.size()); }
            DiskEmbeddingTable table(file, rows, bytes);
            const size_t stride = bytes + 16;
            std::vector<uint32_t> ids(128);
            for (size_t i = 0; i < ids.size(); ++i) ids[i] = i * 17 % rows;
            ids[1] = ids[0]; ids[2] = rows - 1;
            std::vector<uint8_t> output(ids.size() * stride, 0xFE);
            for (const size_t count : {size_t{1}, ids.size()}) {
                table.gather(std::span<const uint32_t>(ids).first(count), output.data(), stride);
                for (size_t i = 0; i < count; ++i) {
                    for (size_t j = 0; j < bytes; ++j) require(output[i * stride + j] == input[ids[i] * bytes + j], "Incorrect gathered row");
                    for (size_t j = bytes; j < stride; ++j) require(output[i * stride + j] == 0xFE, "Row padding overwritten");
                }
            }
            table.gather({}, nullptr, stride);
            const uint32_t invalid = rows;
            rejects([&] { table.gather({&invalid, 1}, output.data(), stride); });
            rejects([&] { table.gather(ids, output.data(), bytes - 1); });
            rejects([&] { DiskEmbeddingTable wrong(file, rows + 1, bytes); });
            std::filesystem::resize_file(file, 0);
            rejects([&] { table.gather(std::span<const uint32_t>(ids).first(1), output.data(), stride); });
            rejects([&] { table.gather(ids, output.data(), stride); });
        }
        unsetenv("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD");
        require(embedding_offload_mode() == EmbeddingOffloadMode::Auto, "default auto mode");
        setenv("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD", "auto", 1);
        require(embedding_offload_mode() == EmbeddingOffloadMode::Auto, "auto mode");
        setenv("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD", "off", 1);
        require(embedding_offload_mode() == EmbeddingOffloadMode::Off, "off mode");
        setenv("SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD", "typo", 1);
        rejects([] { embedding_offload_mode(); });
        const auto disk = root / "nvme0n1";
        std::filesystem::create_directories(disk / "device");
        std::filesystem::create_directories(root / "nvme");
        std::filesystem::create_directory_symlink(root / "nvme", disk / "device/subsystem");
        std::ofstream(disk / "device/transport") << "pcie\n";
        require(embedding_device_on_nvme(disk), "NVMe disk detection");
        const auto part = disk / "nvme0n1p1";
        std::filesystem::create_directory(part);
        std::ofstream(part / "partition") << "1";
        std::filesystem::create_directory_symlink(part, root / "partition-link");
        require(embedding_device_on_nvme(root / "partition-link"), "NVMe partition detection");
        const auto dm = root / "dm-0";
        std::filesystem::create_directories(dm / "slaves");
        std::filesystem::create_directory_symlink(part, dm / "slaves/nvme0n1p1");
        require(!embedding_device_on_nvme(dm), "Device mapper is not a directly attached NVMe disk");
        const auto mmc = root / "mmcblk0";
        std::filesystem::create_directories(mmc / "device");
        std::filesystem::create_directory(root / "mmc");
        std::filesystem::create_directory_symlink(root / "mmc", mmc / "device/subsystem");
        require(!embedding_device_on_nvme(mmc), "eMMC detection");
        std::ofstream(disk / "device/transport") << "tcp\n";
        require(!embedding_device_on_nvme(disk), "Network NVMe must not enable offloading");
        std::filesystem::remove(disk / "device/subsystem");
        require(!embedding_device_on_nvme(disk), "NVMe-looking name is insufficient");
        require(!embedding_device_on_nvme(root / "absent"), "Unknown storage must not enable auto");
        require(!embedding_file_on_nvme(root / "absent"), "Missing file detection");
        // Optional DevKit checks: one NVMe file followed by non-NVMe files.
        if (argc > 1) require(embedding_file_on_nvme(argv[1]), "Expected real NVMe file");
        for (int i = 2; i < argc; ++i)
            require(!embedding_file_on_nvme(argv[i]), "Unexpected offload on non-NVMe file");
        std::filesystem::remove_all(root);
        std::cout << "Embedding offload unit tests passed\n";
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
