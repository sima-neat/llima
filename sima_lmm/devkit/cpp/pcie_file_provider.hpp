#pragma once

#include <filesystem>
#include <functional>
#include <istream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <system_error>

#include "file_provider.hpp"

namespace simaai {
namespace llima {

// FileProvider that pulls model files over PCIe from the host via the
// simaai_svc client (talks to the local simaai_pep_daemon). Lives in the
// cpp_ext / CLI layer, NOT the core runtime lib: only the cpp_ext module
// links simaai_svc.
class PcieFileProvider : public FileProvider {
    public:
        // Returns 0 on success, negative errno on failure. Injectable so unit
        // tests can supply a fake without the daemon. local_name is the
        // destination relative to recv_root, NOT an absolute path: the SoC-side
        // svc client only accepts a name under the daemon's default recv root
        // (an absolute path is refused), and recv_root must be that root.
        using FetchFn = std::function<int(
            const std::string& serve_root,
            const std::string& remote_name,
            const std::string& local_name)>;

        PcieFileProvider(
            std::filesystem::path recv_root,
            std::string serve_root,
            std::string subfolder,
            FetchFn fetch_fn = {});

        std::filesystem::path reserve(std::string_view name) override;   // no pull
        std::filesystem::path get_path(std::string_view name) override;  // pull; missing = throw
        bool exists(std::string_view name) override;                     // pull-probe; missing = false
        // Pull, open, then delete the disk copy right away. The open stream
        // still reads all bytes (Linux keeps an open file's data until it is
        // closed), so the space in recv_root is freed when the stream closes.
        std::unique_ptr<std::istream> open_stream(std::string_view name) override;
        void release(std::string_view name) override;                    // delete the pulled copy
        void fetch(const std::filesystem::path& path) override;
        void evict(const std::filesystem::path& path) override;
        bool pulls_files() const override { return true; }

    private:
        std::string _remote_name_for(const std::filesystem::path& path) const;
        std::string _local_name_for(const std::filesystem::path& path) const;
        // Pull abs unless this provider already pulled it and it is still on
        // disk (so exists() + open_stream() pulls a file only once).
        // Returns the fetch rc (0 = present).
        int _pull(const std::filesystem::path& abs);
        // Delete abs and forget it. Never throws; returns the delete error.
        std::error_code _remove(const std::filesystem::path& abs);

        std::filesystem::path _recv_root;
        std::string _serve_root;
        std::string _subfolder;
        FetchFn _fetch_fn;
        std::mutex _pulled_mutex;
        std::set<std::filesystem::path> _pulled;   // pulled in THIS process, still on disk
};

}  // namespace llima
}  // namespace simaai
