#ifndef _SIMA_LLIMA_FILE_PROVIDER_
#define _SIMA_LLIMA_FILE_PROVIDER_

#include <filesystem>
#include <istream>
#include <memory>
#include <string_view>

namespace simaai {
namespace llima {

// Model assets use root-relative names, e.g. "devkit/vlm_config.json".
// get_path() supplies a local file for path-based readers; open_stream()
// supplies a byte stream. release() permits removal of a fetched temporary file.
class FileProvider {
    public:
        virtual ~FileProvider() = default;

        virtual std::filesystem::path get_path(std::string_view name) = 0;
        virtual std::unique_ptr<std::istream> open_stream(std::string_view name) = 0;
        virtual void release(std::string_view name) { (void)name; }

        // Probe optional assets. Deferred providers must distinguish missing
        // files from transfer errors; required reads must fail when unavailable.
        virtual bool exists(std::string_view name) {
            return std::filesystem::exists(get_path(name));
        }

        // Name a path without fetching it. Deferred providers must override this.
        // reserve("elf_files") must match the constructor's model_path/elf_files.
        virtual std::filesystem::path reserve(std::string_view name) {
            return get_path(name);
        }

        // Load-time hooks using paths returned by reserve()/get_path().
        // Disk providers retain files; deferred providers fetch and evict them.
        virtual void fetch(const std::filesystem::path& path) { (void)path; }
        virtual void evict(const std::filesystem::path& path) { (void)path; }

        // Deferred assets require serial fetch -> MLA load -> evict.
        virtual bool pulls_files() const { return false; }
};

// Local or NFS-backed assets; release(), fetch(), and evict() are no-ops.
class DiskFileProvider : public FileProvider {
    public:
        explicit DiskFileProvider(std::filesystem::path root);

        std::filesystem::path get_path(std::string_view name) override;
        std::unique_ptr<std::istream> open_stream(std::string_view name) override;

    private:
        std::filesystem::path _root;
};

}  // namespace llima
}  // namespace simaai

#endif
