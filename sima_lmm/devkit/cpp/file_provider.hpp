#ifndef _SIMA_LLIMA_FILE_PROVIDER_
#define _SIMA_LLIMA_FILE_PROVIDER_

#include <filesystem>
#include <istream>
#include <memory>
#include <string_view>

namespace simaai {
namespace llima {

// The seam between LLiMa and the files it loads. LLiMa asks a FileProvider
// for each model file, in the order LLiMa chooses (Approach B). Names are
// relative to the model root and include the sub-folder, for example
// "devkit/vlm_config.json" or "elf_files/model.elf". A name must never be
// absolute.
//
// Two ways to ask for a file:
//   - get_path:    a real path on disk. Use it for files opened by a C API
//                  or a library that takes a path (ELF, GGUF, .npy).
//   - open_stream: a byte stream. Use it for everything else (JSON configs,
//                  the HF tokenizer blob, .bin embeddings).
//
// release tells the provider the file is no longer needed. On disk this does
// nothing; over PCIe it deletes the pulled temp file so the next large file
// can land (the explicit-release contract).
class FileProvider {
    public:
        virtual ~FileProvider() = default;

        virtual std::filesystem::path get_path(std::string_view name) = 0;
        virtual std::unique_ptr<std::istream> open_stream(std::string_view name) = 0;
        virtual void release(std::string_view name) { (void)name; }

        // Probe for an OPTIONAL file. Callers with a fallback (e.g. the
        // embeddings ".bin" that falls back to ".npy") must use this, NOT
        // get_path: get_path treats a missing file as a hard error so a
        // genuinely-missing required file — or, over PCIe, a mistyped
        // serve-root/sub-folder — fails loudly instead of masquerading as a
        // corrupt model. The default reproduces the old probe (get_path never
        // fetches on disk, so this is exactly today's behaviour there).
        virtual bool exists(std::string_view name) {
            return std::filesystem::exists(get_path(name));
        }

        // Reserve the path a file WILL occupy, WITHOUT fetching it. Define-time
        // callers (ELF path resolution) use this so no bytes move until load
        // time. Default == get_path, so DiskFileProvider is unchanged.
        virtual std::filesystem::path reserve(std::string_view name) {
            return get_path(name);
        }

        // Load-time hooks keyed by the absolute path reserve()/get_path()
        // returned. fetch() makes the bytes present on disk; evict() removes
        // them. Both are no-ops on the disk provider (files are already local
        // and must survive), so serial load over disk is unchanged.
        virtual void fetch(const std::filesystem::path& path) { (void)path; }
        virtual void evict(const std::filesystem::path& path) { (void)path; }

        // True if files are NOT on disk until fetch() pulls them (PCIe).
        // The MLA loader then must use the serial fetch -> load -> evict path;
        // the parallel batch load would try to load files that are not there.
        virtual bool pulls_files() const { return false; }
};

// The default provider: read straight from a local (or NFS-mounted) folder.
// This is today's behaviour. release is a no-op.
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
