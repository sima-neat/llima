#include "pcie_file_provider.hpp"

#include <cerrno>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <system_error>

#include <dlfcn.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

// The svc header is only in sysroots that have the PCIe stack. Without it we
// still build (so disk-only builds work); only the real PCIe pull is left out,
// and --pcie then fails with a clear message.
#if __has_include("simaai_svc.h")
#include "simaai_svc.h"   // header-only: struct/opts types + SIMAAI_SVC_XF_* flags
#define SIMA_LLIMA_HAVE_SVC 1
#else
#define SIMA_LLIMA_HAVE_SVC 0
#endif

namespace simaai {
namespace llima {

namespace {
#if SIMA_LLIMA_HAVE_SVC
// simaai_svc lives in libsimaaipep.so, which is present only on boards that
// have the PCIe endpoint stack installed. We must NOT hard-link it: this file
// is compiled into cpp_ext, the single module that BOTH the disk-only CLI and
// the --pcie CLI import. A DT_NEEDED on libsimaaipep.so would make cpp_ext fail
// to load on any board without that library — breaking plain disk runs, which
// never touch PCIe. So resolve the svc entry points lazily with dlopen, only
// when a real PCIe fetch happens; a board with no PCIe stack still imports
// cpp_ext and runs from disk, and --pcie fails loudly with a clear message.
struct SvcApi {
    using open_fn  = int (*)(const char*, struct simaai_svc**);
    using getf_fn  = int (*)(struct simaai_svc*, const char*, const char*,
                             const char*, const struct simaai_svc_xfer_opts*,
                             struct simaai_svc_xfer*);
    using close_fn = void (*)(struct simaai_svc*);

    open_fn  open     = nullptr;
    getf_fn  get_file = nullptr;
    close_fn close    = nullptr;
};

// Load libsimaaipep.so once, on first use. The Meyers singleton gives
// thread-safe one-time init (C++11+); if the lib/symbols are missing the init
// throws, so a later call retries and throws again (never a silent half-load).
// The handle is intentionally kept for the process lifetime (no dlclose): svc
// stays usable for every subsequent pull. The SONAME is unversioned
// ("libsimaaipep.so"), so this dlopen resolves the same file the old
// DT_NEEDED did.
const SvcApi& svc_api() {
    static const SvcApi api = [] {
        static constexpr const char* kLib = "libsimaaipep.so";
        // RTLD_LOCAL (NOT GLOBAL): keep libsimaaipep's symbols private to its own
        // dependency subtree. We load it LATE (after the runtime, ggml, opencv,
        // etc. are already in the global scope); with RTLD_GLOBAL its internal
        // large-transfer path (memfd/DMA for big buffers) can bind to a clashing
        // global symbol from an earlier lib and corrupt a big pull mid-stream
        // (observed: small files OK, a 670 MB pull died with -EPROTO). RTLD_LOCAL
        // isolates it, and we still reach the entry points via dlsym on the handle.
        void* h = ::dlopen(kLib, RTLD_NOW | RTLD_LOCAL);
        if (h == nullptr) {
            // Call dlerror() only once: it clears the error, so a second call
            // returns NULL (and fmt throws on a NULL char*).
            const char* err = ::dlerror();
            throw std::runtime_error(fmt::format(
                "PCIe support unavailable: cannot load {} ({}). This board has "
                "no PCIe endpoint stack; --pcie needs it (disk runs do not).",
                kLib, err ? err : "unknown error"));
        }
        SvcApi a;
        a.open     = reinterpret_cast<SvcApi::open_fn>(::dlsym(h, "simaai_svc_open"));
        a.get_file = reinterpret_cast<SvcApi::getf_fn>(::dlsym(h, "simaai_svc_get_file"));
        a.close    = reinterpret_cast<SvcApi::close_fn>(::dlsym(h, "simaai_svc_close"));
        if (a.open == nullptr || a.get_file == nullptr || a.close == nullptr) {
            throw std::runtime_error(fmt::format(
                "PCIe support unavailable: {} is missing simaai_svc symbols", kLib));
        }
        return a;
    }();
    return api;
}

// Real fetch: open a per-call svc client and pull one file. CRC off for the
// large ELF pulls; overwrite so a re-run replaces a stale partial file.
int svc_get_file(const std::string& serve_root,
                 const std::string& remote_name,
                 const std::string& local_name) {
    const SvcApi& svc = svc_api();          // dlopen on first call; throws if absent
    struct simaai_svc* h = nullptr;         // VERIFIED: opaque handle, not simaai_svc_handle_t

    // Per-fetch timing so a run reveals WHERE the time goes. We split it three
    // ways because "PCIe is slower than NFS" has several possible causes and
    // only the breakdown tells them apart:
    //   - stats.elapsed_ms : the daemon's own transfer time -> the pure LINK rate.
    //   - call_ms          : the get_file() wall time -> effective rate, includes
    //                        queue-wait behind other clients + per-call setup.
    //   - open_ms/close_ms : we open a NEW client per file; with many ELFs this
    //                        per-file overhead can dominate the link itself.
    // A big gap between the link rate and the call rate points at queue/setup,
    // not the bus; a small gap with still-low numbers points at the transport.
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    int rc = svc.open(nullptr, &h);
    const auto t1 = clock::now();
    if (rc != 0) return rc;
    struct simaai_svc_xfer stats{};         // VERIFIED: struct simaai_svc_xfer, not ..._stats_t
    // VERIFIED against the soc/EP-side simaai_svc.h (the board runs the
    // simaai-pcie-ep package built from it): the 5th arg is a pointer to
    // simaai_svc_xfer_opts, not a bare flags word. Zero-init gives the
    // library's default drain timings and no progress callback.
    struct simaai_svc_xfer_opts opts{};
    opts.flags = SIMAAI_SVC_XF_OVERWRITE | SIMAAI_SVC_XF_NOCRC;
    // dest == the far daemon's configured root name ("models"); remote_name is
    // relative to it; local_name is a name under our daemon's default recv root
    // (relative, never absolute — the SoC side refuses an absolute dst).
    const auto t2 = clock::now();
    rc = svc.get_file(
        h, serve_root.c_str(), remote_name.c_str(), local_name.c_str(),
        &opts, &stats);
    const auto t3 = clock::now();
    svc.close(h);
    const auto t4 = clock::now();

    if (rc == 0) {
        using ms = std::chrono::duration<double, std::milli>;
        const double open_ms  = ms(t1 - t0).count();
        const double call_ms  = ms(t3 - t2).count();
        const double close_ms = ms(t4 - t3).count();
        constexpr double kMiB = 1024.0 * 1024.0;
        const double link_mibs = stats.elapsed_ms > 0
            ? stats.bytes * 1000.0 / stats.elapsed_ms / kMiB : 0.0;
        const double call_mibs = call_ms > 0.0
            ? stats.bytes / (call_ms / 1000.0) / kMiB : 0.0;
        spdlog::info(
            "PCIe pull {}: {} B | daemon {} ms = {:.1f} MiB/s | call {:.1f} ms = "
            "{:.1f} MiB/s | open {:.1f} ms, close {:.1f} ms",
            remote_name, stats.bytes, stats.elapsed_ms, link_mibs,
            call_ms, call_mibs, open_ms, close_ms);
    }
    return rc;
}
#else
int svc_get_file(const std::string&, const std::string&, const std::string&) {
    throw std::runtime_error(
        "PCIe support unavailable: llima was built without simaai_svc.h, so "
        "--pcie cannot pull files (disk runs do not need it).");
}
#endif
}  // namespace

PcieFileProvider::PcieFileProvider(
    std::filesystem::path recv_root,
    std::string serve_root,
    std::string subfolder,
    FetchFn fetch_fn)
  : _recv_root(std::filesystem::absolute(std::move(recv_root)).lexically_normal()),
    _serve_root(std::move(serve_root)),
    _subfolder(std::move(subfolder)),
    _fetch_fn(fetch_fn ? std::move(fetch_fn) : FetchFn(&svc_get_file)) {}

std::filesystem::path PcieFileProvider::reserve(std::string_view name) {
    return _recv_root / std::filesystem::path(name);
}

std::string PcieFileProvider::_remote_name_for(const std::filesystem::path& path) const {
    const auto rel = _local_name_for(path);
    return _subfolder.empty() ? rel : _subfolder + "/" + rel;
}

// The svc dst: the file's name relative to recv_root. recv_root must be the
// daemon's default recv root, so this relative name is where the pulled file
// actually lands (and where get_path/fetch then read it as an absolute path).
std::string PcieFileProvider::_local_name_for(const std::filesystem::path& path) const {
    const auto abs = std::filesystem::absolute(path).lexically_normal();
    return std::filesystem::relative(abs, _recv_root).generic_string();
}

int PcieFileProvider::_pull(const std::filesystem::path& abs) {
    {
        // Already pulled by us in this run and still there: do not pull again.
        // Only files WE pulled count, so a stale file left by an earlier run
        // is never used without a fresh pull.
        std::lock_guard lock(_pulled_mutex);
        if (_pulled.count(abs) != 0 && std::filesystem::is_regular_file(abs)) return 0;
    }
    std::filesystem::create_directories(abs.parent_path());
    const int rc = _fetch_fn(_serve_root, _remote_name_for(abs), _local_name_for(abs));
    if (rc == 0) {
        std::lock_guard lock(_pulled_mutex);
        _pulled.insert(abs);
    }
    return rc;
}

std::error_code PcieFileProvider::_remove(const std::filesystem::path& abs) {
    std::error_code ec;
    std::filesystem::remove(abs, ec);  // best-effort; do not throw on cleanup
    std::lock_guard lock(_pulled_mutex);
    _pulled.erase(abs);
    return ec;
}

void PcieFileProvider::fetch(const std::filesystem::path& path) {
    const auto abs = std::filesystem::absolute(path).lexically_normal();
    const int rc = _pull(abs);
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "PCIe fetch failed (rc={}) for {}", rc, _remote_name_for(abs)));
    }
}

void PcieFileProvider::evict(const std::filesystem::path& path) {
    const auto ec = _remove(std::filesystem::absolute(path).lexically_normal());
    if (ec) {
        // A failed evict leaves the just-loaded file on disk, so the next
        // fetch stages a second one and the one-file-at-a-time invariant is
        // broken. We must not throw during cleanup, but make the breach
        // visible instead of swallowing it silently.
        spdlog::warn(
            "PCIe evict could not remove {}: {} — the next fetch may leave two "
            "files on disk", path.string(), ec.message());
    }
}

std::filesystem::path PcieFileProvider::get_path(std::string_view name) {
    const auto abs = std::filesystem::absolute(reserve(name)).lexically_normal();
    const int rc = _pull(abs);
    if (rc != 0) {
        // Any nonzero rc is a hard error here. -ENOENT means the file is
        // missing OR the serve-root/sub-folder is unconfigured (the daemon
        // reports both the same way), so throwing keeps a mistyped
        // --pcie-serve-root from looking like a corrupt model. Callers that
        // legitimately probe an OPTIONAL file (a fallback exists) must use
        // exists(), not get_path().
        throw std::runtime_error(fmt::format(
            "PCIe get_path failed (rc={}) for {}", rc, _remote_name_for(abs)));
    }
    return abs;
}

bool PcieFileProvider::exists(std::string_view name) {
    const auto abs = std::filesystem::absolute(reserve(name)).lexically_normal();
    const int rc = _pull(abs);
    // present (and now pulled; a following open_stream/get_path reuses it)
    if (rc == 0) return true;
    if (rc == -ENOENT) return false;     // optional file absent — caller falls back
    throw std::runtime_error(fmt::format(
        "PCIe exists() failed (rc={}) for {}", rc, _remote_name_for(abs)));
}

std::unique_ptr<std::istream> PcieFileProvider::open_stream(std::string_view name) {
    const auto abs = get_path(name);
    auto stream = std::make_unique<std::ifstream>(abs, std::ios::binary);
    if (!*stream) {
        throw std::runtime_error(fmt::format("Cannot open pulled file: {}", abs.string()));
    }
    // The stream holds the file open, so deleting the name now is safe: the
    // bytes stay readable and the space is freed when the stream is closed.
    // This keeps configs and embedding tables from piling up in recv_root.
    if (const auto ec = _remove(abs)) {
        spdlog::warn("PCIe could not remove {} after opening it: {}",
                     abs.string(), ec.message());
    }
    return stream;
}

void PcieFileProvider::release(std::string_view name) {
    const auto abs = std::filesystem::absolute(reserve(name)).lexically_normal();
    if (const auto ec = _remove(abs)) {
        spdlog::warn("PCIe release could not remove {}: {}", abs.string(), ec.message());
    }
}

}  // namespace llima
}  // namespace simaai
