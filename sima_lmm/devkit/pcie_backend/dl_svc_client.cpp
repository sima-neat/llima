// DlSvcClient: loads libsimaaipep.so at run time and wraps the few
// simaai_svc calls the backend needs (open, subscribe, notify, recv).
// It turns the C error codes into RecvStatus or exceptions.
#include "dl_svc_client.hpp"

#include <cerrno>
#include <stdexcept>
#include <vector>

#include <dlfcn.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "simaai_svc.h"   // header only: struct simaai_svc_note, SIMAAI_* constants

namespace simaai {
namespace llima {
namespace pcie_backend {

namespace {
struct SvcApi {
    using open_fn      = int (*)(const char*, struct simaai_svc**);
    using close_fn     = void (*)(struct simaai_svc*);
    using subscribe_fn = int (*)(struct simaai_svc*, const char*);
    using notify_fn    = int (*)(struct simaai_svc*, const struct simaai_svc_note*, unsigned int*);
    using recv_fn      = int (*)(struct simaai_svc*, struct simaai_svc_note*, void*, size_t, int);

    open_fn      open      = nullptr;
    close_fn     close     = nullptr;
    subscribe_fn subscribe = nullptr;
    notify_fn    notify    = nullptr;
    recv_fn      recv      = nullptr;
};

// dlopen, not a link-time dependency: the build needs only the header, and
// on a board without the PCIe stack the program still starts and fails with
// a clear message (not a loader error) when the library is missing.
// Same rule as PcieFileProvider.
// RTLD_LOCAL, not RTLD_GLOBAL: RTLD_GLOBAL once let the library bind
// a clashing global symbol from an earlier library, and a 670 MB pull failed
// with -EPROTO. RTLD_LOCAL keeps its symbols private; dlsym still works.
SvcApi load_api(const char* library) {
    void* h = ::dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (h == nullptr) {
        const char* why = ::dlerror();
        throw std::runtime_error(fmt::format(
            "PCIe support unavailable: cannot load {} ({}). pcie-genai-backend needs the "
            "PCIe endpoint stack on this board.", library, why ? why : "unknown error"));
    }
    SvcApi api;
    api.open      = reinterpret_cast<SvcApi::open_fn>(::dlsym(h, "simaai_svc_open"));
    api.close     = reinterpret_cast<SvcApi::close_fn>(::dlsym(h, "simaai_svc_close"));
    api.subscribe = reinterpret_cast<SvcApi::subscribe_fn>(::dlsym(h, "simaai_svc_subscribe"));
    api.notify    = reinterpret_cast<SvcApi::notify_fn>(::dlsym(h, "simaai_svc_notify"));
    api.recv      = reinterpret_cast<SvcApi::recv_fn>(::dlsym(h, "simaai_svc_recv"));
    if (!api.open || !api.close || !api.subscribe || !api.notify || !api.recv) {
        throw std::runtime_error(fmt::format(
            "PCIe support unavailable: {} is missing simaai_svc symbols", library));
    }
    return api;   // the handle stays open for the process lifetime, like PcieFileProvider
}
}  // namespace

struct DlSvcClient::Impl {
    SvcApi api;
    struct simaai_svc* handle = nullptr;
    std::vector<char> buffer = std::vector<char>(SIMAAI_SVC_PAYLOAD_MAX);
};

DlSvcClient::DlSvcClient(const char* socket, const char* library) : _impl(std::make_unique<Impl>()) {
    _impl->api = load_api(library);
    const int rc = _impl->api.open(socket, &_impl->handle);
    if (rc != 0) {
        throw std::runtime_error(fmt::format(
            "simaai_svc_open failed: rc={} (is simaai-pep-daemon running?)", rc));
    }
}

DlSvcClient::~DlSvcClient() {
    if (_impl && _impl->handle != nullptr) _impl->api.close(_impl->handle);
}

void DlSvcClient::subscribe(const std::string& tag) {
    const int rc = _impl->api.subscribe(_impl->handle, tag.c_str());
    if (rc != 0) throw std::runtime_error(fmt::format("simaai_svc_subscribe({}) failed: rc={}", tag, rc));
}

unsigned DlSvcClient::notify(const std::string& tag, const std::string& payload) {
    struct simaai_svc_note note{};
    note.type = SIMAAI_NOTE_EVENT;
    note.severity = SIMAAI_NOTE_SEV_INFO;
    note.tag = tag.c_str();
    note.payload = payload.empty() ? nullptr : payload.data();
    note.payload_len = payload.size();
    unsigned int subscribers = 0;
    const int rc = _impl->api.notify(_impl->handle, &note, &subscribers);
    if (rc != 0) throw std::runtime_error(fmt::format("simaai_svc_notify({}) failed: rc={}", tag, rc));
    return subscribers;
}

RecvStatus DlSvcClient::recv(SvcNote& out, int timeout_ms) {
    struct simaai_svc_note note{};
    const int rc = _impl->api.recv(_impl->handle, &note, _impl->buffer.data(),
                                   _impl->buffer.size(), timeout_ms);
    // -EAGAIN: nothing arrived in timeout_ms (normal; the caller loops).
    // -ECONNRESET: the local daemon went away; the caller ends its loop.
    // -ENOSPC: the note was bigger than our buffer (see below).
    if (rc == -EAGAIN) return RecvStatus::Timeout;
    if (rc == -ECONNRESET) return RecvStatus::Disconnected;
    if (rc == -ENOSPC) {
        // Consumed but larger than the 1 MiB buffer: skip it. The library has
        // already removed it from the queue, so report a timeout and go on.
        spdlog::warn("pcie-genai-backend: dropped an oversize notification ({} bytes)", note.payload_len);
        return RecvStatus::Timeout;
    }
    if (rc != 0) throw std::runtime_error(fmt::format("simaai_svc_recv failed: rc={}", rc));
    out.tag = note.tag != nullptr ? note.tag : note.tag_buf;
    out.payload.assign(static_cast<const char*>(note.payload), note.payload_len);
    // Some tools send C strings together with their trailing NUL. Our
    // payloads never contain a NUL, so strip it; otherwise the JSON parse
    // or the printed text would get a stray '\0'.
    if (!out.payload.empty() && out.payload.back() == '\0') out.payload.pop_back();
    return RecvStatus::Ok;
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
