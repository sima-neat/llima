// DlSvcClient: the real SvcClient on the card, a handle to the local
// simaai_pep_daemon. BackendLoop and EventBridge use it to receive
// genai.prompt / genai.cancel and to send genai.* events to the host.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_DL_SVC_CLIENT_
#define _SIMA_LLIMA_PCIE_BACKEND_DL_SVC_CLIENT_

#include <memory>

#include "svc_client.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {

// The real SvcClient: one handle to the local simaai_pep_daemon.
// libsimaaipep.so is dlopen'ed RTLD_NOW | RTLD_LOCAL and never linked — the
// same rule as PcieFileProvider (RTLD_GLOBAL let a large pull bind a
// clashing global symbol and fail with -EPROTO). Both dlopen the same file, so
// they share one library handle.
class DlSvcClient final : public SvcClient {
    public:
        // `socket` nullptr = the daemon's default socket.
        explicit DlSvcClient(const char* socket = nullptr, const char* library = "libsimaaipep.so");
        ~DlSvcClient() override;
        DlSvcClient(const DlSvcClient&) = delete;
        DlSvcClient& operator=(const DlSvcClient&) = delete;

        void subscribe(const std::string& tag) override;
        unsigned notify(const std::string& tag, const std::string& payload) override;
        RecvStatus recv(SvcNote& out, int timeout_ms) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> _impl;
};

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
