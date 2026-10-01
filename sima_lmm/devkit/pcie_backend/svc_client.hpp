// SvcClient: the small interface for one simaai_svc handle on the card.
// BackendLoop and EventBridge only use this, so tests can use a fake and
// need no daemon and no PCIe link.
#ifndef _SIMA_LLIMA_PCIE_BACKEND_SVC_CLIENT_
#define _SIMA_LLIMA_PCIE_BACKEND_SVC_CLIENT_

#include <string>

namespace simaai {
namespace llima {
namespace pcie_backend {

// One received notification: the tag it arrived on and its payload bytes.
struct SvcNote {
    std::string tag;
    std::string payload;
};

enum class RecvStatus { Ok, Timeout, Disconnected };

// Minimal view of one simaai_svc client handle. The real one (DlSvcClient)
// talks to the local pep daemon; tests use a scripted fake. The backend logic
// depends only on this, which is what lets it be tested with no daemon.
// Same shape as the host's neat/core/pcie_host/src/genai/SvcClient.h.
class SvcClient {
    public:
        virtual ~SvcClient() = default;
        // Throws on failure.
        virtual void subscribe(const std::string& tag) = 0;
        // Returns the far-side subscriber count. On the card it is always 0
        // (the SoC->host path has no reply), so never treat 0 as an error here.
        virtual unsigned notify(const std::string& tag, const std::string& payload) = 0;
        virtual RecvStatus recv(SvcNote& out, int timeout_ms) = 0;
};

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
