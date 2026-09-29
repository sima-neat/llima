// Scripted stand-in for the simaai_svc daemon, for pcie-genai-backend tests.
#ifndef _SIMA_LLIMA_PCIE_GENAI_FAKE_SVC_CLIENT_
#define _SIMA_LLIMA_PCIE_GENAI_FAKE_SVC_CLIENT_

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "svc_client.hpp"

namespace simaai {
namespace llima {
namespace pcie_backend {
namespace test {

class FakeSvcClient final : public SvcClient {
    public:
        void subscribe(const std::string& tag) override {
            std::lock_guard<std::mutex> lock(_mutex);
            _subscribed.push_back(tag);
        }
        unsigned notify(const std::string& tag, const std::string& payload) override {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _sent.push_back({tag, payload});
            }
            _cv.notify_all();
            return 0;  // like the real card side: SoC->host never reports subscribers
        }
        RecvStatus recv(SvcNote& out, int timeout_ms) override {
            std::unique_lock<std::mutex> lock(_mutex);
            _cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [&] { return !_incoming.empty() || _disconnected; });
            if (!_incoming.empty()) {
                out = _incoming.front();
                _incoming.pop_front();
                return RecvStatus::Ok;
            }
            return _disconnected ? RecvStatus::Disconnected : RecvStatus::Timeout;
        }

        void push(const std::string& tag, const std::string& payload) {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _incoming.push_back({tag, payload});
            }
            _cv.notify_all();
        }
        void disconnect() {
            {
                std::lock_guard<std::mutex> lock(_mutex);
                _disconnected = true;
            }
            _cv.notify_all();
        }
        std::vector<SvcNote> sent() const {
            std::lock_guard<std::mutex> lock(_mutex);
            return _sent;
        }
        std::vector<std::string> subscribed() const {
            std::lock_guard<std::mutex> lock(_mutex);
            return _subscribed;
        }
        // Wait until at least `count` notes were sent on `tag`.
        bool wait_sent(const std::string& tag, std::size_t count = 1, int timeout_ms = 3000) {
            std::unique_lock<std::mutex> lock(_mutex);
            return _cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
                return static_cast<std::size_t>(std::count_if(
                           _sent.begin(), _sent.end(),
                           [&](const SvcNote& n) { return n.tag == tag; })) >= count;
            });
        }

    private:
        mutable std::mutex _mutex;
        std::condition_variable _cv;
        std::deque<SvcNote> _incoming;
        std::vector<SvcNote> _sent;
        std::vector<std::string> _subscribed;
        bool _disconnected = false;
};

}  // namespace test
}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
