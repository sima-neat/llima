#include "language/cache_slots.hpp"

#include <iostream>
#include <stdexcept>

using namespace simaai::llima;
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Error, class Fn> void rejects(Fn&& fn) {
    try { fn(); } catch (const Error&) { return; }
    throw std::runtime_error("Expected cache admission error");
}
}

int main() {
    try {
        rejects<std::invalid_argument>([] { CacheSlots slots(0); });
        CacheSlots slots(2);
        require(slots.count() == 0, "Capacity reservation assigned an identity");
        rejects<std::invalid_argument>([&] { slots.acquire(""); });
        const auto a = slots.acquire("jarvic:router");
        const auto b = slots.acquire("jarvic:planning");
        require(a.created && b.created && a.index != b.index, "Roles share a physical slot");
        for (int i = 0; i < 100; ++i) {
            require(slots.acquire("jarvic:router").index == a.index &&
                    !slots.acquire("jarvic:planning").created, "Role identity changed on reuse");
        }
        rejects<KVCacheCapacityError>([&] { slots.acquire("jarvic:overflow"); });
        rejects<KVCacheCapacityError>([&] { slots.acquire(std::nullopt); });
        require(slots.count() == 2 && slots.acquire("jarvic:router").index == a.index,
                "Capacity failure evicted an existing context");
        require(slots.remove("jarvic:router") == a.index && !slots.remove("missing"),
                "Explicit removal changed the wrong context");
        const auto legacy = slots.acquire(std::nullopt);
        require(legacy.created && legacy.index == a.index && slots.count() == 2,
                "The unnamed context did not consume the freed slot");
        require(!slots.acquire(std::nullopt).created, "Legacy context is not reusable");
        slots.clear();
        require(slots.count() == 0, "Clear retained associations");
        const auto named_legacy = slots.acquire("legacy");
        require(named_legacy.created && slots.acquire(std::nullopt).index != named_legacy.index,
                "A named ID collided with the unnamed legacy context");
        std::cout << "CACHE_SLOTS result=PASS\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
