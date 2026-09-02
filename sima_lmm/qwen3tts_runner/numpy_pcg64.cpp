#include "numpy_pcg64.hpp"

#include <array>
#include <cstddef>

namespace simaai::llima::qwen3tts {
namespace {
constexpr uint32_t kInitA = 0x43b0d7e5U;
constexpr uint32_t kMultA = 0x931e8875U;
constexpr uint32_t kInitB = 0x8b51f9ddU;
constexpr uint32_t kMultB = 0x58f38dedU;
constexpr uint32_t kMixMultL = 0xca01f9ddU;
constexpr uint32_t kMixMultR = 0x4973f715U;
constexpr NumpyPcg64::U128 kMultiplier =
    (static_cast<NumpyPcg64::U128>(2549297995355413924ULL) << 64U) | 4865540595714422341ULL;
}

uint32_t NumpyPcg64::hashmix(uint32_t value, uint32_t& hash_const) {
    value ^= hash_const;
    hash_const *= kMultA;
    value *= hash_const;
    return value ^ (value >> 16U);
}

uint32_t NumpyPcg64::mix(uint32_t x, uint32_t y) {
    uint32_t value = kMixMultL * x - kMixMultR * y;
    return value ^ (value >> 16U);
}

std::array<uint64_t, 4> NumpyPcg64::seed_sequence_state(uint64_t entropy) {
    // This is NumPy SeedSequence(seed).generate_state(4, uint64) for a non-negative integer seed.
    std::array<uint32_t, 4> pool{};
    std::array<uint32_t, 2> source{static_cast<uint32_t>(entropy), static_cast<uint32_t>(entropy >> 32U)};
    std::size_t source_size = entropy > 0xffffffffULL ? 2 : 1;
    uint32_t hash_const = kInitA;
    for (std::size_t i = 0; i < pool.size(); ++i) pool[i] = hashmix(i < source_size ? source[i] : 0U, hash_const);
    for (std::size_t src = 0; src < pool.size(); ++src) {
        for (std::size_t dst = 0; dst < pool.size(); ++dst) {
            if (src != dst) pool[dst] = mix(pool[dst], hashmix(pool[src], hash_const));
        }
    }

    std::array<uint32_t, 8> words{};
    hash_const = kInitB;
    for (std::size_t i = 0; i < words.size(); ++i) {
        uint32_t value = pool[i % pool.size()] ^ hash_const;
        hash_const *= kMultB;
        value *= hash_const;
        words[i] = value ^ (value >> 16U);
    }
    std::array<uint64_t, 4> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<uint64_t>(words[i * 2]) |
                    (static_cast<uint64_t>(words[i * 2 + 1]) << 32U);
    }
    return result;
}

void NumpyPcg64::seed(uint64_t entropy) {
    const auto material = seed_sequence_state(entropy);
    const U128 initial_state = (static_cast<U128>(material[0]) << 64U) | material[1];
    const U128 initial_sequence = (static_cast<U128>(material[2]) << 64U) | material[3];
    state_ = 0;
    increment_ = (initial_sequence << 1U) | 1U;
    state_ = state_ * kMultiplier + increment_;
    state_ += initial_state;
    state_ = state_ * kMultiplier + increment_;
}

uint64_t NumpyPcg64::next_u64() {
    state_ = state_ * kMultiplier + increment_;
    const auto high = static_cast<uint64_t>(state_ >> 64U);
    const auto low = static_cast<uint64_t>(state_);
    const auto value = high ^ low;
    const auto rotation = static_cast<unsigned int>(state_ >> 122U);
    return (value >> rotation) | (value << ((-rotation) & 63U));
}

double NumpyPcg64::next_double() {
    return static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
}

} // namespace simaai::llima::qwen3tts
