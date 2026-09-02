#pragma once

#include <array>
#include <cstdint>

namespace simaai::llima::qwen3tts {

// NumPy 1.26 SeedSequence + PCG64 compatibility for integer CLI seeds.
class NumpyPcg64 {
  public:
    using U128 = unsigned __int128;

    void seed(uint64_t entropy);
    [[nodiscard]] uint64_t next_u64();
    [[nodiscard]] double next_double();

  private:
    static uint32_t hashmix(uint32_t value, uint32_t& hash_const);
    static uint32_t mix(uint32_t x, uint32_t y);
    static std::array<uint64_t, 4> seed_sequence_state(uint64_t entropy);

    U128 state_{};
    U128 increment_{};
};

} // namespace simaai::llima::qwen3tts
