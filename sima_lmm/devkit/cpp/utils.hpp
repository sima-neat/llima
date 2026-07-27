
#ifndef _SIMA_UTILS_
#define _SIMA_UTILS_

#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <omp.h>
#include <optional>
#include <ranges>
#include <ratio>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include <spdlog/spdlog.h>

// Performance profiling code insertion with switch to turn on/off.
#ifndef SIMA_PERF_PROFILE
#define SIMA_PERF_PROFILE 0
#endif

// Make the symbol to be visible by other users of the .so file.
#define EXPORT __attribute__((visibility("default")))

namespace simaai {
namespace llima {

class ChronoTimer {
public:
  ChronoTimer(bool start_now = false) {
    if (start_now)
      start();
    else
      _begin = {};
  };
  void start() {
    _begin = std::chrono::high_resolution_clock::now();
  };

  template <typename T = std::ratio<1, 1>> double stop(bool restart = false) {
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, T>(end - _begin).count();
    if (SIMA_PERF_PROFILE)
      if constexpr (std::is_same_v<T, std::ratio<1, 1>>)
        spdlog::info("[PERF PROFILE] Duration: {:.5f}s", duration);
    if (restart)
      _begin = end;
    return duration;
  };

private:
  std::chrono::high_resolution_clock::time_point _begin;
};

// Useful size/dimension calculation functions.
#define MLA_ROW_SIZE 16

template <typename T>
concept C = std::is_integral_v<T>;

inline auto ceil_div(C auto x, C auto y) {
  return (x + y - 1) / y;
}
inline auto round_up_to(C auto x, C auto y) {
  return ceil_div(x, y) * y;
}
inline auto round_up_to_row(C auto x) {
  return round_up_to(x, MLA_ROW_SIZE);
}

inline std::string trim(std::string in) {
  auto is_space = [](unsigned char c) { return std::isspace(c); };

  auto view = in | std::views::drop_while(is_space) | std::views::reverse |
              std::views::drop_while(is_space) | std::views::reverse;
  return {view.begin(), view.end()};
}

std::vector<uint8_t> base64_decode(std::string_view base64_data);

inline size_t count_regular_files(const std::filesystem::path& dir_path) {
  auto it = std::filesystem::directory_iterator(dir_path);
  return std::count_if(std::filesystem::begin(it), std::filesystem::end(it),
                       [](const auto& entry) { return entry.is_regular_file(); });
}

inline bool get_env_var(const std::string& name, bool default_value = false) {
  auto var_ptr = std::getenv(name.c_str());
  if (var_ptr == nullptr)
    return default_value;
  auto var = std::strcmp(var_ptr, "true") == 0 || std::strcmp(var_ptr, "1") == 0;

  if (var) {
    auto msg = fmt::format("{}: {}", name, var);
    spdlog::info(msg);
    std::cout << msg << std::endl << std::flush;
  }
  return var;
}

inline std::string get_env_var(const std::string& name, std::string default_value = "") {
  auto var_ptr = std::getenv(name.c_str());
  if (var_ptr == nullptr)
    return default_value;
  std::string var(var_ptr);
  if (!var.empty()) {
    auto msg = fmt::format("{}: {}", name, var);
    spdlog::info(msg);
    std::cout << msg << std::endl << std::flush;
  }
  return var;
}

extern std::optional<std::filesystem::path> sample_image_file_name;
extern std::optional<std::filesystem::path> sample_audio_file_name;

void set_sample_image_file_name(std::filesystem::path file_name);
void set_sample_audio_file_name(std::filesystem::path file_name);

template <typename T> struct TopKResult {
  std::vector<std::vector<T>> values;
  std::vector<std::vector<int32_t>> indices;
};

// Portable batched top-K. Heap-based single-row body + OMP across rows. Returns
// (indices, values) per row, sorted descending. k must be in [1, 32].
template <typename T> TopKResult<T> topk(const T* input, size_t rows, size_t cols, int k) {
  if (k < 1 || k > 32) {
    throw std::runtime_error("topk: k out of supported range [1, 32]");
  }
  TopKResult<T> result;
  result.values.resize(rows);
  result.indices.resize(rows);

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < rows; ++i) {
    const T* row = input + i * cols;

    T heap_vals[32];
    int32_t heap_idx[32];

    for (int j = 0; j < k; ++j) {
      heap_vals[j] = row[j];
      heap_idx[j] = j;
    }

    auto sift_down = [&](int root) {
      while (2 * root + 1 < k) {
        int child = 2 * root + 1;
        if (child + 1 < k && heap_vals[child] > heap_vals[child + 1])
          child++;
        if (!(heap_vals[root] > heap_vals[child]))
          return;
        T tv = heap_vals[root];
        heap_vals[root] = heap_vals[child];
        heap_vals[child] = tv;
        int32_t ti = heap_idx[root];
        heap_idx[root] = heap_idx[child];
        heap_idx[child] = ti;
        root = child;
      }
    };
    for (int j = k / 2 - 1; j >= 0; --j)
      sift_down(j);

    for (size_t j = static_cast<size_t>(k); j < cols; ++j) {
      if (row[j] > heap_vals[0]) {
        heap_vals[0] = row[j];
        heap_idx[0] = static_cast<int32_t>(j);
        sift_down(0);
      }
    }

    result.values[i].resize(k);
    result.indices[i].resize(k);
    for (int a = 0; a < k; ++a) {
      int max_i = a;
      for (int b = a + 1; b < k; ++b) {
        if (heap_vals[b] > heap_vals[max_i])
          max_i = b;
      }
      result.values[i][a] = heap_vals[max_i];
      result.indices[i][a] = heap_idx[max_i];
      heap_vals[max_i] = heap_vals[a];
      heap_idx[max_i] = heap_idx[a];
    }
  }
  return result;
}

// Log-softmax over `n` values (converted to float). Returns a fresh vector of length n.
// Numerically stable via the standard max-subtraction trick.
template <typename T> inline std::vector<float> logsoftmax(const T* values, size_t n) {
  float max_v = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < n; ++i) {
    max_v = std::max(max_v, static_cast<float>(values[i]));
  }
  float sum = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    sum += std::exp(static_cast<float>(values[i]) - max_v);
  }
  const float log_sum = std::log(sum) + max_v;
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    out[i] = static_cast<float>(values[i]) - log_sum;
  }
  return out;
}

// Pre-warm OMP + heap paths by running topk on tiny dummy data. Safe to call multiple
// times — only the first call pays the libgomp init cost.
inline void warmup_omp() {
  float dummy[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  (void)topk<float>(dummy, 1, 8, 1);
}

} // namespace llima
} // namespace simaai

#endif
