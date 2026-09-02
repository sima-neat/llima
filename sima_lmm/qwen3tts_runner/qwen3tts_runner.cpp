#include "qwen3tts_runner.hpp"
#include "qwen3_mla_static_batch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <openssl/sha.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Eigen/Dense>

namespace simaai::llima::qwen3tts {

struct NativeTensor {
  std::vector<size_t> shape;
  std::vector<float> values;

  NativeTensor() = default;
  explicit NativeTensor(std::vector<size_t> dimensions, float value = 0.0F)
      : shape(std::move(dimensions)), values(numel(shape), value) {}

  static size_t numel(const std::vector<size_t> &dimensions) {
    return std::accumulate(dimensions.begin(), dimensions.end(), size_t{1},
                           std::multiplies<>());
  }
  size_t numel() const { return values.size(); }
  size_t rank() const { return shape.size(); }
  size_t dim(size_t index) const { return shape.at(index); }
};

struct NativeTensorView {
  const float *values{};
  std::vector<size_t> shape;

  size_t rank() const { return shape.size(); }
  size_t dim(size_t index) const { return shape.at(index); }
};

namespace {
constexpr uint16_t kBackboneLayers = 28;
constexpr uint16_t kCpLayers = 5;
constexpr uint16_t kCodecLayers = 8;
constexpr uint16_t kHidden = 1024;
constexpr uint16_t kHeadDim = 128;
constexpr uint16_t kKvSize = 1024;
constexpr uint16_t kQSize = 2048;
constexpr uint16_t kBackboneMax = 1024;
constexpr uint16_t kCpMax = 17;
constexpr uint16_t kCodecMax = 50;
constexpr uint16_t kCodecCacheMax = 1024;
constexpr uint16_t kCodecHidden = 512;
constexpr uint16_t kCodecLatent = 1024;
constexpr uint16_t kCodecHeads = 16;
constexpr uint16_t kCodecQSize = 1024;
constexpr float kBackboneRopeTheta = 1000000.0F;
constexpr float kCodePredictorRopeTheta = 1000000.0F;
constexpr float kCodecDecoderRopeTheta = 10000.0F;
constexpr int32_t kCodecEos = 2150;
constexpr int32_t kTalkerVocab = 3072;
constexpr int32_t kCpVocab = 2048;
constexpr float kBf16Lowest = -3.38953139e38F;

std::string lower(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::vector<uint32_t> u32(std::initializer_list<uint32_t> values) {
  return {values.begin(), values.end()};
}

uint16_t aligned_cache(uint16_t token) {
  return static_cast<uint16_t>((((token + 1) + 127) / 128) * 128 - 1);
}

std::filesystem::path elf_path(const std::filesystem::path &dir,
                               const std::string &name, const std::string &part,
                               uint16_t layer) {
  return dir /
         (name + "_n1_" + part + std::to_string(layer) + "_stage1_mla.elf");
}

std::filesystem::path cache_path(const std::filesystem::path &dir,
                                 const std::string &name, uint16_t token) {
  return dir / (name + "_n1_cache_token" +
                std::to_string(aligned_cache(token)) + "_stage1_mla.elf");
}

std::filesystem::path n128_elf_path(const std::filesystem::path &dir,
                                    const std::string &name,
                                    const std::string &part, uint16_t layer) {
  return dir /
         (name + "_n128_" + part + std::to_string(layer) + "_stage1_mla.elf");
}

std::string sha256(const void *data, size_t len) {
  std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
  SHA256(reinterpret_cast<const unsigned char *>(data), len, digest.data());
  std::ostringstream stream;
  for (const auto value : digest)
    stream << std::hex << std::setw(2) << std::setfill('0') << int(value);
  return stream.str();
}

void write_u16(std::ofstream &out, uint16_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}
void write_u32(std::ofstream &out, uint32_t value) {
  out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

using RowMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

void require_matrix(const NativeTensor &tensor, const char *name) {
  if (tensor.rank() != 2)
    throw std::runtime_error(std::string(name) + " must be rank 2");
}

NativeTensor row(const NativeTensor &tensor, size_t index) {
  require_matrix(tensor, "tensor");
  if (index >= tensor.dim(0))
    throw std::runtime_error("Tensor row index out of range");
  NativeTensor result({1, tensor.dim(1)});
  std::memcpy(result.values.data(),
              tensor.values.data() + index * tensor.dim(1),
              tensor.dim(1) * sizeof(float));
  return result;
}

NativeTensor row(const NativeTensorView &tensor, size_t index) {
  if (tensor.rank() != 2 || index >= tensor.dim(0)) {
    throw std::runtime_error("Safetensors row index out of range");
  }
  NativeTensor result({1, tensor.dim(1)});
  std::memcpy(result.values.data(), tensor.values + index * tensor.dim(1),
              tensor.dim(1) * sizeof(float));
  return result;
}

NativeTensor gather(const NativeTensorView &tensor,
                    const std::vector<uint32_t> &ids) {
  if (tensor.rank() != 2)
    throw std::runtime_error("Safetensors embedding must be rank 2");
  NativeTensor result({ids.size(), tensor.dim(1)});
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] >= tensor.dim(0))
      throw std::runtime_error("Embedding token out of range");
    std::memcpy(result.values.data() + i * tensor.dim(1),
                tensor.values + static_cast<size_t>(ids[i]) * tensor.dim(1),
                tensor.dim(1) * sizeof(float));
  }
  return result;
}

NativeTensor concatenate_rows(const std::vector<NativeTensor> &pieces) {
  if (pieces.empty())
    return {};
  const auto columns = pieces.front().dim(1);
  size_t rows{};
  for (const auto &piece : pieces) {
    require_matrix(piece, "concatenation input");
    if (piece.dim(1) != columns)
      throw std::runtime_error("Mismatched tensor columns");
    rows += piece.dim(0);
  }
  NativeTensor result({rows, columns});
  size_t cursor{};
  for (const auto &piece : pieces) {
    std::memcpy(result.values.data() + cursor, piece.values.data(),
                piece.values.size() * sizeof(float));
    cursor += piece.values.size();
  }
  return result;
}

NativeTensor add(const NativeTensor &lhs, const NativeTensor &rhs) {
  if (lhs.shape != rhs.shape)
    throw std::runtime_error("Mismatched tensor shapes");
  NativeTensor result(lhs.shape);
  for (size_t i = 0; i < result.numel(); ++i)
    result.values[i] = lhs.values[i] + rhs.values[i];
  return result;
}

void add_inplace(NativeTensor &lhs, const NativeTensor &rhs) {
  if (lhs.shape != rhs.shape)
    throw std::runtime_error("Mismatched tensor shapes");
  for (size_t i = 0; i < lhs.numel(); ++i)
    lhs.values[i] += rhs.values[i];
}

NativeTensor linear(const NativeTensor &input, const NativeTensorView &weight,
                    const NativeTensorView *bias = nullptr) {
  require_matrix(input, "linear input");
  if (weight.rank() != 2 || input.dim(1) != weight.dim(1)) {
    throw std::runtime_error("Invalid linear matrix dimensions");
  }
  if (bias && (bias->rank() != 1 || bias->dim(0) != weight.dim(0))) {
    throw std::runtime_error("Invalid linear bias dimensions");
  }
  NativeTensor result({input.dim(0), weight.dim(0)});
  Eigen::Map<const RowMatrix> x(input.values.data(),
                                static_cast<Eigen::Index>(input.dim(0)),
                                static_cast<Eigen::Index>(input.dim(1)));
  Eigen::Map<const RowMatrix> w(weight.values,
                                static_cast<Eigen::Index>(weight.dim(0)),
                                static_cast<Eigen::Index>(weight.dim(1)));
  Eigen::Map<RowMatrix> out(result.values.data(),
                            static_cast<Eigen::Index>(result.dim(0)),
                            static_cast<Eigen::Index>(result.dim(1)));
  out.noalias() = x * w.transpose();
  if (bias) {
    for (size_t r = 0; r < result.dim(0); ++r) {
      for (size_t c = 0; c < result.dim(1); ++c)
        result.values[r * result.dim(1) + c] += bias->values[c];
    }
  }
  return result;
}

NativeTensor linear_1x1(const NativeTensor &input,
                        const NativeTensorView &weight) {
  if (weight.rank() != 3 || weight.dim(2) != 1)
    throw std::runtime_error("Expected a 1x1 convolution weight");
  NativeTensorView matrix{weight.values, {weight.dim(0), weight.dim(1)}};
  return linear(input, matrix);
}

NativeTensor silu(const NativeTensor &input) {
  NativeTensor result(input.shape);
  for (size_t i = 0; i < input.numel(); ++i)
    result.values[i] = input.values[i] / (1.0F + std::exp(-input.values[i]));
  return result;
}

NativeTensor repeat_row(const NativeTensor &input, size_t rows) {
  if (input.shape != std::vector<size_t>{1, input.dim(1)})
    throw std::runtime_error("Expected a single row");
  NativeTensor result({rows, input.dim(1)});
  for (size_t i = 0; i < rows; ++i) {
    std::memcpy(result.values.data() + i * input.dim(1), input.values.data(),
                input.dim(1) * sizeof(float));
  }
  return result;
}

NativeTensor as_rank4_row(const NativeTensor &input, size_t width) {
  if (input.numel() != width)
    throw std::runtime_error("Invalid MLA row width");
  NativeTensor result({1, 1, 1, width});
  result.values = input.values;
  return result;
}

NativeTensor bf16_rounded(const NativeTensor &input) {
  NativeTensor result(input.shape);
  for (size_t i = 0; i < input.numel(); ++i) {
    result.values[i] =
        static_cast<float>(static_cast<Eigen::bfloat16>(input.values[i]));
  }
  return result;
}

std::string bf16_sha256(const NativeTensor &value) {
  std::vector<Eigen::bfloat16> rounded(value.numel());
  for (size_t i = 0; i < value.numel(); ++i)
    rounded[i] = static_cast<Eigen::bfloat16>(value.values[i]);
  return sha256(rounded.data(), rounded.size() * sizeof(Eigen::bfloat16));
}

NativeTensor rope(uint16_t rows, uint16_t head_dim, float theta, bool cosine) {
  NativeTensor result({rows, static_cast<size_t>(head_dim / 2)});
  for (size_t position = 0; position < rows; ++position) {
    for (size_t dimension = 0; dimension < result.dim(1); ++dimension) {
      const auto angle = static_cast<float>(position) *
                         std::pow(theta, -2.0F * dimension / head_dim);
      result.values[position * result.dim(1) + dimension] =
          cosine ? std::cos(angle) : std::sin(angle);
    }
  }
  return result;
}

NativeTensor causal_conv1d(const NativeTensor &input,
                           const NativeTensorView &weight,
                           const NativeTensorView &bias) {
  require_matrix(input, "convolution input");
  if (weight.rank() != 3 || weight.dim(1) != input.dim(1) || bias.rank() != 1 ||
      bias.dim(0) != weight.dim(0)) {
    throw std::runtime_error("Invalid causal convolution dimensions");
  }
  NativeTensor result({input.dim(0), weight.dim(0)});
  for (size_t frame = 0; frame < input.dim(0); ++frame) {
    for (size_t output = 0; output < weight.dim(0); ++output) {
      float value = bias.values[output];
      for (size_t channel = 0; channel < input.dim(1); ++channel) {
        for (size_t kernel = 0; kernel < weight.dim(2); ++kernel) {
          if (frame + kernel < weight.dim(2) - 1)
            continue;
          const auto source = frame + kernel - (weight.dim(2) - 1);
          value +=
              input.values[source * input.dim(1) + channel] *
              weight.values[(output * weight.dim(1) + channel) * weight.dim(2) +
                            kernel];
        }
      }
      result.values[frame * result.dim(1) + output] = value;
    }
  }
  return result;
}

float rms(const NativeTensor &value) {
  if (value.numel() == 0)
    throw std::runtime_error("Cannot calculate RMS of an empty tensor");
  double sum{};
  for (float element : value.values)
    sum += static_cast<double>(element) * element;
  return static_cast<float>(std::sqrt(sum / value.numel()));
}

MLABufferSlice full(MLABuffer &value) { return MLABufferSlice(&value); }
MLABufferSlice slice(MLABuffer &value, std::vector<uint32_t> begin,
                     std::vector<uint32_t> shape) {
  return MLABufferSlice(&value, std::move(begin), std::move(shape));
}

} // namespace

struct Qwen3TtsRunner::TensorFile {
  explicit TensorFile(const std::filesystem::path &path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      throw std::runtime_error("Unable to open safetensors file: " +
                               path.string());
    struct stat status {};
    if (fstat(fd_, &status) != 0 ||
        status.st_size < static_cast<off_t>(sizeof(uint64_t))) {
      throw std::runtime_error("Invalid safetensors file: " + path.string());
    }
    size_ = static_cast<size_t>(status.st_size);
    mapped_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      throw std::runtime_error("Unable to map safetensors file: " +
                               path.string());
    }
    uint64_t header_size{};
    std::memcpy(&header_size, mapped_, sizeof(header_size));
    if (header_size > size_ - sizeof(header_size)) {
      throw std::runtime_error("Invalid safetensors header length: " +
                               path.string());
    }
    const auto *header =
        static_cast<const char *>(mapped_) + sizeof(header_size);
    const auto metadata = nlohmann::json::parse(header, header + header_size);
    const auto *data = reinterpret_cast<const uint8_t *>(header) + header_size;
    const auto data_size = size_ - sizeof(header_size) - header_size;
    for (auto it = metadata.begin(); it != metadata.end(); ++it) {
      if (it.key() == "__metadata__")
        continue;
      const auto &spec = it.value();
      // Only the explicitly named CPU-side constants are required by this
      // runner. Leave unrelated checkpoint tensors mapped but unparsed.
      if (spec.at("dtype").get<std::string>() != "F32")
        continue;
      const auto shape = spec.at("shape").get<std::vector<size_t>>();
      const auto offsets = spec.at("data_offsets").get<std::array<size_t, 2>>();
      if (offsets[0] > offsets[1] || offsets[1] > data_size ||
          offsets[1] - offsets[0] !=
              NativeTensor::numel(shape) * sizeof(float)) {
        throw std::runtime_error("Invalid safetensors data offsets for " +
                                 it.key());
      }
      tensors_.emplace(
          it.key(),
          NativeTensorView{reinterpret_cast<const float *>(data + offsets[0]),
                           shape});
    }
  }
  ~TensorFile() {
    if (mapped_)
      munmap(mapped_, size_);
    if (fd_ >= 0)
      close(fd_);
  }
  TensorFile(const TensorFile &) = delete;
  TensorFile &operator=(const TensorFile &) = delete;

  const NativeTensorView &tensor(const std::string &key) const {
    const auto it = tensors_.find(key);
    if (it == tensors_.end())
      throw std::runtime_error("Missing safetensors key " + key + " in " +
                               path_.string());
    return it->second;
  }

private:
  std::filesystem::path path_;
  int fd_{-1};
  size_t size_{};
  void *mapped_{};
  std::map<std::string, NativeTensorView> tensors_;
};

struct Qwen3TtsRunner::TailPart {
  uint16_t index{};
  std::filesystem::path elf;
  std::vector<size_t> input_shape;
  std::vector<size_t> output_shape;
};

Qwen3TtsRunner::Qwen3TtsRunner(std::filesystem::path model_dir,
                               std::filesystem::path components_dir,
                               bool preload_models)
    : model_dir_(std::move(model_dir)),
      components_dir_(std::move(components_dir)), elf_dir_(model_dir_ / "mpk"),
      preload_models_(preload_models)
#if defined(QWEN3_CP_SPLIT_HEADS)
      ,
      cp_head_elf_dir_(elf_dir_)
#endif
{
}

Qwen3TtsRunner::~Qwen3TtsRunner() { finalize(); }

MLABuffer &Qwen3TtsRunner::buffer(const std::string &name) const {
  const auto it = buffers_.find(name);
  if (it == buffers_.end())
    throw std::runtime_error("Unknown MLA buffer: " + name);
  return *it->second;
}

NativeTensor Qwen3TtsRunner::bf16_download(const std::string &name) const {
  const auto &value = buffer(name);
  NativeTensor result(value.get_shape());
  std::vector<Eigen::bfloat16> raw(result.numel());
  value.download(raw.data());
  for (size_t i = 0; i < result.numel(); ++i)
    result.values[i] = static_cast<float>(raw[i]);
  return result;
}

void Qwen3TtsRunner::bf16_upload(const std::string &name,
                                 const NativeTensor &value) const {
  auto &target = buffer(name);
  if (value.numel() != target.get_num_elems())
    throw std::runtime_error("MLA upload shape does not match " + name);
  std::vector<Eigen::bfloat16> rounded(value.numel());
  for (size_t i = 0; i < value.numel(); ++i)
    rounded[i] = static_cast<Eigen::bfloat16>(value.values[i]);
  target.upload(rounded.data());
}

#if defined(QWEN3_CP_SPLIT_HEADS)
void Qwen3TtsRunner::fp32_upload(const std::string &name,
                                 const NativeTensor &value) const {
  auto &target = buffer(name);
  if (target.get_elem_size() != sizeof(float) ||
      value.numel() != target.get_num_elems()) {
    throw std::runtime_error("FP32 raw-head upload shape does not match " +
                             name);
  }
  target.upload(value.values.data());
}
#endif

void Qwen3TtsRunner::define_buffers() {
  auto define = [this](const std::string &name, std::vector<size_t> shape,
                       bool align = false) {
    buffers_.emplace(name, std::make_unique<MLABuffer>(name, std::move(shape),
                                                       "bfloat16", align));
  };

  define("backbone_future_token_mask", {2176});
  define("cp_future_token_mask", {176});
  define("codec_dec_future_token_mask", {2176});

  // Direct raw ELFs use rank-4 BF16 edge buffers. K/V edges use HWC16
  // backing so each layer owns a full 1024-token physical cache.
  define("backbone_freq_real", {1, 1, kBackboneMax, kHeadDim / 2});
  define("backbone_freq_imag", {1, 1, kBackboneMax, kHeadDim / 2});
  for (uint16_t i = 0; i < kBackboneLayers; ++i) {
    define("backbone_cache_key_l" + std::to_string(i),
           {1, 8, kBackboneMax, kHeadDim}, true);
    define("backbone_cache_val_l" + std::to_string(i),
           {1, 8, kBackboneMax, kHeadDim}, true);
  }
  define("backbone_n1_buffer1", {1, 1, 1, kHidden});
  define("backbone_n1_buffer2", {1, 16, 1, kHeadDim});
  define("backbone_n1_buffer3", {1, 1, 1, kQSize});
  define("backbone_n1_input_embed", {1, 1, 1, kHidden});

  define("cp_freq_real", {1, 1, kCpMax, kHeadDim / 2});
  define("cp_freq_imag", {1, 1, kCpMax, kHeadDim / 2});
  for (uint16_t i = 0; i < kCpLayers; ++i) {
    define("cp_cache_key_l" + std::to_string(i), {1, 8, kBackboneMax, kHeadDim},
           true);
    define("cp_cache_val_l" + std::to_string(i), {1, 8, kBackboneMax, kHeadDim},
           true);
  }
  define("cp_n1_initial_input", {1, 1, 1, kHidden});
  define("cp_n1_input", {1, 1, 1, kHidden});
  define("cp_n1_buffer1", {1, 1, 1, kHidden});
  define("cp_n1_buffer2", {1, 16, 1, kHeadDim});
  define("cp_n1_buffer3", {1, 1, 1, kQSize});
  for (uint16_t i = 0; i < 15; ++i)
    define("cp_head_logits_cb" + std::to_string(i), {1, 1, 1, kCpVocab});
#if defined(QWEN3_CP_SPLIT_HEADS)
  // MLABuffer has no float32 label; int32 supplies the exact four-byte edge
  // ABI.
  buffers_.emplace("cp_head_input_fp32",
                   std::make_unique<MLABuffer>(
                       "cp_head_input_fp32",
                       std::vector<size_t>{1, 1, 1, kHidden}, "int32", false));
  buffers_.emplace("cp_head_self_attn_fp32",
                   std::make_unique<MLABuffer>(
                       "cp_head_self_attn_fp32",
                       std::vector<size_t>{1, 1, 1, kQSize}, "int32", false));
#endif

  define("codec_dec_freq_real", {1, 1, kCodecCacheMax, 32});
  define("codec_dec_freq_imag", {1, 1, kCodecCacheMax, 32});
  for (uint16_t i = 0; i < kCodecLayers; ++i) {
    define("codec_dec_cache_key_l" + std::to_string(i),
           {1, kCodecHeads, kCodecCacheMax, 64}, true);
    define("codec_dec_cache_val_l" + std::to_string(i),
           {1, kCodecHeads, kCodecCacheMax, 64}, true);
  }
  define("codec_dec_n1_input", {1, 1, 1, kCodecHidden});
  define("codec_dec_n1_buffer1", {1, 1, 1, kCodecHidden});
  define("codec_dec_n1_buffer2", {1, kCodecHeads, 1, 64});
  define("codec_dec_n1_buffer3", {1, 1, 1, kCodecQSize});
  define("codec_dec_output", {1, 1, 1, kCodecLatent});
  define("codec_dec_n128_input", {1, 1, 128, kCodecHidden});
  define("codec_dec_n128_hidden", {1, 1, 128, kCodecHidden});
  define("codec_dec_n128_query", {1, kCodecHeads, 128, 64});
  define("codec_dec_n128_self_attn", {1, 1, 128, kCodecQSize});
  define("codec_dec_n128_output", {1, 1, 128, kCodecLatent});
}

void Qwen3TtsRunner::validate_tail_contract() {
  const auto contract_path =
      model_dir_ / "devkit" / "codec_tail_raw_mla_contract.json";
  std::ifstream stream(contract_path);
  if (!stream)
    throw std::runtime_error("Missing codec-tail contract: " +
                             contract_path.string());
  const auto contract = nlohmann::json::parse(stream);
  if (contract.value("format", "") != "qwen3_codec_tail_raw_mla_micro_v1" ||
      contract.value("dtype", "") != "bfloat16" ||
      contract.value("layout", "") != "NHWC" ||
      contract.value("align_last_dim", true)) {
    throw std::runtime_error("Unsupported codec-tail contract");
  }
  const auto &parts = contract.at("parts");
  if (!parts.is_array() || parts.size() != 27)
    throw std::runtime_error("Codec-tail contract requires 27 stages");
  std::vector<size_t> previous;
  for (size_t i = 0; i < parts.size(); ++i) {
    const auto &item = parts.at(i);
    if (item.value("index", -1) != static_cast<int>(i))
      throw std::runtime_error("Non-sequential tail stage");
    TailPart part;
    part.index = static_cast<uint16_t>(i);
    part.input_shape = item.at("input_shape").get<std::vector<size_t>>();
    part.output_shape = item.at("output_shape").get<std::vector<size_t>>();
    if (part.input_shape.size() != 4 || part.output_shape.size() != 4 ||
        (!previous.empty() && previous != part.input_shape)) {
      throw std::runtime_error("Invalid tail shape chain at stage " +
                               std::to_string(i));
    }
    auto input_numel =
        std::accumulate(part.input_shape.begin(), part.input_shape.end(),
                        size_t{1}, std::multiplies<>());
    auto output_numel =
        std::accumulate(part.output_shape.begin(), part.output_shape.end(),
                        size_t{1}, std::multiplies<>());
    if (item.at("input_bytes").get<size_t>() != input_numel * 2 ||
        item.at("output_bytes").get<size_t>() != output_numel * 2) {
      throw std::runtime_error("Invalid tail byte count at stage " +
                               std::to_string(i));
    }
    const auto name = item.at("elf").get<std::string>();
    if (std::filesystem::path(name).filename() != name)
      throw std::runtime_error("Invalid tail ELF name");
    part.elf = elf_dir_ / name;
    if (!std::filesystem::is_regular_file(part.elf))
      throw std::runtime_error("Missing tail ELF: " + part.elf.string());
    previous = part.output_shape;
    tail_parts_.push_back(std::move(part));
  }
  if (tail_parts_.front().input_shape != std::vector<size_t>{1, 1, 50, 1024} ||
      tail_parts_.back().output_shape != std::vector<size_t>{1, 1, 96000, 16}) {
    throw std::runtime_error("Unexpected C16 tail endpoints");
  }
  for (size_t edge = 0; edge <= tail_parts_.size(); ++edge) {
    const auto &shape = edge == 0 ? tail_parts_.front().input_shape
                                  : tail_parts_[edge - 1].output_shape;
    buffers_.emplace(
        "tail_edge" + std::to_string(edge),
        std::make_unique<MLABuffer>("tail_edge" + std::to_string(edge), shape,
                                    "bfloat16", false));
  }
}

void Qwen3TtsRunner::validate_elfs() const {
  const auto require = [](const std::filesystem::path &path) {
    if (!std::filesystem::is_regular_file(path))
      throw std::runtime_error("Missing required ELF: " + path.string());
  };
  for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
    require(elf_path(elf_dir_, "backbone_language", "pre_layer", layer));
    require(elf_path(elf_dir_, "backbone_language", "post_layer", layer));
  }
  for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
    require(elf_path(elf_dir_, "code_predictor_language", "pre_layer", layer));
#if defined(QWEN3_CP_SPLIT_HEADS)
    if (layer + 1 < kCpLayers)
      require(
          elf_path(elf_dir_, "code_predictor_language", "post_layer", layer));
#else
    require(elf_path(elf_dir_, "code_predictor_language", "post_layer", layer));
#endif
  }
  for (uint16_t layer = 0; layer < kCodecLayers; ++layer) {
    require(elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer));
    require(elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer));
  }
  // The package contains N128 pre/cache/post through layer 6. Layer 7 has
  // no N128 post ELF and therefore remains on the known-good N1 path.
  for (uint16_t layer = 0; layer < kCodecLayers - 1; ++layer) {
    require(
        n128_elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer));
    require(
        n128_elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer));
  }
  require(elf_dir_ / "codec_decoder_language_n128_cache_token0_stage1_mla.elf");
  for (uint16_t cache : {127, 255, 383, 511, 639, 767, 895, 1023})
    require(elf_dir_ / ("backbone_language_n1_cache_token" +
                        std::to_string(cache) + "_stage1_mla.elf"));
  require(elf_dir_ /
          "code_predictor_language_n1_cache_token127_stage1_mla.elf");
#if defined(QWEN3_CP_SPLIT_HEADS)
  for (uint16_t codebook = 0; codebook < 15; ++codebook) {
    require(cp_head_elf_dir_ / ("code_predictor_language_n1_post_layer4_head" +
                                std::to_string(codebook) + "_stage1_mla.elf"));
  }
#endif
  require(elf_dir_ / "codec_decoder_language_n1_cache_token127_stage1_mla.elf");
}

void Qwen3TtsRunner::initialize_static_buffers() {
  auto mask = [](uint16_t active, uint16_t total) {
    NativeTensor result({total}, kBf16Lowest);
    std::fill_n(result.values.begin(), active, 0.0F);
    return result;
  };
  bf16_upload("backbone_future_token_mask", mask(kBackboneMax, 2176));
  bf16_upload("cp_future_token_mask", mask(kCpMax, 176));
  bf16_upload("codec_dec_future_token_mask", mask(kCodecCacheMax, 2176));
  bf16_upload("backbone_freq_real",
              rope(kBackboneMax, kHeadDim, kBackboneRopeTheta, true));
  bf16_upload("backbone_freq_imag",
              rope(kBackboneMax, kHeadDim, kBackboneRopeTheta, false));
  bf16_upload("cp_freq_real",
              rope(kCpMax, kHeadDim, kCodePredictorRopeTheta, true));
  bf16_upload("cp_freq_imag",
              rope(kCpMax, kHeadDim, kCodePredictorRopeTheta, false));
  bf16_upload("codec_dec_freq_real",
              rope(kCodecCacheMax, 64, kCodecDecoderRopeTheta, true));
  bf16_upload("codec_dec_freq_imag",
              rope(kCodecCacheMax, 64, kCodecDecoderRopeTheta, false));
}

void Qwen3TtsRunner::load_host_weights() {
  backbone_weights_ = std::make_unique<TensorFile>(
      components_dir_ / "backbone" / "model.safetensors");
  cp_weights_ = std::make_unique<TensorFile>(
      components_dir_ / "code_predictor" / "model.safetensors");
  codec_weights_ = std::make_unique<TensorFile>(
      components_dir_ / "codec_decoder" / "model.safetensors");
  text_projection_weights_ = std::make_unique<TensorFile>(
      components_dir_ / "text_projection" / "text_projection.safetensors");
  codec_head_weights_ = std::make_unique<TensorFile>(
      components_dir_ / "codec_head" / "codec_head.safetensors");
  text_embeddings_ = &backbone_weights_->tensor("text_embedding.weight");
  codec_embeddings_[0] = &backbone_weights_->tensor("codec_embedding.weight");
  for (uint16_t i = 0; i < 15; ++i)
    codec_embeddings_[i + 1] = &cp_weights_->tensor(
        "model.codec_embedding." + std::to_string(i) + ".weight");
  codec_head_weight_ = &codec_head_weights_->tensor("weight");
  text_fc1_w_ = &text_projection_weights_->tensor("linear_fc1.weight");
  text_fc1_b_ = &text_projection_weights_->tensor("linear_fc1.bias");
  text_fc2_w_ = &text_projection_weights_->tensor("linear_fc2.weight");
  text_fc2_b_ = &text_projection_weights_->tensor("linear_fc2.bias");
  std::ifstream config_stream(components_dir_ / "backbone" / "config.json");
  if (!config_stream)
    throw std::runtime_error("Missing backbone config.json");
  talker_config_ = nlohmann::json::parse(config_stream);
}

void Qwen3TtsRunner::build_and_preload_models() {
  for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
    backbone_pre(layer, 0);
    backbone_post(layer, 0);
    for (uint16_t token = 0; token < kBackboneMax; token += 128)
      backbone_cache(layer, token);
  }
  for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
    cp_pre(layer, 0);
    cp_cache(layer, 0);
    cp_post(layer, 0);
  }
  for (uint16_t layer = 0; layer < kCodecLayers; ++layer) {
    codec_pre(layer, 0);
    codec_cache(layer, 0);
    codec_post(layer, 0);
  }
  if (tail_models_.empty()) {
    for (size_t i = 0; i < tail_parts_.size(); ++i) {
      tail_models_.push_back(std::make_unique<MLAModelWithBuffer>(
          tail_parts_[i].elf,
          std::vector<MLABufferSlice>{
              full(buffer("tail_edge" + std::to_string(i)))},
          std::vector<MLABufferSlice>{
              full(buffer("tail_edge" + std::to_string(i + 1)))}));
    }
  }
  MLAModelWithBuffer::load_all_models();
}

void Qwen3TtsRunner::initialize() {
  if (initialized_)
    return;
  validate_tail_contract();
  validate_elfs();
  define_buffers();
  for (auto &[_, value] : buffers_)
    value->allocate();
  initialize_static_buffers();
  load_host_weights();
  const auto tokenizer_path = components_dir_ / "processor" / "tokenizer.json";
  tokenizer_ = Tokenizer::from_hf_json(tokenizer_path);
  reset_caches();
  if (preload_models_)
    build_and_preload_models();
  initialized_ = true;
}

void Qwen3TtsRunner::finalize() {
  if (!initialized_)
    return;
  tail_models_.clear();
  backbone_pre_.clear();
  backbone_cache_.clear();
  backbone_post_.clear();
  cp_pre_.clear();
  cp_cache_.clear();
  cp_post_.clear();
#if defined(QWEN3_CP_SPLIT_HEADS)
  cp_head_.clear();
#endif
  codec_pre_.clear();
  codec_cache_.clear();
  codec_post_.clear();
  codec_n128_pre_.clear();
  codec_n128_cache_.clear();
  codec_n128_post_.clear();
  codec_final_pre_from_n128_.clear();
  codec_final_post_from_n128_.clear();
  for (auto &[_, value] : buffers_)
    value->free();
  buffers_.clear();
  prefix_snapshots_.clear();
  resident_prefix_key_.reset();
  tokenizer_.reset();
  initialized_ = false;
}

void Qwen3TtsRunner::reset_caches() {
  for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
    buffer("backbone_cache_key_l" + std::to_string(layer)).clear();
    buffer("backbone_cache_val_l" + std::to_string(layer)).clear();
  }
  reset_cp_caches();
  reset_codec_caches();
  backbone_position_ = 0;
}
void Qwen3TtsRunner::reset_cp_caches() {
  for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
    buffer("cp_cache_key_l" + std::to_string(layer)).clear();
    buffer("cp_cache_val_l" + std::to_string(layer)).clear();
  }
}
void Qwen3TtsRunner::reset_codec_caches() {
  for (uint16_t layer = 0; layer < kCodecLayers; ++layer) {
    buffer("codec_dec_cache_key_l" + std::to_string(layer)).clear();
    buffer("codec_dec_cache_val_l" + std::to_string(layer)).clear();
  }
}

NativeTensor Qwen3TtsRunner::text_project(const NativeTensor &input) const {
  return linear(silu(linear(input, *text_fc1_w_, text_fc1_b_)), *text_fc2_w_,
                text_fc2_b_);
}

NativeTensor Qwen3TtsRunner::build_prefill(const std::vector<uint32_t> &ids,
                                           const std::string &speaker,
                                           const std::string &language,
                                           NativeTensor &trailing,
                                           NativeTensor &pad) const {
  if (ids.size() < 8)
    throw std::runtime_error("Unexpectedly short tokenized prompt");
  const auto spk_key = lower(speaker);
  if (!talker_config_.at("spk_id").contains(spk_key))
    throw std::runtime_error("Unsupported speaker: " + speaker);
  const auto spk = talker_config_.at("spk_id").at(spk_key).get<uint32_t>();
  std::optional<uint32_t> language_id;
  if (lower(language) != "auto") {
    const auto lang_key = lower(language);
    if (!talker_config_.at("codec_language_id").contains(lang_key))
      throw std::runtime_error("Unsupported language: " + language);
    language_id =
        talker_config_.at("codec_language_id").at(lang_key).get<uint32_t>();
  }
  const auto special_projected =
      text_project(gather(*text_embeddings_, {151672, 151673, 151671}));
  const auto bos = row(special_projected, 0);
  const auto eos = row(special_projected, 1);
  pad = row(special_projected, 2);
  const auto control = [&](const char *name) {
    return talker_config_.at(name).get<uint32_t>();
  };
  std::vector<uint32_t> prefix_ids;
  if (language_id)
    prefix_ids = {control("codec_think_id"), control("codec_think_bos_id"),
                  *language_id, control("codec_think_eos_id")};
  else
    prefix_ids = {control("codec_nothink_id"), control("codec_think_bos_id"),
                  control("codec_think_eos_id")};
  const auto codec_input = concatenate_rows(
      {gather(*codec_embeddings_[0], prefix_ids),
       row(*codec_embeddings_[0], spk),
       gather(*codec_embeddings_[0],
              {control("codec_pad_id"), control("codec_bos_id")})});
  const auto role =
      text_project(gather(*text_embeddings_, {ids[0], ids[1], ids[2]}));
  const auto body_rows = codec_input.dim(0) - 1;
  const auto body_base =
      concatenate_rows({repeat_row(pad, body_rows - 1), bos});
  NativeTensor body({body_rows, kHidden});
  for (size_t i = 0; i < body.numel(); ++i)
    body.values[i] = body_base.values[i] + codec_input.values[i];
  const auto prefill_base = concatenate_rows({role, body});
  NativeTensor prefill({prefill_base.dim(0) - 1, kHidden});
  std::copy_n(prefill_base.values.begin(), prefill.numel(),
              prefill.values.begin());
  std::vector<uint32_t> main(ids.begin() + 3, ids.end() - 5);
  const auto text_eos =
      concatenate_rows({text_project(gather(*text_embeddings_, main)), eos});
  const auto code_pad = repeat_row(
      row(*codec_embeddings_[0], control("codec_pad_id")), text_eos.dim(0));
  NativeTensor final_text({text_eos.dim(0), kHidden});
  for (size_t i = 0; i < final_text.numel(); ++i)
    final_text.values[i] = text_eos.values[i] + code_pad.values[i];
  const auto codec_bos = row(*codec_embeddings_[0], control("codec_bos_id"));
  NativeTensor final_row({1, kHidden});
  for (size_t i = 0; i < kHidden; ++i)
    final_row.values[i] = pad.values[i] + codec_bos.values[i];
  trailing = pad;
  return bf16_rounded(concatenate_rows({prefill, final_text, final_row}));
}

NativeTensor Qwen3TtsRunner::codec_embedding(int32_t token,
                                             uint32_t codebook) const {
  if (codebook >= codec_embeddings_.size() || !codec_embeddings_[codebook])
    throw std::runtime_error("Codec codebook out of range");
  const auto &embeddings = *codec_embeddings_[codebook];
  if (token < 0 || static_cast<size_t>(token) >= embeddings.dim(0))
    throw std::runtime_error("Codec token out of range");
  return row(embeddings, static_cast<size_t>(token));
}

NativeTensor
Qwen3TtsRunner::backbone_feedback(const std::array<int32_t, 16> &frame) const {
  NativeTensor result({1, kHidden});
  for (size_t codebook = 0; codebook < frame.size(); ++codebook)
    add_inplace(result, codec_embedding(frame[codebook], codebook));
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::backbone_pre(uint16_t layer,
                                                       uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = backbone_pre_[key];
  if (result)
    return result;
  auto &input = layer == 0 ? buffer("backbone_n1_input_embed")
                           : buffer("backbone_n1_buffer1");
  const auto kv_begin = u32({0, 0, token, 0});
  const auto kv_shape = u32({1, 8, 1, kHeadDim});
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "backbone_language", "pre_layer", layer),
      std::vector<MLABufferSlice>{
          full(input),
          slice(buffer("backbone_freq_real"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 64})),
          slice(buffer("backbone_freq_imag"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 64}))},
      std::vector<MLABufferSlice>{
          full(buffer("backbone_n1_buffer2")),
          slice(buffer("backbone_cache_key_l" + std::to_string(layer)),
                kv_begin, kv_shape),
          slice(buffer("backbone_cache_val_l" + std::to_string(layer)),
                kv_begin, kv_shape)});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::backbone_cache(uint16_t layer,
                                                         uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = backbone_cache_[key];
  if (result)
    return result;
  const auto n = static_cast<uint16_t>(aligned_cache(token) + 1);
  result = std::make_unique<MLAModelWithBuffer>(
      cache_path(elf_dir_, "backbone_language", token),
      std::vector<MLABufferSlice>{
          full(buffer("backbone_n1_buffer2")),
          slice(buffer("backbone_cache_key_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, 8, n, kHeadDim})),
          slice(buffer("backbone_future_token_mask"),
                u32({kBackboneMax - (token + 1)}), u32({n})),
          slice(buffer("backbone_cache_val_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, 8, n, kHeadDim}))},
      std::vector<MLABufferSlice>{full(buffer("backbone_n1_buffer3"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::backbone_post(uint16_t layer,
                                                        uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = backbone_post_[key];
  if (result)
    return result;
  auto &input = layer == 0 ? buffer("backbone_n1_input_embed")
                           : buffer("backbone_n1_buffer1");
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "backbone_language", "post_layer", layer),
      std::vector<MLABufferSlice>{full(input),
                                  full(buffer("backbone_n1_buffer3"))},
      std::vector<MLABufferSlice>{full(buffer("backbone_n1_buffer1"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::cp_pre(uint16_t layer,
                                                 uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = cp_pre_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "code_predictor_language", "pre_layer", layer),
      std::vector<MLABufferSlice>{
          full(layer == 0
                   ? buffer(token == 0 ? "cp_n1_initial_input" : "cp_n1_input")
                   : buffer("cp_n1_buffer1")),
          slice(buffer("cp_freq_real"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 64})),
          slice(buffer("cp_freq_imag"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 64}))},
      std::vector<MLABufferSlice>{
          full(buffer("cp_n1_buffer2")),
          slice(buffer("cp_cache_key_l" + std::to_string(layer)),
                u32({0, 0, token, 0}), u32({1, 8, 1, kHeadDim})),
          slice(buffer("cp_cache_val_l" + std::to_string(layer)),
                u32({0, 0, token, 0}), u32({1, 8, 1, kHeadDim}))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::cp_cache(uint16_t layer,
                                                   uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = cp_cache_[key];
  if (result)
    return result;
  constexpr uint16_t kCpPhysicalCache = 128;
  result = std::make_unique<MLAModelWithBuffer>(
      cache_path(elf_dir_, "code_predictor_language", token),
      std::vector<MLABufferSlice>{
          full(buffer("cp_n1_buffer2")),
          slice(buffer("cp_cache_key_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, 8, kCpPhysicalCache, kHeadDim})),
          slice(buffer("cp_future_token_mask"), u32({kCpMax - (token + 1)}),
                u32({kCpPhysicalCache})),
          slice(buffer("cp_cache_val_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, 8, kCpPhysicalCache, kHeadDim}))},
      std::vector<MLABufferSlice>{full(buffer("cp_n1_buffer3"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::cp_post(uint16_t layer,
                                                  uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = cp_post_[key];
  if (result)
    return result;
  std::vector<MLABufferSlice> output;
  auto path =
      elf_path(elf_dir_, "code_predictor_language", "post_layer", layer);
  if (layer + 1 < kCpLayers) {
    output.push_back(full(buffer("cp_n1_buffer1")));
#if defined(QWEN3_CP_SPLIT_HEADS)
  } else if (token > 0) {
    // This is the exact archived direct-raw binding: head N replaces the
    // bundled post-layer4 only at code-predictor token N+1.  Its raw ELF
    // consumes the existing BF16 edges, despite outer MPK cast metadata.
    const auto head = static_cast<uint16_t>(token - 1);
    path = cp_head_elf_dir_ / ("code_predictor_language_n1_post_layer4_head" +
                               std::to_string(head) + "_stage1_mla.elf");
    output.push_back(full(buffer("cp_head_logits_cb" + std::to_string(head))));
#endif
  } else {
    std::vector<uint16_t> order(15);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [](uint16_t a, uint16_t b) {
      return std::to_string(a) < std::to_string(b);
    });
    for (const auto codebook : order)
      output.push_back(
          full(buffer("cp_head_logits_cb" + std::to_string(codebook))));
  }
  result = std::make_unique<MLAModelWithBuffer>(
      path,
      std::vector<MLABufferSlice>{
          full(layer == 0
                   ? buffer(token == 0 ? "cp_n1_initial_input" : "cp_n1_input")
                   : buffer("cp_n1_buffer1")),
          full(buffer("cp_n1_buffer3"))},
      std::move(output));
  return result;
}

#if defined(QWEN3_CP_SPLIT_HEADS)
Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::cp_head(uint16_t codebook) {
  if (codebook >= 15)
    throw std::runtime_error("Invalid code-predictor head index");
  auto &result = cp_head_[codebook];
  if (result)
    return result;
  const auto path =
      cp_head_elf_dir_ / ("code_predictor_language_n1_post_layer4_head" +
                          std::to_string(codebook) + "_stage1_mla.elf");
  result = std::make_unique<MLAModelWithBuffer>(
      path,
      std::vector<MLABufferSlice>{full(buffer("cp_head_input_fp32")),
                                  full(buffer("cp_head_self_attn_fp32"))},
      std::vector<MLABufferSlice>{
          full(buffer("cp_head_logits_cb" + std::to_string(codebook)))});
  return result;
}
#endif

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::codec_pre(uint16_t layer,
                                                    uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = codec_pre_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer),
      std::vector<MLABufferSlice>{
          full(layer == 0 ? buffer("codec_dec_n1_input")
                          : buffer("codec_dec_n1_buffer1")),
          slice(buffer("codec_dec_freq_real"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 32})),
          slice(buffer("codec_dec_freq_imag"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 32}))},
      std::vector<MLABufferSlice>{
          full(buffer("codec_dec_n1_buffer2")),
          slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)),
                u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64})),
          slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)),
                u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64}))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::codec_cache(uint16_t layer,
                                                      uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = codec_cache_[key];
  if (result)
    return result;
  const auto n = static_cast<uint16_t>(aligned_cache(token) + 1);
  result = std::make_unique<MLAModelWithBuffer>(
      cache_path(elf_dir_, "codec_decoder_language", token),
      std::vector<MLABufferSlice>{
          full(buffer("codec_dec_n1_buffer2")),
          slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, kCodecHeads, n, 64})),
          slice(buffer("codec_dec_future_token_mask"),
                u32({kCodecCacheMax - (token + 1)}), u32({n})),
          slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, kCodecHeads, n, 64}))},
      std::vector<MLABufferSlice>{full(buffer("codec_dec_n1_buffer3"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::codec_post(uint16_t layer,
                                                     uint16_t token) {
  const ModelKey key{layer, token};
  auto &result = codec_post_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer),
      std::vector<MLABufferSlice>{full(layer == 0
                                           ? buffer("codec_dec_n1_input")
                                           : buffer("codec_dec_n1_buffer1")),
                                  full(buffer("codec_dec_n1_buffer3"))},
      std::vector<MLABufferSlice>{
          full(buffer(layer + 1 < kCodecLayers ? "codec_dec_n1_buffer1"
                                               : "codec_dec_output"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::codec_n128_pre(uint16_t layer) {
  const ModelKey key{layer, 0};
  auto &result = codec_n128_pre_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      n128_elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer),
      std::vector<MLABufferSlice>{
          full(layer == 0 ? buffer("codec_dec_n128_input")
                          : buffer("codec_dec_n128_hidden")),
          slice(buffer("codec_dec_freq_real"), u32({0, 0, 0, 0}),
                u32({1, 1, 128, 32})),
          slice(buffer("codec_dec_freq_imag"), u32({0, 0, 0, 0}),
                u32({1, 1, 128, 32}))},
      std::vector<MLABufferSlice>{
          full(buffer("codec_dec_n128_query")),
          slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64})),
          slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64}))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::codec_n128_cache(uint16_t layer) {
  const ModelKey key{layer, 0};
  auto &result = codec_n128_cache_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      elf_dir_ / "codec_decoder_language_n128_cache_token0_stage1_mla.elf",
      std::vector<MLABufferSlice>{
          full(buffer("codec_dec_n128_query")),
          slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64})),
          slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)),
                u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64}))},
      std::vector<MLABufferSlice>{full(buffer("codec_dec_n128_self_attn"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &Qwen3TtsRunner::codec_n128_post(uint16_t layer) {
  const ModelKey key{layer, 0};
  auto &result = codec_n128_post_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      n128_elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer),
      std::vector<MLABufferSlice>{full(layer == 0
                                           ? buffer("codec_dec_n128_input")
                                           : buffer("codec_dec_n128_hidden")),
                                  full(buffer("codec_dec_n128_self_attn"))},
      std::vector<MLABufferSlice>{full(buffer("codec_dec_n128_hidden"))});
  return result;
}

Qwen3TtsRunner::ModelPtr &
Qwen3TtsRunner::codec_final_pre_from_n128(uint16_t token) {
  const ModelKey key{kCodecLayers - 1, token};
  auto &result = codec_final_pre_from_n128_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "codec_decoder_language", "pre_layer",
               kCodecLayers - 1),
      std::vector<MLABufferSlice>{
          slice(buffer("codec_dec_n128_hidden"), u32({0, 0, token, 0}),
                u32({1, 1, 1, kCodecHidden})),
          slice(buffer("codec_dec_freq_real"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 32})),
          slice(buffer("codec_dec_freq_imag"), u32({0, 0, token, 0}),
                u32({1, 1, 1, 32}))},
      std::vector<MLABufferSlice>{
          full(buffer("codec_dec_n1_buffer2")),
          slice(buffer("codec_dec_cache_key_l" +
                       std::to_string(kCodecLayers - 1)),
                u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64})),
          slice(buffer("codec_dec_cache_val_l" +
                       std::to_string(kCodecLayers - 1)),
                u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64}))});
  return result;
}

Qwen3TtsRunner::ModelPtr &
Qwen3TtsRunner::codec_final_post_from_n128(uint16_t token) {
  const ModelKey key{kCodecLayers - 1, token};
  auto &result = codec_final_post_from_n128_[key];
  if (result)
    return result;
  result = std::make_unique<MLAModelWithBuffer>(
      elf_path(elf_dir_, "codec_decoder_language", "post_layer",
               kCodecLayers - 1),
      std::vector<MLABufferSlice>{slice(buffer("codec_dec_n128_hidden"),
                                        u32({0, 0, token, 0}),
                                        u32({1, 1, 1, kCodecHidden})),
                                  full(buffer("codec_dec_n1_buffer3"))},
      std::vector<MLABufferSlice>{slice(buffer("codec_dec_n128_output"),
                                        u32({0, 0, token, 0}),
                                        u32({1, 1, 1, kCodecLatent}))});
  return result;
}

void Qwen3TtsRunner::run_backbone_token(uint16_t token) {
  std::vector<MLAModelWithBuffer *> models;
  models.reserve(kBackboneLayers * 3);
  for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
    models.push_back(backbone_pre(layer, token).get());
    models.push_back(backbone_cache(layer, token).get());
    models.push_back(backbone_post(layer, token).get());
  }
  Qwen3MlaStaticBatch(std::move(models)).run();
}

NativeTensor Qwen3TtsRunner::run_backbone_prefill(const NativeTensor &prefill,
                                                  const std::string &speaker,
                                                  const std::string &language,
                                                  RunMetrics &metrics) {
  if (prefill.rank() != 2 || prefill.dim(1) != kHidden || prefill.dim(0) == 0 ||
      prefill.dim(0) > kBackboneMax) {
    throw std::runtime_error("Invalid backbone prefill shape");
  }
  const auto prefix_len =
      static_cast<uint16_t>(lower(language) == "auto" ? 7 : 8);
  metrics.prefix_kv_static_tokens = prefix_len;
  const auto key = std::make_pair(lower(speaker), lower(language));
  const bool complete_static_prefix = prefill.dim(0) > prefix_len;
  if (complete_static_prefix && resident_prefix_key_ &&
      *resident_prefix_key_ == key) {
    metrics.prefix_kv_reused = true;
    metrics.prefix_kv_device_resident = true;
    backbone_position_ = prefix_len;
  } else if (prefix_len >= prefill.dim(0) || !prefix_snapshots_.contains(key)) {
    reset_caches();
    const auto limit = std::min<size_t>(prefix_len, prefill.dim(0));
    for (size_t i = 0; i < limit; ++i) {
      bf16_upload("backbone_n1_input_embed", row(prefill, i));
      run_backbone_token(i);
    }
    if (complete_static_prefix) {
      std::vector<std::vector<uint8_t>> snap;
      const size_t full_head_bytes =
          static_cast<size_t>(kBackboneMax) * kHeadDim * sizeof(uint16_t);
      const size_t prefix_head_bytes =
          static_cast<size_t>(prefix_len) * kHeadDim * sizeof(uint16_t);
      snap.reserve(kBackboneLayers * 2);
      for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
        for (const char *which : {"key", "val"}) {
          auto &cache = buffer("backbone_cache_" + std::string(which) + "_l" +
                               std::to_string(layer));
          std::vector<uint8_t> raw(cache.get_num_elems() * sizeof(uint16_t));
          cache.download(raw.data());
          std::vector<uint8_t> compact(8 * prefix_head_bytes);
          for (size_t head = 0; head < 8; ++head)
            std::memcpy(compact.data() + head * prefix_head_bytes,
                        raw.data() + head * full_head_bytes, prefix_head_bytes);
          snap.push_back(std::move(compact));
        }
      }
      prefix_snapshots_[key] = std::move(snap);
      resident_prefix_key_ = key;
    } else
      resident_prefix_key_.reset();
    backbone_position_ = static_cast<uint16_t>(limit);
  } else {
    reset_caches();
    const auto &snap = prefix_snapshots_.at(key);
    size_t index{};
    for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
      for (const char *which : {"key", "val"}) {
        auto &target = buffer("backbone_cache_" + std::string(which) + "_l" +
                              std::to_string(layer));
        const auto &compact = snap.at(index);
        const size_t full_head_bytes =
            static_cast<size_t>(kBackboneMax) * kHeadDim * sizeof(uint16_t);
        const size_t prefix_head_bytes =
            static_cast<size_t>(prefix_len) * kHeadDim * sizeof(uint16_t);
        for (size_t head = 0; head < 8; ++head)
          target.upload(compact.data() + head * prefix_head_bytes,
                        head * full_head_bytes, prefix_head_bytes, false);
        target.flush_cache();
        ++index;
      }
    }
    metrics.prefix_kv_reused = true;
    resident_prefix_key_ = key;
    backbone_position_ = prefix_len;
  }
  for (size_t token = backbone_position_; token < prefill.dim(0); ++token) {
    bf16_upload("backbone_n1_input_embed", row(prefill, token));
    run_backbone_token(token);
  }
  backbone_position_ = static_cast<uint16_t>(prefill.dim(0));
  const auto downloaded = bf16_download("backbone_n1_buffer1");
  NativeTensor hidden({1, kHidden});
  std::copy_n(downloaded.values.begin(), kHidden, hidden.values.begin());
  return hidden;
}

int32_t Qwen3TtsRunner::select_token(const NativeTensor &logits,
                                     const std::vector<int32_t> &previous,
                                     bool suppress, bool sample, uint32_t top_k,
                                     float top_p, float temperature,
                                     float repetition_penalty) {
  std::vector<float> values = logits.values;
  if (suppress) {
    for (int32_t index = 2048; index < 3072; ++index)
      if (index != kCodecEos)
        values.at(index) = -std::numeric_limits<float>::infinity();
  }
  if (repetition_penalty != 1.0F) {
    std::vector<int32_t> unique = previous;
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    for (const auto token : unique)
      if (token >= 0 && token < static_cast<int32_t>(values.size())) {
        values[token] = values[token] > 0.0F
                            ? values[token] / repetition_penalty
                            : values[token] * repetition_penalty;
      }
  }
  if (!sample)
    return static_cast<int32_t>(std::distance(
        values.begin(), std::max_element(values.begin(), values.end())));
  const float scale = std::max(temperature, 1e-5F);
  for (auto &value : values)
    value /= scale;
  if (top_k > 0 && top_k < values.size()) {
    auto partition = values;
    std::nth_element(partition.begin(), partition.end() - top_k,
                     partition.end());
    const auto threshold = *(partition.end() - top_k);
    for (auto &value : values)
      if (value < threshold)
        value = -std::numeric_limits<float>::infinity();
  }
  if (top_p < 1.0F) {
    std::vector<size_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&values](size_t a, size_t b) { return values[a] > values[b]; });
    float maximum = values.at(order.front());
    double denominator{};
    for (const auto index : order)
      if (std::isfinite(values[index]))
        denominator += std::exp(static_cast<double>(values[index] - maximum));
    double cumulative{};
    for (size_t rank = 0; rank < order.size(); ++rank) {
      const auto index = order[rank];
      const auto probability =
          std::isfinite(values[index])
              ? std::exp(static_cast<double>(values[index] - maximum)) /
                    denominator
              : 0.0;
      cumulative += probability;
      if (rank > 0 && cumulative - probability > top_p)
        values[index] = -std::numeric_limits<float>::infinity();
    }
  }
  const auto maximum = *std::max_element(values.begin(), values.end());
  double total{};
  std::vector<double> probability(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    probability[i] = std::isfinite(values[i])
                         ? std::exp(static_cast<double>(values[i] - maximum))
                         : 0.0;
    total += probability[i];
  }
  const double value = rng_.next_double();
  double cumulative{};
  for (size_t i = 0; i < probability.size(); ++i) {
    cumulative += probability[i] / total;
    if (value < cumulative)
      return static_cast<int32_t>(i);
  }
  return static_cast<int32_t>(probability.size() - 1);
}

std::array<int32_t, 16>
Qwen3TtsRunner::run_code_predictor(const NativeTensor &hidden, int32_t c0,
                                   const RequestOptions &request,
                                   RunMetrics &metrics) {
  if (hidden.shape != std::vector<size_t>{1, kHidden})
    throw std::runtime_error("Invalid code-predictor hidden state");
  const auto code_predictor_start = std::chrono::steady_clock::now();
  auto run_position = [this, &metrics](uint16_t token) {
    std::vector<MLAModelWithBuffer *> models;
    models.reserve(kCpLayers * 3);
    for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
      models.push_back(cp_pre(layer, token).get());
      models.push_back(cp_cache(layer, token).get());
      models.push_back(cp_post(layer, token).get());
    }
    const auto mla_start = std::chrono::steady_clock::now();
    Qwen3MlaStaticBatch(std::move(models)).run();
    metrics.code_predictor_mla_time +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      mla_start)
            .count();
  };
  auto run_initial_and_c0 = [this, &metrics] {
    std::vector<MLAModelWithBuffer *> models;
    models.reserve(kCpLayers * 3 * 2);
    for (uint16_t token = 0; token < 2; ++token) {
      for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
        models.push_back(cp_pre(layer, token).get());
        models.push_back(cp_cache(layer, token).get());
        if (token == 0 && layer + 1 == kCpLayers)
          continue;
        models.push_back(cp_post(layer, token).get());
      }
    }
    const auto mla_start = std::chrono::steady_clock::now();
    Qwen3MlaStaticBatch(std::move(models)).run();
    metrics.code_predictor_mla_time +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      mla_start)
            .count();
  };
  const auto initial_input = as_rank4_row(hidden, kHidden);
  if (metrics.cp_initial_input_sha256.empty())
    metrics.cp_initial_input_sha256 = bf16_sha256(initial_input);
  bf16_upload("cp_n1_initial_input", initial_input);
  std::array<int32_t, 16> frame{};
  frame[0] = c0;
  int32_t current = c0;
  std::vector<int32_t> previous;
  auto token_input = as_rank4_row(codec_embedding(current, 0), kHidden);
  if (metrics.cp_codebook0_input_sha256.empty())
    metrics.cp_codebook0_input_sha256 = bf16_sha256(token_input);
  bf16_upload("cp_n1_input", token_input);
  run_initial_and_c0();
  for (uint16_t codebook = 0; codebook < 15; ++codebook) {
    const auto logits =
        bf16_download("cp_head_logits_cb" + std::to_string(codebook));
    if (codebook == 0 && metrics.cp_codebook0_logits_sha256.empty())
      metrics.cp_codebook0_logits_sha256 = bf16_sha256(logits);
    const auto token =
        select_token(logits, previous, false, request.subtalker_do_sample,
                     request.subtalker_top_k, request.subtalker_top_p,
                     request.subtalker_temperature, 1.0F);
    frame[codebook + 1] = token;
    previous.push_back(token);
    current = token;
    if (codebook + 1 < 15) {
      token_input =
          as_rank4_row(codec_embedding(current, codebook + 1), kHidden);
      bf16_upload("cp_n1_input", token_input);
      run_position(codebook + 2);
    }
  }
  metrics.code_predictor_time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    code_predictor_start)
          .count();
  return frame;
}

NativeTensor Qwen3TtsRunner::run_codec_prefix(
    const std::vector<std::array<int32_t, 16>> &codes) {
  if (codes.empty())
    throw std::runtime_error("Codec codes must not be empty");
  const size_t frames = codes.size();
  auto decode = [this, &codes, frames](const std::string &prefix, size_t start,
                                       size_t count) {
    NativeTensor quantized;
    for (size_t index = 0; index < count; ++index) {
      const auto base =
          prefix + ".vq.layers." + std::to_string(index) + "._codebook";
      const auto &embeddings = codec_weights_->tensor(base + ".embedding_sum");
      const auto &usage = codec_weights_->tensor(base + ".cluster_usage");
      if (embeddings.rank() != 2 || usage.rank() != 1 ||
          usage.dim(0) != embeddings.dim(0)) {
        throw std::runtime_error("Invalid codec quantizer shape");
      }
      if (quantized.numel() == 0)
        quantized = NativeTensor({frames, embeddings.dim(1)});
      if (quantized.dim(1) != embeddings.dim(1))
        throw std::runtime_error("Mismatched codec embedding width");
      for (size_t frame = 0; frame < frames; ++frame) {
        const auto token = codes[frame][start + index];
        if (token < 0 || static_cast<size_t>(token) >= embeddings.dim(0))
          throw std::runtime_error("Codec token out of range");
        const auto scale = 1.0F / std::max(usage.values[token], 1e-5F);
        for (size_t column = 0; column < embeddings.dim(1); ++column) {
          quantized.values[frame * quantized.dim(1) + column] +=
              embeddings.values[static_cast<size_t>(token) * embeddings.dim(1) +
                                column] *
              scale;
        }
      }
    }
    return linear_1x1(quantized,
                      codec_weights_->tensor(prefix + ".output_proj.weight"));
  };
  auto hidden = add(decode("quantizer.rvq_first", 0, 1),
                    decode("quantizer.rvq_rest", 1, 15));
  hidden = causal_conv1d(hidden, codec_weights_->tensor("pre_conv.conv.weight"),
                         codec_weights_->tensor("pre_conv.conv.bias"));
  return linear(hidden,
                codec_weights_->tensor("pre_transformer.input_proj.weight"),
                &codec_weights_->tensor("pre_transformer.input_proj.bias"));
}

float Qwen3TtsRunner::codec_prefix_tail_rms(
    const std::vector<std::array<int32_t, 16>> &frames) {
  constexpr size_t kPrefixContextFrames = 3;
  if (frames.empty())
    throw std::runtime_error("Cannot measure an empty codec prefix");
  const auto begin = frames.size() > kPrefixContextFrames
                         ? frames.size() - kPrefixContextFrames
                         : 0;
  std::vector<std::array<int32_t, 16>> suffix(frames.begin() + begin,
                                              frames.end());
  return rms(row(run_codec_prefix(suffix), suffix.size() - 1));
}

NativeTensor Qwen3TtsRunner::run_codec_transformer(const NativeTensor &hidden) {
  if (hidden.rank() != 2 || hidden.dim(1) != kCodecHidden || hidden.dim(0) == 0)
    throw std::runtime_error("Invalid codec transformer input");
  NativeTensor outputs({hidden.dim(0), kCodecLatent});
  for (size_t offset = 0; offset < hidden.dim(0); offset += kCodecMax) {
    const auto frames = std::min<size_t>(kCodecMax, hidden.dim(0) - offset);
    reset_codec_caches();
    for (size_t token = 0; token < frames; ++token) {
      bf16_upload("codec_dec_n1_input",
                  as_rank4_row(row(hidden, offset + token), kCodecHidden));
      std::vector<MLAModelWithBuffer *> models;
      models.reserve(kCodecLayers * 3);
      for (uint16_t layer = 0; layer < kCodecLayers; ++layer) {
        models.push_back(codec_pre(layer, token).get());
        models.push_back(codec_cache(layer, token).get());
        models.push_back(codec_post(layer, token).get());
      }
      Qwen3MlaStaticBatch(std::move(models)).run();
      const auto output = bf16_download("codec_dec_output");
      std::copy_n(output.values.begin(), kCodecLatent,
                  outputs.values.begin() + (offset + token) * kCodecLatent);
    }
  }
  return outputs;
}

NativeTensor
Qwen3TtsRunner::run_codec_transformer_n128_hybrid(const NativeTensor &hidden) {
  if (hidden.rank() != 2 || hidden.dim(1) != kCodecHidden ||
      hidden.dim(0) == 0 || hidden.dim(0) > 128) {
    throw std::runtime_error("Invalid codec N128 hybrid input");
  }
  reset_codec_caches();
  NativeTensor padded({1, 1, 128, kCodecHidden});
  for (size_t frame = 0; frame < hidden.dim(0); ++frame) {
    std::copy_n(hidden.values.begin() + frame * kCodecHidden, kCodecHidden,
                padded.values.begin() + frame * kCodecHidden);
  }
  bf16_upload("codec_dec_n128_input", padded);
  std::vector<MLAModelWithBuffer *> n128_models;
  n128_models.reserve((kCodecLayers - 1) * 3);
  for (uint16_t layer = 0; layer < kCodecLayers - 1; ++layer) {
    n128_models.push_back(codec_n128_pre(layer).get());
    n128_models.push_back(codec_n128_cache(layer).get());
    n128_models.push_back(codec_n128_post(layer).get());
  }
  Qwen3MlaStaticBatch(std::move(n128_models)).run();
  for (uint16_t token = 0; token < hidden.dim(0); ++token) {
    Qwen3MlaStaticBatch({codec_final_pre_from_n128(token).get(),
                         codec_cache(kCodecLayers - 1, token).get(),
                         codec_final_post_from_n128(token).get()})
        .run();
  }
  const auto full_output = bf16_download("codec_dec_n128_output");
  NativeTensor output({hidden.dim(0), kCodecLatent});
  std::copy_n(full_output.values.begin(), output.numel(),
              output.values.begin());
  return output;
}

std::vector<float> Qwen3TtsRunner::run_codec_tail(const NativeTensor &hidden) {
  if (hidden.rank() != 2 || hidden.dim(1) != kCodecLatent || hidden.dim(0) == 0)
    throw std::runtime_error("Invalid codec-tail input");
  if (tail_models_.empty()) {
    for (size_t i = 0; i < tail_parts_.size(); ++i)
      tail_models_.push_back(std::make_unique<MLAModelWithBuffer>(
          tail_parts_[i].elf,
          std::vector<MLABufferSlice>{
              full(buffer("tail_edge" + std::to_string(i)))},
          std::vector<MLABufferSlice>{
              full(buffer("tail_edge" + std::to_string(i + 1)))}));
  }
  std::vector<MLAModelWithBuffer *> models;
  for (const auto &model : tail_models_)
    models.push_back(model.get());
  Qwen3MlaStaticBatch tail_batch(std::move(models));
  constexpr size_t kTailSamples = 96000;
  const auto samples_per_frame = kTailSamples / kCodecMax;
  std::vector<float> result;
  result.reserve(hidden.dim(0) * samples_per_frame);
  for (size_t offset = 0; offset < hidden.dim(0); offset += kCodecMax) {
    const auto frames = std::min<size_t>(kCodecMax, hidden.dim(0) - offset);
    NativeTensor padded({1, 1, kCodecMax, kCodecLatent});
    std::copy_n(hidden.values.begin() + offset * kCodecLatent,
                frames * kCodecLatent, padded.values.begin());
    bf16_upload("tail_edge0", padded);
    tail_batch.run();
    const auto waveform = bf16_download("tail_edge27");
    for (size_t sample = 0; sample < frames * samples_per_frame; ++sample)
      result.push_back(std::clamp(waveform.values[sample * 16], -1.0F, 1.0F));
  }
  if (!std::all_of(result.begin(), result.end(),
                   [](float value) { return std::isfinite(value); }))
    throw std::runtime_error(
        "Raw MLA codec tail generated non-finite waveform");
  return result;
}

void Qwen3TtsRunner::write_wav_pcm16(const std::filesystem::path &path,
                                     const std::vector<float> &samples) const {
  if (path.empty())
    return;
  std::filesystem::create_directories(
      path.parent_path().empty() ? "." : path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("Unable to open WAV output: " + path.string());
  const uint32_t data_bytes =
      static_cast<uint32_t>(samples.size() * sizeof(int16_t));
  out.write("RIFF", 4);
  write_u32(out, 36 + data_bytes);
  out.write("WAVE", 4);
  out.write("fmt ", 4);
  write_u32(out, 16);
  write_u16(out, 1);
  write_u16(out, 1);
  write_u32(out, 24000);
  write_u32(out, 24000 * 2);
  write_u16(out, 2);
  write_u16(out, 16);
  out.write("data", 4);
  write_u32(out, data_bytes);
  for (const auto value : samples) {
    const auto clipped = std::clamp(value, -1.0F, 1.0F);
    // Match libsndfile's float32-to-PCM16 conversion used by Python
    // soundfile.write.
    const auto scaled = static_cast<int32_t>(std::floor(clipped * 32768.0F));
    const auto pcm = static_cast<int16_t>(std::clamp(scaled, -32768, 32767));
    out.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
  }
}

void Qwen3TtsRunner::set_seed(uint64_t seed) { rng_.seed(seed); }

RunResult Qwen3TtsRunner::run(const RequestOptions &request) {
  if (!initialized_)
    throw std::runtime_error("Qwen3 native engine is not initialized");
  constexpr uint32_t kGenerationSafetyCeiling = 512;
  if (request.max_frames == 0 || request.max_frames > kGenerationSafetyCeiling)
    throw std::runtime_error("--max-frames must be between 1 and 512");
  if (request.endpoint_silence_frames == 0)
    throw std::runtime_error("--endpoint-silence-frames must be positive");
  const auto now = [] { return std::chrono::steady_clock::now(); };
  const auto seconds = [](auto begin, auto end) {
    return std::chrono::duration<double>(end - begin).count();
  };
  const auto e2e_start = now();
  const auto framed = std::string("<|im_start|>assistant\n") + request.prompt +
                      "<|im_end|>\n<|im_start|>assistant\n";
  const auto input_ids = tokenizer_->encode(framed, true);
  std::vector<int64_t> input_ids_i64(input_ids.begin(), input_ids.end());
  NativeTensor trailing, pad;
  const auto prompt_start = now();
  const auto prefill = build_prefill(input_ids, request.speaker,
                                     request.language, trailing, pad);
  RunResult result;
  result.metrics.prompt_tokens = static_cast<uint32_t>(input_ids.size());
  result.metrics.input_ids_sha256 =
      sha256(input_ids_i64.data(), input_ids_i64.size() * sizeof(int64_t));
  result.metrics.prefill_sha256 = bf16_sha256(prefill);
  auto hidden = run_backbone_prefill(prefill, request.speaker, request.language,
                                     result.metrics);
  result.metrics.backbone_prefill_hidden_sha256 = bf16_sha256(hidden);
  const auto prompt_end = now();
  result.metrics.prompt_time = seconds(prompt_start, prompt_end);
  result.metrics.ttft = seconds(e2e_start, prompt_end);
  std::vector<int32_t> previous_c0;
  auto c0 = select_token(linear(hidden, *codec_head_weight_), previous_c0, true,
                         request.do_sample, request.top_k, request.top_p,
                         request.temperature, request.repetition_penalty);
  previous_c0.push_back(c0);
  const bool endpoint_enabled = request.streaming_endpoint &&
                                !request.do_sample &&
                                !request.subtalker_do_sample;
  result.metrics.endpoint_enabled = endpoint_enabled;
  result.metrics.endpoint_silence_rms_threshold = request.endpoint_silence_rms;
  uint32_t silence_run{};
  std::optional<double> ttf_frame;
  const auto generation_start = now();
  for (uint32_t step = 0; step < request.max_frames; ++step) {
    if (c0 == kCodecEos)
      break;
    const auto frame = run_code_predictor(hidden, c0, request, result.metrics);
    result.frames.push_back(frame);
    if (!ttf_frame)
      ttf_frame = seconds(e2e_start, now());
    if (endpoint_enabled) {
      const auto prefix_rms = codec_prefix_tail_rms(result.frames);
      result.metrics.endpoint_prefix_rms.push_back(prefix_rms);
      silence_run =
          prefix_rms < request.endpoint_silence_rms ? silence_run + 1 : 0;
      if (silence_run >= request.endpoint_silence_frames) {
        const auto first_silent = result.frames.size() - silence_run;
        const auto retained_pad =
            std::min<size_t>(request.endpoint_end_pad_frames, silence_run);
        const auto retained_frames = first_silent + retained_pad;
        result.metrics.endpoint_triggered = true;
        result.metrics.endpoint_trigger_frame = step;
        result.metrics.endpoint_retained_pad_frames =
            static_cast<uint32_t>(retained_pad);
        result.metrics.endpoint_discarded_confirmation_frames =
            static_cast<uint32_t>(result.frames.size() - retained_frames);
        result.metrics.generated_frames_before_endpoint =
            static_cast<uint32_t>(result.frames.size());
        result.frames.resize(retained_frames);
        break;
      }
    }
    const auto feedback_start = now();
    auto next = backbone_feedback(frame);
    if (step < trailing.dim(0))
      add_inplace(next, trailing);
    else
      add_inplace(next, pad);
    const auto feedback_end = now();
    bf16_upload("backbone_n1_input_embed", next);
    const auto backbone_mla_start = now();
    run_backbone_token(backbone_position_++);
    const auto backbone_mla_end = now();
    const auto downloaded = bf16_download("backbone_n1_buffer1");
    hidden = NativeTensor({1, kHidden});
    std::copy_n(downloaded.values.begin(), kHidden, hidden.values.begin());
    const auto backbone_decode_end = now();
    result.metrics.backbone_feedback_time +=
        seconds(feedback_start, feedback_end);
    result.metrics.backbone_decode_mla_time +=
        seconds(backbone_mla_start, backbone_mla_end);
    result.metrics.backbone_decode_time +=
        seconds(feedback_end, backbone_decode_end);
    c0 = select_token(linear(hidden, *codec_head_weight_), previous_c0, true,
                      request.do_sample, request.top_k, request.top_p,
                      request.temperature, request.repetition_penalty);
    previous_c0.push_back(c0);
  }
  if (result.frames.empty())
    throw std::runtime_error("No codec frames generated");
  if (!result.metrics.endpoint_triggered)
    result.metrics.generated_frames_before_endpoint =
        static_cast<uint32_t>(result.frames.size());
  const auto generation_end = now();
  result.metrics.frames = static_cast<uint32_t>(result.frames.size());
  result.metrics.generation_time = seconds(generation_start, generation_end);
  result.metrics.ttf_frame = ttf_frame.value_or(result.metrics.ttft);
  std::vector<int32_t> raw_frames(result.frames.size() * 16);
  for (size_t row_index = 0; row_index < result.frames.size(); ++row_index)
    for (size_t column = 0; column < 16; ++column)
      raw_frames[row_index * 16 + column] = result.frames[row_index][column];
  result.metrics.frames_sha256 =
      sha256(raw_frames.data(), raw_frames.size() * sizeof(int32_t));
  const auto codec_start = now();
  const auto prefix = run_codec_prefix(result.frames);
  result.metrics.codec_prefix_sha256 = bf16_sha256(prefix);
  result.metrics.codec_n128_hybrid = request.codec_n128_hybrid;
  const auto tail_input = request.codec_n128_hybrid
                              ? run_codec_transformer_n128_hybrid(prefix)
                              : run_codec_transformer(prefix);
  result.metrics.codec_tail_input_sha256 = bf16_sha256(tail_input);
  result.waveform = run_codec_tail(tail_input);
  result.metrics.codec_tail_chunks =
      static_cast<uint32_t>((tail_input.dim(0) + kCodecMax - 1) / kCodecMax);
  result.metrics.codec_tail_uploads = result.metrics.codec_tail_chunks;
  result.metrics.codec_tail_downloads = result.metrics.codec_tail_chunks;
  const auto codec_end = now();
  result.metrics.codec_to_wav_time = seconds(codec_start, codec_end);
  result.metrics.ttfa = seconds(e2e_start, codec_end);
  const auto wav_start = now();
  write_wav_pcm16(request.output_wav, result.waveform);
  const auto wav_end = now();
  result.metrics.wav_write_time = seconds(wav_start, wav_end);
  result.metrics.wav_path = request.output_wav;
  result.metrics.e2e_time = seconds(e2e_start, wav_end);
  return result;
}

} // namespace simaai::llima::qwen3tts
