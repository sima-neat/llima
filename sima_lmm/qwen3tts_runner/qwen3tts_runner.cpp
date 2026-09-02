#include "qwen3tts_runner.hpp"
#include "qwen3_mla_static_batch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
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

#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/padding.h>

namespace simaai::llima::qwen3tts {
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

namespace F = torch::nn::functional;

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::vector<uint32_t> u32(std::initializer_list<uint32_t> values) {
    return {values.begin(), values.end()};
}

uint16_t aligned_cache(uint16_t token) {
    return static_cast<uint16_t>((((token + 1) + 127) / 128) * 128 - 1);
}

std::filesystem::path elf_path(
    const std::filesystem::path& dir, const std::string& name, const std::string& part, uint16_t layer
) {
    return dir / (name + "_n1_" + part + std::to_string(layer) + "_stage1_mla.elf");
}

std::filesystem::path cache_path(
    const std::filesystem::path& dir, const std::string& name, uint16_t token
) {
    return dir / (name + "_n1_cache_token" + std::to_string(aligned_cache(token)) + "_stage1_mla.elf");
}

std::filesystem::path n128_elf_path(
    const std::filesystem::path& dir, const std::string& name, const std::string& part, uint16_t layer
) {
    return dir / (name + "_n128_" + part + std::to_string(layer) + "_stage1_mla.elf");
}

std::string sha256(const void* data, size_t len) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(data), len, digest.data());
    std::ostringstream stream;
    for (const auto value : digest) stream << std::hex << std::setw(2) << std::setfill('0') << int(value);
    return stream.str();
}

void write_u16(std::ofstream& out, uint16_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
void write_u32(std::ofstream& out, uint32_t value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

torch::Tensor as_float(const torch::Tensor& value) {
    return value.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();
}
torch::Tensor as_bf16(const torch::Tensor& value) {
    return value.to(torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU)).contiguous();
}

// Keep NumPy's ILP64 OpenBLAS private. Linking it into the global namespace
// lets LibTorch's LP64 convolution path bind against incompatible BLAS symbols.
using NumpySgemm64 = void (*)(int, int, int, int64_t, int64_t, int64_t,
                               float, const float*, int64_t, const float*, int64_t,
                               float, float*, int64_t);

std::filesystem::path bundled_runtime_library(const char* filename) {
    return std::filesystem::canonical("/proc/self/exe").parent_path()
           / ".." / "lib" / "numpy.libs" / filename;
}

NumpySgemm64 numpy_sgemm64() {
    static const NumpySgemm64 fn = [] {
        const auto library = bundled_runtime_library(QWEN3_NUMPY_OPENBLAS_NAME);
        void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(std::string("Unable to load NumPy OpenBLAS: ") + dlerror());
        void* symbol = dlsym(handle, "cblas_sgemm64_");
        if (!symbol) throw std::runtime_error(std::string("Missing cblas_sgemm64_ in NumPy OpenBLAS: ") + dlerror());
        return reinterpret_cast<NumpySgemm64>(symbol);
    }();
    return fn;
}

torch::Tensor numpy_openblas_matmul_transpose_rhs(const torch::Tensor& lhs, const torch::Tensor& rhs) {
    const auto x = as_float(lhs).reshape({-1, lhs.size(-1)}).contiguous();
    const auto weight = as_float(rhs).contiguous();
    if (weight.dim() != 2 || x.size(1) != weight.size(1)) {
        throw std::runtime_error("Invalid text-projection matrix dimensions");
    }
    auto result = torch::empty({x.size(0), weight.size(0)}, torch::TensorOptions().dtype(torch::kFloat32));
    constexpr int kRowMajor = 101;
    constexpr int kNoTrans = 111;
    constexpr int kTrans = 112;
    numpy_sgemm64()(kRowMajor, kNoTrans, kTrans,
                   x.size(0), weight.size(0), x.size(1),
                   1.0F, x.data_ptr<float>(), x.size(1),
                   weight.data_ptr<float>(), weight.size(1),
                   0.0F, result.data_ptr<float>(), result.size(1));
    return result;
}

std::string bf16_sha256(const torch::Tensor& value) {
    const auto rounded = as_bf16(value);
    return sha256(rounded.data_ptr(), static_cast<size_t>(rounded.numel()) * sizeof(uint16_t));
}

MLABufferSlice full(MLABuffer& value) { return MLABufferSlice(&value); }
MLABufferSlice slice(MLABuffer& value, std::vector<uint32_t> begin, std::vector<uint32_t> shape) {
    return MLABufferSlice(&value, std::move(begin), std::move(shape));
}

} // namespace

struct Qwen3TtsRunner::TensorFile {
    explicit TensorFile(const std::filesystem::path& path) : path_(path) {
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("Unable to open safetensors file: " + path.string());
        struct stat status {};
        if (fstat(fd_, &status) != 0 || status.st_size < static_cast<off_t>(sizeof(uint64_t))) {
            throw std::runtime_error("Invalid safetensors file: " + path.string());
        }
        size_ = static_cast<size_t>(status.st_size);
        mapped_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped_ == MAP_FAILED) {
            mapped_ = nullptr;
            throw std::runtime_error("Unable to map safetensors file: " + path.string());
        }
        uint64_t header_size{};
        std::memcpy(&header_size, mapped_, sizeof(header_size));
        if (header_size > size_ - sizeof(header_size)) {
            throw std::runtime_error("Invalid safetensors header length: " + path.string());
        }
        const auto* header = static_cast<const char*>(mapped_) + sizeof(header_size);
        metadata_ = nlohmann::json::parse(header, header + header_size);
        data_ = reinterpret_cast<const uint8_t*>(header) + header_size;
        data_size_ = size_ - sizeof(header_size) - header_size;
    }
    ~TensorFile() {
        if (mapped_) munmap(mapped_, size_);
        if (fd_ >= 0) close(fd_);
    }
    TensorFile(const TensorFile&) = delete;
    TensorFile& operator=(const TensorFile&) = delete;

    torch::Tensor tensor(const std::string& key) const {
        if (!metadata_.contains(key)) throw std::runtime_error("Missing safetensors key " + key + " in " + path_.string());
        const auto& spec = metadata_.at(key);
        const auto dtype = spec.at("dtype").get<std::string>();
        torch::ScalarType scalar;
        size_t element_size{};
        if (dtype == "F32") { scalar = torch::kFloat32; element_size = 4; }
        else if (dtype == "I64") { scalar = torch::kInt64; element_size = 8; }
        else throw std::runtime_error("Unsupported safetensors dtype " + dtype + " for " + key);
        std::vector<int64_t> shape = spec.at("shape").get<std::vector<int64_t>>();
        const auto offsets = spec.at("data_offsets").get<std::array<size_t, 2>>();
        if (offsets[0] > offsets[1] || offsets[1] > data_size_) throw std::runtime_error("Invalid data offsets for " + key);
        size_t expected = element_size;
        for (const auto dim : shape) {
            if (dim < 0) throw std::runtime_error("Negative shape in " + key);
            expected *= static_cast<size_t>(dim);
        }
        if (expected != offsets[1] - offsets[0]) throw std::runtime_error("Invalid byte size for " + key);
        auto options = torch::TensorOptions().dtype(scalar).device(torch::kCPU).requires_grad(false);
        return torch::from_blob(const_cast<uint8_t*>(data_ + offsets[0]), shape, options);
    }
    bool has(const std::string& key) const { return metadata_.contains(key); }

  private:
    std::filesystem::path path_;
    int fd_{-1};
    size_t size_{};
    void* mapped_{};
    const uint8_t* data_{};
    size_t data_size_{};
    nlohmann::json metadata_;
};

struct Qwen3TtsRunner::TailPart {
    uint16_t index{};
    std::filesystem::path elf;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
};

#if defined(QWEN3_ENDPOINT_INCREMENTAL)
struct Qwen3TtsRunner::EndpointPrefixRmsState {
    std::array<torch::Tensor, 16> normalized_codebooks;
    torch::Tensor first_output_proj;
    torch::Tensor rest_output_proj;
    torch::Tensor pre_weight;
    torch::Tensor pre_bias;
    torch::Tensor input_proj_weight;
    torch::Tensor input_proj_bias;
    std::deque<torch::Tensor> recent_projected;
};
#endif

Qwen3TtsRunner::Qwen3TtsRunner(
    std::filesystem::path model_dir, std::filesystem::path components_dir, bool preload_models
) : model_dir_(std::move(model_dir)), components_dir_(std::move(components_dir)),
    elf_dir_(model_dir_ / "mpk"), preload_models_(preload_models)
#if defined(QWEN3_CP_SPLIT_HEADS)
    , cp_head_elf_dir_(elf_dir_)
#endif
{}

Qwen3TtsRunner::~Qwen3TtsRunner() { finalize(); }

MLABuffer& Qwen3TtsRunner::buffer(const std::string& name) const {
    const auto it = buffers_.find(name);
    if (it == buffers_.end()) throw std::runtime_error("Unknown MLA buffer: " + name);
    return *it->second;
}

torch::Tensor Qwen3TtsRunner::bf16_download(const std::string& name) const {
    const auto& value = buffer(name);
    std::vector<int64_t> shape(value.get_shape().begin(), value.get_shape().end());
    auto result = torch::empty(shape, torch::TensorOptions().dtype(torch::kBFloat16));
    value.download(result.data_ptr());
    return as_float(result);
}

void Qwen3TtsRunner::bf16_upload(const std::string& name, const torch::Tensor& value) const {
    const auto source = as_bf16(value);
    auto& target = buffer(name);
    if (static_cast<size_t>(source.numel()) != target.get_num_elems()) {
        throw std::runtime_error("MLA upload shape does not match " + name);
    }
    target.upload(source.data_ptr());
}

#if defined(QWEN3_CP_SPLIT_HEADS)
void Qwen3TtsRunner::fp32_upload(const std::string& name, const torch::Tensor& value) const {
    const auto source = as_float(value).contiguous();
    auto& target = buffer(name);
    if (target.get_elem_size() != sizeof(float) ||
        static_cast<size_t>(source.numel()) != target.get_num_elems()) {
        throw std::runtime_error("FP32 raw-head upload shape does not match " + name);
    }
    target.upload(source.data_ptr<float>());
}
#endif

void Qwen3TtsRunner::define_buffers() {
    auto define = [this](const std::string& name, std::vector<size_t> shape, bool align = false) {
        buffers_.emplace(name, std::make_unique<MLABuffer>(name, std::move(shape), "bfloat16", align));
    };

    define("backbone_future_token_mask", {2176});
    define("cp_future_token_mask", {176});
    define("codec_dec_future_token_mask", {2176});

    // Direct raw ELFs use rank-4 BF16 edge buffers. K/V edges use HWC16
    // backing so each layer owns a full 1024-token physical cache.
    define("backbone_freq_real", {1, 1, kBackboneMax, kHeadDim / 2});
    define("backbone_freq_imag", {1, 1, kBackboneMax, kHeadDim / 2});
    for (uint16_t i = 0; i < kBackboneLayers; ++i) {
        define("backbone_cache_key_l" + std::to_string(i), {1, 8, kBackboneMax, kHeadDim}, true);
        define("backbone_cache_val_l" + std::to_string(i), {1, 8, kBackboneMax, kHeadDim}, true);
    }
    define("backbone_n1_buffer1", {1, 1, 1, kHidden});
    define("backbone_n1_buffer2", {1, 16, 1, kHeadDim});
    define("backbone_n1_buffer3", {1, 1, 1, kQSize});
    define("backbone_n1_input_embed", {1, 1, 1, kHidden});

    define("cp_freq_real", {1, 1, kCpMax, kHeadDim / 2});
    define("cp_freq_imag", {1, 1, kCpMax, kHeadDim / 2});
    for (uint16_t i = 0; i < kCpLayers; ++i) {
        define("cp_cache_key_l" + std::to_string(i), {1, 8, kBackboneMax, kHeadDim}, true);
        define("cp_cache_val_l" + std::to_string(i), {1, 8, kBackboneMax, kHeadDim}, true);
    }
    define("cp_n1_initial_input", {1, 1, 1, kHidden});
    define("cp_n1_input", {1, 1, 1, kHidden});
    define("cp_n1_buffer1", {1, 1, 1, kHidden});
    define("cp_n1_buffer2", {1, 16, 1, kHeadDim});
    define("cp_n1_buffer3", {1, 1, 1, kQSize});
    for (uint16_t i = 0; i < 15; ++i) define("cp_head_logits_cb" + std::to_string(i), {1, 1, 1, kCpVocab});
#if defined(QWEN3_CP_SPLIT_HEADS)
    // MLABuffer has no float32 label; int32 supplies the exact four-byte edge ABI.
    buffers_.emplace("cp_head_input_fp32", std::make_unique<MLABuffer>(
        "cp_head_input_fp32", std::vector<size_t>{1, 1, 1, kHidden}, "int32", false));
    buffers_.emplace("cp_head_self_attn_fp32", std::make_unique<MLABuffer>(
        "cp_head_self_attn_fp32", std::vector<size_t>{1, 1, 1, kQSize}, "int32", false));
#endif

    define("codec_dec_freq_real", {1, 1, kCodecCacheMax, 32});
    define("codec_dec_freq_imag", {1, 1, kCodecCacheMax, 32});
    for (uint16_t i = 0; i < kCodecLayers; ++i) {
        define("codec_dec_cache_key_l" + std::to_string(i), {1, kCodecHeads, kCodecCacheMax, 64}, true);
        define("codec_dec_cache_val_l" + std::to_string(i), {1, kCodecHeads, kCodecCacheMax, 64}, true);
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
    const auto contract_path = model_dir_ / "devkit" / "codec_tail_raw_mla_contract.json";
    std::ifstream stream(contract_path);
    if (!stream) throw std::runtime_error("Missing codec-tail contract: " + contract_path.string());
    const auto contract = nlohmann::json::parse(stream);
    if (contract.value("format", "") != "qwen3_codec_tail_raw_mla_micro_v1" ||
        contract.value("dtype", "") != "bfloat16" || contract.value("layout", "") != "NHWC" ||
        contract.value("align_last_dim", true)) {
        throw std::runtime_error("Unsupported codec-tail contract");
    }
    const auto& parts = contract.at("parts");
    if (!parts.is_array() || parts.size() != 27) throw std::runtime_error("Codec-tail contract requires 27 stages");
    std::vector<size_t> previous;
    for (size_t i = 0; i < parts.size(); ++i) {
        const auto& item = parts.at(i);
        if (item.value("index", -1) != static_cast<int>(i)) throw std::runtime_error("Non-sequential tail stage");
        TailPart part;
        part.index = static_cast<uint16_t>(i);
        part.input_shape = item.at("input_shape").get<std::vector<size_t>>();
        part.output_shape = item.at("output_shape").get<std::vector<size_t>>();
        if (part.input_shape.size() != 4 || part.output_shape.size() != 4 || (!previous.empty() && previous != part.input_shape)) {
            throw std::runtime_error("Invalid tail shape chain at stage " + std::to_string(i));
        }
        auto input_numel = std::accumulate(part.input_shape.begin(), part.input_shape.end(), size_t{1}, std::multiplies<>());
        auto output_numel = std::accumulate(part.output_shape.begin(), part.output_shape.end(), size_t{1}, std::multiplies<>());
        if (item.at("input_bytes").get<size_t>() != input_numel * 2 || item.at("output_bytes").get<size_t>() != output_numel * 2) {
            throw std::runtime_error("Invalid tail byte count at stage " + std::to_string(i));
        }
        const auto name = item.at("elf").get<std::string>();
        if (std::filesystem::path(name).filename() != name) throw std::runtime_error("Invalid tail ELF name");
        part.elf = elf_dir_ / name;
        if (!std::filesystem::is_regular_file(part.elf)) throw std::runtime_error("Missing tail ELF: " + part.elf.string());
        previous = part.output_shape;
        tail_parts_.push_back(std::move(part));
    }
    if (tail_parts_.front().input_shape != std::vector<size_t>{1, 1, 50, 1024} ||
        tail_parts_.back().output_shape != std::vector<size_t>{1, 1, 96000, 16}) {
        throw std::runtime_error("Unexpected C16 tail endpoints");
    }
    for (size_t edge = 0; edge <= tail_parts_.size(); ++edge) {
        const auto& shape = edge == 0 ? tail_parts_.front().input_shape : tail_parts_[edge - 1].output_shape;
        buffers_.emplace("tail_edge" + std::to_string(edge), std::make_unique<MLABuffer>(
            "tail_edge" + std::to_string(edge), shape, "bfloat16", false));
    }
}

void Qwen3TtsRunner::validate_elfs() const {
    const auto require = [](const std::filesystem::path& path) {
        if (!std::filesystem::is_regular_file(path)) throw std::runtime_error("Missing required ELF: " + path.string());
    };
    for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
        require(elf_path(elf_dir_, "backbone_language", "pre_layer", layer));
        require(elf_path(elf_dir_, "backbone_language", "post_layer", layer));
    }
    for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
        require(elf_path(elf_dir_, "code_predictor_language", "pre_layer", layer));
#if defined(QWEN3_CP_SPLIT_HEADS)
        if (layer + 1 < kCpLayers) require(elf_path(elf_dir_, "code_predictor_language", "post_layer", layer));
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
        require(n128_elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer));
        require(n128_elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer));
    }
    require(elf_dir_ / "codec_decoder_language_n128_cache_token0_stage1_mla.elf");
    for (uint16_t cache : {127, 255, 383, 511, 639, 767, 895, 1023}) require(elf_dir_ / ("backbone_language_n1_cache_token" + std::to_string(cache) + "_stage1_mla.elf"));
    require(elf_dir_ / "code_predictor_language_n1_cache_token127_stage1_mla.elf");
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
        auto result = torch::full({static_cast<int64_t>(total)}, kBf16Lowest, torch::kFloat32);
        result.slice(0, 0, active).fill_(0);
        return result;
    };
    bf16_upload("backbone_future_token_mask", mask(kBackboneMax, 2176));
    bf16_upload("cp_future_token_mask", mask(kCpMax, 176));
    bf16_upload("codec_dec_future_token_mask", mask(kCodecCacheMax, 2176));
    auto rope = [](int64_t rows, int64_t head_dim, float theta) {
        auto positions = torch::arange(rows, torch::kFloat32);
        auto dimensions = torch::arange(0, head_dim, 2, torch::kFloat32);
        auto inverse = torch::pow(torch::full_like(dimensions, theta), -dimensions / static_cast<float>(head_dim));
        return torch::outer(positions, inverse);
    };
    const auto bb = rope(kBackboneMax, kHeadDim, kBackboneRopeTheta);
    bf16_upload("backbone_freq_real", torch::cos(bb));
    bf16_upload("backbone_freq_imag", torch::sin(bb));
    const auto cp = rope(kCpMax, kHeadDim, kCodePredictorRopeTheta);
    bf16_upload("cp_freq_real", torch::cos(cp));
    bf16_upload("cp_freq_imag", torch::sin(cp));
    const auto codec = rope(kCodecCacheMax, 64, kCodecDecoderRopeTheta);
    bf16_upload("codec_dec_freq_real", torch::cos(codec));
    bf16_upload("codec_dec_freq_imag", torch::sin(codec));
}

void Qwen3TtsRunner::load_host_weights() {
    backbone_weights_ = std::make_unique<TensorFile>(components_dir_ / "backbone" / "model.safetensors");
    cp_weights_ = std::make_unique<TensorFile>(components_dir_ / "code_predictor" / "model.safetensors");
    codec_weights_ = std::make_unique<TensorFile>(components_dir_ / "codec_decoder" / "model.safetensors");
    text_projection_weights_ = std::make_unique<TensorFile>(components_dir_ / "text_projection" / "text_projection.safetensors");
    codec_head_weights_ = std::make_unique<TensorFile>(components_dir_ / "codec_head" / "codec_head.safetensors");
    text_embeddings_ = backbone_weights_->tensor("text_embedding.weight");
    codec_embeddings_ = backbone_weights_->tensor("codec_embedding.weight");
    std::vector<torch::Tensor> cp_embeddings;
    cp_embeddings.push_back(codec_embeddings_);
    for (uint16_t i = 0; i < 15; ++i) cp_embeddings.push_back(cp_weights_->tensor("model.codec_embedding." + std::to_string(i) + ".weight"));
    codec_embeddings_ = torch::cat(cp_embeddings, 0);
    codec_head_weight_ = codec_head_weights_->tensor("weight");
    text_fc1_w_ = text_projection_weights_->tensor("linear_fc1.weight");
    text_fc1_b_ = text_projection_weights_->tensor("linear_fc1.bias");
    text_fc2_w_ = text_projection_weights_->tensor("linear_fc2.weight");
    text_fc2_b_ = text_projection_weights_->tensor("linear_fc2.bias");
    std::ifstream config_stream(components_dir_ / "backbone" / "config.json");
    if (!config_stream) throw std::runtime_error("Missing backbone config.json");
    talker_config_ = nlohmann::json::parse(config_stream);
}

void Qwen3TtsRunner::build_and_preload_models() {
    for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
        backbone_pre(layer, 0); backbone_post(layer, 0);
        for (uint16_t token = 0; token < kBackboneMax; token += 128) backbone_cache(layer, token);
    }
    for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
        cp_pre(layer, 0); cp_cache(layer, 0); cp_post(layer, 0);
    }
    for (uint16_t layer = 0; layer < kCodecLayers; ++layer) {
        codec_pre(layer, 0); codec_cache(layer, 0); codec_post(layer, 0);
    }
    if (tail_models_.empty()) {
        for (size_t i = 0; i < tail_parts_.size(); ++i) {
            tail_models_.push_back(std::make_unique<MLAModelWithBuffer>(tail_parts_[i].elf,
                std::vector<MLABufferSlice>{full(buffer("tail_edge" + std::to_string(i)))},
                std::vector<MLABufferSlice>{full(buffer("tail_edge" + std::to_string(i + 1)))}));
        }
    }
    MLAModelWithBuffer::load_all_models();
}

void Qwen3TtsRunner::initialize() {
    if (initialized_) return;
    validate_tail_contract();
    validate_elfs();
    define_buffers();
    for (auto& [_, value] : buffers_) value->allocate();
    initialize_static_buffers();
    load_host_weights();
    const auto tokenizer_path = components_dir_ / "processor" / "tokenizer.json";
    tokenizer_ = Tokenizer::from_hf_json(tokenizer_path);
    reset_caches();
    if (preload_models_) build_and_preload_models();
    initialized_ = true;
}

void Qwen3TtsRunner::finalize() {
    if (!initialized_) return;
    tail_models_.clear(); backbone_pre_.clear(); backbone_cache_.clear(); backbone_post_.clear();
    cp_pre_.clear(); cp_cache_.clear(); cp_post_.clear();
#if defined(QWEN3_CP_SPLIT_HEADS)
    cp_head_.clear();
#endif
    codec_pre_.clear(); codec_cache_.clear(); codec_post_.clear();
    codec_n128_pre_.clear(); codec_n128_cache_.clear(); codec_n128_post_.clear();
    codec_final_pre_from_n128_.clear(); codec_final_post_from_n128_.clear();
#if defined(QWEN3_ENDPOINT_INCREMENTAL)
    endpoint_prefix_rms_state_.reset();
#endif
    for (auto& [_, value] : buffers_) value->free();
    buffers_.clear(); prefix_snapshots_.clear(); resident_prefix_key_.reset(); tokenizer_.reset();
    initialized_ = false;
}

void Qwen3TtsRunner::reset_caches() {
    for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
        buffer("backbone_cache_key_l" + std::to_string(layer)).clear();
        buffer("backbone_cache_val_l" + std::to_string(layer)).clear();
    }
    reset_cp_caches(); reset_codec_caches(); backbone_position_ = 0;
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

torch::Tensor Qwen3TtsRunner::text_project(const torch::Tensor& input) const {
    auto output = numpy_openblas_matmul_transpose_rhs(input, text_fc1_w_) + text_fc1_b_;
    output = output * torch::sigmoid(output);
    return numpy_openblas_matmul_transpose_rhs(output, text_fc2_w_) + text_fc2_b_;
}

torch::Tensor Qwen3TtsRunner::build_prefill(
    const std::vector<uint32_t>& ids, const std::string& speaker, const std::string& language,
    torch::Tensor& trailing, torch::Tensor& pad
) const {
    if (ids.size() < 8) throw std::runtime_error("Unexpectedly short tokenized prompt");
    const auto spk_key = lower(speaker);
    if (!talker_config_.at("spk_id").contains(spk_key)) throw std::runtime_error("Unsupported speaker: " + speaker);
    const int64_t spk = talker_config_.at("spk_id").at(spk_key).get<int64_t>();
    std::optional<int64_t> language_id;
    if (lower(language) != "auto") {
        const auto lang_key = lower(language);
        if (!talker_config_.at("codec_language_id").contains(lang_key)) throw std::runtime_error("Unsupported language: " + language);
        language_id = talker_config_.at("codec_language_id").at(lang_key).get<int64_t>();
    }
    std::vector<int64_t> ids_i64(ids.begin(), ids.end());
    const auto ids_tensor = torch::from_blob(ids_i64.data(), {static_cast<int64_t>(ids_i64.size())}, torch::kInt64).clone();
    std::vector<int64_t> special_values{151672, 151673, 151671};
    const auto special = torch::from_blob(special_values.data(), {3}, torch::kInt64).clone();
    const auto special_projected = text_project(text_embeddings_.index_select(0, special));
    const auto bos = special_projected.select(0, 0);
    const auto eos = special_projected.select(0, 1);
    pad = special_projected.select(0, 2).contiguous();
    const auto control = [&](const char* name) { return talker_config_.at(name).get<int64_t>(); };
    std::vector<int64_t> prefix_ids;
    if (language_id) prefix_ids = {control("codec_think_id"), control("codec_think_bos_id"), *language_id, control("codec_think_eos_id")};
    else prefix_ids = {control("codec_nothink_id"), control("codec_think_bos_id"), control("codec_think_eos_id")};
    auto tail_ids = std::vector<int64_t>{control("codec_pad_id"), control("codec_bos_id")};
    const auto prefix_tensor = torch::from_blob(prefix_ids.data(), {static_cast<int64_t>(prefix_ids.size())}, torch::kInt64).clone();
    const auto tail_tensor = torch::from_blob(tail_ids.data(), {static_cast<int64_t>(tail_ids.size())}, torch::kInt64).clone();
    auto prefix = codec_embeddings_.index_select(0, prefix_tensor);
    auto tail = codec_embeddings_.index_select(0, tail_tensor);
    auto speaker_tensor = codec_embeddings_.select(0, spk).unsqueeze(0);
    std::vector<torch::Tensor> codec_pieces; codec_pieces.push_back(prefix); codec_pieces.push_back(speaker_tensor); codec_pieces.push_back(tail);
    auto codec_input = torch::cat(codec_pieces, 0);
    auto role = text_project(text_embeddings_.index_select(0, ids_tensor.slice(0, 0, 3)));
    const int64_t body_rows = codec_input.size(0) - 1;
    auto pads = pad.unsqueeze(0).repeat(std::vector<int64_t>{body_rows - 1, 1});
    std::vector<torch::Tensor> body_pieces; body_pieces.push_back(pads); body_pieces.push_back(bos.unsqueeze(0));
    auto body_base = torch::cat(body_pieces, 0);
    std::vector<torch::Tensor> prefill_pieces; prefill_pieces.push_back(role); prefill_pieces.push_back(body_base + codec_input.slice(0, 0, body_rows));
    auto prefill = torch::cat(prefill_pieces, 0).slice(0, 0, -1);
    const auto main = ids_tensor.slice(0, 3, static_cast<int64_t>(ids.size()) - 5);
    std::vector<torch::Tensor> eos_pieces; eos_pieces.push_back(text_project(text_embeddings_.index_select(0, main))); eos_pieces.push_back(eos.unsqueeze(0));
    auto text_eos = torch::cat(eos_pieces, 0);
    auto code_pad = codec_embeddings_.select(0, control("codec_pad_id")).unsqueeze(0).repeat(std::vector<int64_t>{static_cast<int64_t>(text_eos.size(0)), 1});
    auto final_row = (pad + codec_embeddings_.select(0, control("codec_bos_id"))).unsqueeze(0);
    trailing = pad.unsqueeze(0).contiguous();
    std::vector<torch::Tensor> final_pieces; final_pieces.push_back(prefill); final_pieces.push_back(text_eos + code_pad); final_pieces.push_back(final_row);
    return as_bf16(torch::cat(final_pieces, 0));
}

torch::Tensor Qwen3TtsRunner::codec_embedding(int32_t token, uint32_t codebook) const {
    const int64_t offset = codebook == 0 ? 0 : kTalkerVocab + static_cast<int64_t>(codebook - 1) * kCpVocab;
    const int64_t limit = codebook == 0 ? kTalkerVocab : kCpVocab;
    if (token < 0 || token >= limit) throw std::runtime_error("Codec token out of range");
    return codec_embeddings_.select(0, offset + token);
}

torch::Tensor Qwen3TtsRunner::backbone_feedback(const std::array<int32_t, 16>& frame) const {
    auto result = torch::zeros({kHidden}, torch::kFloat32);
    for (uint32_t i = 0; i < frame.size(); ++i) result += codec_embedding(frame[i], i);
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::backbone_pre(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = backbone_pre_[key];
    if (result) return result;
    auto& input = layer == 0 ? buffer("backbone_n1_input_embed") : buffer("backbone_n1_buffer1");
    const auto kv_begin = u32({0, 0, token, 0});
    const auto kv_shape = u32({1, 8, 1, kHeadDim});
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "backbone_language", "pre_layer", layer),
        std::vector<MLABufferSlice>{full(input),
                                    slice(buffer("backbone_freq_real"), u32({0, 0, token, 0}), u32({1, 1, 1, 64})),
                                    slice(buffer("backbone_freq_imag"), u32({0, 0, token, 0}), u32({1, 1, 1, 64}))},
        std::vector<MLABufferSlice>{full(buffer("backbone_n1_buffer2")),
                                    slice(buffer("backbone_cache_key_l" + std::to_string(layer)), kv_begin, kv_shape),
                                    slice(buffer("backbone_cache_val_l" + std::to_string(layer)), kv_begin, kv_shape)}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::backbone_cache(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = backbone_cache_[key];
    if (result) return result;
    const auto n = static_cast<uint16_t>(aligned_cache(token) + 1);
    result = std::make_unique<MLAModelWithBuffer>(
        cache_path(elf_dir_, "backbone_language", token),
        std::vector<MLABufferSlice>{full(buffer("backbone_n1_buffer2")),
                                    slice(buffer("backbone_cache_key_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, 8, n, kHeadDim})),
                                    slice(buffer("backbone_future_token_mask"), u32({kBackboneMax - (token + 1)}), u32({n})),
                                    slice(buffer("backbone_cache_val_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, 8, n, kHeadDim}))},
        std::vector<MLABufferSlice>{full(buffer("backbone_n1_buffer3"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::backbone_post(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = backbone_post_[key];
    if (result) return result;
    auto& input = layer == 0 ? buffer("backbone_n1_input_embed") : buffer("backbone_n1_buffer1");
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "backbone_language", "post_layer", layer),
        std::vector<MLABufferSlice>{full(input), full(buffer("backbone_n1_buffer3"))},
        std::vector<MLABufferSlice>{full(buffer("backbone_n1_buffer1"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::cp_pre(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = cp_pre_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "code_predictor_language", "pre_layer", layer),
        std::vector<MLABufferSlice>{full(layer == 0 ? buffer(token == 0 ? "cp_n1_initial_input" : "cp_n1_input") : buffer("cp_n1_buffer1")),
                                    slice(buffer("cp_freq_real"), u32({0, 0, token, 0}), u32({1, 1, 1, 64})),
                                    slice(buffer("cp_freq_imag"), u32({0, 0, token, 0}), u32({1, 1, 1, 64}))},
        std::vector<MLABufferSlice>{full(buffer("cp_n1_buffer2")),
                                    slice(buffer("cp_cache_key_l" + std::to_string(layer)), u32({0, 0, token, 0}), u32({1, 8, 1, kHeadDim})),
                                    slice(buffer("cp_cache_val_l" + std::to_string(layer)), u32({0, 0, token, 0}), u32({1, 8, 1, kHeadDim}))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::cp_cache(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = cp_cache_[key];
    if (result) return result;
    constexpr uint16_t kCpPhysicalCache = 128;
    result = std::make_unique<MLAModelWithBuffer>(
        cache_path(elf_dir_, "code_predictor_language", token),
        std::vector<MLABufferSlice>{full(buffer("cp_n1_buffer2")),
                                    slice(buffer("cp_cache_key_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, 8, kCpPhysicalCache, kHeadDim})),
                                    slice(buffer("cp_future_token_mask"), u32({kCpMax - (token + 1)}), u32({kCpPhysicalCache})),
                                    slice(buffer("cp_cache_val_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, 8, kCpPhysicalCache, kHeadDim}))},
        std::vector<MLABufferSlice>{full(buffer("cp_n1_buffer3"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::cp_post(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = cp_post_[key];
    if (result) return result;
    std::vector<MLABufferSlice> output;
    auto path = elf_path(elf_dir_, "code_predictor_language", "post_layer", layer);
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
        std::sort(order.begin(), order.end(), [](uint16_t a, uint16_t b) { return std::to_string(a) < std::to_string(b); });
        for (const auto codebook : order) output.push_back(full(buffer("cp_head_logits_cb" + std::to_string(codebook))));
    }
    result = std::make_unique<MLAModelWithBuffer>(
        path,
        std::vector<MLABufferSlice>{full(layer == 0 ? buffer(token == 0 ? "cp_n1_initial_input" : "cp_n1_input") : buffer("cp_n1_buffer1")), full(buffer("cp_n1_buffer3"))}, std::move(output)
    );
    return result;
}

#if defined(QWEN3_CP_SPLIT_HEADS)
Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::cp_head(uint16_t codebook) {
    if (codebook >= 15) throw std::runtime_error("Invalid code-predictor head index");
    auto& result = cp_head_[codebook];
    if (result) return result;
    const auto path = cp_head_elf_dir_ / ("code_predictor_language_n1_post_layer4_head" +
        std::to_string(codebook) + "_stage1_mla.elf");
    result = std::make_unique<MLAModelWithBuffer>(
        path,
        std::vector<MLABufferSlice>{full(buffer("cp_head_input_fp32")), full(buffer("cp_head_self_attn_fp32"))},
        std::vector<MLABufferSlice>{full(buffer("cp_head_logits_cb" + std::to_string(codebook)))}
    );
    return result;
}
#endif

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_pre(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = codec_pre_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer),
        std::vector<MLABufferSlice>{full(layer == 0 ? buffer("codec_dec_n1_input") : buffer("codec_dec_n1_buffer1")),
                                    slice(buffer("codec_dec_freq_real"), u32({0, 0, token, 0}), u32({1, 1, 1, 32})),
                                    slice(buffer("codec_dec_freq_imag"), u32({0, 0, token, 0}), u32({1, 1, 1, 32}))},
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n1_buffer2")),
                                    slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)), u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64})),
                                    slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)), u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64}))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_cache(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = codec_cache_[key];
    if (result) return result;
    const auto n = static_cast<uint16_t>(aligned_cache(token) + 1);
    result = std::make_unique<MLAModelWithBuffer>(
        cache_path(elf_dir_, "codec_decoder_language", token),
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n1_buffer2")),
                                    slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, kCodecHeads, n, 64})),
                                    slice(buffer("codec_dec_future_token_mask"), u32({kCodecCacheMax - (token + 1)}), u32({n})),
                                    slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, kCodecHeads, n, 64}))},
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n1_buffer3"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_post(uint16_t layer, uint16_t token) {
    const ModelKey key{layer, token};
    auto& result = codec_post_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer),
        std::vector<MLABufferSlice>{full(layer == 0 ? buffer("codec_dec_n1_input") : buffer("codec_dec_n1_buffer1")), full(buffer("codec_dec_n1_buffer3"))},
        std::vector<MLABufferSlice>{full(buffer(layer + 1 < kCodecLayers ? "codec_dec_n1_buffer1" : "codec_dec_output"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_n128_pre(uint16_t layer) {
    const ModelKey key{layer, 0};
    auto& result = codec_n128_pre_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        n128_elf_path(elf_dir_, "codec_decoder_language", "pre_layer", layer),
        std::vector<MLABufferSlice>{full(layer == 0 ? buffer("codec_dec_n128_input") : buffer("codec_dec_n128_hidden")),
                                    slice(buffer("codec_dec_freq_real"), u32({0, 0, 0, 0}), u32({1, 1, 128, 32})),
                                    slice(buffer("codec_dec_freq_imag"), u32({0, 0, 0, 0}), u32({1, 1, 128, 32}))},
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n128_query")),
                                    slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64})),
                                    slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64}))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_n128_cache(uint16_t layer) {
    const ModelKey key{layer, 0};
    auto& result = codec_n128_cache_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        elf_dir_ / "codec_decoder_language_n128_cache_token0_stage1_mla.elf",
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n128_query")),
                                    slice(buffer("codec_dec_cache_key_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64})),
                                    slice(buffer("codec_dec_cache_val_l" + std::to_string(layer)), u32({0, 0, 0, 0}), u32({1, kCodecHeads, 128, 64}))},
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n128_self_attn"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_n128_post(uint16_t layer) {
    const ModelKey key{layer, 0};
    auto& result = codec_n128_post_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        n128_elf_path(elf_dir_, "codec_decoder_language", "post_layer", layer),
        std::vector<MLABufferSlice>{full(layer == 0 ? buffer("codec_dec_n128_input") : buffer("codec_dec_n128_hidden")),
                                    full(buffer("codec_dec_n128_self_attn"))},
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n128_hidden"))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_final_pre_from_n128(uint16_t token) {
    const ModelKey key{kCodecLayers - 1, token};
    auto& result = codec_final_pre_from_n128_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "codec_decoder_language", "pre_layer", kCodecLayers - 1),
        std::vector<MLABufferSlice>{slice(buffer("codec_dec_n128_hidden"), u32({0, 0, token, 0}), u32({1, 1, 1, kCodecHidden})),
                                    slice(buffer("codec_dec_freq_real"), u32({0, 0, token, 0}), u32({1, 1, 1, 32})),
                                    slice(buffer("codec_dec_freq_imag"), u32({0, 0, token, 0}), u32({1, 1, 1, 32}))},
        std::vector<MLABufferSlice>{full(buffer("codec_dec_n1_buffer2")),
                                    slice(buffer("codec_dec_cache_key_l" + std::to_string(kCodecLayers - 1)), u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64})),
                                    slice(buffer("codec_dec_cache_val_l" + std::to_string(kCodecLayers - 1)), u32({0, 0, token, 0}), u32({1, kCodecHeads, 1, 64}))}
    );
    return result;
}

Qwen3TtsRunner::ModelPtr& Qwen3TtsRunner::codec_final_post_from_n128(uint16_t token) {
    const ModelKey key{kCodecLayers - 1, token};
    auto& result = codec_final_post_from_n128_[key];
    if (result) return result;
    result = std::make_unique<MLAModelWithBuffer>(
        elf_path(elf_dir_, "codec_decoder_language", "post_layer", kCodecLayers - 1),
        std::vector<MLABufferSlice>{slice(buffer("codec_dec_n128_hidden"), u32({0, 0, token, 0}), u32({1, 1, 1, kCodecHidden})),
                                    full(buffer("codec_dec_n1_buffer3"))},
        std::vector<MLABufferSlice>{slice(buffer("codec_dec_n128_output"), u32({0, 0, token, 0}), u32({1, 1, 1, kCodecLatent}))}
    );
    return result;
}

void Qwen3TtsRunner::run_backbone_token(uint16_t token) {
    std::vector<MLAModelWithBuffer*> models;
    models.reserve(kBackboneLayers * 3);
    for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
        models.push_back(backbone_pre(layer, token).get());
        models.push_back(backbone_cache(layer, token).get());
        models.push_back(backbone_post(layer, token).get());
    }
    Qwen3MlaStaticBatch(std::move(models)).run();
}

torch::Tensor Qwen3TtsRunner::run_backbone_prefill(
    const torch::Tensor& prefill, const std::string& speaker, const std::string& language, RunMetrics& metrics
) {
    if (prefill.dim() != 2 || prefill.size(1) != kHidden || prefill.size(0) == 0 || prefill.size(0) > kBackboneMax) {
        throw std::runtime_error("Invalid backbone prefill shape");
    }
    const auto prefix_len = static_cast<uint16_t>(lower(language) == "auto" ? 7 : 8);
    metrics.prefix_kv_static_tokens = prefix_len;
    const auto key = std::make_pair(lower(speaker), lower(language));
    const bool complete_static_prefix = prefill.size(0) > prefix_len;

    // A repeated request with this speaker/language may keep its static K/V
    // prefix on MLA. Dynamic prompt positions overwrite every suffix slot that
    // the following causal cache reads, so neither clearing nor re-uploading
    // that prefix is needed.
    if (complete_static_prefix && resident_prefix_key_ && *resident_prefix_key_ == key) {
        metrics.prefix_kv_reused = true;
        metrics.prefix_kv_device_resident = true;
        backbone_position_ = prefix_len;
    } else if (prefix_len >= prefill.size(0) || !prefix_snapshots_.contains(key)) {
        reset_caches();
        const auto limit = std::min<int64_t>(prefix_len, prefill.size(0));
        for (int64_t i = 0; i < limit; ++i) { bf16_upload("backbone_n1_input_embed", prefill.select(0, i)); run_backbone_token(i); }
        if (complete_static_prefix) {
            std::vector<std::vector<uint8_t>> snap;
            const size_t full_head_bytes = static_cast<size_t>(kBackboneMax) * kHeadDim * sizeof(uint16_t);
            const size_t prefix_head_bytes = static_cast<size_t>(prefix_len) * kHeadDim * sizeof(uint16_t);
            snap.reserve(kBackboneLayers * 2);
            for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
                for (const char* which : {"key", "val"}) {
                    auto& cache = buffer("backbone_cache_" + std::string(which) + "_l" + std::to_string(layer));
                    std::vector<uint8_t> raw(cache.get_num_elems() * sizeof(uint16_t));
                    cache.download(raw.data());
                    std::vector<uint8_t> compact(8 * prefix_head_bytes);
                    for (size_t head = 0; head < 8; ++head) {
                        std::memcpy(compact.data() + head * prefix_head_bytes,
                                    raw.data() + head * full_head_bytes, prefix_head_bytes);
                    }
                    snap.push_back(std::move(compact));
                }
            }
            prefix_snapshots_[key] = std::move(snap);
            resident_prefix_key_ = key;
        } else {
            resident_prefix_key_.reset();
        }
        backbone_position_ = limit;
    } else {
        reset_caches();
        const auto& snap = prefix_snapshots_.at(key);
        size_t index{};
        for (uint16_t layer = 0; layer < kBackboneLayers; ++layer) {
            for (const char* which : {"key", "val"}) {
                auto& target = buffer("backbone_cache_" + std::string(which) + "_l" + std::to_string(layer));
                const auto& compact = snap.at(index);
                const size_t full_head_bytes = static_cast<size_t>(kBackboneMax) * kHeadDim * sizeof(uint16_t);
                const size_t prefix_head_bytes = static_cast<size_t>(prefix_len) * kHeadDim * sizeof(uint16_t);
                for (size_t head = 0; head < 8; ++head) {
                    target.upload(compact.data() + head * prefix_head_bytes,
                                  head * full_head_bytes, prefix_head_bytes, false);
                }
                target.flush_cache();
                ++index;
            }
        }
        metrics.prefix_kv_reused = true;
        resident_prefix_key_ = key;
        backbone_position_ = prefix_len;
    }
    for (uint16_t token = backbone_position_; token < prefill.size(0); ++token) {
        bf16_upload("backbone_n1_input_embed", prefill.select(0, token));
        run_backbone_token(token);
    }
    backbone_position_ = static_cast<uint16_t>(prefill.size(0));
    return bf16_download("backbone_n1_buffer1");
}

int32_t Qwen3TtsRunner::select_token(
    torch::Tensor logits, const std::vector<int32_t>& previous, bool suppress, bool sample,
    uint32_t top_k, float top_p, float temperature, float repetition_penalty
) {
    logits = as_float(logits).reshape({-1});
    std::vector<float> values(logits.numel());
    std::memcpy(values.data(), logits.data_ptr<float>(), values.size() * sizeof(float));
    if (suppress) {
        for (int32_t index = 2048; index < 3072; ++index) if (index != kCodecEos) values.at(index) = -std::numeric_limits<float>::infinity();
    }
    if (repetition_penalty != 1.0F) {
        std::vector<int32_t> unique = previous;
        std::sort(unique.begin(), unique.end()); unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (const auto token : unique) if (token >= 0 && token < static_cast<int32_t>(values.size())) {
            values[token] = values[token] > 0.0F ? values[token] / repetition_penalty : values[token] * repetition_penalty;
        }
    }
    if (!sample) return static_cast<int32_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
    const float scale = std::max(temperature, 1e-5F);
    for (auto& value : values) value /= scale;
    if (top_k > 0 && top_k < values.size()) {
        auto partition = values;
        std::nth_element(partition.begin(), partition.end() - top_k, partition.end());
        const auto threshold = *(partition.end() - top_k);
        for (auto& value : values) if (value < threshold) value = -std::numeric_limits<float>::infinity();
    }
    if (top_p < 1.0F) {
        std::vector<size_t> order(values.size()); std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&values](size_t a, size_t b) { return values[a] > values[b]; });
        float maximum = values.at(order.front());
        double denominator{};
        for (const auto index : order) if (std::isfinite(values[index])) denominator += std::exp(static_cast<double>(values[index] - maximum));
        double cumulative{};
        for (size_t rank = 0; rank < order.size(); ++rank) {
            const auto index = order[rank];
            const auto probability = std::isfinite(values[index]) ? std::exp(static_cast<double>(values[index] - maximum)) / denominator : 0.0;
            cumulative += probability;
            if (rank > 0 && cumulative - probability > top_p) values[index] = -std::numeric_limits<float>::infinity();
        }
    }
    const auto maximum = *std::max_element(values.begin(), values.end());
    double total{};
    std::vector<double> probability(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        probability[i] = std::isfinite(values[i]) ? std::exp(static_cast<double>(values[i] - maximum)) : 0.0;
        total += probability[i];
    }
    const double value = rng_.next_double();
    double cumulative{};
    for (size_t i = 0; i < probability.size(); ++i) {
        cumulative += probability[i] / total;
        if (value < cumulative) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(probability.size() - 1);
}

torch::Tensor Qwen3TtsRunner::run_code_predictor(
    const torch::Tensor& hidden, int32_t c0, const RequestOptions& request, RunMetrics& metrics
) {
    const auto code_predictor_start = std::chrono::steady_clock::now();
    // Every logical K/V slot read by this frame is rewritten by its matching
    // pre-layer before cache attention consumes it; future slots are masked.
    // Keep the initialized backing buffers resident across frames.
    auto run_position = [this, &metrics](uint16_t token) {
        std::vector<MLAModelWithBuffer*> models;
        models.reserve(kCpLayers * 3);
        for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
            models.push_back(cp_pre(layer, token).get());
            models.push_back(cp_cache(layer, token).get());
            models.push_back(cp_post(layer, token).get());
        }
        const auto mla_start = std::chrono::steady_clock::now();
        Qwen3MlaStaticBatch(std::move(models)).run();
        metrics.code_predictor_mla_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - mla_start).count();
    };
    auto run_initial_and_c0 = [this, &metrics] {
        std::vector<MLAModelWithBuffer*> models;
        models.reserve(kCpLayers * 3 * 2);
        for (uint16_t token = 0; token < 2; ++token) {
            for (uint16_t layer = 0; layer < kCpLayers; ++layer) {
                models.push_back(cp_pre(layer, token).get());
                models.push_back(cp_cache(layer, token).get());
                // Position 0 establishes only K/V context. Its final logits are
                // overwritten by position 1 before sampling codebook 0.
                if (token == 0 && layer + 1 == kCpLayers) continue;
                models.push_back(cp_post(layer, token).get());
            }
        }
        const auto mla_start = std::chrono::steady_clock::now();
        Qwen3MlaStaticBatch(std::move(models)).run();
        metrics.code_predictor_mla_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - mla_start).count();
    };
    const auto initial_input = hidden.reshape({1, 1, 1, kHidden});
    if (metrics.cp_initial_input_sha256.empty()) metrics.cp_initial_input_sha256 = bf16_sha256(initial_input);
    bf16_upload("cp_n1_initial_input", initial_input);
    std::array<int32_t, 16> frame{};
    frame[0] = c0;
    int32_t current = c0;
    std::vector<int32_t> previous;
    auto token_input = codec_embedding(current, 0).reshape({1, 1, 1, kHidden});
    if (metrics.cp_codebook0_input_sha256.empty()) metrics.cp_codebook0_input_sha256 = bf16_sha256(token_input);
    bf16_upload("cp_n1_input", token_input);
    run_initial_and_c0();
    for (uint16_t codebook = 0; codebook < 15; ++codebook) {
        const auto logits = bf16_download("cp_head_logits_cb" + std::to_string(codebook));
        if (codebook == 0 && metrics.cp_codebook0_logits_sha256.empty()) {
            metrics.cp_codebook0_logits_sha256 = bf16_sha256(logits);
        }
        const auto token = select_token(logits, previous, false, request.subtalker_do_sample, request.subtalker_top_k,
                                        request.subtalker_top_p, request.subtalker_temperature, 1.0F);
        frame[codebook + 1] = token;
        previous.push_back(token);
        current = token;
        if (codebook + 1 < 15) {
            token_input = codec_embedding(current, codebook + 1).reshape({1, 1, 1, kHidden});
            bf16_upload("cp_n1_input", token_input);
            run_position(codebook + 2);
        }
    }
    auto result = torch::empty({16}, torch::kInt64);
    auto* out = result.data_ptr<int64_t>();
    for (size_t i = 0; i < frame.size(); ++i) out[i] = frame[i];
    metrics.code_predictor_time += std::chrono::duration<double>(std::chrono::steady_clock::now() - code_predictor_start).count();
    return result;
}

torch::Tensor Qwen3TtsRunner::run_codec_prefix(const torch::Tensor& codes) {
    if (codes.dim() != 2 || codes.size(1) != 16) throw std::runtime_error("Codec codes must be [frames,16]");
    const auto t = codes.size(0);
    auto code_t = codes.transpose(0, 1).unsqueeze(0).contiguous();
    auto decode = [this, &code_t, t](const std::string& prefix, int64_t start, int64_t count) {
        torch::Tensor quantized;
        for (int64_t index = 0; index < count; ++index) {
            const auto base = prefix + ".vq.layers." + std::to_string(index) + "._codebook";
            auto embeddings = codec_weights_->tensor(base + ".embedding_sum") /
                codec_weights_->tensor(base + ".cluster_usage").clamp_min(1e-5F).unsqueeze(1);
            const auto ids = code_t.select(1, start + index).reshape({-1});
            auto current = embeddings.index_select(0, ids).reshape({1, t, embeddings.size(1)}).transpose(1, 2);
            quantized = quantized.defined() ? quantized + current : current;
        }
        const auto weight = codec_weights_->tensor(prefix + ".output_proj.weight");
        return F::conv1d(quantized, weight, F::Conv1dFuncOptions().bias(torch::Tensor()).stride(1).padding(0).dilation(1).groups(1));
    };
    auto hidden = decode("quantizer.rvq_first", 0, 1) + decode("quantizer.rvq_rest", 1, 15);
    const auto pre_weight = codec_weights_->tensor("pre_conv.conv.weight");
    const auto pre_bias = codec_weights_->tensor("pre_conv.conv.bias");
    hidden = torch::constant_pad_nd(hidden, {2, 0}, 0.0);
    hidden = F::conv1d(hidden, pre_weight, F::Conv1dFuncOptions().bias(pre_bias).stride(1).padding(0).dilation(1).groups(1));
    hidden = hidden.transpose(1, 2);
    hidden = torch::matmul(hidden, codec_weights_->tensor("pre_transformer.input_proj.weight").transpose(0, 1)) +
        codec_weights_->tensor("pre_transformer.input_proj.bias");
    return as_float(hidden.squeeze(0));
}

float Qwen3TtsRunner::codec_prefix_tail_rms(
    const std::vector<std::array<int32_t, 16>>& frames
) {
    // pre_conv is causal with a width-three receptive field. Evaluating only
    // the final three codec frames preserves the exact current prefix state
    // without decoding waveform samples or transferring MLA state.
    constexpr size_t kPrefixContextFrames = 3;
    if (frames.empty()) throw std::runtime_error("Cannot measure an empty codec prefix");
    const auto begin = frames.size() > kPrefixContextFrames ? frames.size() - kPrefixContextFrames : 0;
    const auto rows = frames.size() - begin;
    std::vector<int64_t> values(rows * 16);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < 16; ++column) {
            values[row * 16 + column] = frames[begin + row][column];
        }
    }
    const auto codes = torch::from_blob(
        values.data(), {static_cast<int64_t>(rows), 16}, torch::kInt64
    ).clone();
    const auto tail = run_codec_prefix(codes).select(0, static_cast<int64_t>(rows - 1));
    return torch::sqrt(torch::mean(torch::square(tail))).item<float>();
}

#if defined(QWEN3_ENDPOINT_INCREMENTAL)
void Qwen3TtsRunner::reset_endpoint_prefix_rms_state() {
    if (!endpoint_prefix_rms_state_) {
        auto state = std::make_unique<EndpointPrefixRmsState>();
        for (size_t codebook = 0; codebook < state->normalized_codebooks.size(); ++codebook) {
            const std::string prefix = codebook == 0 ? "quantizer.rvq_first" : "quantizer.rvq_rest";
            const auto layer = codebook == 0 ? 0 : codebook - 1;
            const auto base = prefix + ".vq.layers." + std::to_string(layer) + "._codebook";
            state->normalized_codebooks[codebook] = codec_weights_->tensor(base + ".embedding_sum") /
                codec_weights_->tensor(base + ".cluster_usage").clamp_min(1e-5F).unsqueeze(1);
        }
        state->first_output_proj = codec_weights_->tensor("quantizer.rvq_first.output_proj.weight").select(2, 0);
        state->rest_output_proj = codec_weights_->tensor("quantizer.rvq_rest.output_proj.weight").select(2, 0);
        state->pre_weight = codec_weights_->tensor("pre_conv.conv.weight");
        state->pre_bias = codec_weights_->tensor("pre_conv.conv.bias");
        state->input_proj_weight = codec_weights_->tensor("pre_transformer.input_proj.weight");
        state->input_proj_bias = codec_weights_->tensor("pre_transformer.input_proj.bias");
        endpoint_prefix_rms_state_ = std::move(state);
    }
    endpoint_prefix_rms_state_->recent_projected.clear();
}

float Qwen3TtsRunner::codec_prefix_tail_rms_incremental(const std::array<int32_t, 16>& frame) {
    auto& state = *endpoint_prefix_rms_state_;
    const auto first = state.normalized_codebooks[0].select(0, frame[0]);
    auto rest = torch::zeros_like(first);
    for (size_t codebook = 1; codebook < state.normalized_codebooks.size(); ++codebook) {
        rest.add_(state.normalized_codebooks[codebook].select(0, frame[codebook]));
    }
    auto projected = torch::matmul(state.first_output_proj, first) +
        torch::matmul(state.rest_output_proj, rest);
    state.recent_projected.push_back(std::move(projected));
    while (state.recent_projected.size() > 3) state.recent_projected.pop_front();

    // Exact causal width-three pre_conv window for the newest prefix output.
    auto pre_conv = state.pre_bias.clone();
    const auto first_kernel = 3 - state.recent_projected.size();
    for (size_t index = 0; index < state.recent_projected.size(); ++index) {
        pre_conv.add_(torch::matmul(
            state.pre_weight.select(2, static_cast<int64_t>(first_kernel + index)),
            state.recent_projected[index]
        ));
    }
    const auto tail = torch::matmul(state.input_proj_weight, pre_conv) + state.input_proj_bias;
    return torch::sqrt(torch::mean(torch::square(tail))).item<float>();
}
#endif

torch::Tensor Qwen3TtsRunner::run_codec_transformer(const torch::Tensor& hidden) {
    if (hidden.dim() != 2 || hidden.size(1) != kCodecHidden || hidden.size(0) <= 0) {
        throw std::runtime_error("Invalid codec transformer input");
    }
    std::vector<torch::Tensor> outputs;
    outputs.reserve(hidden.size(0));

    // The N1 codec-transformer ELFs expose positions 0..49 only. Process the
    // latent stream in independent fixed-size windows, as we do for the raw
    // codec tail below, so generation is not capped at the first 50 frames.
    for (int64_t offset = 0; offset < hidden.size(0); offset += kCodecMax) {
        const auto frames = std::min<int64_t>(kCodecMax, hidden.size(0) - offset);
        reset_codec_caches();
        for (uint16_t token = 0; token < frames; ++token) {
            bf16_upload("codec_dec_n1_input", hidden.select(0, offset + token));
            std::vector<MLAModelWithBuffer*> models;
            models.reserve(kCodecLayers * 3);
            for (uint16_t layer = 0; layer < kCodecLayers; ++layer) {
                models.push_back(codec_pre(layer, token).get());
                models.push_back(codec_cache(layer, token).get());
                models.push_back(codec_post(layer, token).get());
            }
            Qwen3MlaStaticBatch(std::move(models)).run();
            outputs.push_back(bf16_download("codec_dec_output").reshape({1, kCodecLatent}));
        }
    }
    return torch::cat(outputs, 0);
}

torch::Tensor Qwen3TtsRunner::run_codec_transformer_n128_hybrid(const torch::Tensor& hidden) {
    if (hidden.dim() != 2 || hidden.size(1) != kCodecHidden || hidden.size(0) == 0 || hidden.size(0) > 128) {
        throw std::runtime_error("Invalid codec N128 hybrid input");
    }
    reset_codec_caches();

    // Causal N128 artifacts see the same first T positions as N1. The zero
    // suffix is invisible to those valid positions and never leaves MLA.
    auto padded = torch::zeros({1, 1, 128, kCodecHidden}, torch::kFloat32);
    padded.slice(2, 0, hidden.size(0)).copy_(hidden.reshape({1, 1, hidden.size(0), kCodecHidden}));
    bf16_upload("codec_dec_n128_input", padded);

    std::vector<MLAModelWithBuffer*> n128_models;
    n128_models.reserve((kCodecLayers - 1) * 3);
    for (uint16_t layer = 0; layer < kCodecLayers - 1; ++layer) {
        n128_models.push_back(codec_n128_pre(layer).get());
        n128_models.push_back(codec_n128_cache(layer).get());
        n128_models.push_back(codec_n128_post(layer).get());
    }
    Qwen3MlaStaticBatch(std::move(n128_models)).run();

    // The package has no N128 final-post ELF. Keep layer 7 causal/N1, but
    // bind its input and its final output directly to the N128 edge buffers.
    for (uint16_t token = 0; token < hidden.size(0); ++token) {
        Qwen3MlaStaticBatch({codec_final_pre_from_n128(token).get(),
                              codec_cache(kCodecLayers - 1, token).get(),
                              codec_final_post_from_n128(token).get()}).run();
    }
    return bf16_download("codec_dec_n128_output").slice(2, 0, hidden.size(0)).reshape({hidden.size(0), kCodecLatent});
}

std::vector<float> Qwen3TtsRunner::run_codec_tail(const torch::Tensor& hidden) {
    if (hidden.dim() != 2 || hidden.size(1) != kCodecLatent || hidden.size(0) <= 0) {
        throw std::runtime_error("Invalid codec-tail input");
    }
    if (tail_models_.empty()) {
        for (size_t i = 0; i < tail_parts_.size(); ++i) {
            tail_models_.push_back(std::make_unique<MLAModelWithBuffer>(tail_parts_[i].elf,
                std::vector<MLABufferSlice>{full(buffer("tail_edge" + std::to_string(i)))},
                std::vector<MLABufferSlice>{full(buffer("tail_edge" + std::to_string(i + 1)))}));
        }
    }
    std::vector<MLAModelWithBuffer*> models;
    models.reserve(tail_models_.size());
    for (const auto& model : tail_models_) models.push_back(model.get());
    Qwen3MlaStaticBatch tail_batch(std::move(models));

    constexpr int64_t kTailSamples = 96000;
    if (kTailSamples % kCodecMax != 0) {
        throw std::runtime_error("Codec-tail waveform length is not divisible by its fixed-frame contract");
    }
    const auto samples_per_frame = kTailSamples / kCodecMax;
    std::vector<float> result;
    result.reserve(static_cast<size_t>(hidden.size(0) * samples_per_frame));

    // The tail ELFs have a fixed 50-frame transport ABI, while the codec
    // transformer has a 1024-frame cache. Decode the completed latent stream
    // in fixed-size tail transactions rather than stopping generation at the
    // first tail boundary.
    for (int64_t offset = 0; offset < hidden.size(0); offset += kCodecMax) {
        const auto frames = std::min<int64_t>(kCodecMax, hidden.size(0) - offset);
        auto padded = torch::zeros({1, 1, kCodecMax, kCodecLatent}, torch::kFloat32);
        padded.slice(2, 0, frames).copy_(hidden.slice(0, offset, offset + frames).unsqueeze(0).unsqueeze(0));
        bf16_upload("tail_edge0", padded);
        tail_batch.run();

        auto waveform = bf16_download("tail_edge27").select(3, 0).reshape({kTailSamples})
            .slice(0, 0, frames * samples_per_frame).clamp(-1.0F, 1.0F).contiguous();
        const auto prior_size = result.size();
        result.resize(prior_size + static_cast<size_t>(waveform.numel()));
        std::memcpy(result.data() + prior_size, waveform.data_ptr<float>(),
                    static_cast<size_t>(waveform.numel()) * sizeof(float));
    }
    if (!std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("Raw MLA codec tail generated non-finite waveform");
    }
    return result;
}

void Qwen3TtsRunner::write_wav_pcm16(const std::filesystem::path& path, const std::vector<float>& samples) const {
    if (path.empty()) return;
    std::filesystem::create_directories(path.parent_path().empty() ? "." : path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("Unable to open WAV output: " + path.string());
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    out.write("RIFF", 4); write_u32(out, 36 + data_bytes); out.write("WAVE", 4);
    out.write("fmt ", 4); write_u32(out, 16); write_u16(out, 1); write_u16(out, 1); write_u32(out, 24000);
    write_u32(out, 24000 * 2); write_u16(out, 2); write_u16(out, 16);
    out.write("data", 4); write_u32(out, data_bytes);
    for (const auto value : samples) {
        const auto clipped = std::clamp(value, -1.0F, 1.0F);
        // Match libsndfile's float32-to-PCM16 conversion used by Python soundfile.write.
        const auto scaled = static_cast<int32_t>(std::floor(clipped * 32768.0F));
        const auto pcm = static_cast<int16_t>(std::clamp(scaled, -32768, 32767));
        out.write(reinterpret_cast<const char*>(&pcm), sizeof(pcm));
    }
}

void Qwen3TtsRunner::set_seed(uint64_t seed) {
    rng_.seed(seed);
}

RunResult Qwen3TtsRunner::run(const RequestOptions& request) {
    if (!initialized_) throw std::runtime_error("Qwen3 native engine is not initialized");
    constexpr uint32_t kGenerationSafetyCeiling = 512;
    if (request.max_frames == 0 || request.max_frames > kGenerationSafetyCeiling) {
        throw std::runtime_error("--max-frames must be between 1 and 512");
    }
    if (request.endpoint_silence_frames == 0) {
        throw std::runtime_error("--endpoint-silence-frames must be positive");
    }
    const auto now = [] { return std::chrono::steady_clock::now(); };
    const auto seconds = [](auto begin, auto end) { return std::chrono::duration<double>(end - begin).count(); };
    const auto e2e_start = now();
    const auto framed = std::string("<|im_start|>assistant\n") + request.prompt + "<|im_end|>\n<|im_start|>assistant\n";
    const auto input_ids = tokenizer_->encode(framed, true);
    std::vector<int64_t> input_ids_i64(input_ids.begin(), input_ids.end());
    torch::Tensor trailing, pad;
    const auto prompt_start = now();
    const auto prefill = build_prefill(input_ids, request.speaker, request.language, trailing, pad);
    RunResult result;
    result.metrics.prompt_tokens = static_cast<uint32_t>(input_ids.size());
    result.metrics.input_ids_sha256 = sha256(input_ids_i64.data(), input_ids_i64.size() * sizeof(int64_t));
    result.metrics.prefill_sha256 = bf16_sha256(prefill);
    auto hidden = run_backbone_prefill(prefill, request.speaker, request.language, result.metrics);
    result.metrics.backbone_prefill_hidden_sha256 = bf16_sha256(hidden);
    const auto prompt_end = now();
    result.metrics.prompt_time = seconds(prompt_start, prompt_end);
    result.metrics.ttft = seconds(e2e_start, prompt_end);
    std::vector<int32_t> previous_c0;
    auto c0 = select_token(torch::matmul(hidden.reshape({kHidden}), codec_head_weight_.transpose(0, 1)), previous_c0,
                           true, request.do_sample, request.top_k, request.top_p, request.temperature, request.repetition_penalty);
    previous_c0.push_back(c0);
    const bool endpoint_enabled = request.streaming_endpoint && !request.do_sample && !request.subtalker_do_sample;
    result.metrics.endpoint_enabled = endpoint_enabled;
    result.metrics.endpoint_silence_rms_threshold = request.endpoint_silence_rms;
#if defined(QWEN3_ENDPOINT_INCREMENTAL)
    if (endpoint_enabled) reset_endpoint_prefix_rms_state();
#endif
    uint32_t silence_run{};
    std::optional<double> ttf_frame;
    const auto generation_start = now();
    for (uint32_t step = 0; step < request.max_frames; ++step) {
        if (c0 == kCodecEos) break;
        const auto frame_t = run_code_predictor(hidden, c0, request, result.metrics);
        std::array<int32_t, 16> frame{};
        const auto* values = frame_t.data_ptr<int64_t>();
        for (size_t i = 0; i < frame.size(); ++i) frame[i] = static_cast<int32_t>(values[i]);
        result.frames.push_back(frame);
        if (!ttf_frame) ttf_frame = seconds(e2e_start, now());
        if (endpoint_enabled) {
#if defined(QWEN3_ENDPOINT_INCREMENTAL)
            const auto prefix_rms = codec_prefix_tail_rms_incremental(frame);
#else
            const auto prefix_rms = codec_prefix_tail_rms(result.frames);
#endif
            result.metrics.endpoint_prefix_rms.push_back(prefix_rms);
            silence_run = prefix_rms < request.endpoint_silence_rms ? silence_run + 1 : 0;
            if (silence_run >= request.endpoint_silence_frames) {
                const auto first_silent = result.frames.size() - silence_run;
                const auto retained_pad = std::min<size_t>(request.endpoint_end_pad_frames, silence_run);
                const auto retained_frames = first_silent + retained_pad;
                result.metrics.endpoint_triggered = true;
                result.metrics.endpoint_trigger_frame = step;
                result.metrics.endpoint_retained_pad_frames = static_cast<uint32_t>(retained_pad);
                result.metrics.endpoint_discarded_confirmation_frames =
                    static_cast<uint32_t>(result.frames.size() - retained_frames);
                result.metrics.generated_frames_before_endpoint = static_cast<uint32_t>(result.frames.size());
                result.frames.resize(retained_frames);
                break;
            }
        }
        const auto feedback_start = now();
        auto next = backbone_feedback(frame);
        next += step < trailing.size(0) ? trailing.select(0, step) : pad;
        const auto feedback_end = now();
        bf16_upload("backbone_n1_input_embed", next);
        const auto backbone_mla_start = now();
        run_backbone_token(backbone_position_++);
        const auto backbone_mla_end = now();
        hidden = bf16_download("backbone_n1_buffer1");
        const auto backbone_decode_end = now();
        result.metrics.backbone_feedback_time += seconds(feedback_start, feedback_end);
        result.metrics.backbone_decode_mla_time += seconds(backbone_mla_start, backbone_mla_end);
        result.metrics.backbone_decode_time += seconds(feedback_end, backbone_decode_end);
        c0 = select_token(torch::matmul(hidden.reshape({kHidden}), codec_head_weight_.transpose(0, 1)), previous_c0,
                          true, request.do_sample, request.top_k, request.top_p, request.temperature, request.repetition_penalty);
        previous_c0.push_back(c0);
    }
    if (result.frames.empty()) throw std::runtime_error("No codec frames generated");
    if (!result.metrics.endpoint_triggered) {
        result.metrics.generated_frames_before_endpoint = static_cast<uint32_t>(result.frames.size());
    }
    const auto generation_end = now();
    result.metrics.frames = static_cast<uint32_t>(result.frames.size());
    result.metrics.generation_time = seconds(generation_start, generation_end);
    result.metrics.ttf_frame = ttf_frame.value_or(result.metrics.ttft);
    std::vector<int32_t> raw_frames(result.frames.size() * 16);
    for (size_t row = 0; row < result.frames.size(); ++row) for (size_t col = 0; col < 16; ++col) raw_frames[row * 16 + col] = result.frames[row][col];
    result.metrics.frames_sha256 = sha256(raw_frames.data(), raw_frames.size() * sizeof(int32_t));
    const auto codec_start = now();
    auto codes = torch::from_blob(raw_frames.data(), {static_cast<int64_t>(result.frames.size()), 16}, torch::kInt32).to(torch::kInt64).clone();
    auto prefix = run_codec_prefix(codes);
    result.metrics.codec_prefix_sha256 = bf16_sha256(prefix);
    result.metrics.codec_n128_hybrid = request.codec_n128_hybrid;
    auto tail_input = request.codec_n128_hybrid ? run_codec_transformer_n128_hybrid(prefix) : run_codec_transformer(prefix);
    result.metrics.codec_tail_input_sha256 = bf16_sha256(tail_input);
    result.waveform = run_codec_tail(tail_input);
    result.metrics.codec_tail_chunks = static_cast<uint32_t>(
        (tail_input.size(0) + kCodecMax - 1) / kCodecMax
    );
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
