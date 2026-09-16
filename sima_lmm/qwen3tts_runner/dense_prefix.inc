// Included after TensorFile in the isolated runner's namespace.
// Same dense-prefix algorithm as the board-validated replay probe.
struct Qwen3TtsRunner::DensePrefix {
  MLABuffer first_{"prefix_first", {1, 1, 52, 256}, "bfloat16", true};
  MLABuffer rest_{"prefix_rest", {1, 1, 52, 256}, "bfloat16", true};
  MLABuffer output_{"prefix_output", {1, 1, 50, 512}, "bfloat16", true};
  MLAModelWithBuffer model_;
  explicit DensePrefix(const std::filesystem::path &elf)
      : model_(elf, {full(first_), full(rest_)}, {full(output_)}) {
    first_.allocate();
    rest_.allocate();
    output_.allocate();
    model_.load();
  }
  NativeTensor run(const TensorFile &weights,
                   const std::vector<std::array<int32_t, 16>> &codes) {
    const size_t n = codes.size();
    if (!n)
      throw std::runtime_error("Codec codes must not be empty");
    std::array<NativeTensor, 2> sums{NativeTensor({n, 256}),
                                     NativeTensor({n, 256})};
    for (size_t group = 0; group < 2; ++group) {
      const std::string prefix =
          group ? "quantizer.rvq_rest" : "quantizer.rvq_first";
      for (size_t cb = 0; cb < (group ? 15 : 1); ++cb) {
        const auto base =
            prefix + ".vq.layers." + std::to_string(cb) + "._codebook";
        const auto &embedding = weights.tensor(base + ".embedding_sum");
        const auto &usage = weights.tensor(base + ".cluster_usage");
        if (embedding.shape != std::vector<size_t>{2048, 256} ||
            usage.shape != std::vector<size_t>{2048})
          throw std::runtime_error("Unexpected codebook dimensions");
        for (size_t t = 0; t < n; ++t) {
          const auto token = codes[t][cb + (group ? 1 : 0)];
          if (token < 0 || token >= 2048)
            throw std::runtime_error("Invalid codec token");
          const float scale = 1.0F / std::max(usage.values[token], 1e-5F);
          for (size_t c = 0; c < 256; ++c)
            sums[group].values[t * 256 + c] +=
                embedding.values[token * 256 + c] * scale;
        }
      }
    }
    NativeTensor result({n, 512});
    for (size_t offset = 0; offset < n; offset += 50) {
      for (size_t group = 0; group < 2; ++group) {
        std::vector<Eigen::bfloat16> input(52 * 256, Eigen::bfloat16(0.0F));
        for (size_t t = 0; t < 52; ++t) {
          const auto source = static_cast<int64_t>(offset + t) - 2;
          if (source < 0 || source >= static_cast<int64_t>(n))
            continue;
          for (size_t c = 0; c < 256; ++c)
            input[t * 256 + c] =
                Eigen::bfloat16(sums[group].values[source * 256 + c]);
        }
        (group ? rest_ : first_).upload(input.data());
      }
      Qwen3MlaStaticBatch({&model_}).run();
      std::vector<Eigen::bfloat16> output(50 * 512);
      output_.download(output.data());
      for (size_t i = 0; i < std::min<size_t>(50, n - offset) * 512; ++i)
        result.values[offset * 512 + i] = static_cast<float>(output[i]);
    }
    return result;
  }
};
