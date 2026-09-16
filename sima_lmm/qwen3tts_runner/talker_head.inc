// Actual MLA-driven projection: no CPU logits are computed on the MLA path.
struct Qwen3TtsRunner::TalkerHead {
  MLABuffer input{"talker_head_input", {1, 1, 1, 1024}, "bfloat16", true};
  MLABuffer output{"talker_head_output", {1, 1, 1, 3072}, "bfloat16", true};
  MLAModelWithBuffer model;
  explicit TalkerHead(const std::filesystem::path &elf)
      : model(elf, {full(input)}, {full(output)}) {
    input.allocate();
    output.allocate();
    model.load();
  }
  NativeTensor run(const NativeTensor &hidden) {
    if (hidden.numel() != 1024)
      throw std::runtime_error("Expected 1024 hidden values");
    std::vector<Eigen::bfloat16> data(1024);
    for (size_t i = 0; i < 1024; ++i)
      data[i] = Eigen::bfloat16(hidden.values[i]);
    input.upload(data.data());
    Qwen3MlaStaticBatch({&model}).run();
    data.resize(3072);
    output.download(data.data());
    NativeTensor result({1, 3072});
    for (size_t i = 0; i < 3072; ++i)
      result.values[i] = static_cast<float>(data[i]);
    return result;
  }
};

NativeTensor Qwen3TtsRunner::project_talker(const NativeTensor &hidden,
                                            RunMetrics &metrics) {
  const auto start = std::chrono::steady_clock::now();
  if (!talker_head_)
    talker_head_ = std::make_unique<TalkerHead>(talker_head_elf_);
  const auto result = talker_head_->run(hidden);
  if (!std::all_of(result.values.begin(), result.values.end(),
                   [](float v) { return std::isfinite(v); }))
    throw std::runtime_error("Nonfinite talker logits");
  metrics.talker_projection_time +=
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  ++metrics.talker_projection_calls;
  return result;
}
