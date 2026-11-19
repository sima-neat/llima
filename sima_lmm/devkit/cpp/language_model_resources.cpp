#include <cnpy.h>
#include <fmt/std.h>
#include <fstream>

#include "language_model_state.hpp"

namespace simaai::llima {
namespace {
// Reject trailing bytes as well as truncated embedding tables.
void load_whole_file(MLABuffer &buffer, std::istream &stream,
                     const std::string &name) {
  buffer.load_stream(stream);
  if (stream.peek() != std::char_traits<char>::eof()) {
    throw std::runtime_error(
        fmt::format("Invalid size for {}: file is larger than buffer {}", name,
                    buffer.get_name()));
  }
}
} // namespace

std::shared_ptr<LanguageResources>
LanguageResources::acquire(const std::filesystem::path &path,
                           std::shared_ptr<FileProvider> files) {
  auto package = ModelResources<VlmConfig>::acquire(path, std::move(files));
  static std::mutex mutex;
  static std::map<std::pair<ModelResources<VlmConfig> *, EmbeddingOffloadMode>,
                  std::weak_ptr<LanguageResources>>
      resources;
  std::lock_guard lock(mutex);
  auto &weak = resources[{package.get(), embedding_offload_mode()}];
  if (auto shared = weak.lock())
    return shared;
  auto shared = std::make_shared<LanguageResources>(std::move(package));
  weak = shared;
  return shared;
}

size_t LanguageModel::shared_resource_bytes() const {
  size_t bytes = 0;
  for (const auto &[name, buffer] : _resources->buffers)
    bytes += buffer.get_allocation_size();
  return bytes;
}

size_t LanguageModel::workspace_memory_bytes() const {
  size_t bytes = 0;
  for (const auto &[name, buffer] : _workspace->buffers)
    bytes += buffer.get_allocation_size();
  return bytes;
}

void LanguageResources::prepare_embeddings() {
  if (ready || cfg.lm_cfg.is_speculative_draft())
    return;
  const auto mode = embedding_offload_mode();
  if (mode == EmbeddingOffloadMode::Off) {
    package->logger->info("Embedding offload disabled: resident DRAM");
    return;
  }
  const auto normal = package->devkit_dir / (cfg.language_model_name + "_embeddings.bin");
  const auto per_layer = package->devkit_dir / (cfg.language_model_name + "_per_layer_embeddings.bin");
  const bool has_per_layer = cfg.model_type == "vlm-gemma4" && cfg.lm_cfg.hidden_size_per_layer_input > 0;
  if (mode != EmbeddingOffloadMode::On &&
      (!std::filesystem::is_regular_file(normal) ||
       (has_per_layer && !std::filesystem::is_regular_file(per_layer)))) {
    package->logger->info("Embedding offload auto: resident DRAM (raw tables are unavailable)");
    return;
  }
  auto offload = std::make_shared<EmbeddingOffload>();
  const size_t elem = cfg.pipeline_cfg.quantize_embeddings ? 1 : 2;
  const size_t vocab = cfg.lm_cfg.token_cfg.vocab_size;
  offload->normal = std::make_unique<DiskEmbeddingTable>(normal, vocab, cfg.lm_cfg.hidden_size * elem);
  if (has_per_layer)
    offload->per_layer = std::make_unique<DiskEmbeddingTable>(
        per_layer, vocab, static_cast<size_t>(cfg.lm_cfg.num_hidden_layers) *
                              cfg.lm_cfg.hidden_size_per_layer_input * elem);
  embedding_offload = std::move(offload);
  package->logger->info("Embedding offload enabled (scales remain in DRAM)");
  if (!embedding_file_on_nvme(normal) || (has_per_layer && !embedding_file_on_nvme(per_layer))) {
    package->logger->warn("Embedding offload is enabled on non-NVMe storage; performance may be reduced. "
                          "Set SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off to keep embeddings in DRAM.");
  }
}

void LanguageResources::load() {
  if (ready)
    return;
  const bool draft_model = cfg.lm_cfg.is_speculative_draft();
  // Upload language embeddings (drafts use the target's embeddings, so skip).
  const bool is_eagle3_draft = cfg.lm_cfg.is_eagle3_draft();
  if (!draft_model) {
    if (!embedding_offload) {
      const auto embeddings_bin =
          "devkit/" + cfg.language_model_name + "_embeddings.bin";
      // The .bin file is optional. Use exists() to check softly: if it
      // is there, exists() pulls it and we use it; if it is missing we
      // fall back to the required .npy below with get_path(), which
      // pulls it and fails loudly if it is not there either. Both go
      // through package->files, so PCIe and disk take the same path. Do NOT use
      // get_path() for the optional check — it throws when the file is
      // missing.
      if (package->files->exists(embeddings_bin)) {
        load_whole_file(buffers.at("embeddings"),
                        *package->files->open_stream(embeddings_bin),
                        embeddings_bin);
      } else {
        // Compatibility with packages generated before raw embedding files were
        // introduced.
        const auto embeddings_npy =
            "devkit/" + cfg.language_model_name + "_embeddings.npy";
        auto embeddings_tensor =
            cnpy::npy_load(package->files->get_path(embeddings_npy));
        // cnpy has read the whole file into memory, so the disk copy
        // is no longer needed (over PCIe this frees recv_root space).
        package->files->release(embeddings_npy);
        buffers.at("embeddings").upload(embeddings_tensor.data<void>());
      }
    }
    if (cfg.pipeline_cfg.quantize_embeddings) {
      const auto scale_file =
          "devkit/" + cfg.language_model_name + "_embedding_scales.bin";
      load_whole_file(buffers.at("embedding_scales"),
                      *package->files->open_stream(scale_file), scale_file);
    }
  } else if (is_eagle3_draft) {
    // Load d2t mapping (int64 in npy, narrows to int32 — values fit easily).
    auto d2t_tensor =
        cnpy::npy_load(package->files->get_path("devkit/d2t.npy"));
    package->files->release("devkit/d2t.npy"); // already in memory
    const int64_t *src = d2t_tensor.data<int64_t>();
    const size_t n = d2t_tensor.num_vals;
    d2t.resize(n);
    for (size_t i = 0; i < n; ++i) {
      d2t[i] = static_cast<int32_t>(src[i]);
    }
    package->logger->info("Loaded d2t mapping with {} entries", d2t.size());
  } else if (cfg.lm_cfg.speculative_decoding_cfg &&
             cfg.lm_cfg.speculative_decoding_cfg->use_ordered_embeddings) {
    const auto ordering_file_name =
        package->files->get_path("devkit/gemma4_token_ordering.npy");
    const auto ordering_tensor = cnpy::npy_load(ordering_file_name);
    package->files->release("devkit/gemma4_token_ordering.npy");
    const size_t vocab_size = cfg.lm_cfg.token_cfg.vocab_size;
    if (ordering_tensor.word_size != sizeof(int64_t) ||
        ordering_tensor.num_vals != vocab_size) {
      throw std::runtime_error(fmt::format(
          "Invalid Gemma4 MTP token ordering in {}: expected {} int64 entries",
          ordering_file_name.string(), vocab_size));
    }
    if (cfg.lm_cfg.speculative_decoding_cfg->num_centroids == 0 ||
        vocab_size % cfg.lm_cfg.speculative_decoding_cfg->num_centroids != 0 ||
        cfg.lm_cfg.speculative_decoding_cfg->centroid_intermediate_top_k == 0 ||
        cfg.lm_cfg.speculative_decoding_cfg->centroid_intermediate_top_k >
            cfg.lm_cfg.speculative_decoding_cfg->num_centroids) {
      throw std::runtime_error(
          "Invalid Gemma4 MTP ordered-embedding configuration");
    }

    const int64_t *source = ordering_tensor.data<int64_t>();
    std::vector<uint8_t> seen(vocab_size, 0);
    gemma4_token_ordering.resize(vocab_size);
    for (size_t index = 0; index < vocab_size; ++index) {
      const int64_t token_id = source[index];
      if (token_id < 0 || static_cast<size_t>(token_id) >= vocab_size ||
          seen[static_cast<size_t>(token_id)] != 0) {
        throw std::runtime_error(fmt::format(
            "Invalid Gemma4 MTP token ordering entry at index {}", index));
      }
      seen[static_cast<size_t>(token_id)] = 1;
      gemma4_token_ordering[index] = static_cast<uint32_t>(token_id);
    }
    package->logger->info("Loaded Gemma4 MTP token ordering with {} entries",
                          gemma4_token_ordering.size());
  }

  // Upload attention sinks (gpt_oss): broadcast each layer's per-head vector
  // across its buffer rows. Raw bf16, shape (num_layers, num_heads).
  if (cfg.lm_cfg.uses_attention_sinks()) {
    const auto sinks_asset = "devkit/" + cfg.language_model_name + "_sinks.bin";
    const auto sinks_file_name = package->files->get_path(sinks_asset);
    std::ifstream sinks_file(sinks_file_name, std::ios::binary);
    if (!sinks_file) {
      throw std::runtime_error(fmt::format("Missing attention sinks file: {}",
                                           sinks_file_name.string()));
    }
    const uint16_t num_heads = cfg.lm_cfg.attn_cfg.num_attention_heads;
    std::vector<uint16_t> sinks( // raw bf16, 2 bytes/elem
        static_cast<size_t>(cfg.lm_cfg.num_hidden_layers) * num_heads);
    // A stale or truncated file would leave entries zeroed, and a zero sink is
    // not neutral: it is a logit in the softmax, so it silently reweights
    // attention. Require an exact match.
    const auto expected_bytes =
        static_cast<std::uintmax_t>(sinks.size() * sizeof(uint16_t));
    const auto actual_bytes = std::filesystem::file_size(sinks_file_name);
    if (actual_bytes != expected_bytes) {
      throw std::runtime_error(fmt::format(
          "Attention sinks file {} has {} bytes, expected {} for {} layers x "
          "{} heads. "
          "The file is stale or truncated; recompile the model.",
          sinks_file_name.string(), actual_bytes, expected_bytes,
          static_cast<unsigned>(cfg.lm_cfg.num_hidden_layers), num_heads));
    }
    sinks_file.read(reinterpret_cast<char *>(sinks.data()),
                    static_cast<std::streamsize>(expected_bytes));
    sinks_file.close();
    package->files->release(sinks_asset);
    for (uint8_t layer = 0; layer < cfg.lm_cfg.num_hidden_layers; ++layer) {
      auto &buf = buffers.at(fmt::format("sinks_l{}", layer));
      const size_t rows = buf.get_shape()[0];
      std::vector<uint16_t> host(rows * num_heads);
      for (size_t r = 0; r < rows; ++r) {
        std::copy_n(sinks.data() + static_cast<size_t>(layer) * num_heads,
                    num_heads, host.begin() + r * num_heads);
      }
      buf.upload(host.data());
    }
  }

  // Upload freq real and imag.
  auto rope_table = calc_freq_real_imag(
      cfg.pipeline_cfg.max_num_tokens,
      cfg.lm_cfg.rope_cfg.rope_scaling.rope_type,
      cfg.lm_cfg.rope_cfg.rope_theta,
      cfg.lm_cfg.rope_cfg.get_rope_dimension_count("full_attention"),
      cfg.lm_cfg.attn_cfg.get_head_dim("full_attention"),
      cfg.lm_cfg.rope_cfg.rope_scaling);
  // Save a pristine host copy for sourcing tree-RoPE rows without re-reading
  // the (mutated) device buffer.
  global_freq_host = rope_table;
  if (cfg.lm_cfg.attn_cfg.swa_enable) {
    // A distinct local base freq (Gemma3) means an unscaled local rope; when it
    // matches rope_theta the sliding layers share the global scaling (gpt-oss
    // YaRN).
    const std::string local_rope_type =
        cfg.lm_cfg.rope_cfg.rope_local_base_freq !=
                cfg.lm_cfg.rope_cfg.rope_theta
            ? "default"
            : cfg.lm_cfg.rope_cfg.rope_scaling.rope_type;
    rope_table = calc_freq_real_imag(
        cfg.pipeline_cfg.max_num_tokens, local_rope_type,
        cfg.lm_cfg.rope_cfg.rope_local_base_freq,
        cfg.lm_cfg.rope_cfg.get_rope_dimension_count("sliding_attention"),
        cfg.lm_cfg.attn_cfg.get_head_dim("sliding_attention"),
        cfg.lm_cfg.rope_cfg.rope_scaling);
    // Save a pristine host copy for sourcing tree-RoPE rows in spec verify
    // (mirrors global_freq_host).
    local_freq_host = rope_table;
  }
  if (cfg.lm_cfg.rope_cfg.rope_scaling.rope_type == "mrope") {
    assert(!cfg.lm_cfg.attn_cfg.swa_enable);
    master_rope_table = std::move(rope_table);
  }

  if ((cfg.model_type == "vlm-gemma4" &&
       cfg.lm_cfg.hidden_size_per_layer_input > 0)) {
    load_per_layer_embeddings();
  }

  ready = true;
}

void LanguageResources::load_per_layer_embeddings() {
  const auto file_rel =
      "devkit/" + cfg.language_model_name + "_per_layer_embeddings.bin";
  const auto file_name = package->files->get_path(file_rel);
  const size_t vocab_size = cfg.lm_cfg.token_cfg.vocab_size;
  const size_t out_dim = static_cast<size_t>(cfg.lm_cfg.num_hidden_layers) *
                         cfg.lm_cfg.hidden_size_per_layer_input;
  const size_t elem_size = cfg.pipeline_cfg.quantize_embeddings
                               ? sizeof(int8_t)
                               : sizeof(Eigen::bfloat16);
  const size_t token_row_size = out_dim * elem_size;
  const size_t expected_size = vocab_size * token_row_size;
  const size_t actual_size = std::filesystem::file_size(file_name);
  if (actual_size != expected_size) {
    throw std::runtime_error(
        fmt::format("Invalid size for {}: expected {} bytes, got {}", file_name,
                    expected_size, actual_size));
  }

  // Resident mode streams complete tables; offload mode has no table shards.
  std::unique_ptr<std::istream> stream;
  if (!embedding_offload)
    stream = package->files->open_stream(file_rel);
  for (auto *shard : per_layer_embedding_shards) {
    if (shard->get_buf_len() != shard->get_shape()[0] * token_row_size) {
      throw std::runtime_error(fmt::format(
          "Per-layer embedding rows require unsupported MLA padding: {}",
          shard->get_name()));
    }
    shard->load_stream(*stream);
  }

  if (cfg.pipeline_cfg.quantize_embeddings) {
    const auto scale_file =
        "devkit/" + cfg.language_model_name + "_per_layer_embedding_scales.bin";
    load_whole_file(buffers.at("per_layer_embedding_scales"),
                    *package->files->open_stream(scale_file), scale_file);
  }
}

} // namespace simaai::llima
