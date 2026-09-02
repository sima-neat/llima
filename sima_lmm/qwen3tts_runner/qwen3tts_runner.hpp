#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "mla_buffer.hpp"
#include "mla_model.hpp"
#include "numpy_pcg64.hpp"
#include "tokenizer.hpp"

#include <torch/torch.h>

namespace simaai::llima::qwen3tts {

struct RequestOptions {
    std::string prompt;
    std::string speaker{"Vivian"};
    std::string language{"English"};
    uint32_t seed{1};
    uint32_t max_frames{50};
    bool do_sample{false};
    uint32_t top_k{50};
    float top_p{1.0F};
    float temperature{0.9F};
    float repetition_penalty{1.05F};
    uint32_t subtalker_top_k{50};
    float subtalker_top_p{1.0F};
    float subtalker_temperature{0.9F};
    bool subtalker_do_sample{true};
    std::string prefill_mode{"prefix_kv"};
    // Batch codec-decoder layers 0..6 with the compiled causal
    // N128 ELFs, then retain exact N1 execution for layer 7.
    bool codec_n128_hybrid{};
    // The talker can continue emitting codec tokens after the causal codec
    // prefix has entered terminal silence. EOS remains the primary stop.
    bool streaming_endpoint{true};
    float endpoint_silence_rms{0.040F};
    uint32_t endpoint_silence_frames{5};
    uint32_t endpoint_end_pad_frames{2};
    std::filesystem::path output_wav;
};

struct RunMetrics {
    uint32_t prompt_tokens{};
    uint32_t frames{};
    double prompt_time{};
    double generation_time{};
    double code_predictor_time{};
    double code_predictor_mla_time{};
    double backbone_feedback_time{};
    double backbone_decode_time{};
    double backbone_decode_mla_time{};
    double codec_to_wav_time{};
    double wav_write_time{};
    double ttft{};
    double ttf_frame{};
    double ttfa{};
    double e2e_time{};
    uint32_t codec_tail_uploads{};
    uint32_t codec_tail_downloads{};
    uint32_t codec_tail_chunks{};
    bool prefix_kv_reused{};
    bool prefix_kv_device_resident{};
    bool codec_n128_hybrid{};
    uint32_t prefix_kv_static_tokens{};
    uint32_t generated_frames_before_endpoint{};
    bool endpoint_enabled{};
    bool endpoint_triggered{};
    uint32_t endpoint_trigger_frame{};
    uint32_t endpoint_retained_pad_frames{};
    uint32_t endpoint_discarded_confirmation_frames{};
    float endpoint_silence_rms_threshold{};
    std::vector<float> endpoint_prefix_rms;
    std::string frames_sha256;
    // Diagnostic-only boundaries for cross-runtime equivalence checks.
    std::string input_ids_sha256;
    std::string prefill_sha256;
    std::string backbone_prefill_hidden_sha256;
    std::string cp_initial_input_sha256;
    std::string cp_codebook0_input_sha256;
    std::string cp_codebook0_logits_sha256;
    std::string codec_prefix_sha256;
    std::string codec_tail_input_sha256;
    std::filesystem::path wav_path;
};

struct RunResult {
    std::vector<std::array<int32_t, 16>> frames;
    std::vector<float> waveform;
    RunMetrics metrics;
};

class Qwen3TtsRunner {
  public:
    Qwen3TtsRunner(std::filesystem::path model_dir, std::filesystem::path components_dir,
                          bool preload_models);
    ~Qwen3TtsRunner();
    void initialize();
    void finalize();
    void set_seed(uint64_t seed);
    RunResult run(const RequestOptions& request);

  private:
    struct TensorFile;
    struct TailPart;
#if defined(QWEN3_ENDPOINT_INCREMENTAL)
    struct EndpointPrefixRmsState;
#endif
    struct ModelKey {
        uint16_t layer{};
        uint16_t token{};
        bool operator<(const ModelKey& other) const {
            return std::tie(layer, token) < std::tie(other.layer, other.token);
        }
    };
    using ModelPtr = std::unique_ptr<MLAModelWithBuffer>;

    void load_host_weights();
    void define_buffers();
    void initialize_static_buffers();
    void build_and_preload_models();
    void validate_tail_contract();
    void validate_elfs() const;
    void reset_caches();
    void reset_cp_caches();
    void reset_codec_caches();
    torch::Tensor text_project(const torch::Tensor& input) const;
    torch::Tensor build_prefill(const std::vector<uint32_t>& ids, const std::string& speaker,
                                const std::string& language, torch::Tensor& trailing,
                                torch::Tensor& pad) const;
    torch::Tensor run_backbone_prefill(const torch::Tensor& prefill, const std::string& speaker,
                                       const std::string& language, RunMetrics& metrics);
    void run_backbone_token(uint16_t token);
    torch::Tensor run_code_predictor(const torch::Tensor& hidden, int32_t c0, const RequestOptions& request,
                                     RunMetrics& metrics);
    torch::Tensor run_codec_prefix(const torch::Tensor& codes);
    float codec_prefix_tail_rms(const std::vector<std::array<int32_t, 16>>& frames);
#if defined(QWEN3_ENDPOINT_INCREMENTAL)
    void reset_endpoint_prefix_rms_state();
    float codec_prefix_tail_rms_incremental(const std::array<int32_t, 16>& frame);
#endif
    torch::Tensor run_codec_transformer(const torch::Tensor& hidden);
    torch::Tensor run_codec_transformer_n128_hybrid(const torch::Tensor& hidden);
    std::vector<float> run_codec_tail(const torch::Tensor& hidden);
    int32_t select_token(torch::Tensor logits, const std::vector<int32_t>& previous, bool suppress,
                         bool sample, uint32_t top_k, float top_p, float temperature,
                         float repetition_penalty);
    torch::Tensor codec_embedding(int32_t token, uint32_t codebook) const;
    torch::Tensor backbone_feedback(const std::array<int32_t, 16>& frame) const;
    torch::Tensor bf16_download(const std::string& name) const;
    void bf16_upload(const std::string& name, const torch::Tensor& value) const;
#if defined(QWEN3_CP_SPLIT_HEADS)
    // Archived one-head ELFs expose FP32 edge tensors.  int32 storage has the
    // required four-byte ABI in MLABuffer and carries FP32 byte values here.
    void fp32_upload(const std::string& name, const torch::Tensor& value) const;
#endif
    MLABuffer& buffer(const std::string& name) const;
    ModelPtr& backbone_pre(uint16_t layer, uint16_t token);
    ModelPtr& backbone_cache(uint16_t layer, uint16_t token);
    ModelPtr& backbone_post(uint16_t layer, uint16_t token);
    ModelPtr& cp_pre(uint16_t layer, uint16_t token);
    ModelPtr& cp_cache(uint16_t layer, uint16_t token);
    ModelPtr& cp_post(uint16_t layer, uint16_t token);
#if defined(QWEN3_CP_SPLIT_HEADS)
    ModelPtr& cp_head(uint16_t codebook);
#endif
    ModelPtr& codec_pre(uint16_t layer, uint16_t token);
    ModelPtr& codec_cache(uint16_t layer, uint16_t token);
    ModelPtr& codec_post(uint16_t layer, uint16_t token);
    ModelPtr& codec_n128_pre(uint16_t layer);
    ModelPtr& codec_n128_cache(uint16_t layer);
    ModelPtr& codec_n128_post(uint16_t layer);
    ModelPtr& codec_final_pre_from_n128(uint16_t token);
    ModelPtr& codec_final_post_from_n128(uint16_t token);
    void write_wav_pcm16(const std::filesystem::path& path, const std::vector<float>& samples) const;

    std::filesystem::path model_dir_, components_dir_, elf_dir_;
#if defined(QWEN3_CP_SPLIT_HEADS)
    std::filesystem::path cp_head_elf_dir_;
#endif
    bool preload_models_{};
    bool initialized_{};
    uint16_t backbone_position_{};
    std::unique_ptr<Tokenizer> tokenizer_;
    std::map<std::string, std::unique_ptr<MLABuffer>> buffers_;
    std::map<ModelKey, ModelPtr> backbone_pre_, backbone_cache_, backbone_post_;
    std::map<ModelKey, ModelPtr> cp_pre_, cp_cache_, cp_post_;
#if defined(QWEN3_CP_SPLIT_HEADS)
    std::map<uint16_t, ModelPtr> cp_head_;
#endif
    std::map<ModelKey, ModelPtr> codec_pre_, codec_cache_, codec_post_;
    std::map<ModelKey, ModelPtr> codec_n128_pre_, codec_n128_cache_, codec_n128_post_;
    std::map<ModelKey, ModelPtr> codec_final_pre_from_n128_, codec_final_post_from_n128_;
    std::vector<ModelPtr> tail_models_;
    std::vector<TailPart> tail_parts_;
    std::map<std::pair<std::string, std::string>, std::vector<std::vector<uint8_t>>> prefix_snapshots_;
    std::optional<std::pair<std::string, std::string>> resident_prefix_key_;
    NumpyPcg64 rng_;
    std::unique_ptr<TensorFile> backbone_weights_, cp_weights_, codec_weights_, text_projection_weights_, codec_head_weights_;
#if defined(QWEN3_ENDPOINT_INCREMENTAL)
    std::unique_ptr<EndpointPrefixRmsState> endpoint_prefix_rms_state_;
#endif
    torch::Tensor text_embeddings_, codec_embeddings_, codec_head_weight_, text_fc1_w_, text_fc1_b_, text_fc2_w_, text_fc2_b_;
    nlohmann::json talker_config_;
};

} // namespace simaai::llima::qwen3tts
