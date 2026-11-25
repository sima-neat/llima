#include "language/embedding_offload.hpp"
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>
#include <string_view>

#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include "language/speculative/eagle_helpers.hpp"
#include "language/speculative/gemma4_mtp_helpers.hpp"
#include "language/language_model.hpp"
#include "language/layer_executor.hpp"
#include "language/language_model_state.hpp"
#include "language/request_scope.hpp"
#include "utils.hpp"

namespace simaai {
namespace llima {

LanguageModel::~LanguageModel() { _finalize(); }

void LanguageModel::stop_model() { _run->interrupts.fetch_add(1, std::memory_order_relaxed); }

const std::set<uint32_t>& LanguageModel::get_stop_token_ids() const { return _run->limits().stop_token_ids; }

void LanguageModel::set_kv_cache_len(uint16_t len) { _cache->kv_cache_len = len; }


LanguageModel::LanguageModel(
    std::filesystem::path model_path,
    std::set<uint32_t> stop_token_ids,
    std::optional<uint32_t> image_token_id,
    std::optional<uint32_t> pad_token_id,
    TextStreamer& text_streamer,
    std::shared_ptr<FileProvider> file_provider
) : _resources(LanguageAssets::acquire(model_path, std::move(file_provider))),
    _cfg(_resources->cfg),
    _elf_dir(_resources->package->elf_dir),
    _devkit_dir(_resources->package->devkit_dir),
    _files(_resources->package->files),
    _logger(_resources->package->logger),
    _cache(std::make_unique<SequenceCache>()),
    _workspace(std::make_unique<LanguageWorkspace>()),
    _run(std::make_unique<RequestControl>(std::move(stop_token_ids),
        _cfg.pipeline_cfg.max_num_tokens, text_streamer)),
    _image_token_id(image_token_id),
    _pad_token_id(pad_token_id)
{
    std::lock_guard execution_lock(RuntimeContext::instance().execution_mutex);
    if (
        _cfg.pipeline_cfg.max_num_tokens == 0
        || _cfg.pipeline_cfg.max_num_tokens % MAX_NUM_TOKENS_ALIGNMENT
    ) {
        throw std::runtime_error(
            "max_num_tokens must be a positive multiple of 1024"
        );
    }
    if (
        _cfg.lm_cfg.is_spec_decode()
        && _cfg.lm_cfg.speculative_decoding_cfg.value().is_eagle3()
        && _cfg.lm_cfg.attn_cfg.swa_enable
    ) {
        throw std::runtime_error(
            "EAGLE3 speculative decoding does not support sliding-window attention"
        );
    }

    _use_group_token_models = (
        _cfg.pipeline_cfg.input_token_group_offsets.has_value()
        && _cfg.pipeline_cfg.input_token_group_offsets.value().size() > 0
        // Gemma4 MTP assistants only execute their speculative-width path. Their
        // serialized pipeline config retains target group offsets, but the compiler
        // intentionally does not emit grouped assistant pre/cache/post models.
        && !_cfg.lm_cfg.is_gemma4_mtp_draft()
    );
    _need_argmax = _cfg.lm_cfg.lm_head_num_splits > 1 || _cfg.pipeline_cfg.return_logits;

    _init_cache_layout();

    // EAGLE3: build the constant tree_mask_init (eye(topk)) once. Only the
    // draft uses it (consumed in topk_generate's depth loop seed and concat).
    if (_cfg.lm_cfg.is_eagle3_draft()) {
        const int topk = _cfg.lm_cfg.speculative_decoding_cfg.value().speculative_budget;
        _workspace->eagle3_tree_mask_init.data = std::vector<std::vector<std::vector<std::vector<float>>>>(
            1, std::vector<std::vector<std::vector<float>>>(
                1, std::vector<std::vector<float>>(
                    topk, std::vector<float>(topk, 0.0f)
                )
            )
        );
        for (int i = 0; i < topk; ++i) _workspace->eagle3_tree_mask_init.data[0][0][i][i] = 1.0f;

        _workspace->eagle3_position_ids.assign(topk, 0);
    }

    _initialize();
}


void LanguageModel::_initialize() {
    _logger->info("Language model initialize starting ...");
    _resources->prepare_embeddings();
    _define_buffers();
    for (auto* buffers : {&_resources->buffers, &_workspace->buffers}) {
        for (auto& [name, buffer] : *buffers) buffer.try_allocate();
    }

    for (auto& [name, buffer] : _cache->buffers) {
        buffer.allocate();
        buffer.clear();
    }

    // Define and load the models in parallel.
    _define_models();
    MLAModelWithBuffer::load_all_models(_elf_dir / _cfg.language_model_name, _files.get());

    // Resolve the MoE host round-trip handles once (buffers allocated above).
    if (_cfg.lm_cfg.is_moe()) {
        _init_moe_host_cache();
    }

    _resources->load();

    get_buffer("global_freq_real").upload(_resources->global_freq_host.re.data());
    get_buffer("global_freq_imag").upload(_resources->global_freq_host.im.data());
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        get_buffer("local_freq_real").upload(_resources->local_freq_host.re.data());
        get_buffer("local_freq_imag").upload(_resources->local_freq_host.im.data());
    }

    // Upload the future token mask.
    const uint16_t max_future_token_mask_size = _get_max_future_token_mask_size();
    if (max_future_token_mask_size > 1) {
        std::vector<Eigen::bfloat16> future_token_mask(
            max_future_token_mask_size - 1,
            std::numeric_limits<Eigen::bfloat16>::lowest()
        );
        MLABuffer& future_token_mask_buf = get_buffer("future_token_mask");
        future_token_mask_buf.clear();
        // The pointwise Gemma4 MTP draft has exactly one max_num_tokens-wide
        // mask row, with no legacy tail space. Its runtime mask is populated
        // immediately before every draft execution.
        if (!_cfg.lm_cfg.is_gemma4_mtp_draft()) {
            future_token_mask_buf.upload_raw(
                future_token_mask.data(),
                _cfg.pipeline_cfg.max_num_tokens * 2,
                future_token_mask.size() * 2
            );
        }
    }
    _logger->info("Language model initialize completed");
}

bool LanguageModel::_has_linear_attention_layers() const {
    return std::find(
        _cfg.lm_cfg.layer_types.begin(),
        _cfg.lm_cfg.layer_types.end(),
        LayerType::LinearAttention
    ) != _cfg.lm_cfg.layer_types.end();
}


const LinearAttentionConfig& LanguageModel::_linear_attn_cfg() const {
    if (!_cfg.lm_cfg.linear_attn_cfg.has_value()) {
        throw std::runtime_error("linear_attention layers require linear_attn_cfg");
    }
    const auto& linear_cfg = _cfg.lm_cfg.linear_attn_cfg.value();
    if (linear_cfg.conv_kernel_dim <= 1) {
        throw std::runtime_error("linear_attn_cfg.conv_kernel_dim must be greater than 1");
    }
    return linear_cfg;
}


std::optional<std::vector<uint32_t>> LanguageModel::run_model(
    std::span<const uint32_t> input_token_ids, const TokenRunOptions& options
) {
    if (options.stop.stop_requested()) {
        if (options.performance) *options.performance = {};
        return std::nullopt;
    }
    RequestScope request(*this, nullptr, {}, options.stop, options.stop_token_ids);
    create_input_buffers(input_token_ids);
    auto output = _generate_tokens(request, input_token_ids, options.max_total_tokens,
        options.timer_ttft, options.performance, options.stable_prefix_token_count);
    request.finish();
    return request.cancelled() ? std::nullopt : output;
}

std::optional<std::vector<uint32_t>> LanguageModel::_generate_tokens(
    RequestScope& request, std::span<const uint32_t> ids, std::optional<uint16_t> limit,
    std::optional<ChronoTimer> timer, GenerationPerformanceResult* performance, uint16_t stable_prefix
) {
    _run->require_active_request();
    if (request.cancelled()) {
        if (performance) *performance = {};
        return std::nullopt;
    }
    auto& max_tokens = _run->limits().max_num_tokens;
    if (limit) max_tokens = std::min(*limit, _cfg.pipeline_cfg.max_num_tokens);
    auto* draft = request.draft();
    if (!draft) return _run_model_ordinary(ids, timer, stable_prefix, performance);
    const auto& spec = *_cfg.lm_cfg.speculative_decoding_cfg;
    if (spec.is_gemma4_mtp())
        return _run_model_gemma4_mtp(*draft, ids, max_tokens, timer, performance);
    if (spec.is_dflash())
        return _run_model_dflash_speculative_decoding(*draft, ids, max_tokens, timer,
                                                      performance, stable_prefix);
    return _run_model_eagle3_speculative_decoding(*draft, ids, max_tokens, timer, performance);
}

std::optional<std::vector<uint32_t>> LanguageModel::_run_model_ordinary(
    std::span<const uint32_t> input_token_ids, std::optional<ChronoTimer> timer_ttft,
    uint16_t stable_prefix_token_count, GenerationPerformanceResult* performance
) {
    _run->require_active_request();
    if (performance) *performance = {};
    auto* durations = performance ? &performance->token_durations : nullptr;

    std::optional<std::vector<uint32_t>> output_token_ids{};

    // If the input is already greater than the cache size, no need to run the model.
    if (input_token_ids.size() > _run->limits().max_num_tokens) {
        _notify_cache_full();
        output_token_ids = std::vector<uint32_t>();
    } else {
        _init_state_checkpoint_capture(input_token_ids.size(), stable_prefix_token_count);

        // Create input embeds from input token ids and image embeds.
        auto num_cached_tokens = _set_input_text_embeds(input_token_ids);

        // Prefill.
        auto token_id = _prefill(input_token_ids, num_cached_tokens, timer_ttft, durations);
        const auto output_token_id_begin = input_token_ids.size();
        if (_run->cancelled()) {
            return std::nullopt;
        }
        if (input_token_ids.size() == _run->limits().max_num_tokens) {
            _notify_cache_full();
            output_token_ids = std::vector<uint32_t>();
        } else if (_run->limits().stop_token_ids.contains(token_id)) {
            _notify_stop();
            output_token_ids = std::vector<uint32_t>{token_id};
        } else {
            // Decode.
            _decode(input_token_ids.size(), token_id, durations);
            output_token_ids = std::vector<uint32_t>(
                _cache->cached_token_ids.begin() + output_token_id_begin, _cache->cached_token_ids.end()
            );
        }
    }

    if (performance) performance->generated_tokens = durations->size();
    return output_token_ids;
}


std::optional<std::vector<uint32_t>> LanguageModel::run_model_speculative_decoding(
    LanguageModel& draft_lm, std::span<const uint32_t> input_token_ids,
    const TokenRunOptions& options
) {
    if (options.stop.stop_requested()) {
        if (options.performance) *options.performance = {};
        return std::nullopt;
    }
    RequestScope request(*this, &draft_lm, {}, options.stop, options.stop_token_ids);
    create_input_buffers(input_token_ids);
    auto output = _generate_tokens(request, input_token_ids, options.max_total_tokens,
        options.timer_ttft, options.performance, options.stable_prefix_token_count);
    request.finish();
    return request.cancelled() ? std::nullopt : output;
}

namespace {
void validate_dflash_pair(const VlmConfig& target_cfg, const VlmConfig& draft_cfg) {
    const auto& target = *target_cfg.lm_cfg.speculative_decoding_cfg;
    const auto& draft = *draft_cfg.lm_cfg.speculative_decoding_cfg;
    if (target.speculative_budget != draft.speculative_budget ||
        target.target_layer_ids != draft.target_layer_ids)
        throw std::invalid_argument("DFlash target and draft block configuration does not match");
    if (target_cfg.pipeline_cfg.input_token_group_size != draft_cfg.pipeline_cfg.input_token_group_size)
        throw std::invalid_argument("DFlash target and draft language group sizes do not match");
    if (target_cfg.pipeline_cfg.input_token_group_size <= target.speculative_budget)
        throw std::invalid_argument("DFlash language group size must exceed the block size");
    if (draft_cfg.pipeline_cfg.max_num_tokens < target_cfg.pipeline_cfg.max_num_tokens)
        throw std::invalid_argument("DFlash draft cache is smaller than the target cache");
    if (target_cfg.pipeline_cfg.quantize_embeddings != draft_cfg.pipeline_cfg.quantize_embeddings)
        throw std::invalid_argument("DFlash target and draft embedding quantization modes do not match");
    if (target_cfg.pipeline_cfg.future_token_mask_size <= 1 || draft_cfg.pipeline_cfg.future_token_mask_size <= 1)
        throw std::invalid_argument("DFlash models require a multi-token future-mask configuration");
    if (target.mask_token_id != draft.mask_token_id || draft.mask_token_id < 0 ||
        static_cast<uint32_t>(draft.mask_token_id) >= target_cfg.lm_cfg.token_cfg.vocab_size)
        throw std::invalid_argument("DFlash target and draft mask token IDs do not match");
}
} // namespace

void LanguageModel::attach_draft(LanguageModel& draft_lm) {
    std::lock_guard lock(RuntimeContext::instance().execution_mutex);
    if (_run->active || draft_lm._run->active)
        throw std::logic_error("Cannot change the target/draft pair during generation");
    if (!_cfg.lm_cfg.speculative_decoding_cfg || !draft_lm._cfg.lm_cfg.speculative_decoding_cfg)
        throw std::invalid_argument("Both models must contain speculative-decoding configuration");
    const auto& target = *_cfg.lm_cfg.speculative_decoding_cfg;
    const auto& draft = *draft_lm._cfg.lm_cfg.speculative_decoding_cfg;
    if (target.is_draft || !draft.is_draft)
        throw std::invalid_argument("Speculative decoding requires a target paired with a draft");
    if (_workspace->speculative_pair && _workspace->speculative_pair == draft_lm._workspace->speculative_pair)
        return;
    if (target.method != draft.method)
        throw std::invalid_argument("Target and draft speculative methods do not match");

    if (target.is_gemma4_mtp()) {
        if (!_cfg.lm_cfg.is_gemma4_mtp_target() || !draft_lm._cfg.lm_cfg.is_gemma4_mtp_draft())
            throw std::invalid_argument("Gemma4 MTP requires a target and an assistant");
        if (_cfg.lm_cfg.hidden_size != draft.backbone_hidden_size)
            throw std::invalid_argument("Gemma4 MTP target hidden size must match assistant backbone size");
        const auto width = _cfg.lm_cfg.get_single_num_tokens();
        if (!draft.speculative_budget || width != draft.speculative_budget + 1
            || draft_lm._cfg.lm_cfg.get_single_num_tokens() != 1)
            throw std::invalid_argument("Gemma4 MTP requires an N1 assistant and budget-plus-one target width");
        _workspace->mtp_target_kv = gemma4_mtp_helpers::resolve_target_kv(
            _cfg, draft_lm._cfg, _workspace->attention_binding_buffers);
    } else {
        if (_cfg.lm_cfg.hidden_size != draft_lm._cfg.lm_cfg.hidden_size ||
            _cfg.lm_cfg.token_cfg.vocab_size != draft_lm._cfg.lm_cfg.token_cfg.vocab_size)
            throw std::invalid_argument("Target and draft hidden or vocabulary sizes do not match");
        if (target.method == "dflash")
            validate_dflash_pair(_cfg, draft_lm._cfg);
        else if (target.is_eagle3())
            _validate_eagle3_target();
        else
            throw std::invalid_argument("Unsupported speculative-decoding method: " + target.method);
    }
    auto pair = std::make_shared<LanguageWorkspace::SpeculativePair>();
    invalidate_prompt_reuse();
    draft_lm.invalidate_prompt_reuse();
    _workspace->speculative_pair = pair;
    draft_lm._workspace->speculative_pair = std::move(pair);
}

uint32_t LanguageModel::_prefill(
    std::span<const uint32_t> input_token_ids, uint16_t num_cached_tokens,
    std::optional<ChronoTimer> timer_ttft, std::vector<double>* durations
) {
    _run->require_active_request();
    if (_uses_per_layer_inputs())
        _workspace->prompt_per_layer_token_ids = _get_per_layer_token_ids(input_token_ids);
    if (!timer_ttft.has_value())
        timer_ttft = ChronoTimer{true};
    const uint16_t num_input_tokens = input_token_ids.size();
    const uint16_t group_size = _cfg.pipeline_cfg.input_token_group_size;
    uint16_t token_idx{};
    uint32_t next_token_id{};
    uint16_t last_group_valid_tokens = 0;
    num_cached_tokens = _prepare_state_checkpoints_for_prefill(num_cached_tokens);

    if (num_input_tokens == num_cached_tokens) {
        // All input tokens are already cached.
        if (num_cached_tokens < _cache->cached_token_ids.size())
            next_token_id = _cache->cached_token_ids[num_cached_tokens];
        else
            next_token_id = _cache->cached_first_generated_token;
    } else if (_use_group_token_models) {
        const auto& offsets = _cfg.pipeline_cfg.input_token_group_offsets.value();
        const auto& num_tokens = _cfg.pipeline_cfg.input_token_group_size;
        auto it = offsets.end();
        while (it != offsets.begin()) {
            --it;
            token_idx = *it;
            if (*it > num_cached_tokens)
                continue;
            if (token_idx <= num_cached_tokens && token_idx + num_tokens > num_cached_tokens)
                break;
            // Number of cached tokens are greater than the supported group offsets.
            token_idx = num_cached_tokens;
            it = offsets.end();
            break;
        }

        while (token_idx < num_input_tokens) {
            if (it != offsets.end()) {
                assert(token_idx >= *it && token_idx < *it + num_tokens);
                token_idx = *it;
                last_group_valid_tokens = std::min(
                    num_input_tokens, static_cast<uint16_t>(token_idx + num_tokens)
                ) - token_idx;
                if (_uses_per_layer_inputs()) {
                    _compute_and_upload_per_layer_inputs_prefill(
                        num_tokens, token_idx, num_input_tokens
                    );
                }
                next_token_id = run_model_once(num_tokens, token_idx, num_input_tokens, 0);
                ++it;
                token_idx += num_tokens;
            } else {
                if (_uses_per_layer_inputs()) {
                    _compute_and_upload_per_layer_inputs_prefill(1, token_idx, num_input_tokens);
                }
                next_token_id = run_model_once(1, token_idx, num_input_tokens, 0);
                token_idx += 1;
            }
            if (_run->cancelled()) {
                if (!_cfg.lm_cfg.is_gemma4_mtp_target()) _notify_interrupt();
                return next_token_id;
            }
        }
        if (
            !_cache->cached_states.empty()
            && last_group_valid_tokens > 0
            && last_group_valid_tokens < group_size
        ) {
            _move_state_tail_for_decode(last_group_valid_tokens);
        }
    } else {
        for (token_idx = num_cached_tokens; token_idx < num_input_tokens; ++token_idx) {
            if (_uses_per_layer_inputs()) {
                _compute_and_upload_per_layer_inputs_prefill(1, token_idx, num_input_tokens);
            }
            next_token_id = run_model_once(1, token_idx, num_input_tokens, 0);
            if (_run->cancelled()) {
                if (!_cfg.lm_cfg.is_gemma4_mtp_target()) _notify_interrupt();
                return next_token_id;
            }
        }
    }
    _cache->cached_token_ids.assign(input_token_ids.begin(), input_token_ids.end());
    auto duration = timer_ttft.value().stop();
    if (num_input_tokens < _run->limits().max_num_tokens) {
        if (durations) durations->push_back(duration);
        _notify_first_token(next_token_id, duration);
    }
    _cache->cached_first_generated_token = next_token_id;
    return next_token_id;
}


void LanguageModel::_decode(uint16_t num_input_tokens, uint32_t token_id,
                            std::vector<double>* durations) {
    _run->require_active_request();
    ChronoTimer timer_tps(true);
    uint16_t token_idx;
    for (token_idx = num_input_tokens; token_idx < _run->limits().max_num_tokens; ++token_idx ) {
        if (_run->cancelled()) {
            _notify_interrupt();
            return;
        }
        auto next_token_id = run_model_once(1, token_idx, num_input_tokens, token_id);
        if (_run->cancelled()) {
            _notify_interrupt();
            return;
        }
        _cache->cached_token_ids.emplace_back(token_id);
        token_id = next_token_id;

        auto duration = timer_tps.stop(true);
        if (token_idx + 1 < _run->limits().max_num_tokens) {
            if (durations) durations->push_back(duration);
            _notify_new_token(next_token_id, duration);
            if (_run->limits().stop_token_ids.contains(token_id)) {
                _notify_stop();
                return;
            }
        }
    }

    // Check if cache is full.
    if (token_idx == _run->limits().max_num_tokens) {
        _notify_cache_full();
    }
}


void LanguageModel::_upload_group_future_token_masks(
    uint16_t num_tokens,
    uint16_t token_idx
) {
    if (
        num_tokens != _cfg.pipeline_cfg.input_token_group_size
        || num_tokens == _cfg.lm_cfg.get_single_num_tokens()
    ) {
        return;
    }

    auto upload_group_mask = [&](
        const std::string& name,
        LayerType layer_type,
        uint16_t cache_token_idx_begin
    ) {
        const uint16_t effective_context = token_idx + num_tokens - cache_token_idx_begin;
        const uint16_t mask_size = _get_cache_mask_size(
            layer_type, effective_context, true
        );
        if (mask_size <= num_tokens) {
    return;
}
        const uint16_t effective_token_idx = token_idx - cache_token_idx_begin;
        const uint16_t aligned_context = std::min<uint16_t>(
            round_up_to(effective_context, mask_size),
            _cfg.pipeline_cfg.max_num_tokens
        );
        std::vector<Eigen::bfloat16> mask(
            num_tokens * aligned_context,
            std::numeric_limits<Eigen::bfloat16>::lowest()
        );
        for (uint16_t row = 0; row < num_tokens; ++row) {
            std::fill_n(
                mask.begin() + row * aligned_context,
                effective_token_idx + row + 1,
                Eigen::bfloat16{0.0f}
            );
        }
        get_buffer(name).upload_raw(mask.data(), 0, mask.size() * sizeof(Eigen::bfloat16));
    };

    upload_group_mask(
        "group_future_token_mask", LayerType::FullAttention, 0
    );
    if (_cfg.lm_cfg.attn_cfg.swa_enable) {
        const uint16_t cache_token_idx_begin = _get_cache_token_idx_begin(
            num_tokens, token_idx, LayerType::SlidingAttention
        );
        upload_group_mask(
            "group_sliding_future_token_mask", LayerType::SlidingAttention, cache_token_idx_begin
        );
    }
}


uint32_t LanguageModel::run_model_once(
    uint16_t num_tokens,
    uint16_t token_idx,
    uint16_t num_input_tokens,
    uint32_t token_id,
    std::vector<Eigen::bfloat16>* logits_ptr
) {
    _run->require_active_request();
    auto execution = LayerExecutor::lock_execution();
    if (token_idx == 0 && logits_ptr)
        _prepare_state_checkpoints_for_prefill(0);

    uint16_t next_token_idx;
    if (num_tokens > 1) {
        next_token_idx = std::min(num_input_tokens, uint16_t(token_idx + num_tokens));
    } else {
        next_token_idx = token_idx + 1;
    }
    auto use_input_tokens = token_idx < num_input_tokens;
    // PrefillTrace tests parse this message prefix.
    _logger->info("Processing token no. {}-{}", token_idx, next_token_idx);

    _upload_group_future_token_masks(num_tokens, token_idx);

    MLABuffer* normal_input_buf;
    MLABuffer* normal_scale_buf = nullptr;
    uint32_t normal_input_row;
    uint32_t normal_input_num_tokens;
    if (use_input_tokens) {
        normal_input_buf = &get_buffer("input_embeds");
        if (_cfg.pipeline_cfg.quantize_embeddings)
            normal_scale_buf = &get_buffer("input_embedding_scales");
        if (_resources->embedding_offload) _prepare_offloaded_prompt(num_tokens, token_idx);
        normal_input_row = _resources->embedding_offload && _workspace->offload_chunk_staging ? 0 : token_idx;
        normal_input_num_tokens = num_tokens;
    } else {
        normal_input_buf = &get_buffer(_resources->embedding_offload ? "offload_decode_embeds" : "embeddings");
        if (_cfg.pipeline_cfg.quantize_embeddings)
            normal_scale_buf = &get_buffer(_resources->embedding_offload ? "offload_decode_scales" : "embedding_scales");
        normal_input_row = _resources->embedding_offload ? 0 : token_id;
        normal_input_num_tokens = 1;
        if (_resources->embedding_offload) _gather_embedding_rows({&token_id, 1}, *normal_input_buf, normal_scale_buf);
    }

    if (_uses_per_layer_inputs()) {
        auto inputs = LayerExecutor::per_layer_inputs(*this, normal_input_buf,
            normal_scale_buf, normal_input_row, normal_input_num_tokens,
            use_input_tokens ? std::nullopt : std::optional<uint32_t>(token_id));
        _workspace->per_layer_model_map.at({num_tokens, 0, 0}).add_to_queue(&inputs);
    }

    if (num_tokens > 1 && _has_linear_attention_layers()) {
        const uint16_t valid_tokens = next_token_idx - token_idx;
        std::vector<Eigen::bfloat16> valid_mask(num_tokens, Eigen::bfloat16(0.0f));
        std::fill_n(valid_mask.begin(), valid_tokens, Eigen::bfloat16(1.0f));
        get_buffer("linear_valid_mask").upload(valid_mask.data());
    }

    const bool direct_dflash_capture = (
        _cfg.lm_cfg.is_dflash()
        && !_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft
        && num_tokens == _cfg.pipeline_cfg.input_token_group_size
    );
    MLABuffer* dflash_layer_input = nullptr;
    const auto final_output = num_input_tokens > next_token_idx
        ? LayerExecutor::LayerOutput::CacheOnly : LayerExecutor::LayerOutput::LogitsRow;
    const uint16_t last_row = next_token_idx - token_idx - 1;
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        const bool last_layer = layer_idx + 1 == _cfg.lm_cfg.num_hidden_layers;
        LayerExecutor::Overrides inputs;
        if (layer_idx == 0) {
            inputs.emplace(0, MLABufferSlice{normal_input_buf, {normal_input_row, 0},
                                             {normal_input_num_tokens, _cfg.lm_cfg.hidden_size}});
        } else if (dflash_layer_input) {
            inputs.emplace(0, MLABufferSlice{dflash_layer_input, {0, 0},
                                             {num_tokens, _cfg.lm_cfg.hidden_size}});
        }

        MLABuffer* dflash_capture_buffer = direct_dflash_capture && !last_layer
            ? _get_dflash_hidden_buffer(num_tokens, layer_idx) : nullptr;
        LayerExecutor::Overrides outputs;
        if (dflash_capture_buffer) {
            outputs.emplace(0, MLABufferSlice{dflash_capture_buffer, {0, 0},
                                              {num_tokens, _cfg.lm_cfg.hidden_size}});
        }
        LayerExecutor::queue_layer(*this, num_tokens, token_idx, layer_idx, normal_scale_buf,
            normal_input_row, {.input = &inputs, .post_input = &inputs, .output = &outputs},
            last_layer ? final_output : LayerExecutor::LayerOutput::AllRows, last_row);
        dflash_layer_input = dflash_capture_buffer;

        if (
            _cfg.lm_cfg.is_spec_decode()
            && !_cfg.lm_cfg.speculative_decoding_cfg.value().is_draft
            && _cfg.lm_cfg.speculative_decoding_cfg->is_eagle3()
            && layer_idx < _cfg.lm_cfg.num_hidden_layers - 1
        ) {
            const uint8_t num_layers = _cfg.lm_cfg.num_hidden_layers;
            const std::vector<uint8_t> capture_layers{
                2,
                static_cast<uint8_t>(num_layers / 2),
                static_cast<uint8_t>(num_layers - 3),
            };
            const auto it = std::find(
                capture_layers.begin(), capture_layers.end(), layer_idx
            );
            if (it != capture_layers.end()) {
                MLAModelWithBuffer::run_queue();
                const size_t capture_idx = std::distance(capture_layers.begin(), it);
                if (_workspace->eagle3_intermediate_hidden_states.size() < capture_layers.size()) {
                    _workspace->eagle3_intermediate_hidden_states.resize(capture_layers.size());
                }
                auto& buffer = get_buffer(fmt::format("n{}_buffer1", num_tokens));
                auto& capture = _workspace->eagle3_intermediate_hidden_states[capture_idx];
                capture.resize(static_cast<size_t>(num_tokens) * _cfg.lm_cfg.hidden_size);
                buffer.download(capture.data());
            }
        }
    }

    // Run all the queued models.
    MLAModelWithBuffer::run_queue();
    for (uint8_t layer_idx = 0; layer_idx < _cfg.lm_cfg.num_hidden_layers; ++layer_idx) {
        if (_cfg.lm_cfg.layer_types[layer_idx] == LayerType::LinearAttention) {
            get_buffer(fmt::format("linear_delta_state_history_l{}", layer_idx)).swap_storage(
                get_buffer(fmt::format("linear_delta_state_history_alt_l{}", layer_idx))
            );
        }
    }
    // If this run landed exactly on a checkpoint boundary, save the tail.
    const bool save_checkpoint = (
        _workspace->capture_state_checkpoints
        && std::binary_search(
            _cache->checkpoint_boundaries.begin(), _cache->checkpoint_boundaries.end(), next_token_idx
        )
    );
    if (save_checkpoint) {
        // A partial prefill group can land exactly on a checkpoint boundary.
        _save_state_checkpoint(
            next_token_idx, num_tokens, next_token_idx - token_idx,
            token_idx < num_input_tokens
        );
    }

    if (logits_ptr) {
        assert(num_tokens == 1 && _cfg.pipeline_cfg.return_logits);
        MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
        buf_ptr->invalidate_cache(
            0, static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) *
                sizeof(Eigen::bfloat16)
        );

        // Append the logits.
        auto* ptr = reinterpret_cast<Eigen::bfloat16*>(buf_ptr->get_virtual_addr());
        logits_ptr->insert(logits_ptr->end(), ptr, ptr + _cfg.lm_cfg.token_cfg.vocab_size);

        // Return dummy token id.
        return 0;
    }

    // Find the next token id.
    uint32_t next_token_id;
    if (num_input_tokens <= next_token_idx) {
        if (_cfg.lm_cfg.is_spec_decode()) {
            const auto window = LayerExecutor::logits_window(_cfg.lm_cfg, num_tokens, last_row);
            const uint32_t row = last_row - window.first;
            const uint16_t logits_num_tokens = window.width;
            const uint32_t vocab_size = _cfg.lm_cfg.get_lm_head_output_size();

            if (_cfg.lm_cfg.lm_head_num_splits == 1) {
                MLABuffer* buf_ptr = &get_buffer(
                    fmt::format("n{}_buffer4", logits_num_tokens)
                );
                const size_t row_bytes =
                    static_cast<size_t>(vocab_size) * sizeof(Eigen::bfloat16);
                const size_t row_offset = static_cast<size_t>(row) * row_bytes;
                buf_ptr->invalidate_cache(row_offset, row_bytes);
                auto* ptr = reinterpret_cast<Eigen::bfloat16*>(
                    static_cast<uint8_t*>(buf_ptr->get_virtual_addr()) + row_offset
                );
                uint32_t best_idx = 0;
                Eigen::bfloat16 best_val = ptr[0];
                for (uint32_t i = 1; i < vocab_size; ++i) {
                    if (ptr[i] > best_val) {
                        best_val = ptr[i];
                        best_idx = i;
                    }
                }
                next_token_id = best_idx;
            } else {
                // Splits: argmax across all split buffers at this row.
                const auto& split_dim = _cfg.lm_cfg.lm_head_split_dim;
                uint32_t best_idx = 0;
                Eigen::bfloat16 best_val = std::numeric_limits<Eigen::bfloat16>::lowest();
                uint32_t i = 0;
                for (uint32_t split_begin = 0; split_begin < vocab_size; split_begin += split_dim, ++i) {
                    uint32_t split_size = std::min(vocab_size, split_begin + split_dim) - split_begin;
                    MLABuffer* buf_ptr = &get_buffer(
                        fmt::format("n{}_lm_split{}", logits_num_tokens, i)
                    );
                    const size_t row_bytes =
                        static_cast<size_t>(split_size) * sizeof(Eigen::bfloat16);
                    const size_t row_offset = static_cast<size_t>(row) * row_bytes;
                    buf_ptr->invalidate_cache(row_offset, row_bytes);
                    auto* ptr = reinterpret_cast<Eigen::bfloat16*>(
                        static_cast<uint8_t*>(buf_ptr->get_virtual_addr()) + row_offset
                    );
                    for (uint32_t j = 0; j < split_size; ++j) {
                        if (ptr[j] > best_val) {
                            best_val = ptr[j];
                            best_idx = split_begin + j;
                        }
                    }
                }
                next_token_id = best_idx;
            }
            if (_cfg.lm_cfg.is_gemma4_mtp_target()) {
                auto hidden_rows = _read_gemma4_mtp_target_hidden_rows(
                    logits_num_tokens, static_cast<uint16_t>(row + 1)
                );
                const size_t hidden_size = _cfg.lm_cfg.hidden_size;
                const auto row_begin = hidden_rows.begin()
                    + static_cast<std::ptrdiff_t>(row * hidden_size);
                _workspace->gemma4_mtp_prefill_hidden_state.emplace(
                    row_begin,
                    row_begin + static_cast<std::ptrdiff_t>(hidden_size)
                );
            }
        } else {
            // Non-spec: read from single n1_buffer4
            MLABuffer* buf_ptr = &get_buffer("n1_buffer4");
            const size_t result_bytes = _need_argmax
                ? static_cast<size_t>(_cfg.lm_cfg.token_cfg.vocab_size) *
                      sizeof(Eigen::bfloat16)
                : sizeof(uint32_t);
            buf_ptr->invalidate_cache(0, result_bytes);
            if (_need_argmax) {
                next_token_id = _calc_next_token_id(buf_ptr);
            } else {
                auto* ptr = reinterpret_cast<uint32_t*>(buf_ptr->get_virtual_addr());
                next_token_id = ptr[0];
            }
        }
    } else {
        next_token_id = 0;
    }
    return next_token_id;
}


void LanguageModel::_finalize() {
    if (!_workspace) return;
    _logger->info("Language model finalize starting ...");
    // Destroy bindings before their cache and workspace buffers and shared resources.
    _workspace.reset();
    _cache = nullptr;
    _resources.reset();
    _logger->info("Language model finalize completed");
}


uint16_t LanguageModel::_get_cache_token_idx_begin(
    uint16_t num_tokens, uint16_t token_idx, LayerType layer_type
) const {
    if (layer_type != LayerType::SlidingAttention) {
        return 0;
    }
    // Retain the first query's full window for prefill and verification.
    return std::max(
        0, token_idx + 1 - static_cast<int>(_cfg.lm_cfg.attn_cfg.sliding_window.value())
    );
}


LanguageModelMapKey LanguageModel::_get_cache_model_key(
    uint16_t num_tokens, uint16_t token_idx, uint8_t layer_idx
) const {
    const auto& layer_type = _cfg.lm_cfg.layer_types[layer_idx];
    const uint16_t cache_token_idx_begin = _get_cache_token_idx_begin(
        num_tokens, token_idx, layer_type
    );

    const uint16_t eff_token_idx = token_idx - cache_token_idx_begin;
    const uint16_t eff_num_cached_tokens = token_idx + num_tokens - cache_token_idx_begin;
    const uint16_t single_num_tokens = _cfg.lm_cfg.get_single_num_tokens();
    const bool is_single_model = num_tokens == single_num_tokens;
    const uint16_t sliding_window = _cfg.lm_cfg.attn_cfg.sliding_window.value_or(0);
    const bool separate_sliding_cache = (
        layer_type == LayerType::SlidingAttention
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.has_value()
        && _cfg.lm_cfg.attn_cfg.sliding_head_dim.value() != _cfg.lm_cfg.attn_cfg.head_dim
    );
    // Bucket differences matter only when the sliding window truncates the context.
    const bool sliding_cache_mask_differs = (
        layer_type == LayerType::SlidingAttention
        && !separate_sliding_cache
        && sliding_window < _cfg.pipeline_cfg.max_num_tokens
        && (
            _get_cache_mask_size(LayerType::FullAttention, sliding_window, true)
                != _get_cache_mask_size(LayerType::SlidingAttention, sliding_window, true)
            || _get_cache_mask_size(LayerType::FullAttention, sliding_window, false)
                != _get_cache_mask_size(LayerType::SlidingAttention, sliding_window, false)
        )
    );
    const bool use_sliding_cache = (
        separate_sliding_cache
        || (layer_type == LayerType::SlidingAttention
            && (!is_single_model || _cfg.lm_cfg.is_spec_decode())
            && token_idx + num_tokens > sliding_window)
        || (is_single_model && sliding_cache_mask_differs
            && eff_num_cached_tokens >= sliding_window)
    );
    const LayerType cache_layer_type = (
        use_sliding_cache ? LayerType::SlidingAttention : LayerType::FullAttention
    );
    const uint16_t cache_mask_size = _get_cache_mask_size(
        cache_layer_type, eff_num_cached_tokens, !is_single_model
    );
    const bool use_single_future_token_mask = is_single_model && cache_mask_size > 1;
    const bool use_group_future_token_mask = (
        !is_single_model && cache_mask_size > num_tokens
    );

    uint16_t aligned_eff_token_idx;
    if (use_single_future_token_mask) {
        aligned_eff_token_idx = std::min(
            round_up_to(eff_num_cached_tokens, cache_mask_size) - 1,
            _cfg.pipeline_cfg.max_num_tokens - 1
        );
    } else if (use_group_future_token_mask) {
        const uint16_t aligned_eff_num_cached_tokens = std::min<uint16_t>(
            round_up_to(eff_num_cached_tokens, cache_mask_size),
            _cfg.pipeline_cfg.max_num_tokens
        );
        aligned_eff_token_idx = aligned_eff_num_cached_tokens - num_tokens;
    } else {
        aligned_eff_token_idx = eff_token_idx;
    }
    if (use_sliding_cache && is_single_model && _cfg.lm_cfg.is_spec_decode()) {
        aligned_eff_token_idx = std::min<uint16_t>(
            aligned_eff_token_idx, sliding_window + num_tokens - 2
        );
    }

    // The second key field is a cache executable kind, not a transformer layer.
    return LanguageModelMapKey{
        num_tokens,
        static_cast<uint8_t>(use_sliding_cache),
        aligned_eff_token_idx
    };
}


void LanguageModel::invalidate_prompt_reuse() {
    _cache->invalidate_reuse();
    _workspace->clear_cached_scratch();
    _workspace->speculative_pair.reset();
}

void LanguageModel::set_image_identity(std::string identity) {
    if (_cache->image_identity != identity) {
        _cache->invalidate_reuse();
        _cache->image_identity = std::move(identity);
    }
}

uint32_t LanguageModel::_calc_next_token_id(MLABuffer* buf_ptr) {
    const auto* values = reinterpret_cast<const Eigen::bfloat16*>(
        buf_ptr->get_virtual_addr()
    );
    const uint32_t vocab_size = _cfg.lm_cfg.token_cfg.vocab_size;
    float best_value = static_cast<float>(values[0]);
    uint32_t best_index = 0;
    #pragma omp parallel for num_threads(4)
    for (uint32_t part = 0; part < 4; ++part) {
        const uint32_t begin = static_cast<uint64_t>(vocab_size) * part / 4;
        const uint32_t end = static_cast<uint64_t>(vocab_size) * (part + 1) / 4;
        const auto [local_value, local_offset] = argmax_bf16(
            values + begin, end - begin
        );
        const uint32_t local_index = begin + local_offset;
        #pragma omp critical
        {
            if (local_value > best_value
                || (local_value == best_value && local_index < best_index)) {
                best_value = local_value;
                best_index = local_index;
            }
        }
    }
    return best_index;
}


void LanguageModel::_notify_first_token(uint32_t token_id, double duration) {
    _logger->info("Time to the first token: {:d} in {:.5f}s", token_id, duration);
    _run->text_streamer.push(DecodeCallbackType::TTFT, token_id, duration);
}


void LanguageModel::_notify_new_token(
    uint32_t token_id, double duration, bool from_draft
) {
    _logger->info("Got token: {:d} in {:.5f}s", token_id, duration);
    _run->text_streamer.push(DecodeCallbackType::TPS, token_id, duration, from_draft);
}


void LanguageModel::_notify_cache_full() const {
    if (_run->active) _run->active->length_limit_reached = true;
    _logger->info("Cache full");
    _run->text_streamer.push(DecodeCallbackType::CACHE_FULL, 0, 0);
}


void LanguageModel::_notify_stop() const {
    _logger->info("Got stop token");
    _run->text_streamer.push(DecodeCallbackType::STOP, 0, 0);
}


void LanguageModel::_notify_interrupt() const {
    _logger->info("Interrupted by user");
    _run->text_streamer.push(DecodeCallbackType::STOP, 0, 0);
}


}
}
