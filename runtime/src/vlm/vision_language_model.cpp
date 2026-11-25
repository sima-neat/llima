#include <spdlog/spdlog.h>
#include <openssl/sha.h>

#include <stdexcept>
#include <utility>

#include "reasoning_parser.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"
#include "language/language_model_state.hpp"
#include "language/request_scope.hpp"
#include "vlm/vision_model.hpp"
#include "vlm/vlm_helper.hpp"

namespace simaai {
namespace llima {

// Owns encoded image bytes and their identity.
// A weak reference to the vision package validates later reuse.
struct EncodedImageData {
    std::weak_ptr<ModelResources<VlmConfig>> resources;
    std::string identity;
    std::vector<std::vector<std::byte>> outputs;
};

// Shared language model, prompt helper, streamer and request lock.
// Targets retain an attached draft's runtime after the draft wrapper is destroyed.
struct LanguageRuntime {
    friend class VisionLanguageModel;
    LanguageRuntime(std::filesystem::path path, std::shared_ptr<FileProvider> provider,
                    std::optional<std::string> system, std::optional<std::string> chat_template)
        : resources(ModelResources<VlmConfig>::acquire(path, std::move(provider))),
          _cfg(resources->cfg), _files(resources->files),
          vlm_helper(_cfg, *_files, std::move(system), std::move(chat_template)),
          text_streamer(vlm_helper.get_tokenizer(), std::nullopt, std::nullopt) {}
    PreservedTokens structural_tokens(const Chat& chat);
    std::shared_ptr<ModelResources<VlmConfig>> resources;
    const VlmConfig& _cfg;
    const std::shared_ptr<FileProvider>& _files;
    VlmHelper vlm_helper;
    TextStreamer text_streamer;
    ToolCallFormat tool_call_format = ToolCallFormat::GenericJson;
    std::unique_ptr<LanguageModel> language_model_ptr;
    std::mutex run_mutex;
    std::shared_ptr<LanguageRuntime> draft;
};

namespace {

// Keeps the target and any attached draft alive and locks them for this call.
class ModelLock {
  public:
    explicit ModelLock(std::shared_ptr<LanguageRuntime> target);
    LanguageModel& target() const { return *_target->language_model_ptr; }
    LanguageModel* draft() const { return _draft ? _draft->language_model_ptr.get() : nullptr; }

  private:
    std::shared_ptr<LanguageRuntime> _target;
    std::shared_ptr<LanguageRuntime> _draft;
    std::unique_lock<std::mutex> _target_lock;
    std::unique_lock<std::mutex> _draft_lock;
};

ModelLock::ModelLock(std::shared_ptr<LanguageRuntime> target)
    : _target(std::move(target)), _target_lock(_target->run_mutex, std::defer_lock) {
    for (;;) {
        _target_lock.lock();
        _draft = _target->draft;
        if (!_draft)
            break;
        _draft_lock = std::unique_lock(_draft->run_mutex, std::defer_lock);
        _target_lock.unlock();
        std::lock(_target_lock, _draft_lock);
        if (_target->draft == _draft)
            break;
        _draft_lock.unlock();
        _target_lock.unlock();
    }
}

TextStreamer::RequestOptions stream_options(
    const GenerationOptions& options, PreservedTokens markers = {}, bool streaming = false
) {
    return {options.callbacks, std::move(markers),
            options.performance_summary, streaming};
}

std::optional<uint16_t> context_limit(size_t prompt_size, std::optional<uint16_t> max_new_tokens,
                                      uint16_t capacity) {
    if (!max_new_tokens) return std::nullopt;
    return std::min<size_t>(capacity, prompt_size + *max_new_tokens);
}

std::unique_lock<std::mutex> idle_lock(std::mutex& mutex) {
    std::unique_lock lock(mutex, std::try_to_lock);
    if (!lock.owns_lock())
        throw std::logic_error("Model settings cannot change during a request");
    return lock;
}
} // namespace

namespace {
std::string image_digest(const std::vector<Eigen::bfloat16>& tensor) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(tensor.data()),
           tensor.size() * sizeof(Eigen::bfloat16), digest);
    return {reinterpret_cast<const char*>(digest), sizeof(digest)};
}
}


bool VisionLanguageModel::support_image() const { return _state->_cfg.support_image(); }
std::string_view VisionLanguageModel::model_type() const { return _state->_cfg.model_type; }

Chat VisionLanguageModel::create_chat() {
    return Chat(_state->vlm_helper);
}

ToolCallFormat VisionLanguageModel::tool_call_format() const { return _state->tool_call_format; }

void VisionLanguageModel::set_reloc(const std::string& name) {
    auto lock = idle_lock(_state->run_mutex);
    _state->language_model_ptr->set_reloc(name);
}

void VisionLanguageModel::unset_reloc() {
    auto lock = idle_lock(_state->run_mutex);
    _state->language_model_ptr->unset_reloc();
}

VisionLanguageModel::~VisionLanguageModel() = default;

VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template
) : VisionLanguageModel(
        std::move(model_path), std::move(system_prompt), std::move(chat_template), nullptr
    ) {}

VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    std::shared_ptr<FileProvider> file_provider
) : VisionLanguageModel(std::move(model_path), std::move(system_prompt), std::move(chat_template),
                        std::move(file_provider), {}) {}

VisionLanguageModel::VisionLanguageModel(
    std::filesystem::path model_path,
    std::optional<std::string> system_prompt,
    std::optional<std::string> chat_template,
    std::shared_ptr<FileProvider> file_provider,
    ModelOptions options
) : _state([&] {
    if (!options.max_kv_cache_slots)
        throw std::invalid_argument("max_kv_cache_slots must be positive");
    return std::make_shared<LanguageRuntime>(model_path, file_provider, system_prompt, chat_template);
}())
{
    _state->tool_call_format = tool_call_format_for_model(_state->_cfg.model_type);
    if (_state->_cfg.support_image()) {
        // VisionModel loads its stage1 ELFs and vlm_config.json through the
        // same FileProvider as the language path, so over PCIe they are pulled
        // (fetch -> load -> evict), one ELF on disk at a time.
        _vision_model = std::make_unique<VisionModel>(model_path, _state->_files);
    }
    _state->language_model_ptr = std::make_unique<LanguageModel>(
        model_path,
        _state->vlm_helper.get_stop_token_ids(),
        _state->vlm_helper.get_image_token_id(),
        _state->vlm_helper.get_pad_token_id(),
        _state->text_streamer,
        _state->_files, options.max_kv_cache_slots
    );

    // Dummy-query warmup. Skipped in spec mode: drafts have no standalone
    // run_model path.
    if (!_state->_cfg.lm_cfg.is_spec_decode()) {
        Chat chat = create_chat();
        if (_state->_cfg.support_image() && sample_image_file_name.has_value()) {
            chat.add_image(sample_image_file_name.value());
            chat.add_query("Describe what's in the image");
        } else {
            chat.add_query("Hi");
        }
        _state->text_streamer.disable();
        generate(chat, {.max_new_tokens = 2});
        _state->text_streamer.enable();
    }

    // Pre-warm OMP thread pool to avoid libgomp first-call cost.
    warmup_omp();
    // Constructor warmup is not an application cache association.
    _state->language_model_ptr->clear_kv_caches();
}


std::vector<EncodedImage> VisionLanguageModel::encode_images(std::span<const cv::Mat> images) {
    if (!_vision_model) throw std::logic_error("Model does not support images");
    std::lock_guard lock(_state->run_mutex);
    std::vector<EncodedImage> encoded;
    encoded.reserve(images.size());
    for (const auto& pixels : images) {
        auto tensor = _state->vlm_helper.preprocess_image(pixels);
        auto data = std::make_shared<EncodedImageData>();
        data->resources = _vision_model->_resources;
        data->identity = image_digest(tensor);
        data->outputs = _vision_model->encode(tensor);
        EncodedImage image;
        image._data = std::move(data);
        encoded.emplace_back(std::move(image));
    }
    return encoded;
}

ChatResult VisionLanguageModel::generate(const Chat& chat, const GenerationOptions& options) {
    ChatResult result;
    ModelLock models(_state);
    if (options.stop.stop_requested()) {
        result.finish = ChatResult::Finish::Cancelled;
        return result;
    }

    auto markers = _state->structural_tokens(chat);
    if (!options.preserve_structural_tokens) markers.clear();
    RequestScope request(models.target(), models.draft(), stream_options(options, markers, true),
                         options.stop, options.stop_token_ids, options.cache_id);

    ChronoTimer timer_ttft(true);
    auto input = prepare_chat_inputs(request, chat, options);
    const auto limit = context_limit(input.input_token_ids.size(), options.max_new_tokens,
                                    config().pipeline_cfg.max_num_tokens);
    auto ids = models.target()._generate_tokens(request, input.input_token_ids, limit,
        timer_ttft, &result.performance, input.stable_prefix_token_count);
    request.finish();

    if (request.cancelled()) {
        result.finish = ChatResult::Finish::Cancelled;
        result.performance = {};
        return result;
    }
    if (request.length_limit_reached() || !ids) result.finish = ChatResult::Finish::Length;
    if (ids) {
        result.token_ids = std::move(*ids);
        result.performance.generated_tokens = result.token_ids.size();
        auto& tokenizer = *_state->vlm_helper.get_tokenizer();
        result.text = markers.empty() ? tokenizer.decode(result.token_ids, true)
            : decode_tool_call_output(tokenizer, result.token_ids, markers);
    }
    return result;
}

const VlmConfig& VisionLanguageModel::config() const {
    return _state->_cfg;
}

std::optional<std::vector<uint32_t>> VisionLanguageModel::benchmark_chat(
    const Chat& chat, const GenerationOptions& options, ChatBenchmark& benchmark
) {
    ModelLock models(_state);
    if (options.stop.stop_requested()) return std::nullopt;
    auto markers = _state->structural_tokens(chat);
    if (!options.preserve_structural_tokens) markers.clear();
    RequestScope request(models.target(), models.draft(), stream_options(options, markers, true),
                         options.stop, options.stop_token_ids, options.cache_id);

    ChronoTimer timer_ttft(true);
    auto input = prepare_chat_inputs(request, chat, options, &benchmark);
    const auto limit = context_limit(input.input_token_ids.size(), options.max_new_tokens,
                                    config().pipeline_cfg.max_num_tokens);
    auto ids = models.target()._generate_tokens(request, input.input_token_ids, limit,
        timer_ttft, benchmark.performance, input.stable_prefix_token_count);
    request.finish();
    return request.cancelled() ? std::nullopt : std::move(ids);
}

PreservedTokens LanguageRuntime::structural_tokens(const Chat& chat) {
    PreservedTokens preserved_tokens;
    auto* tokenizer = vlm_helper.get_tokenizer();
    if (chat.has_tools()) {
        preserved_tokens = resolve_tool_call_special_tokens(tool_call_format, *tokenizer);
    }
    const auto reasoning_format = reasoning_format_for_model(_cfg.model_type);
    if (chat.get_enable_thinking() && reasoning_format == ReasoningFormat::None) {
        throw std::invalid_argument(
            "Thinking is not supported for model type '" + _cfg.model_type + "'"
        );
    }
    // gpt_oss always emits harmony channel tokens; preserve them even with
    // thinking off so the parser can strip to the final channel.
    if (chat.get_enable_thinking() || reasoning_format == ReasoningFormat::GptOss) {
        for (const auto marker : reasoning_special_tokens(reasoning_format)) {
            try {
                preserved_tokens.emplace_back(
                    tokenizer->token_to_id(std::string(marker)), marker
                );
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "Reasoning token '" + std::string(marker) + "' is missing from the tokenizer"
                );
            }
        }
    }
    return preserved_tokens;
}

PreprocessedChat VisionLanguageModel::prepare_chat_inputs(
    RequestScope& request, const Chat& chat, const GenerationOptions& options,
    ChatBenchmark* benchmark
) {
    if (request.cancelled()) return {};

    // Preprocess the chat.
    ChronoTimer timer_preprocess(true);
    if (!benchmark)
        _state->resources->logger->info("Chat:\n{}", chat.get_messages().dump(4));
    const bool supplied_images = !options.images.empty() || !options.encoded_images.empty();
    auto preprocessed_data = _state->vlm_helper.preprocess(chat, !supplied_images);
    const auto prompt_size = preprocessed_data.input_token_ids.size();
    if (options.reject_oversized_prompt) {
        const auto context_size = _state->_cfg.pipeline_cfg.max_num_tokens;
        if (prompt_size >= context_size) {
            throw std::length_error(fmt::format(
                "Prompt uses {} tokens; model context is {}. Start a new conversation or shorten the input.",
                prompt_size, context_size));
        }
        if (options.max_new_tokens && prompt_size + *options.max_new_tokens > UINT16_MAX)
            throw std::length_error("Requested token budget exceeds the LLiMa token limit");
    }
    if (supplied_images) {
        if (options.images.size() + options.encoded_images.size() != preprocessed_data.image_count)
            throw std::invalid_argument("Supplied image count must match the chat image placeholders");
        for (const auto& image : options.images)
            preprocessed_data.image_tensors.emplace_back(_state->vlm_helper.preprocess_image(image));
    }
    if (benchmark) {
        if (benchmark->max_context > _state->_cfg.pipeline_cfg.max_num_tokens)
            throw std::invalid_argument("Requested context exceeds the compiled model context");
        const size_t input_tokens = preprocessed_data.input_token_ids.size();
        if (!options.max_new_tokens || *options.max_new_tokens == 0 || input_tokens == 0 ||
            input_tokens + *options.max_new_tokens > benchmark->max_context)
            throw std::invalid_argument("Image/text prompt plus output budget exceeds context; no truncation applied");
        // Each benchmark sample measures cold prefill for the selected target/draft pair.
        _state->language_model_ptr->invalidate_prompt_reuse();
        if (auto* draft = request.draft()) draft->invalidate_prompt_reuse();
        benchmark->input_tokens = input_tokens;
        benchmark->formatted_prompt = std::move(preprocessed_data.formatted_prompt);
        _state->resources->logger->info("Benchmark prompt: {} input tokens, {} images", input_tokens,
                                       preprocessed_data.image_count);
    } else {
        _state->resources->logger->info("Formatted prompt: {}", preprocessed_data.formatted_prompt);
    }
    // Placeholder IDs alone do not identify the pixels encoded into the cache.
    std::string image_identity;
    for (const auto& tensor : preprocessed_data.image_tensors) image_identity += image_digest(tensor);
    for (const auto& image : options.encoded_images) {
        if (!image._data || !_vision_model ||
            image._data->resources.lock() != _vision_model->_resources)
            throw std::invalid_argument("Encoded image belongs to a different model package");
        image_identity += image._data->identity;
    }
    _state->language_model_ptr->set_image_identity(image_identity);
    if (auto* draft = request.draft()) draft->set_image_identity(std::move(image_identity));
    _state->resources->logger->info("Preprocess time: {:.2f}s", timer_preprocess.stop());

    if (request.cancelled()) return preprocessed_data;

    // Create buffers for the input embeds and deepstack features (for qwen3) and ofm maps for
    // vision models.
    auto vision_ofm_maps = _state->language_model_ptr->create_input_buffers(
        preprocessed_data.input_token_ids
    );

    // Encode the image tensors to embeddings. The image embeddings are directly written to the
    // input embeddings for the language models.
    ChronoTimer timer_image_encode(true);
    for (size_t i = 0; i < preprocessed_data.image_tensors.size(); ++i) {
        if (request.cancelled()) return preprocessed_data;
        _vision_model->run_model(preprocessed_data.image_tensors[i], &vision_ofm_maps[i]);
    }
    for (size_t i = 0; i < options.encoded_images.size(); ++i) {
        const auto& outputs = options.encoded_images[i]._data->outputs;
        const auto& destinations = vision_ofm_maps.at(preprocessed_data.image_tensors.size() + i);
        if (outputs.size() != destinations.size()) throw std::invalid_argument("Encoded image output count mismatch");
        for (const auto& [port, slice] : destinations) {
            auto* buffer = slice.get_buf_ptr();
            const auto& bytes = outputs.at(port);
            if (bytes.size() != buffer->get_buf_len(slice.get_buf_shapes()))
                throw std::invalid_argument("Encoded image output shape mismatch");
            buffer->upload_raw(bytes.data(), buffer->get_buf_addr_offset(slice.get_buf_begins()), bytes.size());
        }
    }
    _state->resources->logger->info("Image encode time: {:.2f}s", timer_image_encode.stop());

    return preprocessed_data;
}


std::vector<uint32_t> VisionLanguageModel::run_model(
    std::span<const uint32_t> input_token_ids,
    const GenerationOptions& options
) {
    ModelLock models(_state);
    if (options.stop.stop_requested()) return {};
    RequestScope request(models.target(), models.draft(), stream_options(options),
                         options.stop, options.stop_token_ids, options.cache_id);
    if (request.cancelled()) return {};
    auto& language = models.target();
    language.set_image_identity({});
    if (auto* draft = models.draft()) draft->set_image_identity({});
    language.create_input_buffers(input_token_ids);
    const auto limit = context_limit(input_token_ids.size(), options.max_new_tokens,
                                    config().pipeline_cfg.max_num_tokens);
    auto output = language._generate_tokens(request, input_token_ids, limit, {}, nullptr, 0);
    request.finish();
    if (request.cancelled()) return {};
    return output.value_or(std::vector<uint32_t>());
}


std::vector<Eigen::bfloat16> VisionLanguageModel::run_model_for_logits(
    std::span<const uint32_t> input_token_ids, const GenerationOptions& options
) {
    ModelLock models(_state);
    if (options.stop.stop_requested()) throw std::runtime_error("Scoring request was interrupted");
    RequestScope request(models.target(), models.draft(), stream_options(options),
                         options.stop, options.stop_token_ids, options.cache_id);
    if (request.cancelled()) throw std::runtime_error("Scoring request was interrupted");
    auto& language = models.target();
    language.set_image_identity({});
    if (auto* draft = models.draft()) draft->set_image_identity({});
    language.invalidate_prompt_reuse();
    std::vector<Eigen::bfloat16> logits;
    for (size_t i = 0; i < input_token_ids.size(); ++i) {
        if (request.cancelled()) throw std::runtime_error("Scoring request was interrupted");
        language.run_model_once(1, i, 0, input_token_ids[i], &logits);
    }
    request.finish();
    if (request.cancelled()) throw std::runtime_error("Scoring request was interrupted");
    return logits;
}


LogLikelihoodResult VisionLanguageModel::run_model_for_loglikelihood(
    std::span<const uint32_t> input_token_ids,
    size_t continuation_start,
    std::span<const uint32_t> continuation_token_ids,
    const GenerationOptions& options, bool use_group_prefill
) {
    ModelLock models(_state);
    if (options.stop.stop_requested()) throw std::runtime_error("Scoring request was interrupted");
    RequestScope request(models.target(), models.draft(), stream_options(options),
                         options.stop, options.stop_token_ids, options.cache_id);
    if (request.cancelled()) throw std::runtime_error("Scoring request was interrupted");
    auto& language = models.target();
    language.set_image_identity({});
    if (auto* draft = models.draft()) draft->set_image_identity({});
    language.invalidate_prompt_reuse();
    auto result = language._score(input_token_ids, continuation_start, continuation_token_ids, use_group_prefill);
    request.finish();
    if (request.cancelled()) throw std::runtime_error("Scoring request was interrupted");
    return result;
}


GenerationPerformanceResult VisionLanguageModel::run_model_for_ttnt(
    std::span<const uint32_t> input_token_ids,
    const GenerationOptions& options
) {
    ModelLock models(_state);
    if (options.stop.stop_requested()) throw std::runtime_error("Performance request was interrupted");
    RequestScope request(models.target(), models.draft(), stream_options(options),
                         options.stop, options.stop_token_ids, options.cache_id);
    if (request.cancelled()) throw std::runtime_error("Performance request was interrupted");
    auto& language = models.target();
    language.set_image_identity({});
    language.invalidate_prompt_reuse();
    if (auto* draft = models.draft()) {
        draft->set_image_identity({});
        draft->invalidate_prompt_reuse();
    }
    language.create_input_buffers(input_token_ids);
    GenerationPerformanceResult result;
    const auto limit = context_limit(input_token_ids.size(), options.max_new_tokens,
                                    config().pipeline_cfg.max_num_tokens);
    auto output = language._generate_tokens(request, input_token_ids, limit, ChronoTimer(true), &result, 0);
    request.finish();
    if (request.cancelled() || !output)
        throw std::runtime_error("Performance request was interrupted or has insufficient token capacity");
    // Profiling requests leave no reusable sequence state behind.
    language.invalidate_prompt_reuse();
    if (auto* draft = models.draft()) draft->invalidate_prompt_reuse();
    return result;
}


void VisionLanguageModel::stop_model() {
    _state->language_model_ptr->stop_model();
}

void VisionLanguageModel::set_draft_vlm(VisionLanguageModel* draft) {
    auto lock = idle_lock(_state->run_mutex);
    if (draft == this)
        throw std::invalid_argument("A model cannot draft for itself");
    const auto next = draft ? draft->_state : nullptr;
    if (_state->draft == next) return;
    if (next) {
        auto next_lock = idle_lock(next->run_mutex);
        _state->language_model_ptr->attach_draft(*next->language_model_ptr);
    } else {
        _state->language_model_ptr->invalidate_all_prompt_reuse();
    }
    _state->draft = next;
}

size_t VisionLanguageModel::shared_resource_bytes() const {
    return _state->language_model_ptr->shared_resource_bytes();
}
size_t VisionLanguageModel::workspace_memory_bytes() const {
    return _state->language_model_ptr->workspace_memory_bytes();
}

size_t VisionLanguageModel::kv_cache_count() const {
    auto lock = idle_lock(_state->run_mutex);
    return _state->language_model_ptr->kv_cache_count();
}

size_t VisionLanguageModel::kv_cache_bytes_per_slot() const {
    return _state->language_model_ptr->kv_cache_bytes_per_slot();
}

bool VisionLanguageModel::remove_kv_cache(const std::string& id) {
    auto lock = idle_lock(_state->run_mutex);
    std::unique_lock<std::mutex> draft_lock;
    if (_state->draft) draft_lock = idle_lock(_state->draft->run_mutex);
    const bool removed = _state->language_model_ptr->remove_kv_cache(id);
    if (_state->draft) _state->draft->language_model_ptr->remove_kv_cache(id);
    return removed;
}

void VisionLanguageModel::clear_kv_caches() {
    auto lock = idle_lock(_state->run_mutex);
    std::unique_lock<std::mutex> draft_lock;
    if (_state->draft) draft_lock = idle_lock(_state->draft->run_mutex);
    _state->language_model_ptr->clear_kv_caches();
    if (_state->draft) _state->draft->language_model_ptr->clear_kv_caches();
}

}
}
