#include <spdlog/spdlog.h>
#include <openssl/sha.h>

#include <stdexcept>
#include <utility>

#include "reasoning_parser.hpp"
#include "utils.hpp"
#include "vision_language_model.hpp"
#include "language_model_state.hpp"
#include "language_model_executor.hpp"
#include "vision_model.hpp"
#include "vlm_helper.hpp"

namespace simaai {
namespace llima {

struct EncodedImageData {
    std::weak_ptr<ModelResources<VlmConfig>> resources;
    std::string identity;
    std::map<uint8_t, std::vector<std::byte>> outputs;
};

// Language execution state retained independently of the vision facade.
struct LanguageRunnerState {
    friend class VisionLanguageModel;
    LanguageRunnerState(std::filesystem::path path, std::shared_ptr<FileProvider> provider,
                        std::optional<std::string> system, std::optional<std::string> chat_template)
        : resources(ModelResources<VlmConfig>::acquire(path, std::move(provider))),
          _cfg(resources->cfg), _files(resources->files),
          vlm_helper(_cfg, *_files, std::move(system), std::move(chat_template)),
          text_streamer(vlm_helper.get_tokenizer(), std::nullopt, std::nullopt) {}
    std::optional<std::vector<uint32_t>>
    run_language_model(std::span<const uint32_t> input_token_ids,
                       std::optional<uint16_t> max_num_tokens,
                       std::optional<ChronoTimer> timer_ttft = std::nullopt,
                       GenerationPerformanceResult* performance_result = nullptr,
                       uint16_t stable_prefix_token_count = 0);
    std::shared_ptr<ModelResources<VlmConfig>> resources;
    const VlmConfig& _cfg;
    const std::shared_ptr<FileProvider>& _files;
    VlmHelper vlm_helper;
    TextStreamer text_streamer;
    ToolCallFormat tool_call_format = ToolCallFormat::GenericJson;
    std::unique_ptr<LanguageModel> language_model_ptr;
    std::mutex run_mutex;
    mutable std::mutex defaults_mutex;
    std::shared_ptr<LanguageRunnerState> draft;
    // Cancellation reads this lease without taking the request mutex.
    std::atomic<std::shared_ptr<LanguageRunnerState>> active_draft;
    std::weak_ptr<LanguageSession> default_draft_context;
    // Keep borrowed bindings alive without locking another runner during teardown.
    std::shared_ptr<LanguageSession> selected_draft_context;
};

// Locks and leases both workspaces before preparing prompt/image data.
class VisionLanguageRequest {
  public:
    VisionLanguageRequest(std::shared_ptr<LanguageRunnerState> state, SessionId session,
                          bool streaming = true, const GenerationOptions* options = nullptr);
    ~VisionLanguageRequest();
    void clear_images();
    bool cancelled() const;

  private:
    std::shared_ptr<LanguageRunnerState> _target;
    std::shared_ptr<LanguageRunnerState> _draft;
    std::unique_lock<std::mutex> _target_lock;
    std::unique_lock<std::mutex> _draft_lock;
    bool _streaming;
    bool _was_enabled;
    std::pair<TextStreamer::InfoCallback, TextStreamer::TextCallback> _callbacks;
    std::unique_ptr<LanguageExecutor::Run> _execution;
};

VisionLanguageRequest::VisionLanguageRequest(std::shared_ptr<LanguageRunnerState> state,
                                             SessionId session, bool streaming,
                                             const GenerationOptions* options)
    : _target(std::move(state)), _target_lock(_target->run_mutex, std::defer_lock),
      _streaming(streaming) {
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
    const auto previous_target = _target->language_model_ptr->active_session();
    auto* previous_draft = _draft ? _draft->language_model_ptr->_session : nullptr;
    const auto previous_context = _draft ? _draft->selected_draft_context : nullptr;
    _callbacks = _target->text_streamer.callbacks();
    _was_enabled = _target->text_streamer.enabled();
    try {
        _target->language_model_ptr->select_session(session);
        _target->selected_draft_context.reset();
        if (_draft) {
            auto& context = _target->language_model_ptr->_sessions->get(session).draft;
            if (!context) {
                auto& initial = _draft->language_model_ptr->_sessions->get(0);
                if (_draft->default_draft_context.expired()) {
                    initial.reset();
                    // Lease the existing cache; the owner keeps its tokenizer and buffers alive.
                    context = std::shared_ptr<LanguageSession>(
                        &initial, [owner = _draft](auto*) mutable { owner.reset(); });
                    _draft->default_draft_context = context;
                } else {
                    auto created = std::make_shared<LanguageSession>();
                    created->init_layout(initial);
                    context = std::move(created);
                }
            }
            _draft->language_model_ptr->_select_session(*context);
            // The default context already belongs to the runner; retaining its alias would cycle.
            _draft->selected_draft_context =
                context.get() == &_draft->language_model_ptr->_sessions->get(0) ? nullptr : context;
        }
        if (options) {
            if (options->info_callback)
                _target->text_streamer.set_info_callback(*options->info_callback);
            if (options->text_callback)
                _target->text_streamer.set_text_callback(*options->text_callback);
        }
        if (!streaming)
            _target->text_streamer.disable();
        _target->active_draft.store(_draft);
        _execution = std::make_unique<LanguageExecutor::Run>(
            *_target->language_model_ptr, _draft ? _draft->language_model_ptr.get() : nullptr);
    } catch (...) {
        _target->active_draft.store(nullptr);
        _target->text_streamer.set_info_callback(std::move(_callbacks.first));
        _target->text_streamer.set_text_callback(std::move(_callbacks.second));
        if (_was_enabled)
            _target->text_streamer.enable();
        else
            _target->text_streamer.disable();
        _target->language_model_ptr->select_session(previous_target);
        if (_draft) {
            _draft->language_model_ptr->_select_session(*previous_draft);
            _draft->selected_draft_context = previous_context;
        }
        throw;
    }
}

bool VisionLanguageRequest::cancelled() const {
    return _target->language_model_ptr->_run->cancelled.load(std::memory_order_relaxed);
}

void VisionLanguageRequest::clear_images() {
    _target->language_model_ptr->set_image_identity({});
    if (_draft)
        _draft->language_model_ptr->set_image_identity({});
}

VisionLanguageRequest::~VisionLanguageRequest() {
    if (!_streaming)
        _target->text_streamer.disable();
    _execution.reset();
    _target->text_streamer.set_info_callback(std::move(_callbacks.first));
    _target->text_streamer.set_text_callback(std::move(_callbacks.second));
    _target->active_draft.store(nullptr);
    if (_was_enabled)
        _target->text_streamer.enable();
    else
        _target->text_streamer.disable();
}

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
    std::lock_guard lock(_state->defaults_mutex);
    return Chat(_state->vlm_helper);
}

ToolCallFormat VisionLanguageModel::tool_call_format() const { return _state->tool_call_format; }

void VisionLanguageModel::set_info_callback(TextStreamer::InfoCallback callback) {
    std::unique_lock lock(_state->run_mutex, std::try_to_lock);
    if (!lock.owns_lock()) throw std::logic_error("Callback defaults cannot change during a request");
    _state->text_streamer.set_info_callback(std::move(callback));
}

void VisionLanguageModel::set_text_callback(TextStreamer::TextCallback callback) {
    std::unique_lock lock(_state->run_mutex, std::try_to_lock);
    if (!lock.owns_lock()) throw std::logic_error("Callback defaults cannot change during a request");
    _state->text_streamer.set_text_callback(std::move(callback));
}

void VisionLanguageModel::wait_for_streamer_completion() {
    _state->text_streamer.wait_streaming();
}

void VisionLanguageModel::set_reloc(const std::string& name) { set_reloc(0, name); }
void VisionLanguageModel::unset_reloc() { unset_reloc(0); }

void VisionLanguageModel::set_reloc(SessionId session, const std::string& name) {
    VisionLanguageRequest request(_state, session, false);
    _state->language_model_ptr->set_reloc(name);
    if (auto draft = _state->active_draft.load()) draft->language_model_ptr->invalidate_prompt_reuse();
}

void VisionLanguageModel::unset_reloc(SessionId session) {
    VisionLanguageRequest request(_state, session, false);
    _state->language_model_ptr->unset_reloc();
    if (auto draft = _state->active_draft.load()) draft->language_model_ptr->invalidate_prompt_reuse();
}

void VisionLanguageModel::set_enable_thinking(bool enable_thinking) {
    std::lock_guard lock(_state->defaults_mutex);
    _state->vlm_helper.set_enable_thinking(enable_thinking);
}

bool VisionLanguageModel::get_enable_thinking() const {
    std::lock_guard lock(_state->defaults_mutex);
    return _state->vlm_helper.get_enable_thinking();
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
) : _state(std::make_shared<LanguageRunnerState>(model_path, file_provider, system_prompt, chat_template))
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
        _state->_files
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
        run_model(chat, 2);
        _state->text_streamer.enable();
    }

    // Pre-warm OMP thread pool to avoid libgomp first-call cost.
    warmup_omp();
}


std::vector<EncodedImage> VisionLanguageModel::encode_images(std::span<const cv::Mat> images) {
    if (!_vision_model) throw std::logic_error("Model does not support images");
    VisionLanguageRequest request(_state, 0, false);
    const auto tokens_per_image = _state->_cfg.mm_cfg->mm_tokens_per_image;
    if (images.size() > _state->_cfg.pipeline_cfg.max_num_tokens / tokens_per_image)
        throw std::invalid_argument("Image count exceeds the compiled context");
    std::vector<uint32_t> placeholders(images.size() * tokens_per_image,
                                      *_state->vlm_helper.get_image_token_id());
    auto destinations = _state->language_model_ptr->create_input_buffers(placeholders);
    std::vector<EncodedImage> encoded;
    for (size_t i = 0; i < images.size(); ++i) {
        auto tensor = _state->vlm_helper.preprocess_image(images[i]);
        _vision_model->run_model(tensor, &destinations.at(i));
        auto data = std::make_shared<EncodedImageData>();
        data->resources = _vision_model->_resources;
        data->identity = image_digest(tensor);
        for (const auto& [port, slice] : destinations.at(i)) {
            auto* buffer = slice.get_buf_ptr();
            const auto offset = buffer->get_buf_addr_offset(slice.get_buf_begins());
            const auto length = buffer->get_buf_len(slice.get_buf_shapes());
            buffer->invalidate_cache(offset, length);
            const auto* bytes = static_cast<const std::byte*>(buffer->get_virtual_addr()) + offset;
            data->outputs.emplace(port, std::vector<std::byte>(bytes, bytes + length));
        }
        EncodedImage image;
        image._data = std::move(data);
        encoded.emplace_back(std::move(image));
    }
    return encoded;
}

std::optional<std::string> VisionLanguageModel::run_model(const Chat& chat, std::optional<uint16_t> max_new_tokens) {
    return run_model(0, chat, max_new_tokens);
}

std::vector<uint32_t> VisionLanguageModel::run_model(std::span<const uint32_t> input_token_ids, std::optional<uint16_t> override_max_num_tokens, std::optional<std::set<uint32_t>> override_stop_token_ids) {
    return run_model(0, input_token_ids, override_max_num_tokens, override_stop_token_ids);
}

std::vector<Eigen::bfloat16> VisionLanguageModel::run_model_for_logits(std::span<const uint32_t> input_token_ids) {
    return run_model_for_logits(0, input_token_ids);
}

LogLikelihoodResult VisionLanguageModel::run_model_for_loglikelihood(std::span<const uint32_t> input_token_ids, size_t continuation_start, std::span<const uint32_t> continuation_token_ids, bool use_group_prefill) {
    return run_model_for_loglikelihood(0, input_token_ids, continuation_start, continuation_token_ids, use_group_prefill);
}

GenerationPerformanceResult VisionLanguageModel::run_model_for_ttnt(std::span<const uint32_t> input_token_ids, std::optional<uint16_t> override_max_num_tokens, std::optional<std::set<uint32_t>> override_stop_token_ids) {
    return run_model_for_ttnt(0, input_token_ids, override_max_num_tokens, override_stop_token_ids);
}

std::optional<std::string> VisionLanguageModel::run_model(
    SessionId session, const Chat& chat, std::optional<uint16_t> max_new_tokens
) {
    GenerationOptions options;
    options.max_new_tokens = max_new_tokens;
    return run_model(session, chat, options);
}

std::optional<std::string> VisionLanguageModel::run_model(
    const Chat& chat, const GenerationOptions& options
) {
    return run_model(0, chat, options);
}

std::optional<std::string> VisionLanguageModel::run_model(
    SessionId session, const Chat& chat, const GenerationOptions& options
) {
    VisionLanguageRequest request(_state, session, true, &options);
    if (options.stop_token_ids) _state->language_model_ptr->set_stop_token_ids(options.stop_token_ids);
    if (request.cancelled()) return std::nullopt;
    std::vector<std::pair<uint32_t, std::string>> preserved_tokens;
    auto* tokenizer = _state->vlm_helper.get_tokenizer();
    if (chat.has_tools()) {
        preserved_tokens = resolve_tool_call_special_tokens(_state->tool_call_format, *tokenizer);
    }
    const auto reasoning_format = reasoning_format_for_model(_state->_cfg.model_type);
    if (chat.get_enable_thinking() && reasoning_format == ReasoningFormat::None) {
        throw std::invalid_argument(
            "Thinking is not supported for model type '" + _state->_cfg.model_type + "'"
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
    const bool preserve_structural_tokens = !preserved_tokens.empty();
    _state->text_streamer.set_preserved_token_ids(std::move(preserved_tokens));
    // Keep the existing setter for ABI compatibility; it now gates all
    // request-specific structural markers, not only tool-call markers.
    _state->text_streamer.set_tool_call_enabled(preserve_structural_tokens);
    
    ChronoTimer timer_ttft(true);

    // Preprocess the chat.
    ChronoTimer timer_preprocess(true);
    _state->resources->logger->info("Chat:\n{}", chat.get_messages().dump(4));
    const bool supplied_images = !options.images.empty() || !options.encoded_images.empty();
    auto preprocessed_data = _state->vlm_helper.preprocess(chat, !supplied_images);
    if (supplied_images) {
        if (options.images.size() + options.encoded_images.size() != preprocessed_data.image_count)
            throw std::invalid_argument("Supplied image count must match the chat image placeholders");
        for (const auto& image : options.images)
            preprocessed_data.image_tensors.emplace_back(_state->vlm_helper.preprocess_image(image));
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
    if (auto draft = _state->active_draft.load())
        draft->language_model_ptr->set_image_identity(std::move(image_identity));
    _state->resources->logger->info("Formatted prompt: {}", preprocessed_data.formatted_prompt);
    _state->resources->logger->info("Preprocess time: {:.2f}s", timer_preprocess.stop());

    if (request.cancelled()) return std::nullopt;

    // Create buffers for the input embeds and deepstack features (for qwen3) and ofm maps for
    // vision models.
    auto vision_ofm_maps = _state->language_model_ptr->create_input_buffers(
        preprocessed_data.input_token_ids
    );

    // Encode the image tensors to embeddings. The image embeddings are directly written to the
    // input embeddings for the language models.
    ChronoTimer timer_image_encode(true);
    for (size_t i = 0; i < preprocessed_data.image_tensors.size(); ++i) {
        if (request.cancelled()) return std::nullopt;
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

    // Decode the text and image embeddings. When a draft VLM is configured
    // (set_draft_vlm), dispatch to speculative decoding; otherwise run the
    // normal language-model decode loop.
    std::optional<uint16_t> max_num_tokens{};
    if (options.max_new_tokens)
        max_num_tokens = std::min<size_t>(_state->_cfg.pipeline_cfg.max_num_tokens,
            preprocessed_data.input_token_ids.size() + *options.max_new_tokens);

    auto output_token_ids = _state->run_language_model(
        preprocessed_data.input_token_ids, max_num_tokens, timer_ttft,
        nullptr, preprocessed_data.stable_prefix_token_count
    );

    if (output_token_ids.has_value())
        return _state->vlm_helper.get_tokenizer()->decode(output_token_ids.value(), true);
    return std::nullopt;
}


std::optional<std::vector<uint32_t>> LanguageRunnerState::run_language_model(
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> max_num_tokens,
    std::optional<ChronoTimer> timer_ttft,
    GenerationPerformanceResult* performance_result,
    uint16_t stable_prefix_token_count
) {
    if (!active_draft.load()) {
        return language_model_ptr->run_model(
            input_token_ids, timer_ttft, max_num_tokens, std::nullopt, stable_prefix_token_count
        );
    }
    if (!_cfg.lm_cfg.speculative_decoding_cfg.has_value()) {
        throw std::runtime_error(
            "Draft VLM is configured but target model lacks speculative decoding config"
        );
    }
    return language_model_ptr->run_model_speculative_decoding(
        *active_draft.load()->language_model_ptr, input_token_ids,
        max_num_tokens, timer_ttft, performance_result, stable_prefix_token_count
    );
}


std::vector<uint32_t> VisionLanguageModel::run_model(
    SessionId session,
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids
) {
    VisionLanguageRequest request(_state, session, false);
    request.clear_images();
    _state->text_streamer.set_tool_call_enabled(false);
    _state->language_model_ptr->set_stop_token_ids(override_stop_token_ids);
    _state->language_model_ptr->create_input_buffers(input_token_ids);
    auto output = _state->run_language_model(input_token_ids, override_max_num_tokens);
    return output.value_or(std::vector<uint32_t>());
}


std::vector<Eigen::bfloat16> VisionLanguageModel::run_model_for_logits(
    SessionId session,
    std::span<const uint32_t> input_token_ids
) {
    VisionLanguageRequest request(_state, session, false);
    request.clear_images();
    _state->language_model_ptr->invalidate_prompt_reuse();
    std::vector<Eigen::bfloat16> logits;
    for (size_t i = 0; i < input_token_ids.size(); ++i) {
        if (request.cancelled()) throw std::runtime_error("Scoring request was interrupted");
        _state->language_model_ptr->run_model_once(1, i, 0, input_token_ids[i], &logits);
    }
    return logits;
}


LogLikelihoodResult VisionLanguageModel::run_model_for_loglikelihood(
    SessionId session,
    std::span<const uint32_t> input_token_ids,
    size_t continuation_start,
    std::span<const uint32_t> continuation_token_ids,
    bool use_group_prefill
) {
    VisionLanguageRequest request(_state, session, false);
    request.clear_images();
    _state->language_model_ptr->invalidate_prompt_reuse();
    return _state->language_model_ptr->run_model_for_loglikelihood(
        input_token_ids, continuation_start, continuation_token_ids, use_group_prefill);
}


GenerationPerformanceResult VisionLanguageModel::run_model_for_ttnt(
    SessionId session,
    std::span<const uint32_t> input_token_ids,
    std::optional<uint16_t> override_max_num_tokens,
    std::optional<std::set<uint32_t>> override_stop_token_ids
) {
    VisionLanguageRequest request(_state, session, false);
    request.clear_images();
    // Given a list of input token ids, return a list of time in seconds for each generated token.
    // The text streamer is disabled for this mode.

    _state->language_model_ptr->create_input_buffers(input_token_ids);
    GenerationPerformanceResult result;

    _state->language_model_ptr->set_stop_token_ids(override_stop_token_ids);
    if (auto draft = _state->active_draft.load()) {
        auto output = _state->run_language_model(input_token_ids, override_max_num_tokens,
                                          std::nullopt, &result);
        if (!output) throw std::runtime_error("Speculative performance request was interrupted or has insufficient token capacity");
        _state->language_model_ptr->invalidate_prompt_reuse();
        draft->language_model_ptr->invalidate_prompt_reuse();
        return result;
    }

    std::vector<double> ttnt;

    ChronoTimer timer(true);

    // Generate the first token.
    auto prefill_result = _state->language_model_ptr->run_model(
        input_token_ids, std::nullopt, input_token_ids.size(), override_stop_token_ids
    );
    ttnt.emplace_back(timer.stop(true));
    if (!prefill_result.has_value()) {
        _state->language_model_ptr->invalidate_prompt_reuse();
        throw std::runtime_error("Performance request was interrupted during prefill");
    }
    auto next_token_id = _state->language_model_ptr->get_cached_first_generated_token();

    if (prefill_result->empty() && !override_stop_token_ids.value_or(
            _state->language_model_ptr->get_stop_token_ids()).contains(next_token_id)) {
        // Override the max_num_tokens and stop_token_ids.
        _state->language_model_ptr->set_max_num_tokens(override_max_num_tokens);

        // Generate the new tokens until the stop token.
        for (
            size_t i = input_token_ids.size(); i < _state->language_model_ptr->get_max_num_tokens(); ++i
        ) {
            if (request.cancelled()) throw std::runtime_error("Performance request was interrupted");
            next_token_id = _state->language_model_ptr->run_model_once(
                1, i, input_token_ids.size(), next_token_id
            );
            ttnt.emplace_back(timer.stop(true));
            _state->resources->logger->info("next_token_id: {}", next_token_id);
            if (_state->language_model_ptr->get_stop_token_ids().contains(next_token_id))
                break;
        }

    }

    // Clear the token id cache to have correct measurement.
    _state->language_model_ptr->invalidate_prompt_reuse();

    result.token_durations = std::move(ttnt);
    result.generated_tokens = result.token_durations.size();
    return result;
}


void VisionLanguageModel::stop_model() {
    _state->language_model_ptr->stop_model();
    if (auto draft = _state->active_draft.load()) draft->language_model_ptr->stop_model();
}


namespace {
std::unique_lock<std::mutex> idle_lock(std::mutex& mutex) {
    std::unique_lock lock(mutex, std::try_to_lock);
    if (!lock.owns_lock())
        throw std::logic_error("Session settings cannot change during a request");
    return lock;
}
} // namespace

void VisionLanguageModel::set_draft_vlm(VisionLanguageModel* draft) {
    auto lock = idle_lock(_state->run_mutex);
    if (draft == this)
        throw std::invalid_argument("A model cannot draft for itself");
    const auto previous = _state->draft;
    const auto next = draft ? draft->_state : nullptr;
    if (previous == next)
        return;
    std::unique_lock<std::mutex> previous_lock;
    if (previous)
        previous_lock = idle_lock(previous->run_mutex);
    std::unique_lock<std::mutex> next_lock;
    if (next) {
        next_lock = idle_lock(next->run_mutex);
        _state->language_model_ptr->attach_draft(*next->language_model_ptr);
    }
    if (previous) {
        previous->language_model_ptr->select_session(previous->language_model_ptr->active_session());
        previous->selected_draft_context.reset();
    }
    for (auto id : _state->language_model_ptr->session_ids()) {
        auto& session = _state->language_model_ptr->_sessions->get(id);
        session.draft.reset();
        session.invalidate_reuse();
    }
    _state->draft = next;
}

SessionId VisionLanguageModel::create_session() {
    auto lock = idle_lock(_state->run_mutex);
    return _state->language_model_ptr->create_session();
}

void VisionLanguageModel::reset_session(SessionId id) {
    auto lock = idle_lock(_state->run_mutex);
    std::unique_lock<std::mutex> draft_lock;
    if (_state->draft)
        draft_lock = idle_lock(_state->draft->run_mutex);
    _state->language_model_ptr->reset_session(id);
}

void VisionLanguageModel::release_session(SessionId id) {
    auto lock = idle_lock(_state->run_mutex);
    std::unique_lock<std::mutex> draft_lock;
    if (const auto& draft = _state->draft) {
        draft_lock = idle_lock(draft->run_mutex);
        const auto& context = _state->language_model_ptr->_sessions->get(id).draft;
        if (context && draft->language_model_ptr->_session == context.get()) {
            draft->language_model_ptr->select_session(draft->language_model_ptr->active_session());
            draft->selected_draft_context.reset();
        }
    }
    _state->language_model_ptr->release_session(id);
}

void VisionLanguageModel::set_session_limit(size_t limit) {
    auto lock = idle_lock(_state->run_mutex);
    _state->language_model_ptr->set_session_limit(limit);
}

size_t VisionLanguageModel::session_memory_bytes(SessionId id) const {
    auto lock = idle_lock(_state->run_mutex);
    return _state->language_model_ptr->session_memory_bytes(id);
}
size_t VisionLanguageModel::shared_resource_bytes() const {
    return _state->language_model_ptr->shared_resource_bytes();
}
size_t VisionLanguageModel::workspace_memory_bytes() const {
    return _state->language_model_ptr->workspace_memory_bytes();
}

}
}
