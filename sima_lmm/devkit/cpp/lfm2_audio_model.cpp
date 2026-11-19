#include "lfm2_audio_model.hpp"
#include "mla_buffer.hpp"
#include "mla_model.hpp"
#include "tokenizer.hpp"
#include <minja/chat-template.hpp>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace simaai::llima {
namespace {
using Json = nlohmann::json;

struct View {
    MLABuffer* buffer = nullptr;
    std::vector<uint32_t> begin;
    std::vector<uint32_t> shape;
    MLABufferSlice slice() const {
        if (!buffer) throw std::logic_error("Audio buffer is not bound");
        return MLABufferSlice(buffer, begin, shape);
    }
};

void upload(MLABuffer& buffer, std::span<const float> values) {
    if (values.size() != buffer.get_num_elems()) throw std::invalid_argument("Audio input size differs from its buffer");
    std::vector<Eigen::bfloat16> packed(values.size());
    std::transform(values.begin(), values.end(), packed.begin(), [](float value) { return Eigen::bfloat16(value); });
    buffer.upload(packed.data());
}

std::vector<float> read_rows(const View& view, size_t start, size_t count, size_t width) {
    std::vector<Eigen::bfloat16> packed(view.buffer->get_num_elems());
    view.buffer->download(packed.data());
    if ((start + count) * width > packed.size()) throw std::out_of_range("Audio output row is outside its buffer");
    std::vector<float> values(count * width);
    std::transform(packed.begin() + start * width, packed.begin() + (start + count) * width,
                   values.begin(), [](auto value) { return static_cast<float>(value); });
    return values;
}

class Table {
public:
    Table(const std::filesystem::path& directory, const Json& description) {
        auto shape = description.at("shape").get<std::vector<size_t>>();
        if (shape.empty() || shape.size() > 2) throw std::invalid_argument("Unsupported audio lookup shape");
        _rows = shape.size() == 2 ? shape[0] : 1;
        _width = shape.back();
        const auto dtype = description.at("dtype").get<std::string>();
        _bf16 = dtype == "bfloat16";
        if (!_bf16 && dtype != "float32") throw std::invalid_argument("Unsupported audio lookup dtype");
        _bytes = _rows * _width * (_bf16 ? 2 : 4);
        auto path = directory / description.at("path").get<std::string>();
        if (std::filesystem::file_size(path) != _bytes) throw std::invalid_argument("Audio lookup size mismatch");
        const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor < 0) throw std::runtime_error("Cannot open audio lookup table");
        _data = mmap(nullptr, _bytes, PROT_READ, MAP_PRIVATE, descriptor, 0);
        close(descriptor);
        if (_data == MAP_FAILED) { _data = nullptr; throw std::runtime_error("Cannot map audio lookup table"); }
    }
    ~Table() { if (_data) munmap(_data, _bytes); }
    Table(const Table&) = delete;
    Table& operator=(const Table&) = delete;
    size_t width() const { return _width; }
    std::vector<float> row(size_t index) const {
        if (index >= _rows) throw std::out_of_range("Audio token is outside its lookup table");
        std::vector<float> result(_width);
        if (_bf16) {
            const auto* values = static_cast<const Eigen::bfloat16*>(_data) + index * _width;
            std::transform(values, values + _width, result.begin(), [](auto value) { return static_cast<float>(value); });
        } else {
            const auto* values = static_cast<const float*>(_data) + index * _width;
            std::copy(values, values + _width, result.begin());
        }
        return result;
    }
private:
    void* _data = nullptr;
    size_t _bytes = 0;
    size_t _rows = 0;
    size_t _width = 0;
    bool _bf16 = false;
};

class Unit {
public:
    Unit(const std::filesystem::path& directory, std::string name, const Json& description)
        : _name(std::move(name)), _description(description) {
        auto allocate = [&](const Json& specs, auto& buffers, auto& views) {
            for (const auto& spec : specs) {
                if (spec.at("dtype").get<std::string>().find("bfloat16") == std::string::npos) {
                    throw std::invalid_argument("Native audio interfaces must use BF16");
                }
                const bool aligned = spec.at("dram_layout").get<std::string>().find("HWC16") != std::string::npos;
                auto buffer = std::make_unique<MLABuffer>(
                    _name + "/" + spec.at("name").get<std::string>(),
                    spec.at("dram_shape").get<std::vector<size_t>>(), "bfloat16", aligned
                );
                buffer->allocate();
                buffer->clear();
                views.push_back({buffer.get(), std::vector<uint32_t>(4, 0), spec.at("shape").get<std::vector<uint32_t>>()});
                buffers.push_back(std::move(buffer));
            }
        };
        allocate(description.at("interfaces").at("mla_inputs"), _inputs, _input_views);
        allocate(description.at("interfaces").at("mla_outputs"), _outputs, _output_views);
        std::vector<MLABufferSlice> inputs, outputs;
        for (const auto& view : _input_views) inputs.push_back(view.slice());
        for (const auto& view : _output_views) outputs.push_back(view.slice());
        _model = std::make_unique<MLAModelWithBuffer>(
            directory / description.at("elf").get<std::string>(), std::move(inputs), std::move(outputs)
        );
    }
    ~Unit() { if (_loaded) _model->free(); }
    MLABuffer& input(const std::string& name) { return *_inputs.at(input_index(name)); }
    View output(size_t index = 0) { return _output_views.at(index); }
    void run(const std::map<std::string, View>& inputs = {}, const std::map<uint8_t, View>& outputs = {}) {
        if (!_loaded) { _model->load(); _loaded = true; }
        std::map<uint8_t, MLABufferSlice> ifms, ofms;
        for (const auto& [name, view] : inputs) ifms.emplace(input_index(name), view.slice());
        for (const auto& [index, view] : outputs) ofms.emplace(index, view.slice());
        _model->run(&ifms, &ofms);
    }
private:
    uint8_t input_index(const std::string& name) const {
        const auto& inputs = _description.at("interfaces").at("inputs");
        for (size_t index = 0; index < inputs.size(); ++index) {
            if (inputs[index].at("name") == name) return static_cast<uint8_t>(index);
        }
        throw std::invalid_argument("Unknown audio model input: " + name);
    }
    std::string _name;
    Json _description;
    std::vector<std::unique_ptr<MLABuffer>> _inputs, _outputs;
    std::vector<View> _input_views, _output_views;
    std::unique_ptr<MLAModelWithBuffer> _model;
    bool _loaded = false;
};

struct Family {
    std::string name;
    std::vector<std::string> layers;
    size_t width = 0, heads = 0, kv_heads = 0, capacity = 0, cached = 0, position = 0, window = 0;
    float theta = 1000000.f;
    std::vector<std::unique_ptr<MLABuffer>> keys, values, tails;
};
} // namespace

struct Lfm2AudioModel::Impl {
    std::filesystem::path directory;
    Json config;
    std::map<std::string, std::unique_ptr<Table>> tables;
    std::map<std::string, std::unique_ptr<Unit>> units;
    Family backbone, depth, decoder;
    View last_hidden;
    std::unique_ptr<AudioWaveformSynthesizer> waveform;
    std::unique_ptr<AudioFeatureExtractor> features;
    std::unique_ptr<Tokenizer> tokenizer;
    std::unique_ptr<minja::chat_template> chat_template;
    std::vector<std::array<uint32_t, 8>> pending_codes;
    std::recursive_mutex mutex;
    bool waveform_closed = false;
    AudioDecodeTimings decode_profile;

    Impl(const std::filesystem::path& path) : directory(path) {
        std::ifstream file(directory / "audio_runtime.json");
        if (!file) throw std::runtime_error("Missing audio_runtime.json");
        file >> config;
        if (config.at("model_type") != "lfm2_audio" || config.at("precision") != "A_BF16_W_INT8") {
            throw std::invalid_argument("Unsupported native audio model bundle");
        }
        tokenizer = Tokenizer::from_hf_json(directory / "tokenizer.json");
        Json tokenizer_config;
        std::ifstream(directory / "tokenizer_config.json") >> tokenizer_config;
        auto special_token = [&](const char* name) {
            const auto& value = tokenizer_config.at(name);
            return value.is_string() ? value.get<std::string>() : value.at("content").get<std::string>();
        };
        std::ifstream template_file(directory / "chat_template.jinja");
        if (!template_file) throw std::runtime_error("Missing audio chat template");
        std::string template_text((std::istreambuf_iterator<char>(template_file)), {});
        chat_template = std::make_unique<minja::chat_template>(template_text, special_token("bos_token"), special_token("eos_token"));
        if (tokenizer->token_to_id(special_token("eos_token")) != 7 ||
            tokenizer->token_to_id("<|audio_start|>") != 128 ||
            tokenizer->token_to_id("<|text_end|>") != 130) {
            throw std::invalid_argument("Audio tokenizer special IDs differ from the generation contract");
        }
        for (auto it = config.at("tables").begin(); it != config.at("tables").end(); ++it) {
            tables.emplace(it.key(), std::make_unique<Table>(directory / "tables", it.value()));
        }
        waveform = std::make_unique<AudioWaveformSynthesizer>(tables.at("window")->row(0));
        std::vector<float> filters;
        for (size_t index = 0; index < 128; ++index) {
            auto row = tables.at("mel_filters")->row(index);
            filters.insert(filters.end(), row.begin(), row.end());
        }
        features = std::make_unique<AudioFeatureExtractor>(tables.at("analysis_window")->row(0), filters);
        connect_mla_rt({});
        const auto& source = config.at("source_config");
        if (source.at("lfm").at("conv_L_cache") != 3 || config.at("detokenizer_config").at("conv_L_cache") != 3) {
            throw std::invalid_argument("Native audio runtime currently requires convolution cache length three");
        }
        const auto& options = config.at("options");
        initialize_family(backbone, "backbone", source.at("lfm"), options.at("context_length"));
        Json depth_config = {
            {"hidden_size", source.at("depthformer").at("dim")}, {"num_attention_heads", 32},
            {"num_key_value_heads", 8}, {"layer_types", std::vector<std::string>(
                source.at("depthformer").at("layers"), "full_attention")}
        };
        initialize_family(depth, "depth", depth_config, 8);
        const auto& detok = config.at("detokenizer_config");
        const size_t decoder_tokens = options.at("decoder_frames").get<size_t>() * 6;
        const size_t window = detok.value("sliding_window", 30);
        initialize_family(decoder, "detokenizer", detok, std::max<size_t>(128, decoder_tokens + window - 1));
        decoder.window = window;
    }
    ~Impl() {
        // Free only this model's registered executables. The process-global MLA
        // context can also be in use by another model in this process.
        units.clear();
    }
    Unit& unit(const std::string& name) {
        auto found = units.find(name);
        if (found == units.end()) {
            found = units.emplace(name, std::make_unique<Unit>(directory, name, config.at("units").at(name))).first;
        }
        return *found->second;
    }
    void initialize_family(Family& family, const std::string& name, const Json& source, size_t capacity) {
        family.name = name;
        family.width = source.at("hidden_size");
        family.heads = source.at("num_attention_heads");
        family.kv_heads = source.at("num_key_value_heads");
        family.layers = source.at("layer_types").get<std::vector<std::string>>();
        family.theta = source.value("rope_theta", 1000000.f);
        family.capacity = capacity;
        for (size_t index = 0; index < family.layers.size(); ++index) {
            auto make = [&](const std::string& kind, std::vector<size_t> shape) {
                auto buffer = std::make_unique<MLABuffer>(name + "/" + kind + std::to_string(index), shape, "bfloat16", true);
                buffer->allocate(); buffer->clear(); return buffer;
            };
            const bool conv = family.layers[index] == "conv";
            family.keys.push_back(conv ? nullptr : make("keys", {1, family.kv_heads, capacity, family.width / family.heads}));
            family.values.push_back(conv ? nullptr : make("values", {1, family.kv_heads, capacity, family.width / family.heads}));
            family.tails.push_back(conv ? make("tail", {1, 1, 2, family.width}) : nullptr);
        }
    }
    void reset_family(Family& family) {
        for (auto* buffers : {&family.keys, &family.values, &family.tails}) {
            for (auto& buffer : *buffers) if (buffer) buffer->clear();
        }
        family.cached = family.position = 0;
    }
    void shift_cache(MLABuffer& buffer, size_t begin, size_t count, const Family& family) {
        const size_t dim = family.width / family.heads;
        buffer.invalidate_cache();
        auto* data = static_cast<Eigen::bfloat16*>(buffer.get_virtual_addr());
        for (size_t head = 0; head < family.kv_heads; ++head) {
            auto* base = data + head * family.capacity * dim;
            std::memmove(base, base + begin * dim, count * dim * sizeof(Eigen::bfloat16));
        }
        buffer.flush_cache();
    }
    View run_family(Family& family, std::span<const float> embeddings, size_t length, size_t valid) {
        if (valid > length || embeddings.size() != length * family.width || family.cached + length > family.capacity) {
            throw std::invalid_argument("Audio transformer input exceeds its compiled cache contract");
        }
        const size_t dim = family.width / family.heads;
        const std::string prefix = family.name + "_n" + std::to_string(length);
        auto& cache = unit(prefix + "_cache" + std::to_string(family.capacity));
        std::vector<float> mask(length * family.capacity, -0x1.fep127f);
        for (size_t row = 0; row < length; ++row) {
            for (size_t key = 0; key < family.cached + valid; ++key) {
                const int64_t distance = static_cast<int64_t>(family.cached + row) - static_cast<int64_t>(key);
                if (distance >= 0 && (!family.window || distance < static_cast<int64_t>(family.window))) {
                    mask[row * family.capacity + key] = 0.f;
                }
            }
        }
        upload(cache.input("attn_mask"), mask);
        std::vector<float> real(length * dim / 2), imaginary(real.size());
        for (size_t row = 0; row < length; ++row) {
            for (size_t pair = 0; pair < dim / 2; ++pair) {
                const float phase = static_cast<float>(family.position + row)
                    / std::pow(family.theta, static_cast<float>(2 * pair) / dim);
                real[row * dim / 2 + pair] = std::cos(phase);
                imaginary[row * dim / 2 + pair] = std::sin(phase);
            }
        }
        View current;
        for (size_t index = 0; index < family.layers.size(); ++index) {
            const auto layer = prefix + "_layer" + std::to_string(index);
            if (family.layers[index] == "conv") {
                auto& conv = unit(layer + "_conv");
                if (!current.buffer) { upload(conv.input("input"), embeddings); }
                std::map<std::string, View> inputs{{"conv_cache", {family.tails[index].get(), {0, 0, 0, 0}, {1, 1, 2, static_cast<uint32_t>(family.width)}}}};
                if (current.buffer) inputs.emplace("input", current);
                conv.run(inputs);
                auto tail = conv.output(1);
                tail.buffer->invalidate_cache();
                const auto* raw = static_cast<const Eigen::bfloat16*>(tail.buffer->get_virtual_addr());
                // Padded prefill must retain the last two real input rows,
                // rather than advancing convolution state through padding.
                family.tails[index]->upload(raw + (valid - 1) * family.width);
                current = conv.output();
                if (index + 1 == family.layers.size()) {
                    auto& final = unit(layer + "_final");
                    final.run({{"input", current}});
                    current = final.output();
                }
            } else {
                auto& pre = unit(layer + "_pre");
                if (!current.buffer) upload(pre.input("input"), embeddings);
                upload(pre.input("freq_real"), real);
                upload(pre.input("freq_imag"), imaginary);
                auto key = View{family.keys[index].get(), {0, 0, 0, 0},
                    {1, static_cast<uint32_t>(family.kv_heads), static_cast<uint32_t>(family.capacity), static_cast<uint32_t>(dim)}};
                auto value = key; value.buffer = family.values[index].get();
                auto key_delta = key, value_delta = value;
                key_delta.begin[2] = value_delta.begin[2] = family.cached;
                key_delta.shape[2] = value_delta.shape[2] = length;
                pre.run(current.buffer ? std::map<std::string, View>{{"input", current}} : std::map<std::string, View>{},
                        {{1, key_delta}, {2, value_delta}});
                cache.run({{"input", pre.output()}, {"cached_keys", key}, {"cached_values", value}});
                auto& post = unit(layer + "_post");
                post.run({{"input", current.buffer ? current : View{&pre.input("input"), {0, 0, 0, 0},
                    {1, 1, static_cast<uint32_t>(length), static_cast<uint32_t>(family.width)}}}, {"self_attn", cache.output()}});
                current = post.output();
            }
        }
        family.position += valid;
        family.cached += valid;
        if (family.window) {
            const auto keep = std::min(family.cached, family.window - 1);
            for (size_t index = 0; index < family.layers.size(); ++index) if (family.keys[index]) {
                shift_cache(*family.keys[index], family.cached - keep, keep, family);
                shift_cache(*family.values[index], family.cached - keep, keep, family);
            }
            family.cached = keep;
        }
        return current;
    }
};

Lfm2AudioModel::Lfm2AudioModel(const std::filesystem::path& directory) : _impl(std::make_unique<Impl>(directory)) {}
Lfm2AudioModel::~Lfm2AudioModel() = default;
size_t Lfm2AudioModel::hidden_size() const { return _impl->backbone.width; }
void Lfm2AudioModel::reset() {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    for (auto* family : {&self.backbone, &self.depth, &self.decoder}) self.reset_family(*family);
    self.last_hidden = {}; self.pending_codes.clear(); self.waveform->reset(); self.waveform_closed = false;
    self.decode_profile = {};
}

std::vector<float> Lfm2AudioModel::forward(std::span<const float> embeddings) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    const size_t width = hidden_size(), group = self.config.at("options").at("group_size");
    if (embeddings.empty() || embeddings.size() % width) throw std::invalid_argument("Invalid audio backbone input");
    size_t offset = 0, last_valid = 0;
    while (offset < embeddings.size()) {
        const size_t remaining = (embeddings.size() - offset) / width;
        const size_t length = remaining >= group ||
            (remaining > 1 && self.backbone.cached + group <= self.backbone.capacity) ? group : 1;
        last_valid = std::min(remaining, length);
        auto input = embeddings.subspan(offset, last_valid * width);
        std::vector<float> padded;
        if (last_valid < length) {
            padded.resize(length * width, 0.f);
            std::copy(input.begin(), input.end(), padded.begin());
            input = padded;
        }
        self.last_hidden = self.run_family(self.backbone, input, length, last_valid);
        offset += last_valid * width;
    }
    auto values = read_rows(self.last_hidden, last_valid - 1, 1, width);
    self.last_hidden.begin[2] = last_valid - 1;
    self.last_hidden.shape[2] = 1;
    return values;
}

std::vector<float> Lfm2AudioModel::text_logits(std::span<const float> hidden) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    if (hidden.size() != hidden_size() || !self.last_hidden.buffer) throw std::invalid_argument("No audio backbone hidden state");
    std::vector<float> logits;
    for (size_t index = 0; ; ++index) {
        const auto name = "backbone_text_head" + std::to_string(index);
        if (!self.config.at("units").contains(name)) break;
        auto& head = self.unit(name); head.run({{"hidden", self.last_hidden}});
        const auto view = head.output();
        auto part = read_rows(view, 0, 1, view.shape.back());
        logits.insert(logits.end(), part.begin(), part.end());
    }
    return logits;
}

std::array<uint32_t, 8> Lfm2AudioModel::audio_codes(
    std::span<const float> hidden, const AudioSamplingOptions& options, std::mt19937_64& random
) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    if (hidden.size() != hidden_size() || !self.last_hidden.buffer) throw std::invalid_argument("No audio backbone hidden state");
    self.reset_family(self.depth);
    auto& projection = self.unit("depth_input"); projection.run({{"hidden", self.last_hidden}});
    auto input = read_rows(projection.output(), 0, 1, self.depth.width * 8);
    std::vector<float> previous(self.depth.width, 0.f);
    std::array<uint32_t, 8> result{};
    for (size_t index = 0; index < result.size(); ++index) {
        std::vector<float> embedding(self.depth.width);
        for (size_t column = 0; column < embedding.size(); ++column) embedding[column] = input[index * embedding.size() + column] + previous[column];
        auto output = self.run_family(self.depth, embedding, 1, 1);
        auto& head = self.unit("depth_head" + std::to_string(index)); head.run({{"hidden", output}});
        auto logits = read_rows(head.output(), 0, 1, 2049);
        result[index] = sample_audio_model_token(logits, options, random);
        previous = self.tables.at("depth_embeddings_" + std::to_string(index))->row(result[index]);
    }
    return result;
}

std::vector<float> Lfm2AudioModel::text_embedding(uint32_t token) {
    return _impl->tables.at("text_embeddings")->row(token);
}
std::vector<float> Lfm2AudioModel::audio_embedding(const std::array<uint32_t, 8>& codes) {
    std::vector<float> result(hidden_size(), 0.f);
    for (size_t index = 0; index < codes.size(); ++index) {
        if (codes[index] > 2048) throw std::out_of_range("Invalid audio generation code");
        auto row = _impl->tables.at("audio_embeddings")->row(index * 2049 + codes[index]);
        for (size_t column = 0; column < result.size(); ++column) result[column] += row[column];
    }
    for (auto& value : result) value = static_cast<float>(Eigen::bfloat16(value));
    return result;
}

std::vector<float> Lfm2AudioModel::make_prompt(std::span<const AudioPromptMessage> messages) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    if (messages.empty()) throw std::invalid_argument("Audio prompt needs at least one message");
    minja::chat_template_inputs inputs;
    inputs.messages = Json::array();
    std::vector<std::string> markers;
    for (size_t index = 0; index < messages.size(); ++index) {
        const auto& message = messages[index];
        if ((message.role != "system" && message.role != "user" && message.role != "assistant") ||
            (message.role == "system" && index != 0) || message.audio_embeddings.size() % hidden_size()) {
            throw std::invalid_argument("Invalid audio prompt role or dense embedding shape");
        }
        markers.push_back("\x1fLLIMA_AUDIO_CONTENT_" + std::to_string(index) + "\x1f");
        inputs.messages.push_back({{"role", message.role}, {"content", markers.back()}});
    }
    auto rendered = self.chat_template->apply(inputs);
    std::vector<float> embeddings;
    auto append_text = [&](const std::string& text) {
        for (auto token : self.tokenizer->encode(text, false)) {
            auto row = text_embedding(token);
            embeddings.insert(embeddings.end(), row.begin(), row.end());
        }
    };
    size_t offset = 0;
    for (size_t index = 0; index < messages.size(); ++index) {
        auto found = rendered.find(markers[index], offset);
        if (found == std::string::npos) throw std::runtime_error("Audio chat template discarded a message");
        append_text(rendered.substr(offset, found - offset));
        append_text(messages[index].text);
        const auto& audio = messages[index].audio_embeddings;
        embeddings.insert(embeddings.end(), audio.begin(), audio.end());
        offset = found + markers[index].size();
    }
    append_text(rendered.substr(offset));
    return embeddings;
}

std::string Lfm2AudioModel::decode_text(const std::vector<uint32_t>& tokens) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    return self.tokenizer->decode(tokens, true);
}

std::vector<float> Lfm2AudioModel::encode_mel(std::span<const float> features, size_t frames, size_t physical_frames) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    const size_t capacity = self.config.at("options").at("audio_frames");
    const size_t channels = self.config.at("source_config").at("encoder").at("feat_in");
    if (!frames || frames > physical_frames || physical_frames > capacity || features.size() != channels * frames) {
        throw std::invalid_argument("Audio features exceed the compiled encoder shape");
    }
    std::vector<float> padded(capacity * channels, 0.f);
    for (size_t time = 0; time < frames; ++time) for (size_t channel = 0; channel < channels; ++channel) {
        padded[time * channels + channel] = features[channel * frames + time];
    }
    auto& stem = self.unit("encoder_subsampling_f" + std::to_string(capacity));
    upload(stem.input("features"), padded);
    for (size_t index = 0, divisor = 2; index < 3; ++index, divisor *= 2) {
        std::vector<float> valid((capacity + divisor - 1) / divisor, 0.f);
        std::fill_n(valid.begin(), (physical_frames + divisor - 1) / divisor, 1.f);
        upload(stem.input("stage" + std::to_string(index + 1) + "_valid"), valid);
    }
    stem.run(); auto current = stem.output();
    const size_t encoded = (capacity + 7) / 8, valid = (frames + 7) / 8;
    std::vector<float> tokens(encoded, 0.f), attention(encoded * encoded, 0.f);
    std::fill_n(tokens.begin(), valid, 1.f);
    for (size_t row = 0; row < valid; ++row) std::fill_n(attention.begin() + row * encoded, valid, 1.f);
    const size_t layers = self.config.at("source_config").at("encoder").at("n_layers");
    for (size_t index = 0; index < layers; ++index) {
        auto& layer = self.unit("encoder_n" + std::to_string(encoded) + "_layer" + std::to_string(index));
        upload(layer.input("attention_valid"), attention); upload(layer.input("tokens_valid"), tokens);
        layer.run({{"hidden", current}}); current = layer.output();
    }
    auto& adapter = self.unit("encoder_n" + std::to_string(encoded) + "_adapter");
    adapter.run({{"hidden", current}});
    return read_rows(adapter.output(), 0, valid, hidden_size());
}

std::vector<float> Lfm2AudioModel::encode_pcm(std::span<const float> pcm) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    auto mel = self.features->process(pcm);
    return encode_mel(mel.values, mel.frames, mel.frames);
}

std::vector<float> Lfm2AudioModel::decode_codes(std::span<const std::array<uint32_t, 8>> codes, bool finish) {
    auto& self = *_impl; std::lock_guard lock(self.mutex);
    if (self.waveform_closed) throw std::logic_error("Audio output is finished; reset before another response");
    for (const auto& frame : codes) for (auto value : frame) if (value >= 2048) throw std::out_of_range("End-of-audio codes cannot be decoded");
    self.pending_codes.insert(self.pending_codes.end(), codes.begin(), codes.end());
    const size_t chunk = self.config.at("options").at("decoder_frames");
    std::vector<float> waveform;
    while (self.pending_codes.size() >= chunk || (finish && !self.pending_codes.empty())) {
        const size_t count = std::min(chunk, self.pending_codes.size()), width = self.decoder.width;
        std::vector<float> input(chunk * 6 * width, 0.f);
        for (size_t frame = 0; frame < count; ++frame) {
            std::vector<float> mean(width, 0.f);
            for (size_t codebook = 0; codebook < 8; ++codebook) {
                auto row = self.tables.at("decoder_embeddings")->row(codebook * 2048 + self.pending_codes[frame][codebook]);
                for (size_t column = 0; column < width; ++column) mean[column] += row[column];
            }
            for (size_t repeat = 0; repeat < 6; ++repeat) for (size_t column = 0; column < width; ++column) {
                input[(frame * 6 + repeat) * width + column] = mean[column] / 8.f;
            }
        }
        const auto network_started = std::chrono::steady_clock::now();
        auto hidden = self.run_family(self.decoder, input, chunk * 6, count * 6);
        auto& spectrum = self.unit("detokenizer_n" + std::to_string(chunk * 6) + "_spectrum");
        spectrum.run({{"hidden", hidden}});
        auto spectral = read_rows(spectrum.output(), 0, count * 6, 1282);
        const auto waveform_started = std::chrono::steady_clock::now();
        auto samples = self.waveform->push(spectral); waveform.insert(waveform.end(), samples.begin(), samples.end());
        self.decode_profile.network_ms += std::chrono::duration<double, std::milli>(waveform_started-network_started).count();
        self.decode_profile.waveform_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-waveform_started).count();
        self.pending_codes.erase(self.pending_codes.begin(), self.pending_codes.begin() + count);
    }
    if (finish) {
        const auto waveform_started = std::chrono::steady_clock::now();
        auto tail = self.waveform->finish(); waveform.insert(waveform.end(), tail.begin(), tail.end());
        self.decode_profile.waveform_cpu_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-waveform_started).count();
        self.waveform_closed = true;
    }
    return waveform;
}

AudioDecodeTimings Lfm2AudioModel::decode_timings() const {
    std::lock_guard lock(_impl->mutex);
    return _impl->decode_profile;
}

} // namespace simaai::llima
