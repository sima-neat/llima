#include <algorithm>
#include <cmath>
#include <iterator>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <cnpy.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "utils.hpp"
#include "whisper_model.hpp"


namespace simaai {
namespace llima {

WhisperPreprocessor::WhisperPreprocessor(const std::filesystem::path& devkit_dir) {
    // Load mel filters.
    auto json = nlohmann::json::parse(std::ifstream(devkit_dir / "preprocessor_config.json"));
    auto mel_filters_json = json["mel_filters"];
    _mel_filters.resize(mel_filters_json.size(), mel_filters_json[0].size());
    for (uint32_t i = 0; i < mel_filters_json.size(); ++i) {
        _mel_filters.row(i) = Eigen::Map<Eigen::RowVectorXd>(
            mel_filters_json[i].get<std::vector<double>>().data(), mel_filters_json[i].size()
        );
    }

    // Precompute the periodic Hann window used by Transformers/OpenAI Whisper.
    _hanning_window.resize(N_FFT);
    for (uint32_t i = 0; i < N_FFT; ++i)
        _hanning_window[i] = 0.5 - 0.5 * std::cos(2.0 * std::numbers::pi * i / N_FFT);

    _padded_audio_tensor.resize(N_SAMPLES + N_FFT);

    // FFT.
    _fft_in = fftw_alloc_real(N_FFT);
    if (!_fft_in) throw std::bad_alloc();
    _fft_out = fftw_alloc_complex(N_FRAMES * (N_FFT / 2 + 1));
    if (!_fft_out) throw std::bad_alloc();
    _fft_plan = fftw_plan_dft_r2c_1d(N_FFT, _fft_in, _fft_out, FFTW_PATIENT);
    if (!_fft_plan) {
        fftw_free(_fft_in);
        fftw_free(_fft_out);
        throw std::runtime_error("FFTW plan creation failed");
    }
}


WhisperPreprocessor::~WhisperPreprocessor() {
    fftw_free(_fft_in);
    fftw_free(_fft_out);
    fftw_destroy_plan(_fft_plan);
}


ArrayXXbf WhisperPreprocessor::preprocess(
    const std::filesystem::path& audio_file_name
) {
    auto audio_tensor = _load_audio_ffmpeg(audio_file_name);
    auto log_spec = _log_mel_spectrogram(audio_tensor);
    return log_spec;
}


ArrayXXbf WhisperPreprocessor::preprocess_pcm(
    std::span<const float> pcm,
    uint32_t sample_rate
) {
    if (sample_rate != SAMPLE_RATE) {
        throw std::runtime_error("WhisperPreprocessor::preprocess_pcm requires 16 kHz PCM");
    }
    ArrayXf audio_tensor = ArrayXf::Zero(N_SAMPLES);
    const auto copy_len = std::min<size_t>(pcm.size(), N_SAMPLES);
    std::copy_n(pcm.data(), copy_len, audio_tensor.data());
    return _log_mel_spectrogram(audio_tensor);
}


namespace {
// Helper deleters for FFmpeg RAII
struct FFmpegDeleter {
    void operator()(AVFormatContext* p) { avformat_close_input(&p); }
    void operator()(AVCodecContext* p) { avcodec_free_context(&p); }
    void operator()(SwrContext* p) { swr_free(&p); }
    void operator()(AVPacket* p) { av_packet_free(&p); }
    void operator()(AVFrame* p) { av_frame_free(&p); }
};
template<typename T> using ScopedPtr = std::unique_ptr<T, FFmpegDeleter>;
} 

ArrayXf WhisperPreprocessor::_load_audio_ffmpeg(
    const std::filesystem::path& audio_file_name
) {
    av_log_set_level(AV_LOG_ERROR);

    // 1. Open Input
    AVFormatContext* ctx_raw = nullptr;
    if (avformat_open_input(&ctx_raw, audio_file_name.string().c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error(fmt::format("LibAv: Open failed: {}", audio_file_name.string()));
    }
    ScopedPtr<AVFormatContext> fmt_ctx(ctx_raw);

    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0)
        throw std::runtime_error("LibAv: No stream info");

    // 2. Find Stream & Codec
    int stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_idx < 0) throw std::runtime_error("LibAv: No audio stream");

    AVCodecParameters* params = fmt_ctx->streams[stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(params->codec_id);
    if (!codec) throw std::runtime_error("LibAv: Decoder not found");

    // 3. Setup Codec
    ScopedPtr<AVCodecContext> codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) throw std::runtime_error("LibAv: Codec alloc failed");

    if (avcodec_parameters_to_context(codec_ctx.get(), params) < 0)
        throw std::runtime_error("LibAv: Params copy failed");
    if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) 
        throw std::runtime_error("LibAv: Codec open failed");

    // 4. Setup Resampler
    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
    SwrContext* swr_raw = nullptr;
    int ret = swr_alloc_set_opts2(&swr_raw, &out_layout, AV_SAMPLE_FMT_S16, 16000,
                                  &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate, 
                                  0, nullptr);
    if (ret < 0 || !swr_raw || swr_init(swr_raw) < 0) throw std::runtime_error("LibAv: Swr setup failed");
    ScopedPtr<SwrContext> swr_ctx(swr_raw);

    // 5. Decoding Loop
    ScopedPtr<AVPacket> packet(av_packet_alloc());
    ScopedPtr<AVFrame> frame(av_frame_alloc());
    if (!packet || !frame) throw std::runtime_error("LibAv: Allocation failed");

    std::vector<int16_t> samples(N_SAMPLES, 0);
    size_t total = 0;
    std::vector<int16_t> pcm_buf;

    auto process_data = [&](const uint8_t** in_data, int in_count) {
        if (total >= N_SAMPLES) return;
        int delay = swr_get_delay(swr_ctx.get(), codec_ctx->sample_rate);
        int out_count = av_rescale_rnd(delay + in_count, 16000, codec_ctx->sample_rate, AV_ROUND_UP);
        
        if (out_count > 0) {
            if (pcm_buf.size() < static_cast<size_t>(out_count)) {
                pcm_buf.resize(out_count);
            }
            uint8_t* out_ptrs[1] = { reinterpret_cast<uint8_t*>(pcm_buf.data()) };
            
            int ret = swr_convert(swr_ctx.get(), out_ptrs, out_count, in_data, in_count);
            if (ret > 0) {
                size_t copy_len = std::min((size_t)ret, N_SAMPLES - total);
                std::memcpy(samples.data() + total, pcm_buf.data(), copy_len * sizeof(int16_t));
                total += copy_len;
            }
        }
    };

    while (av_read_frame(fmt_ctx.get(), packet.get()) >= 0) {
        if (packet->stream_index == stream_idx) {
            if (avcodec_send_packet(codec_ctx.get(), packet.get()) >= 0) {
                while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
                    process_data((const uint8_t**)frame->data, frame->nb_samples);
                }
            }
        }
        av_packet_unref(packet.get());
        if (total >= N_SAMPLES) break;
    }

    // Flush resampler
    if (total < N_SAMPLES) process_data(nullptr, 0);

    // 6. Return
    Eigen::Map<ArrayXi16> map(samples.data(), N_SAMPLES);
    return map.cast<float>() * (1.0f / 32768.0f);
}


ArrayXXbf WhisperPreprocessor::_log_mel_spectrogram(ArrayXf& audio_tensor) {
    // Reflect pad the signal.
    _padded_audio_tensor.segment(N_FFT / 2, N_SAMPLES) = audio_tensor;
    _padded_audio_tensor.head(N_FFT / 2) = audio_tensor.segment(1, N_FFT / 2).reverse();
    _padded_audio_tensor.tail(N_FFT / 2) = (
        audio_tensor.segment(N_SAMPLES - 1 - N_FFT / 2, N_FFT / 2).reverse()
    );

    // Compute stft.
    Eigen::Map<Eigen::ArrayXd> fft_in_map(_fft_in, N_FFT);
    for (uint32_t i = 0; i < N_FRAMES; ++i) {
        fft_in_map = (
            _padded_audio_tensor.segment(i * HOP_LENGTH, N_FFT).cast<double>() * _hanning_window
        );
        fftw_execute_dft_r2c(_fft_plan, _fft_in, _fft_out + i * (N_FFT / 2 + 1));
    }
    Eigen::Map<ArrayXXcd> stft_result(
        reinterpret_cast<std::complex<double>*>(_fft_out), N_FRAMES, N_FFT / 2 + 1
    );

    // Post process.
    ArrayXXd tmp = (
        (stft_result.abs2().matrix() * _mel_filters.matrix().transpose())
        .array().cwiseMax(1e-10).log10()
    );
    return ((tmp.cwiseMax(tmp.maxCoeff() - 8.0) + 4.0) / 4.0).cast<Eigen::bfloat16>();
}


const std::map<std::string, std::string> WhisperModel::_TO_LANGUAGE_CODE = {
    {"english", "en"}, {"chinese", "zh"}, {"german", "de"}, {"spanish", "es"}, {"russian", "ru"},
    {"korean", "ko"}, {"french", "fr"}, {"japanese", "ja"}, {"portuguese", "pt"}, {"turkish", "tr"},
    {"polish", "pl"}, {"catalan", "ca"}, {"dutch", "nl"}, {"arabic", "ar"}, {"swedish", "sv"},
    {"italian", "it"}, {"indonesian", "id"}, {"hindi", "hi"}, {"finnish", "fi"},
    {"vietnamese", "vi"}, {"hebrew", "he"}, {"ukrainian", "uk"}, {"greek", "el"}, {"malay", "ms"},
    {"czech", "cs"}, {"romanian", "ro"}, {"danish", "da"}, {"hungarian", "hu"}, {"tamil", "ta"},
    {"norwegian", "no"}, {"thai", "th"}, {"urdu", "ur"}, {"croatian", "hr"}, {"bulgarian", "bg"},
    {"lithuanian", "lt"}, {"latin", "la"}, {"maori", "mi"}, {"malayalam", "ml"}, {"welsh", "cy"},
    {"slovak", "sk"}, {"telugu", "te"}, {"persian", "fa"}, {"latvian", "lv"}, {"bengali", "bn"},
    {"serbian", "sr"}, {"azerbaijani", "az"}, {"slovenian", "sl"}, {"kannada", "kn"},
    {"estonian", "et"}, {"macedonian", "mk"}, {"breton", "br"}, {"basque", "eu"},
    {"icelandic", "is"}, {"armenian", "hy"}, {"nepali", "ne"}, {"mongolian", "mn"},
    {"bosnian", "bs"}, {"kazakh", "kk"}, {"albanian", "sq"}, {"swahili", "sw"}, {"galician", "gl"},
    {"marathi", "mr"}, {"punjabi", "pa"}, {"sinhala", "si"}, {"khmer", "km"}, {"shona", "sn"},
    {"yoruba", "yo"}, {"somali", "so"}, {"afrikaans", "af"}, {"occitan", "oc"}, {"georgian", "ka"},
    {"belarusian", "be"}, {"tajik", "tg"}, {"sindhi", "sd"}, {"gujarati", "gu"}, {"amharic", "am"},
    {"yiddish", "yi"}, {"lao", "lo"}, {"uzbek", "uz"}, {"faroese", "fo"}, {"haitian creole", "ht"},
    {"pashto", "ps"}, {"turkmen", "tk"}, {"nynorsk", "nn"}, {"maltese", "mt"}, {"sanskrit", "sa"},
    {"luxembourgish", "lb"}, {"myanmar", "my"}, {"tibetan", "bo"}, {"tagalog", "tl"},
    {"malagasy", "mg"}, {"assamese", "as"}, {"tatar", "tt"}, {"hawaiian", "haw"}, {"lingala", "ln"},
    {"hausa", "ha"}, {"bashkir", "ba"}, {"javanese", "jw"}, {"sundanese", "su"},
    {"cantonese", "yue"}, {"burmese", "my"}, {"valencian", "ca"}, {"flemish", "nl"},
    {"haitian", "ht"}, {"letzeburgesch", "lb"}, {"pushto", "ps"}, {"panjabi", "pa"},
    {"moldavian", "ro"}, {"moldovan", "ro"}, {"sinhalese", "si"}, {"castilian", "es"},
    {"mandarin", "zh"}
};


WhisperModel::WhisperModel(std::filesystem::path model_path) : BaseModel(model_path),
    _preprocessor(_devkit_dir),
    _is_running(false)
{
    _tokenizer_ptr = Tokenizer::from_hf_json(_devkit_dir / "tokenizer.json");
    _text_streamer = std::make_unique<TextStreamer>(
        _tokenizer_ptr.get(),
        [](const std::string&, double) {},
        [](const std::string&, bool, bool) {}
    );
    _initialize();

    // Run one dummy query to cache the system prompt and warm up other libraries.
    _text_streamer->disable();
    run_model(sample_audio_file_name.value(), "en");
    _text_streamer->enable();
}


WhisperModel::TranscriptionResult WhisperModel::run_model(
    const std::filesystem::path& audio_file_name,
    const std::string& language,
    const std::string& task
) {
    std::lock_guard<std::mutex> lock(_mutex);
    _logger->info("Audio file: {}", audio_file_name);

    ArrayXXbf audio_tensor = _preprocessor.preprocess(audio_file_name);
    return _run_model(audio_tensor, language, task);
}


WhisperModel::TranscriptionResult WhisperModel::run_model_from_pcm(
    std::span<const float> pcm,
    uint32_t sample_rate,
    const std::string& language,
    const std::string& task
) {
    std::lock_guard<std::mutex> lock(_mutex);
    ArrayXXbf mel = _preprocessor.preprocess_pcm(pcm, sample_rate);
    return _run_model(mel, language, task);
}


WhisperModel::TranscriptionResult WhisperModel::_run_model(
    const ArrayXXbf& mel,
    const std::string& language,
    const std::string& task
) {
    _is_running.store(true, std::memory_order_relaxed);
    struct RunningGuard {
        std::atomic<bool>& running;
        ~RunningGuard() {
            running.store(false, std::memory_order_relaxed);
        }
    } running_guard{_is_running};

    ChronoTimer timer_ttft(true);
    _logger->info("Language: {}", language);

    // Upload audio_tensor and run the encoder model.
    get_buffer("encoder_ifm").upload(mel.data());
    _encoder_model_ptr->run();

    auto language_detect_result = _run_language_detect();

    // Update language token id.
    std::string resolved_language;
    if (_is_auto_language(language)) {
        auto detected_language_index = language_detect_result.language_index;
        auto detected_language_token = _language_token_from_index(detected_language_index);
        _set_language_token(detected_language_token);
        resolved_language = _language_code_from_token(detected_language_token);
        _logger->info(
            "Detected language: {} token={} index={}",
            resolved_language,
            detected_language_token,
            detected_language_index
        );
    } else {
        resolved_language = _update_language(language);
    }
    auto resolved_task = _update_task(task);

    // Upload the decoder input embeds.
    const uint8_t* token_embeddings_ptr = reinterpret_cast<const uint8_t*>(
        get_buffer("token_embeddings").get_virtual_addr()
    );
    auto token_embed_size = _cfg.d_model * 2;
    auto& decoder_init_buf = get_buffer("decoder_init");
    for (uint32_t i = 0; i < _input_token_ids.size(); ++i) {
        decoder_init_buf.upload(
            token_embeddings_ptr + _input_token_ids[i] * token_embed_size,
            i * token_embed_size,
            token_embed_size,
            i + 1 == _input_token_ids.size()
        );
    }

    auto& new_token_buf = get_buffer("new_token");
    uint32_t* new_token_ptr = reinterpret_cast<uint32_t*>(new_token_buf.get_virtual_addr());
    std::vector<uint32_t> new_tokens;
    float logprob_sum = 0.0f;
    uint32_t logprob_count = 0;

    // Run decoder init model to generate the first token.
    for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
        if (_cfg.log_probe_enabled && layer_idx == _cfg.decoder_layers - 1)
            _decoder_init_log_probe_model_map.at(layer_idx).add_to_queue();
        else
            _decoder_init_model_map.at(layer_idx).add_to_queue();
    }
    MLAModelWithBuffer::run_queue();
    new_token_buf.invalidate_cache();
    new_tokens.emplace_back(new_token_ptr[0]);
    if (_cfg.log_probe_enabled) {
        auto& logits_buf = get_buffer("decoder_logits");
        logits_buf.invalidate_cache();
        if (new_tokens.back() != _stop_token_id) {
            logprob_sum += _compute_token_logprob(logits_buf, new_tokens.back());
            ++logprob_count;
        }
    }
    const double ttft = timer_ttft.stop();
    _logger->info("Time to the first token: {:d} in {:.5f}s", new_tokens.back(), ttft);
    _text_streamer->push(DecodeCallbackType::TTFT, new_tokens.back(), ttft);
    if (new_tokens.back() == _stop_token_id) {
        _text_streamer->push(DecodeCallbackType::STOP, 0, 0);
        _text_streamer->wait_streaming();
        return {
            _tokenizer_ptr->decode(new_tokens, true),
            resolved_language,
            resolved_task,
            language_detect_result.no_speech_prob,
            _cfg.log_probe_enabled ? std::optional<float>{0.0f} : std::optional<float>{}
        };
    }

    // Run decoder pre/cache/post models to generate other tokens.
    ChronoTimer timer_tps(true);
    for (
        uint32_t token_idx = _input_token_ids.size();
        token_idx < _cfg.max_target_positions;
        ++token_idx
    ) {
        if (!_is_running.load(std::memory_order_relaxed))
            break;

        for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
            WhisperDecoderModelMapKey model_key{layer_idx, token_idx};

            std::map<uint8_t, MLABufferSlice> ifm_map;
            if (layer_idx == 0) {
                ifm_map.emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(0),
                    std::forward_as_tuple(
                        &get_buffer("token_embeddings"),
                        std::vector<uint32_t>{new_tokens.back(), 0},
                        std::vector<uint32_t>{1, _cfg.d_model}
                    )
                );
            }
            _decoder_pre_model_map.at(model_key).add_to_queue(&ifm_map);
            _decoder_cache_model_map.at(model_key).add_to_queue();
            _decoder_post_model_map.at(model_key).add_to_queue(&ifm_map);
        }

        MLAModelWithBuffer::run_queue();
        new_token_buf.invalidate_cache();
        new_tokens.emplace_back(new_token_ptr[0]);
        if (_cfg.log_probe_enabled) {
            auto& logits_buf = get_buffer("decoder_logits");
            logits_buf.invalidate_cache();
            if (new_tokens.back() != _stop_token_id) {
                logprob_sum += _compute_token_logprob(logits_buf, new_tokens.back());
                ++logprob_count;
            }
        }
        const double ttnt = timer_tps.stop(true);
        _logger->info("Got token {:d} in {:.5f}s", new_tokens.back(), ttnt);

        if (new_tokens.back() == _stop_token_id)
            break;

        _text_streamer->push(DecodeCallbackType::TPS, new_tokens.back(), ttnt);
    }
    _text_streamer->push(DecodeCallbackType::STOP, 0, 0);
    _text_streamer->wait_streaming();
    std::optional<float> avg_logprob = std::nullopt;
    if (_cfg.log_probe_enabled)
        avg_logprob = logprob_count > 0 ? logprob_sum / logprob_count : 0.0f;
    return {
        _tokenizer_ptr->decode(new_tokens, true),
        resolved_language,
        resolved_task,
        language_detect_result.no_speech_prob,
        avg_logprob
    };
}


void WhisperModel::stop_model() {
    _is_running.store(false, std::memory_order_relaxed);
}


void WhisperModel::_initialize() {
    _logger->info("Whisper model initialize starting ...");
    // Find the input token ids. This is a constant because we are only using this model to perform
    // transcription. Default to transcribe to English.
    _input_token_ids = {
        _tokenizer_ptr->token_to_id("<|startoftranscript|>"),
        _tokenizer_ptr->token_to_id("<|en|>"),
        _tokenizer_ptr->token_to_id("<|transcribe|>"),
        _tokenizer_ptr->token_to_id("<|notimestamps|>")
    };
    _stop_token_id = _tokenizer_ptr->token_to_id("<|endoftext|>");

    // Define and allocate buffers.
    BaseModel::_initialize();

    // Define and load the models in parallel.
    _define_models();
    MLAModelWithBuffer::load_all_models(_elf_dir);

    // Upload token and position embeddings.
    auto token_embeddings_file_name = (
        _devkit_dir / fmt::format("{}_token_embeddings.npy", _cfg.model_name)
    );
    auto token_embeddings_tensor = cnpy::npy_load(token_embeddings_file_name);
    get_buffer("token_embeddings").upload(token_embeddings_tensor.data<void>());

    auto position_embeddings_file_name = (
        _devkit_dir / fmt::format("{}_position_embeddings.npy", _cfg.model_name)
    );
    auto position_embeddings_tensor = cnpy::npy_load(position_embeddings_file_name);
    get_buffer("position_embeddings").upload(position_embeddings_tensor.data<void>());

    // Populate the future token mask.
    std::vector<Eigen::bfloat16> decoder_future_token_mask(
        _cfg.max_target_positions, std::numeric_limits<Eigen::bfloat16>::lowest()
    );
    auto& future_token_mask_buf = get_buffer("decoder_future_token_mask");
    future_token_mask_buf.clear(false);
    future_token_mask_buf.upload(
        decoder_future_token_mask.data(),
        _cfg.max_target_positions * 2,
        decoder_future_token_mask.size() * 2,
        true
    );

    // Clear the KV caches.
    for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
        get_buffer(fmt::format("decoder_cache_key{}", layer_idx)).clear();
        get_buffer(fmt::format("decoder_cache_val{}", layer_idx)).clear();
    }
    _logger->info("Whisper model initialize completed");
}


void WhisperModel::_finalize() {
    _logger->info("Whisper model finalize starting ...");
    MLAModelWithBuffer::free_all_models(_elf_dir);
    BaseModel::_finalize();
    _logger->info("Whisper model finalize completed");
}


void WhisperModel::_define_buffers() {
    // Encoder.
    define_buffer("encoder_ifm", {2 * _cfg.max_source_positions, _cfg.num_mel_bins});
    define_buffer("encoder_ofm", {_cfg.max_source_positions, _cfg.d_model});

    // Embedding tables.
    define_buffer("token_embeddings", {_cfg.vocab_size, _cfg.d_model});
    define_buffer("position_embeddings", {_cfg.max_target_positions, _cfg.d_model});

    // Cache.
    for (uint32_t i = 0; i < _cfg.decoder_layers; ++i) {
        define_buffer(
            fmt::format("encoder_cache_key{}", i),
            {_cfg.decoder_attention_heads, _cfg.max_source_positions, _cfg.get_decoder_head_dim()}
        );
        define_buffer(
            fmt::format("encoder_cache_val{}", i),
            {_cfg.decoder_attention_heads, _cfg.max_source_positions, _cfg.get_decoder_head_dim()}
        );
        define_buffer(
            fmt::format("decoder_cache_key{}", i), {_cfg.max_target_positions, _cfg.d_model}
        );
        define_buffer(
            fmt::format("decoder_cache_val{}", i), {_cfg.max_target_positions, _cfg.d_model}
        );
    }

    // Other buffers.
    define_buffer("decoder_future_token_mask", {2 * _cfg.max_target_positions});
    define_buffer("decoder_init", {_input_token_ids.size(), _cfg.d_model});

    // Pre input (for layer_idx > 0 or init), post input and post output.
    define_buffer("decoder_n1_buffer1", {1, _cfg.d_model});
    // Pre output and cache input.
    define_buffer(
        "decoder_n1_buffer2", {_cfg.decoder_attention_heads, 1, _cfg.get_decoder_head_dim()}
    );
    // Cache output and post input.
    define_buffer("decoder_n1_buffer3", {1, _cfg.d_model});
    // Post output for last layer.
    define_buffer("new_token", {1}, "int32");
    define_buffer("language_detect_logits", {1, _cfg.vocab_size});
    define_buffer("decoder_logits", {1, _cfg.vocab_size});
}


void WhisperModel::_define_model(
    const std::string& model_type,
    std::variant<std::monostate, uint8_t, WhisperDecoderModelMapKey> key,
    const std::filesystem::path& model_path,
    const std::vector<MLABufferSlice>& ifms,
    const std::vector<MLABufferSlice>& ofms
) {
    if (model_type == "encoder") {
        _encoder_model_ptr = std::make_unique<MLAModelWithBuffer>(model_path, ifms, ofms);
    } else if (model_type == "decoder_language_detect") {
        _decoder_language_detect_model_ptr = std::make_unique<MLAModelWithBuffer>(
            model_path, ifms, ofms
        );
    } else if (model_type == "decoder_init") {
        _decoder_init_model_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::get<uint8_t>(key)),
            std::forward_as_tuple(model_path, ifms, ofms)
        );
    } else if (model_type == "decoder_init_log_probe") {
        _decoder_init_log_probe_model_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::get<uint8_t>(key)),
            std::forward_as_tuple(model_path, ifms, ofms)
        );
    } else if (model_type == "decoder_pre") {
        _decoder_pre_model_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::get<WhisperDecoderModelMapKey>(key)),
            std::forward_as_tuple(model_path, ifms, ofms)
        );
    } else if (model_type == "decoder_cache") {
        _decoder_cache_model_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::get<WhisperDecoderModelMapKey>(key)),
            std::forward_as_tuple(model_path, ifms, ofms)
        );
    } else if (model_type == "decoder_post") {
        _decoder_post_model_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::get<WhisperDecoderModelMapKey>(key)),
            std::forward_as_tuple(model_path, ifms, ofms)
        );
    } else {
        throw std::runtime_error(fmt::format("Invalid model type: {}", model_type));
    }
}


void WhisperModel::_define_models() {
    // Encoder model.
    _define_model(
        "encoder",
        {},
        _get_elf_path_encoder(),
        std::vector<MLABufferSlice>{{&get_buffer("encoder_ifm")}},
        std::vector<MLABufferSlice>{{&get_buffer("encoder_ofm")}}
    );

    _define_model(
        "decoder_language_detect",
        {},
        _get_elf_path_decoder_language_detect(),
        std::vector<MLABufferSlice>{{&get_buffer("encoder_ofm")}},
        std::vector<MLABufferSlice>{
            {&get_buffer("new_token")},
            {&get_buffer("language_detect_logits")}
        }
    );

    // Decoder init model.
    uint32_t num_input_tokens = _input_token_ids.size();
    for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
        std::vector<MLABufferSlice> ifms = {&get_buffer("decoder_init")};
        ifms.emplace_back(&get_buffer("encoder_ofm"));

        std::vector<MLABufferSlice> ofms;
        if (layer_idx < _cfg.decoder_layers - 1)
            ofms.emplace_back(&get_buffer("decoder_init"));
        else
            ofms.emplace_back(&get_buffer("new_token"));
        ofms.emplace_back(
            &get_buffer(fmt::format("decoder_cache_key{}", layer_idx)),
            std::vector<uint32_t>{0, 0},
            std::vector<uint32_t>{num_input_tokens, _cfg.d_model}
        );
        ofms.emplace_back(
            &get_buffer(fmt::format("decoder_cache_val{}", layer_idx)),
            std::vector<uint32_t>{0, 0},
            std::vector<uint32_t>{num_input_tokens, _cfg.d_model}
        );
        ofms.emplace_back(
            &get_buffer(fmt::format("encoder_cache_key{}", layer_idx)),
            std::vector<uint32_t>{0, 0, 0},
            std::vector<uint32_t>{
                _cfg.decoder_attention_heads,
                _cfg.max_source_positions,
                _cfg.get_decoder_head_dim()
            }
        );
        ofms.emplace_back(
            &get_buffer(fmt::format("encoder_cache_val{}", layer_idx)),
            std::vector<uint32_t>{0, 0, 0},
            std::vector<uint32_t>{
                _cfg.decoder_attention_heads,
                _cfg.max_source_positions,
                _cfg.get_decoder_head_dim()
            }
        );
        _define_model("decoder_init", layer_idx, _get_elf_path_decoder_init(layer_idx), ifms, ofms);

        if (_cfg.log_probe_enabled && layer_idx == _cfg.decoder_layers - 1) {
            std::vector<MLABufferSlice> log_probe_ofms;
            log_probe_ofms.emplace_back(&get_buffer("new_token"));
            log_probe_ofms.emplace_back(&get_buffer("decoder_logits"));
            log_probe_ofms.emplace_back(
                &get_buffer(fmt::format("decoder_cache_key{}", layer_idx)),
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_input_tokens, _cfg.d_model}
            );
            log_probe_ofms.emplace_back(
                &get_buffer(fmt::format("decoder_cache_val{}", layer_idx)),
                std::vector<uint32_t>{0, 0},
                std::vector<uint32_t>{num_input_tokens, _cfg.d_model}
            );
            log_probe_ofms.emplace_back(
                &get_buffer(fmt::format("encoder_cache_key{}", layer_idx)),
                std::vector<uint32_t>{0, 0, 0},
                std::vector<uint32_t>{
                    _cfg.decoder_attention_heads,
                    _cfg.max_source_positions,
                    _cfg.get_decoder_head_dim()
                }
            );
            log_probe_ofms.emplace_back(
                &get_buffer(fmt::format("encoder_cache_val{}", layer_idx)),
                std::vector<uint32_t>{0, 0, 0},
                std::vector<uint32_t>{
                    _cfg.decoder_attention_heads,
                    _cfg.max_source_positions,
                    _cfg.get_decoder_head_dim()
                }
            );
            _define_model(
                "decoder_init_log_probe",
                layer_idx,
                _get_elf_path_decoder_init_log_probe(layer_idx),
                ifms,
                log_probe_ofms
            );
        }
    }

    // Decoder pre/cache/post.
    for (
        uint32_t token_idx = num_input_tokens;
        token_idx < _cfg.max_target_positions;
        ++token_idx
    ) {
        for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
            WhisperDecoderModelMapKey model_key{layer_idx, token_idx};
            std::vector<MLABufferSlice> pre_ifms;
            std::vector<MLABufferSlice> pre_ofms;
            if (layer_idx == 0) {
                pre_ifms.emplace_back(&get_buffer("token_embeddings"));
                pre_ifms.emplace_back(
                    &get_buffer("position_embeddings"),
                    std::vector<uint32_t>{token_idx, 0},
                    std::vector<uint32_t>{1, _cfg.d_model}
                );
            } else {
                pre_ifms.emplace_back(&get_buffer("decoder_n1_buffer1"));
            }
            pre_ofms.emplace_back(&get_buffer("decoder_n1_buffer2"));
            pre_ofms.emplace_back(
                &get_buffer(fmt::format("decoder_cache_key{}", layer_idx)),
                std::vector<uint32_t>{token_idx, 0},
                std::vector<uint32_t>{1, _cfg.d_model}
            );
            pre_ofms.emplace_back(
                &get_buffer(fmt::format("decoder_cache_val{}", layer_idx)),
                std::vector<uint32_t>{token_idx, 0},
                std::vector<uint32_t>{1, _cfg.d_model}
            );
            _define_model(
                "decoder_pre", model_key, _get_elf_path_decoder_pre(layer_idx), pre_ifms, pre_ofms
            );

            std::vector<MLABufferSlice> cache_ofms{&get_buffer("decoder_n1_buffer3")};
            if (_cfg.decoder_use_future_token_mask) {
                std::vector<MLABufferSlice> cache_ifms{
                    pre_ofms[0],
                    {
                        &get_buffer(fmt::format("decoder_cache_key{}", layer_idx)),
                        {0, 0},
                        {_cfg.max_target_positions, _cfg.d_model}
                    },
                    {
                        &get_buffer("decoder_future_token_mask"),
                        {_cfg.max_target_positions - token_idx - 1},
                        {_cfg.max_target_positions}
                    },
                    {
                        &get_buffer(fmt::format("decoder_cache_val{}", layer_idx)),
                        {0, 0},
                        {_cfg.max_target_positions, _cfg.d_model}
                    }
                };
                _define_model(
                    "decoder_cache",
                    model_key,
                    _get_elf_path_decoder_cache(_cfg.max_target_positions - 1),
                    cache_ifms,
                    cache_ofms
                );
            } else {
                std::vector<MLABufferSlice> cache_ifms{
                    pre_ofms[0],
                    {
                        &get_buffer(fmt::format("decoder_cache_key{}", layer_idx)),
                        {0, 0},
                        {token_idx + 1, _cfg.d_model}
                    },
                    {
                        &get_buffer(fmt::format("decoder_cache_val{}", layer_idx)),
                        {0, 0},
                        {token_idx + 1, _cfg.d_model}
                    }
                };
                _define_model(
                    "decoder_cache",
                    model_key,
                    _get_elf_path_decoder_cache(token_idx),
                    cache_ifms,
                    cache_ofms
                );
            }

            std::vector<MLABufferSlice> post_ifms{
                pre_ifms[0],
                cache_ofms[0],
                {&get_buffer(fmt::format("encoder_cache_key{}", layer_idx))},
                {&get_buffer(fmt::format("encoder_cache_val{}", layer_idx))},
            };
            std::vector<MLABufferSlice> post_ofms;
            if (layer_idx < _cfg.decoder_layers - 1) {
                post_ofms.emplace_back(&get_buffer("decoder_n1_buffer1"));
            } else {
                post_ofms.emplace_back(&get_buffer("new_token"));
                if (_cfg.log_probe_enabled)
                    post_ofms.emplace_back(&get_buffer("decoder_logits"));
            }
            _define_model(
                "decoder_post",
                model_key,
                _get_elf_path_decoder_post(layer_idx),
                post_ifms,
                post_ofms
            );
        }
    }
}


std::filesystem::path WhisperModel::_get_elf_path_encoder() const {
    return _elf_dir / fmt::format("{}_encoder_stage1_mla.elf", _cfg.model_name);
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_init(uint8_t layer_idx) const {
    auto elf_file_name = fmt::format(
        "{}_decoder_init_layer{}_stage1_mla.elf", _cfg.model_name, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_init_log_probe(uint8_t layer_idx) const {
    auto elf_file_name = fmt::format(
        "{}_decoder_init_log_probe_layer{}_stage1_mla.elf", _cfg.model_name, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_pre(uint8_t layer_idx) const {
    auto elf_file_name = fmt::format(
        "{}_decoder_n1_pre_layer{}_stage1_mla.elf", _cfg.model_name, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_cache(uint16_t token_idx) const {
    auto elf_file_name = fmt::format(
        "{}_decoder_n1_cache_token{}_stage1_mla.elf", _cfg.model_name, token_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_post(uint8_t layer_idx) const {
    auto elf_file_name = fmt::format(
        "{}_decoder_n1_post_layer{}_stage1_mla.elf", _cfg.model_name, layer_idx
    );
    return _elf_dir / elf_file_name;
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_language_detect() const {
    return _elf_dir / fmt::format(
        "{}_decoder_language_detect_stage1_mla.elf", _cfg.model_name
    );
}


bool WhisperModel::_is_auto_language(const std::string& language) const {
    return language.empty() || language == "auto";
}


WhisperModel::LanguageDetectResult WhisperModel::_run_language_detect() {
    _decoder_language_detect_model_ptr->run();
    auto& new_token_buf = get_buffer("new_token");
    new_token_buf.invalidate_cache();
    auto* token_ptr = reinterpret_cast<uint32_t*>(new_token_buf.get_virtual_addr());
    // HF tokenizers name this token <|nocaptions|>; Whisper generation treats it as
    // the no-speech token at the id immediately before <|notimestamps|>.
    auto no_speech_token_id = _tokenizer_ptr->token_to_id("<|notimestamps|>") - 1;
    auto& logits_buf = get_buffer("language_detect_logits");
    logits_buf.invalidate_cache();
    return {
        token_ptr[0],
        _compute_token_prob(logits_buf, no_speech_token_id)
    };
}


float WhisperModel::_compute_token_logprob(MLABuffer& logits_buf, uint32_t token_id) {
    if (token_id >= _cfg.vocab_size)
        throw std::runtime_error(fmt::format("Invalid token id for logprob: {}", token_id));

    auto* logits_ptr = reinterpret_cast<Eigen::bfloat16*>(logits_buf.get_virtual_addr());

    float max_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < _cfg.vocab_size; ++i) {
        max_logit = std::max(max_logit, static_cast<float>(logits_ptr[i]));
    }

    float exp_sum = 0.0f;
    for (uint32_t i = 0; i < _cfg.vocab_size; ++i) {
        exp_sum += expf(static_cast<float>(logits_ptr[i]) - max_logit);
    }
    return static_cast<float>(logits_ptr[token_id]) - (max_logit + logf(exp_sum));
}


float WhisperModel::_compute_token_prob(MLABuffer& logits_buf, uint32_t token_id) {
    return expf(_compute_token_logprob(logits_buf, token_id));
}


uint32_t WhisperModel::_language_token_from_index(uint32_t language_idx) const {
    if (language_idx >= _cfg.language_token_ids.size()) {
        throw std::runtime_error(
            fmt::format("Invalid detected language index: {}", language_idx)
        );
    }
    return _cfg.language_token_ids[language_idx];
}


void WhisperModel::_set_language_token(uint32_t token_id) {
    _input_token_ids[1] = token_id;
}


std::string WhisperModel::_language_code_from_token(uint32_t token_id) const {
    auto it = std::find(
        _cfg.language_token_ids.begin(), _cfg.language_token_ids.end(), token_id
    );
    if (it == _cfg.language_token_ids.end())
        return "unknown";

    auto idx = static_cast<size_t>(std::distance(_cfg.language_token_ids.begin(), it));
    return _cfg.language_codes[idx];
}


std::string WhisperModel::_update_language(const std::string& language) {
    std::string language_code;
    if (auto it = _TO_LANGUAGE_CODE.find(language); it != _TO_LANGUAGE_CODE.end())
        language_code = it->second;
    else
        language_code = language;
    if (
        auto it = std::find(_cfg.language_codes.begin(), _cfg.language_codes.end(), language_code);
        it != _cfg.language_codes.end()
    ) {
        auto idx = static_cast<size_t>(std::distance(_cfg.language_codes.begin(), it));
        _set_language_token(_cfg.language_token_ids[idx]);
        return language_code;
    } else {
        throw std::runtime_error("Invalid language: " + language);
    }
}


std::string WhisperModel::_update_task(const std::string& task) {
    if (task == "transcribe") {
        _input_token_ids[2] = _tokenizer_ptr->token_to_id("<|transcribe|>");
        return task;
    }
    if (task == "translate") {
        _input_token_ids[2] = _tokenizer_ptr->token_to_id("<|translate|>");
        return task;
    }
    throw std::runtime_error("Invalid Whisper task: " + task);
}


}
}
