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

WhisperPreprocessor::WhisperPreprocessor(const std::filesystem::path& devkit_dir)
    : WhisperPreprocessor(DiskFileProvider(devkit_dir).open_stream("preprocessor_config.json")) {}

WhisperPreprocessor::WhisperPreprocessor(std::shared_ptr<FileProvider> files)
    : WhisperPreprocessor(files->open_stream("devkit/preprocessor_config.json")) {}

WhisperPreprocessor::WhisperPreprocessor(std::unique_ptr<std::istream> config) {
    // Load mel filters.
    auto json = nlohmann::json::parse(*config);
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
    auto audio = _load_audio_ffmpeg(audio_file_name);
    return preprocess_pcm({audio.data(), static_cast<size_t>(audio.size())}, SAMPLE_RATE);
}


ArrayXf WhisperPreprocessor::load_audio(const std::filesystem::path& audio_file_name) {
    return _load_audio_ffmpeg(audio_file_name);
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

    std::vector<int16_t> samples;
    std::vector<int16_t> pcm_buf;

    auto process_data = [&](const uint8_t** in_data, int in_count) {
        int delay = swr_get_delay(swr_ctx.get(), codec_ctx->sample_rate);
        int out_count = av_rescale_rnd(delay + in_count, 16000, codec_ctx->sample_rate, AV_ROUND_UP);
        
        if (out_count > 0) {
            if (pcm_buf.size() < static_cast<size_t>(out_count)) {
                pcm_buf.resize(out_count);
            }
            uint8_t* out_ptrs[1] = { reinterpret_cast<uint8_t*>(pcm_buf.data()) };
            
            int ret = swr_convert(swr_ctx.get(), out_ptrs, out_count, in_data, in_count);
            if (ret > 0) {
                if (samples.size() + ret > MAX_AUDIO_SAMPLES) {
                    throw std::invalid_argument(fmt::format(
                        "Audio longer than {} s is not supported", MAX_AUDIO_SECONDS
                    ));
                }
                samples.insert(samples.end(), pcm_buf.begin(), pcm_buf.begin() + ret);
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
    }

    // Flush resampler
    process_data(nullptr, 0);

    // 6. Return
    Eigen::Map<ArrayXi16> map(samples.data(), samples.size());
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


WhisperModel::WhisperModel(std::filesystem::path model_path)
    : WhisperModel(std::move(model_path), nullptr) {}

WhisperModel::WhisperModel(std::filesystem::path model_path, std::shared_ptr<FileProvider> files)
    : BaseModel(model_path, std::move(files)),
    _preprocessor(_files),
    _is_running(false)
{
    _tokenizer_ptr = Tokenizer::from_hf_json(_files->get_path("devkit/tokenizer.json"));
    _files->release("devkit/tokenizer.json");
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
    return run_model(audio_file_name, TranscriptionOptions{language, task});
}


WhisperModel::TranscriptionResult WhisperModel::run_model(
    const std::filesystem::path& audio_file_name,
    const TranscriptionOptions& options
) {
    std::lock_guard<std::mutex> lock(_mutex);
    _logger->info("Audio file: {}", audio_file_name);

    ArrayXf audio = _preprocessor.load_audio(audio_file_name);
    return _transcribe(
        {audio.data(), static_cast<size_t>(audio.size())},
        WhisperPreprocessor::SAMPLE_RATE,
        options
    );
}


WhisperModel::TranscriptionResult WhisperModel::run_model_from_pcm(
    std::span<const float> pcm,
    uint32_t sample_rate,
    const std::string& language,
    const std::string& task
) {
    return run_model_from_pcm(pcm, sample_rate, TranscriptionOptions{language, task});
}


WhisperModel::TranscriptionResult WhisperModel::run_model_from_pcm(
    std::span<const float> pcm,
    uint32_t sample_rate,
    const TranscriptionOptions& options
) {
    std::lock_guard<std::mutex> lock(_mutex);
    return _transcribe(pcm, sample_rate, options);
}


WhisperModel::TranscriptionResult WhisperModel::_transcribe(
    std::span<const float> pcm,
    uint32_t sample_rate,
    const TranscriptionOptions& options
) {
    _is_running.store(true, std::memory_order_relaxed);
    struct RunningGuard {
        std::atomic<bool>& running;
        ~RunningGuard() {
            running.store(false, std::memory_order_relaxed);
        }
    } running_guard{_is_running};

    _logger->info("Language: {}", options.language);
    const auto prompt_ids = _encode_prompt(options.prompt);
    if (pcm.size() > WhisperPreprocessor::MAX_AUDIO_SAMPLES) {
        throw std::invalid_argument(fmt::format(
            "Audio longer than {} s is not supported", WhisperPreprocessor::MAX_AUDIO_SECONDS
        ));
    }

    // Audio longer than one encoder input runs as consecutive fixed windows. A trailing
    // remainder under 100 ms, such as resampler rounding, cannot hold speech and is skipped.
    constexpr size_t window_samples = WhisperPreprocessor::N_SAMPLES;
    constexpr size_t min_window_samples = WhisperPreprocessor::SAMPLE_RATE / 10;
    size_t num_windows = pcm.size() / window_samples;
    if (num_windows == 0 || pcm.size() % window_samples >= min_window_samples)
        ++num_windows;
    if (num_windows > 1)
        _logger->info("Whisper long-form audio: {} windows", num_windows);

    const size_t max_context_tokens = _cfg.max_target_positions / 2 - 1;
    std::vector<uint32_t> text_ids;
    // Text before this index is not carried into later windows.
    size_t context_begin = 0;
    std::string language = options.language;
    TranscriptionResult result{.no_speech_prob = 1.0f};
    float logprob_sum = 0.0f;
    uint32_t logprob_count = 0;
    try {
        for (size_t window_idx = 0; window_idx < num_windows; ++window_idx) {
            if (window_idx > 0 && !_is_running.load(std::memory_order_relaxed))
                break;

            // The prompt applies to every window, and preceding text fills the remaining context.
            std::vector<uint32_t> context_ids = prompt_ids;
            if (options.condition_on_previous_text) {
                const size_t budget = max_context_tokens - prompt_ids.size();
                const size_t begin = std::max(
                    context_begin, text_ids.size() - std::min(budget, text_ids.size())
                );
                context_ids.insert(context_ids.end(), text_ids.begin() + begin, text_ids.end());
            }

            const size_t offset = window_idx * window_samples;
            const auto window = pcm.subspan(offset, std::min(window_samples, pcm.size() - offset));
            auto window_result = _run_window(
                _preprocessor.preprocess_pcm(window, sample_rate),
                language,
                options.task,
                context_ids,
                window_idx == 0
            );
            if (window_idx == 0) {
                // Later windows keep the language detected in the first one.
                language = window_result.language;
                result.language = window_result.language;
                result.task = window_result.task;
            }
            result.no_speech_prob = std::min(result.no_speech_prob, window_result.no_speech_prob);
            logprob_sum += window_result.logprob_sum;
            logprob_count += window_result.logprob_count;
            text_ids.insert(
                text_ids.end(), window_result.tokens.begin(), window_result.tokens.end()
            );
            // A window cut off at the token limit is often a repetition loop; don't carry it on.
            if (!window_result.finished)
                context_begin = text_ids.size();
        }
    } catch (...) {
        // End the stream so streamed text is flushed and the next request starts clean.
        _text_streamer->push(DecodeCallbackType::STOP, 0, 0);
        _text_streamer->wait_streaming();
        throw;
    }
    _text_streamer->push(DecodeCallbackType::STOP, 0, 0);
    _text_streamer->wait_streaming();

    result.text = _tokenizer_ptr->decode(text_ids, true);
    if (_cfg.log_probe_enabled)
        result.avg_logprob = logprob_count > 0 ? logprob_sum / logprob_count : 0.0f;
    return result;
}


WhisperModel::WindowResult WhisperModel::_run_window(
    const ArrayXXbf& mel,
    const std::string& language,
    const std::string& task,
    const std::vector<uint32_t>& prompt_ids,
    bool first_window
) {
    ChronoTimer timer_ttft(true);

    // Upload audio_tensor and run the encoder model.
    get_buffer("encoder_ifm").upload(mel.data());
    _run_encoder();

    auto language_detect_result = _run_language_detect();
    WindowResult result{
        .finished = false,
        .no_speech_prob = language_detect_result.no_speech_prob,
        .logprob_sum = 0.0f,
        .logprob_count = 0
    };

    // Update language token id.
    if (_is_auto_language(language)) {
        auto detected_language_index = language_detect_result.language_index;
        auto detected_language_token = _language_token_from_index(detected_language_index);
        _set_language_token(detected_language_token);
        result.language = _language_code_from_token(detected_language_token);
        _logger->info(
            "Detected language: {} token={} index={}",
            result.language,
            detected_language_token,
            detected_language_index
        );
    } else {
        result.language = _update_language(language);
    }
    result.task = _update_task(task);

    // A prompt precedes the start sequence as <|startofprev|> context, as in OpenAI Whisper.
    std::vector<uint32_t> decoder_input_ids;
    if (!prompt_ids.empty()) {
        decoder_input_ids.reserve(1 + prompt_ids.size() + _input_token_ids.size());
        _logger->info("Whisper prompt tokens: {}", prompt_ids.size());
        decoder_input_ids.emplace_back(_tokenizer_ptr->token_to_id("<|startofprev|>"));
        decoder_input_ids.insert(decoder_input_ids.end(), prompt_ids.begin(), prompt_ids.end());
    }
    decoder_input_ids.insert(
        decoder_input_ids.end(), _input_token_ids.begin(), _input_token_ids.end()
    );

    // Upload the decoder input embeds.
    const uint8_t* token_embeddings_ptr = reinterpret_cast<const uint8_t*>(
        get_buffer("token_embeddings").get_virtual_addr()
    );
    auto token_embed_size = _cfg.d_model * 2;
    auto& decoder_init_buf = get_buffer("decoder_init");
    for (uint32_t i = 0; i < _input_token_ids.size(); ++i) {
        decoder_init_buf.upload_raw(
            token_embeddings_ptr + decoder_input_ids[i] * token_embed_size,
            i * token_embed_size,
            token_embed_size
        );
    }

    auto& new_token_buf = get_buffer("new_token");
    uint32_t* new_token_ptr = reinterpret_cast<uint32_t*>(new_token_buf.get_virtual_addr());
    auto read_new_token = [&]() {
        new_token_buf.invalidate_cache();
        const uint32_t token = new_token_ptr[0];
        if (_cfg.log_probe_enabled) {
            auto& logits_buf = get_buffer("decoder_logits");
            logits_buf.invalidate_cache();
            result.logprob_sum += _compute_token_logprob(logits_buf, token);
            ++result.logprob_count;
        }
        return token;
    };

    // The init model consumes the first input tokens and single-token steps consume the rest
    // of a prompted input, all in one queue. Only the prediction after the last one is kept.
    for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
        _decoder_init_model_map.at(layer_idx).add_to_queue();
    }
    for (
        uint32_t token_idx = _input_token_ids.size();
        token_idx < decoder_input_ids.size();
        ++token_idx
    ) {
        _queue_decoder_step(token_idx, decoder_input_ids[token_idx]);
    }
    MLAModelWithBuffer::run_queue();
    uint32_t token = read_new_token();
    const double ttft = timer_ttft.stop();
    if (first_window) {
        _logger->info("Time to the first token: {:d} in {:.5f}s", token, ttft);
        _text_streamer->push(DecodeCallbackType::TTFT, token, ttft);
    } else {
        _logger->info("Got window token {:d} in {:.5f}s", token, ttft);
        if (token != _stop_token_id)
            _text_streamer->push(DecodeCallbackType::TPS, token, ttft);
    }

    // Run decoder pre/cache/post models to generate other tokens.
    ChronoTimer timer_tps(true);
    for (
        uint32_t token_idx = decoder_input_ids.size();
        token != _stop_token_id;
        ++token_idx
    ) {
        if (token < _stop_token_id)
            result.tokens.emplace_back(token);
        if (token_idx >= _cfg.max_target_positions || !_is_running.load(std::memory_order_relaxed))
            return result;

        _queue_decoder_step(token_idx, token);
        MLAModelWithBuffer::run_queue();
        token = read_new_token();
        const double ttnt = timer_tps.stop(true);
        _logger->info("Got token {:d} in {:.5f}s", token, ttnt);

        if (token != _stop_token_id)
            _text_streamer->push(DecodeCallbackType::TPS, token, ttnt);
    }
    result.finished = true;
    return result;
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
    MLAModelWithBuffer::load_all_models(_elf_dir, _files.get());

    // Upload token and position embeddings.
    auto token_embeddings_file_name = (
        "devkit/" + fmt::format("{}_token_embeddings.npy", _cfg.model_name)
    );
    auto token_embeddings_tensor = cnpy::npy_load(_files->get_path(token_embeddings_file_name));
    _files->release(token_embeddings_file_name);
    get_buffer("token_embeddings").upload(token_embeddings_tensor.data<void>());

    auto position_embeddings_file_name = (
        "devkit/" + fmt::format("{}_position_embeddings.npy", _cfg.model_name)
    );
    auto position_embeddings_tensor = cnpy::npy_load(_files->get_path(position_embeddings_file_name));
    _files->release(position_embeddings_file_name);
    get_buffer("position_embeddings").upload(position_embeddings_tensor.data<void>());

    // Populate the future token mask.
    std::vector<Eigen::bfloat16> decoder_future_token_mask(
        _cfg.max_target_positions, std::numeric_limits<Eigen::bfloat16>::lowest()
    );
    auto& future_token_mask_buf = get_buffer("decoder_future_token_mask");
    future_token_mask_buf.clear();
    future_token_mask_buf.upload_raw(
        decoder_future_token_mask.data(),
        _cfg.max_target_positions * 2,
        decoder_future_token_mask.size() * 2
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

    // Layer 0 positioned residual, pre input for later layers, and post input/output.
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
    if (model_type == "encoder_layer") {
        _encoder_layer_model_map.emplace(
            std::piecewise_construct,
            std::forward_as_tuple(std::get<uint8_t>(key)),
            std::forward_as_tuple(model_path, ifms, ofms)
        );
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
    constexpr size_t positioned_residual_output_idx = 3;

    std::vector<uint32_t> missing_encoder_layers;
    for (uint8_t layer_idx = 0; layer_idx < _cfg.encoder_layers; ++layer_idx) {
        // Deferred providers validate each ELF at serial load time, without
        // fetching all encoder weights merely to check their presence.
        if (!_files->pulls_files() &&
            !std::filesystem::is_regular_file(_get_elf_path_encoder_layer(layer_idx)))
            missing_encoder_layers.emplace_back(layer_idx);
    }
    if (!missing_encoder_layers.empty()) {
        const auto legacy_encoder_path = _elf_dir / fmt::format(
            "{}_encoder_stage1_mla.elf", _cfg.model_name
        );
        if (
            missing_encoder_layers.size() == _cfg.encoder_layers
            && std::filesystem::is_regular_file(legacy_encoder_path)
        ) {
            throw std::runtime_error(fmt::format(
                "Unsupported legacy Whisper model at '{}': found monolithic encoder '{}'. "
                "This LLiMa version requires {} layered encoder ELF files named "
                "'{}_encoder_layer<N>_stage1_mla.elf'. Download a layered model or "
                "recompile this model with the current LLiMa version.",
                _devkit_dir.parent_path().string(),
                legacy_encoder_path.filename().string(),
                _cfg.encoder_layers,
                _cfg.model_name
            ));
        }
        const auto encoder_layer_elf_count =
            _cfg.encoder_layers - missing_encoder_layers.size();
        throw std::runtime_error(fmt::format(
            "Incomplete layered Whisper encoder at '{}': found {} of {} expected ELF "
            "files; missing layer indices [{}]. Expected files named "
            "'{}_encoder_layer<N>_stage1_mla.elf'.",
            _elf_dir.string(),
            encoder_layer_elf_count,
            _cfg.encoder_layers,
            fmt::join(missing_encoder_layers, ", "),
            _cfg.model_name
        ));
    }

    _logger->info("Using layered Whisper encoder ({} ELF files)", _cfg.encoder_layers);
    for (uint8_t layer_idx = 0; layer_idx < _cfg.encoder_layers; ++layer_idx) {
        auto& ifm = get_buffer(layer_idx == 0 ? "encoder_ifm" : "encoder_ofm");
        _define_model(
            "encoder_layer",
            layer_idx,
            _get_elf_path_encoder_layer(layer_idx),
            std::vector<MLABufferSlice>{{&ifm}},
            std::vector<MLABufferSlice>{{&get_buffer("encoder_ofm")}}
        );
    }

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
        else {
            ofms.emplace_back(&get_buffer("new_token"));
            if (_cfg.log_probe_enabled)
                ofms.emplace_back(&get_buffer("decoder_logits"));
        }
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
            if (layer_idx == 0)
                pre_ofms.emplace_back(&get_buffer("decoder_n1_buffer1"));
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
                layer_idx == 0
                    ? pre_ofms.at(positioned_residual_output_idx)
                    : pre_ifms[0],
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


std::filesystem::path WhisperModel::_get_elf_path_encoder_layer(uint8_t layer_idx) const {
    return _elf_dir / fmt::format(
        "{}_encoder_layer{}_stage1_mla.elf", _cfg.model_name, layer_idx
    );
}


std::filesystem::path WhisperModel::_get_elf_path_decoder_init(uint8_t layer_idx) const {
    auto elf_file_name = fmt::format(
        "{}_decoder_init_layer{}_stage1_mla.elf", _cfg.model_name, layer_idx
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


void WhisperModel::_run_encoder() {
    for (auto& [layer_idx, encoder_layer_model] : _encoder_layer_model_map) {
        _logger->debug("Queueing Whisper encoder layer {}", layer_idx);
        encoder_layer_model.add_to_queue();
    }
    MLAModelWithBuffer::run_queue();
}


std::vector<uint32_t> WhisperModel::_encode_prompt(const std::string& prompt) const {
    constexpr const char* whitespace = " \t\n\v\f\r";
    const auto begin = prompt.find_first_not_of(whitespace);
    if (begin == std::string::npos)
        return {};
    const auto end = prompt.find_last_not_of(whitespace);

    // Match OpenAI Whisper: encode the stripped text after a space and keep the most recent
    // tokens that fit in half of the decoder context together with <|startofprev|>.
    auto prompt_ids = _tokenizer_ptr->encode(" " + prompt.substr(begin, end - begin + 1), false);
    // Whisper vocabularies place all special and timestamp tokens from <|endoftext|> onward.
    if (std::ranges::any_of(prompt_ids, [this](uint32_t id) { return id >= _stop_token_id; })) {
        throw std::invalid_argument("Whisper prompt must not contain special tokens");
    }
    const size_t max_prompt_tokens = _cfg.max_target_positions / 2 - 1;
    if (prompt_ids.size() > max_prompt_tokens)
        prompt_ids.erase(prompt_ids.begin(), prompt_ids.end() - max_prompt_tokens);
    return prompt_ids;
}


void WhisperModel::_queue_decoder_step(uint32_t token_idx, uint32_t token_id) {
    for (uint8_t layer_idx = 0; layer_idx < _cfg.decoder_layers; ++layer_idx) {
        WhisperDecoderModelMapKey model_key{layer_idx, token_idx};

        std::map<uint8_t, MLABufferSlice> ifm_map;
        if (layer_idx == 0) {
            ifm_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(0),
                std::forward_as_tuple(
                    &get_buffer("token_embeddings"),
                    std::vector<uint32_t>{token_id, 0},
                    std::vector<uint32_t>{1, _cfg.d_model}
                )
            );
        }
        _decoder_pre_model_map.at(model_key).add_to_queue(&ifm_map);
        _decoder_cache_model_map.at(model_key).add_to_queue();
        _decoder_post_model_map.at(model_key).add_to_queue();
    }
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
