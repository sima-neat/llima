#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <complex>
#include <cmath>
#include <fmt/std.h>
#include <nlohmann/json.hpp>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

#include "audio_preprocessing.hpp"

namespace simaai::llima {

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

void check_audio_input(int error, const std::string& message) {
    if (error >= 0) return;
    if (error == AVERROR_INVALIDDATA || error == AVERROR_EOF || error == AVERROR(EINVAL)
        || error == AVERROR_DEMUXER_NOT_FOUND || error == AVERROR_STREAM_NOT_FOUND
        || error == AVERROR_DECODER_NOT_FOUND) {
        throw std::invalid_argument(message);
    }
    throw std::runtime_error(message);
}
}

Gemma4AudioPreprocessor::Gemma4AudioPreprocessor(const AudioModelConfig& config, FileProvider& files)
    : _cfg(config), _fft_input(config.fft_length), _fft_output(config.fft_length / 2 + 1),
      _fft_plan(nullptr) {
    const auto data = nlohmann::json::parse(*files.open_stream("devkit/audio_preprocessor_config.json"));
    const auto window = data.at("window").get<std::vector<float>>();
    if (window.size() != _cfg.frame_length || _cfg.fft_length < _cfg.frame_length
        || !_cfg.hop_length || !_cfg.max_tokens || !_cfg.attention_chunk_size) {
        throw std::invalid_argument("Invalid Gemma audio frontend dimensions");
    }
    _window = Eigen::Map<const ArrayXf>(window.data(), window.size());
    const auto& filters = data.at("mel_filters");
    if (filters.size() != _fft_output.size()) throw std::invalid_argument("Invalid audio mel filters");
    _mel_filters.resize(filters.size(), _cfg.num_mel_bins);
    for (size_t row = 0; row < filters.size(); ++row) {
        const auto values = filters[row].get<std::vector<double>>();
        if (values.size() != _cfg.num_mel_bins) throw std::invalid_argument("Invalid audio mel filter width");
        _mel_filters.row(row) = Eigen::Map<const ArrayXd>(values.data(), values.size());
    }
    auto read_bins = [&](const char* key, double fallback) {
        const auto values = data.at(key).is_null()
            ? std::vector<double>(_cfg.num_mel_bins, fallback) : data.at(key).get<std::vector<double>>();
        if (values.size() != _cfg.num_mel_bins) throw std::invalid_argument("Invalid audio feature normalization");
        return ArrayXd(Eigen::Map<const ArrayXd>(values.data(), values.size()));
    };
    _mean = read_bins("per_bin_mean", 0.0);
    _stddev = read_bins("per_bin_stddev", 1.0);
    _mel_floor = data.at("mel_floor").get<double>();
    if ((_stddev <= 0).any() || _mel_floor <= 0) throw std::invalid_argument("Invalid audio feature scale");
    _fft_plan = fftw_plan_dft_r2c_1d(_cfg.fft_length, _fft_input.data(),
        reinterpret_cast<fftw_complex*>(_fft_output.data()), FFTW_ESTIMATE);
    if (!_fft_plan) throw std::runtime_error("Could not create Gemma audio FFT plan");
}

Gemma4AudioPreprocessor::~Gemma4AudioPreprocessor() { fftw_destroy_plan(_fft_plan); }

AudioInput Gemma4AudioPreprocessor::preprocess(const ArrayXf& samples) {
    if (samples.size() == 0 || !samples.isFinite().all()) throw std::invalid_argument("Audio is empty or nonfinite");
    if (samples.size() > std::min<uint32_t>(30 * _cfg.sampling_rate, _cfg.num_frames() * _cfg.hop_length))
        throw std::invalid_argument("Audio exceeds the compiled token budget");
    AudioInput result{{}, {}, 0};
    result.features.resize(static_cast<size_t>(_cfg.num_frames()) * _cfg.num_mel_bins, Eigen::bfloat16{0.0f});
    result.mask.resize(_cfg.num_frames(), Eigen::bfloat16{0.0f});
    const int64_t left = _cfg.frame_length / 2;
    ArrayXd magnitude(_fft_output.size());
    for (uint32_t frame = 0; frame < _cfg.num_frames(); ++frame) {
        const int64_t start = static_cast<int64_t>(frame) * _cfg.hop_length - left;
        // The source mask checks the extra analysis-window sample as well.
        if (start + _cfg.frame_length >= samples.size()) break;
        std::fill(_fft_input.begin(), _fft_input.end(), 0.0);
        for (uint32_t i = 0; i < _cfg.frame_length; ++i) {
            if (start + i >= 0) _fft_input[i] = samples[start + i] * _window[i];
        }
        fftw_execute(_fft_plan);
        for (size_t i = 0; i < _fft_output.size(); ++i) magnitude[i] = std::abs(_fft_output[i]);
        ArrayXd mel = (magnitude.matrix() * _mel_filters.matrix()).array();
        mel = ((mel + _mel_floor).log() - _mean) / _stddev;
        for (uint32_t i = 0; i < _cfg.num_mel_bins; ++i)
            result.features[static_cast<size_t>(frame) * _cfg.num_mel_bins + i] = Eigen::bfloat16(mel[i]);
        result.mask[frame] = Eigen::bfloat16{1.0f};
        result.num_tokens += frame % 4 == 0;
    }
    if (!result.num_tokens) throw std::invalid_argument("Audio is too short to produce a valid audio token");
    return result;
}

ArrayXf load_audio(std::span<const uint8_t> data, size_t max_samples) {
    if (data.empty()) throw std::invalid_argument("Audio input is empty");
    const std::unique_ptr<FILE, int (*)(FILE*)> file(std::tmpfile(), &std::fclose);
    if (!file) throw std::runtime_error("Could not create temporary audio file");
    if (std::fwrite(data.data(), 1, data.size(), file.get()) != data.size()
        || std::fflush(file.get()) != 0) {
        throw std::runtime_error("Could not write temporary audio file");
    }
    // Linux exposes the anonymous file to FFmpeg without keeping a named file
    // around for the lifetime of a streaming request or chat history.
    return load_audio(fmt::format("/proc/self/fd/{}", fileno(file.get())), max_samples);
}

ArrayXf load_audio(
    const std::filesystem::path& audio_file_name, size_t max_samples, bool truncate
) {
    if (max_samples == 0) throw std::invalid_argument("Audio sample budget must be positive");
    av_log_set_level(AV_LOG_ERROR);

    // 1. Open Input
    AVFormatContext* ctx_raw = nullptr;
    check_audio_input(avformat_open_input(&ctx_raw, audio_file_name.string().c_str(), nullptr, nullptr),
                      fmt::format("LibAv: Open failed: {}", audio_file_name.string()));
    ScopedPtr<AVFormatContext> fmt_ctx(ctx_raw);

    check_audio_input(avformat_find_stream_info(fmt_ctx.get(), nullptr), "LibAv: No stream info");

    // 2. Find Stream & Codec
    int stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    check_audio_input(stream_idx, "LibAv: No audio stream");

    AVCodecParameters* params = fmt_ctx->streams[stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(params->codec_id);
    if (!codec) throw std::invalid_argument("LibAv: Audio codec is not supported");

    // 3. Setup Codec
    ScopedPtr<AVCodecContext> codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) throw std::runtime_error("LibAv: Codec alloc failed");

    check_audio_input(avcodec_parameters_to_context(codec_ctx.get(), params), "LibAv: Params copy failed");
    check_audio_input(avcodec_open2(codec_ctx.get(), codec, nullptr), "LibAv: Codec open failed");

    // 4. Setup Resampler
    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;
    SwrContext* swr_raw = nullptr;
    int ret = swr_alloc_set_opts2(&swr_raw, &out_layout, AV_SAMPLE_FMT_S16, 16000,
                                  &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate,
                                  0, nullptr);
    ScopedPtr<SwrContext> swr_ctx(swr_raw);
    check_audio_input(ret, "LibAv: Swr setup failed");
    if (!swr_ctx) throw std::runtime_error("LibAv: Swr allocation failed");
    check_audio_input(swr_init(swr_ctx.get()), "LibAv: Swr setup failed");

    // 5. Decoding Loop
    ScopedPtr<AVPacket> packet(av_packet_alloc());
    ScopedPtr<AVFrame> frame(av_frame_alloc());
    if (!packet || !frame) throw std::runtime_error("LibAv: Allocation failed");

    std::vector<int16_t> samples(max_samples, 0);
    size_t total = 0;
    std::vector<int16_t> pcm_buf;

    auto process_data = [&](const uint8_t** in_data, int in_count) {
        if (truncate && total >= max_samples) return;
        int delay = swr_get_delay(swr_ctx.get(), codec_ctx->sample_rate);
        int out_count = av_rescale_rnd(delay + in_count, 16000, codec_ctx->sample_rate, AV_ROUND_UP);

        if (out_count > 0) {
            if (pcm_buf.size() < static_cast<size_t>(out_count)) {
                pcm_buf.resize(out_count);
            }
            uint8_t* out_ptrs[1] = { reinterpret_cast<uint8_t*>(pcm_buf.data()) };

            int ret = swr_convert(swr_ctx.get(), out_ptrs, out_count, in_data, in_count);
            check_audio_input(ret, "LibAv: Resampling failed");
            if (ret > 0) {
                if (!truncate && static_cast<size_t>(ret) > max_samples - total) {
                    throw std::invalid_argument("Audio exceeds the compiled token budget");
                }
                size_t copy_len = std::min(static_cast<size_t>(ret), max_samples - total);
                std::memcpy(samples.data() + total, pcm_buf.data(), copy_len * sizeof(int16_t));
                total += copy_len;
            }
        }
    };

    while (av_read_frame(fmt_ctx.get(), packet.get()) >= 0) {
        if (packet->stream_index == stream_idx) {
            if (avcodec_send_packet(codec_ctx.get(), packet.get()) >= 0) {
                while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
                    process_data((const uint8_t**)frame->extended_data, frame->nb_samples);
                }
            }
        }
        av_packet_unref(packet.get());
        if (truncate && total >= max_samples) break;
    }

    // Drain delayed decoder frames before flushing the resampler.
    if (!truncate || total < max_samples) {
        if (!truncate && avcodec_send_packet(codec_ctx.get(), nullptr) >= 0) {
            while (avcodec_receive_frame(codec_ctx.get(), frame.get()) >= 0) {
                process_data((const uint8_t**)frame->extended_data, frame->nb_samples);
            }
        }
        process_data(nullptr, 0);
    }

    // 6. Return
    if (total == 0 && !truncate) throw std::invalid_argument("Audio input is empty");
    Eigen::Map<ArrayXi16> map(samples.data(), total);
    return map.cast<float>() * (1.0f / 32768.0f);
}


}
