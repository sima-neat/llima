#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fmt/std.h>
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
    ScopedPtr<SwrContext> swr_ctx(swr_raw);
    if (ret < 0 || !swr_ctx || swr_init(swr_ctx.get()) < 0) throw std::runtime_error("LibAv: Swr setup failed");

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
            if (ret < 0) throw std::runtime_error("LibAv: Resampling failed");
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
