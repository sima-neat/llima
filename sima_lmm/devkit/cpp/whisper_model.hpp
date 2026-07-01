#ifndef _SIMA_LLIMA_WHISPER_MODEL_
#define _SIMA_LLIMA_WHISPER_MODEL_

#include <filesystem>
#include <map>
#include <memory>
#include <tuple>
#include <atomic>
#include <mutex>
#include <span>
#include <vector>

#include <fftw3.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswresample/swresample.h>
}

#include "base_model.hpp"
#include "eigen_types.hpp"
#include "text_streamer.hpp"
#include "tokenizer.hpp"
#include "whisper_config.hpp"


namespace simaai {
namespace llima {


class WhisperPreprocessor {
    public:
        WhisperPreprocessor(const std::filesystem::path& devkit_dir);
        ~WhisperPreprocessor();

        ArrayXXbf preprocess(const std::filesystem::path& audio_file_name);
        ArrayXXbf preprocess_pcm(std::span<const float> pcm, uint32_t sample_rate);

        static constexpr uint32_t SAMPLE_RATE = 16000;
        static constexpr uint32_t N_FFT = 400;
        static constexpr uint32_t HOP_LENGTH = 160;
        static constexpr uint32_t CHUNK_LENGTH = 30;
        static constexpr uint32_t N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE;
        static constexpr uint32_t N_FRAMES = N_SAMPLES / HOP_LENGTH;

    private:
        ArrayXf _load_audio_ffmpeg(const std::filesystem::path& audio_file_name);
        ArrayXXbf _log_mel_spectrogram(ArrayXf& audio_tensor);

        ArrayXXd _mel_filters;
        ArrayXd _hanning_window;
        ArrayXf _padded_audio_tensor;
        double* _fft_in;
        fftw_complex* _fft_out;
        fftw_plan _fft_plan;
};


// Key to access the whisper decoder model map: (layer_idx, token_idx).
using WhisperDecoderModelMapKey = std::tuple<uint8_t, uint16_t>;
using WhisperDecoderModelMap = std::map<WhisperDecoderModelMapKey, MLAModelWithBuffer>;

class WhisperModel : public BaseModel<WhisperConfig> {
    public:
        WhisperModel(std::filesystem::path model_path);
        virtual ~WhisperModel() { _finalize(); };

        std::string run_model(
            const std::filesystem::path& audio_file_name, const std::string& language
        );
        std::string run_model_from_pcm(
            std::span<const float> pcm, uint32_t sample_rate, const std::string& language
        );
        void set_info_callback(TextStreamer::InfoCallback callback) {
            _text_streamer->set_info_callback(callback);
        }
        void set_text_callback(TextStreamer::TextCallback callback) {
            _text_streamer->set_text_callback(callback);
        }
        void wait_for_streamer_completion() {
            _text_streamer->wait_streaming();
        }
        void stop_model();

    private:
        virtual void _initialize() override;
        virtual void _finalize() override;
        std::string _run_model(const ArrayXXbf& mel, const std::string& language);

        virtual void _define_buffers() override;
        void _define_model(
            const std::string& model_type,
            std::variant<std::monostate, uint8_t, WhisperDecoderModelMapKey> key,
            const std::filesystem::path& model_path,
            const std::vector<MLABufferSlice>& ifms,
            const std::vector<MLABufferSlice>& ofms
        );
        void _define_models();

        std::filesystem::path _get_elf_path_encoder() const;
        std::filesystem::path _get_elf_path_decoder_init(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_pre(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_cache(uint16_t token_idx) const;
        std::filesystem::path _get_elf_path_decoder_post(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_language_detect() const;
        bool _is_auto_language(const std::string& language) const;
        uint32_t _detect_language_token();
        void _set_language_token(uint32_t token_id);
        bool _is_language_token(uint32_t token_id) const;
        std::string _language_code_from_token(uint32_t token_id) const;
        void _update_language(const std::string& language);
        
        std::mutex _mutex;

        WhisperPreprocessor _preprocessor;
        std::unique_ptr<Tokenizer> _tokenizer_ptr;
        std::unique_ptr<TextStreamer> _text_streamer;
        std::atomic<bool> _is_running;

        std::unique_ptr<MLAModelWithBuffer> _encoder_model_ptr;
        std::unique_ptr<MLAModelWithBuffer> _decoder_language_detect_model_ptr;
        std::map<uint8_t, MLAModelWithBuffer> _decoder_init_model_map;
        WhisperDecoderModelMap _decoder_pre_model_map;
        WhisperDecoderModelMap _decoder_cache_model_map;
        WhisperDecoderModelMap _decoder_post_model_map;
        std::vector<uint32_t> _input_token_ids;
        uint32_t _stop_token_id;
        bool _language_detect_available = false;

        static const std::vector<std::string> _LANGUAGE_CODES;
        static const std::map<std::string, std::string> _TO_LANGUAGE_CODE;
};


}
}

#endif
