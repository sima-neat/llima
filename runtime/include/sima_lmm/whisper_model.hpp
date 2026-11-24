#ifndef _SIMA_LLIMA_WHISPER_MODEL_
#define _SIMA_LLIMA_WHISPER_MODEL_

#include <cstdint>
#include <filesystem>
#include <istream>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <fftw3.h>
#include <spdlog/logger.h>



#include "eigen_types.hpp"
#include "file_provider.hpp"
#include "mla_buffer.hpp"
#include "mla_model.hpp"
#include "text_streamer.hpp"
#include "tokenizer.hpp"
#include "whisper_config.hpp"


namespace simaai {
namespace llima {


struct WhisperPreprocessResources;
struct WhisperWorkspace;
struct WhisperDecodeResult;

class WhisperPreprocessor {
    public:
        WhisperPreprocessor(const std::filesystem::path& devkit_dir);
        explicit WhisperPreprocessor(std::shared_ptr<FileProvider> files);
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
        friend struct WhisperWorkspace;
        WhisperPreprocessor(std::shared_ptr<FileProvider> files, std::filesystem::path identity);
        explicit WhisperPreprocessor(std::unique_ptr<std::istream> config);
        explicit WhisperPreprocessor(std::shared_ptr<const WhisperPreprocessResources> resources);
        ArrayXf _load_audio_ffmpeg(const std::filesystem::path& audio_file_name);
        ArrayXXbf _log_mel_spectrogram(ArrayXf& audio_tensor);

        std::shared_ptr<const WhisperPreprocessResources> _resources;
        ArrayXf _padded_audio_tensor;
        double* _fft_in;
        fftw_complex* _fft_out;
        fftw_plan _fft_plan;
};


// Key to access the whisper decoder model map: (layer_idx, token_idx).
using WhisperDecoderModelMapKey = std::tuple<uint8_t, uint16_t>;
using WhisperDecoderModelMap = std::map<WhisperDecoderModelMapKey, MLAModelWithBuffer>;

template<class Config> struct ModelResources;

struct TranscriptionOptions {
    std::string language = "en";
    std::string task = "transcribe";
    StreamCallbacks callbacks;
};

class WhisperModel {
    public:
        struct TranscriptionResult {
            std::string text;
            std::string language;
            std::string task;
            float no_speech_prob;
            std::optional<float> avg_logprob;
        };

        WhisperModel(std::filesystem::path model_path);
        WhisperModel(std::filesystem::path model_path, std::shared_ptr<FileProvider> files);
        virtual ~WhisperModel();

        TranscriptionResult run_model(
            const std::filesystem::path& audio_file_name,
            const std::string& language,
            const std::string& task = "transcribe"
        );
        TranscriptionResult run_model_from_pcm(
            std::span<const float> pcm,
            uint32_t sample_rate,
            const std::string& language,
            const std::string& task = "transcribe"
        );
        TranscriptionResult run_model(const std::filesystem::path& audio_file,
                                      const TranscriptionOptions& options);
        TranscriptionResult run_model_from_pcm(std::span<const float> pcm, uint32_t sample_rate,
                                               const TranscriptionOptions& options);
        void set_info_callback(InfoCallback callback);
        void set_text_callback(TextCallback callback);
        void set_performance_summary_enabled(bool enabled);
        void wait_for_streamer_completion();
        void stop_model();

    private:
        std::shared_ptr<ModelResources<WhisperConfig>> _resources;
        const WhisperConfig& _cfg;
        const std::filesystem::path& _elf_dir;
        const std::filesystem::path& _devkit_dir;
        const std::shared_ptr<FileProvider>& _files;
        std::shared_ptr<spdlog::logger> _logger;
        std::unique_ptr<WhisperWorkspace> _workspace;
        void define_buffer(const std::string& name, const std::vector<size_t>& shape,
                           const std::string& dtype = "bfloat16", bool align_last_dim = true);
        void _define_resource_buffer(const std::string& name, const std::vector<size_t>& shape);
        MLABuffer& get_buffer(const std::string& name);

        void _initialize();
        void _finalize();
        WhisperDecodeResult _run_model(
            const ArrayXXbf& mel,
            const std::string& language,
            const std::string& task
        );

        void _define_buffers();
        void _define_model(
            const std::string& model_type,
            std::variant<std::monostate, uint8_t, WhisperDecoderModelMapKey> key,
            const std::filesystem::path& model_path,
            const std::vector<MLABufferSlice>& ifms,
            const std::vector<MLABufferSlice>& ofms
        );
        void _define_models();

        std::filesystem::path _get_elf_path_encoder_layer(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_init(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_pre(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_cache(uint16_t token_idx) const;
        std::filesystem::path _get_elf_path_decoder_post(uint8_t layer_idx) const;
        std::filesystem::path _get_elf_path_decoder_language_detect() const;
        void _run_encoder();
        bool _is_auto_language(const std::string& language) const;
        struct LanguageDetectResult {
            uint32_t language_index;
            float no_speech_prob;
        };
        LanguageDetectResult _run_language_detect();
        float _compute_token_logprob(MLABuffer& logits_buf, uint32_t token_id);
        float _compute_token_prob(MLABuffer& logits_buf, uint32_t token_id);
        uint32_t _language_token_from_index(uint32_t language_idx) const;
        void _set_language_token(uint32_t token_id);
        std::string _language_code_from_token(uint32_t token_id) const;
        std::string _update_language(const std::string& language);
        std::string _update_task(const std::string& task);
        



        static const std::map<std::string, std::string> _TO_LANGUAGE_CODE;
};


}
}

#endif
