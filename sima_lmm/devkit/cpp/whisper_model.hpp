//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2025 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#ifndef _SIMA_LLIMA_WHISPER_MODEL_
#define _SIMA_LLIMA_WHISPER_MODEL_

#include <filesystem>
#include <map>
#include <memory>
#include <tuple>
#include <mutex>
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
#include "tokenizer.hpp"
#include "whisper_config.hpp"


namespace simaai {
namespace llima {


class WhisperPreprocessor {
    public:
        WhisperPreprocessor(const std::filesystem::path& devkit_dir);
        ~WhisperPreprocessor();

        ArrayXXbf preprocess(const std::filesystem::path& audio_file_name);

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
        WhisperModel(std::filesystem::path model_path, bool do_parallel_load);
        virtual ~WhisperModel() { _finalize(); };

        std::string run_model(
            const std::filesystem::path& audio_file_name, const std::string& language
        );

    private:
        virtual void _initialize() override;
        virtual void _finalize() override;

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
        void _update_language(const std::string& language);
        
        std::mutex _mutex;

        WhisperPreprocessor _preprocessor;
        bool _do_parallel_load;
        std::unique_ptr<Tokenizer> _tokenizer_ptr;

        std::unique_ptr<MLAModelWithBuffer> _encoder_model_ptr;
        std::map<uint8_t, MLAModelWithBuffer> _decoder_init_model_map;
        WhisperDecoderModelMap _decoder_pre_model_map;
        WhisperDecoderModelMap _decoder_cache_model_map;
        WhisperDecoderModelMap _decoder_post_model_map;
        std::vector<uint32_t> _input_token_ids;
        uint32_t _stop_token_id;

        static const std::vector<std::string> _LANGUAGE_CODES;
        static const std::map<std::string, std::string> _TO_LANGUAGE_CODE;
};


}
}

#endif
