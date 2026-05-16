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

#ifndef _SIMA_LLIMA_TEXT_STREAMER_
#define _SIMA_LLIMA_TEXT_STREAMER_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include <spdlog/spdlog.h>

#include "tokenizer.hpp"

namespace simaai {
namespace llima {

// Implement the simplified c++ version of the TextStreamer from Huggingface.
// https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
// It also implement extra features to accept TTFT, TPS information.


enum class DecodeCallbackType : uint32_t {
    TTFT = 1,
    TPS = 2,
    CACHE_FULL = 3,
    STOP = 4
};


struct DecodeCallbackData {
    DecodeCallbackType type;
    uint32_t token_id;
    double duration;
};


class TextStreamer {
    public:
        // Callback for timing/status information (ttft, tps, FULL, END)
        // Parameters: metric_type ("ttft", "tps", "FULL", "END"), metric_value (duration or rate)
        using InfoCallback = std::function<void(const std::string&, double)>;
        
        // Callback for finalized text chunks
        using TextCallback = std::function<void(const std::string&, bool)>;

        TextStreamer(
            Tokenizer* tokenizer_ptr,
            std::optional<InfoCallback> info_callback = std::nullopt,
            std::optional<TextCallback> text_callback = std::nullopt
        );
        ~TextStreamer() {};

        void push(DecodeCallbackData data);
        void push(DecodeCallbackType type, uint32_t token_id, double duration) {
            push({type, token_id, duration});
        }
        void pop_forever(std::stop_token thread_stop_token);
        
        void set_info_callback(InfoCallback callback) {
            std::lock_guard<std::mutex> lock(_mutex);
            _callback_info = callback;
        }
        
        void set_text_callback(TextCallback callback) {
            std::lock_guard<std::mutex> lock(_mutex);
            _callback_finalize_text = callback;
        }

        void put(uint32_t token_id);
        void end();

        void enable() { _is_enabled = true; }
        void disable() { _is_enabled = false; }
        void wait_streaming() {
            _is_streaming.wait(true);
        }

    private:
        void _on_finalized_text(const std::string& text, bool stream_end = false);
        bool _ends_with_chinese_char(const std::string& text);

        Tokenizer* _tokenizer_ptr;
        std::vector<uint32_t> _cached_token_ids;
        uint32_t _print_len;
        bool _is_enabled;

        double _time_to_first_token;
        std::vector<double> _time_to_next_token_vec;

        std::queue<DecodeCallbackData> _queue;
        std::mutex _mutex;
        std::condition_variable_any _cv;

        InfoCallback _callback_info;
        TextCallback _callback_finalize_text;
        std::jthread _pop_thread;
        std::atomic<bool> _is_streaming;

        std::shared_ptr<spdlog::logger> _logger;
};


}
}

#endif
