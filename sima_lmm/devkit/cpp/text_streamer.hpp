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
#include <utility>
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
    bool from_draft = false;
};


class TextStreamer {
    public:
        // Callback for timing/status information (ttft, tps, FULL, END)
        // Parameters: metric_type ("ttft", "tps", "FULL", "END"), metric_value (duration or rate)
        using InfoCallback = std::function<void(const std::string&, double)>;
        
        // Callback for finalized text chunks.
        // Parameters: text, stream_end, from_draft (true when the chunk came
        // from draft-accepted tokens during speculative decoding). Lets the
        // sink decide how to render — ANSI for terminal, HTML/metadata for
        // web, ignored for plain output.
        using TextCallback = std::function<void(const std::string&, bool, bool)>;

        TextStreamer(
            Tokenizer* tokenizer_ptr,
            std::optional<InfoCallback> info_callback = std::nullopt,
            std::optional<TextCallback> text_callback = std::nullopt
        );
        ~TextStreamer() {};

        void push(DecodeCallbackData data);
        void push(DecodeCallbackType type, uint32_t token_id, double duration) {
            push({type, token_id, duration, false});
        }
        void push(
            DecodeCallbackType type, uint32_t token_id, double duration, bool from_draft
        ) {
            push({type, token_id, duration, from_draft});
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

        void set_preserved_token_ids(
            std::vector<std::pair<uint32_t, std::string>> tokens
        );
        void set_tool_call_enabled(bool enabled);

        void put(uint32_t token_id, bool from_draft = false);
        void end();

        void enable() { _is_enabled = true; }
        void disable() { _is_enabled = false; }
        void wait_streaming() {
            _is_streaming.wait(true);
        }

    private:
        void _on_finalized_text(const std::string& text, bool stream_end = false);
        void _flush_cached_text(bool stream_end = false);
        bool _ends_with_chinese_char(const std::string& text);

        Tokenizer* _tokenizer_ptr;
        std::vector<uint32_t> _cached_token_ids;
        uint32_t _print_len;
        bool _is_enabled;
        // from_draft flag for the tokens currently buffered in
        // _cached_token_ids. Set when the buffer transitions from empty to
        // non-empty; used to color the next emit. put() force-flushes the
        // buffer when an incoming token's from_draft differs, so a single
        // emitted chunk is always all-draft or all-target.
        bool _chunk_from_draft = false;
        // Gate for ANSI green wrap on draft-accepted text. Read once at
        // construction from SIMA_LLIMA_ENABLE_DRAFT_HIGHLIGHT; default off.
        bool _highlight_draft_tokens = false;
        // Count of tokens accepted from the draft model this generation;
        // reported in end()'s stats and reset there.
        uint32_t _draft_token_count = 0;

        double _time_to_first_token;
        std::vector<double> _time_to_next_token_vec;

        std::queue<DecodeCallbackData> _queue;
        std::mutex _mutex;
        std::condition_variable_any _cv;

        InfoCallback _callback_info;
        TextCallback _callback_finalize_text;
        std::vector<std::pair<uint32_t, std::string>> _preserved_tokens;
        bool _tool_call_enabled = false;
        std::jthread _pop_thread;
        std::atomic<bool> _is_streaming;

        std::shared_ptr<spdlog::logger> _logger;
};


}
}

#endif
