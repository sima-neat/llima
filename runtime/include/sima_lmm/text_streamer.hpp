#ifndef _SIMA_LLIMA_TEXT_STREAMER_
#define _SIMA_LLIMA_TEXT_STREAMER_

#include <atomic>
#include <condition_variable>
#include <functional>
#include <exception>
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

#include "generation_types.hpp"
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
    STOP = 4,
    CACHED_PROMPT_TOKENS = 5,
    CACHE_CREATED = 6
};


// Queued token or status event consumed by the streaming worker.
struct DecodeCallbackData {
    DecodeCallbackType type;
    uint32_t token_id;
    double duration;
    bool from_draft = false;
};


// Decodes queued tokens and delivers text and status callbacks on a worker thread.
// Keeps request settings separate from defaults and waits for callbacks when finishing.
class TextStreamer {
    public:
        // Per-request overrides for callbacks, preserved markers, reporting and streaming.
        struct RequestOptions {
            StreamCallbacks callbacks;
            std::optional<PreservedTokens> preserved_tokens;
            std::optional<bool> performance_summary;
            bool streaming = true;
        };

        TextStreamer(
            Tokenizer* tokenizer_ptr,
            std::optional<InfoCallback> info_callback = std::nullopt,
            std::optional<TextCallback> text_callback = std::nullopt
        );
        ~TextStreamer();

        void begin();
        void begin(StreamCallbacks callbacks);
        void begin_request(RequestOptions options);
        void finish();
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
        
        std::pair<InfoCallback, TextCallback> callbacks() {
            std::lock_guard<std::mutex> lock(_mutex);
            return {_defaults.info, _defaults.text};
        }

        void set_info_callback(InfoCallback callback) {
            std::lock_guard<std::mutex> lock(_mutex);
            _defaults.info = std::move(callback);
        }
        
        void set_text_callback(TextCallback callback) {
            std::lock_guard<std::mutex> lock(_mutex);
            _defaults.text = std::move(callback);
        }

        void set_preserved_token_ids(
            std::vector<std::pair<uint32_t, std::string>> tokens
        );
        void set_performance_summary_enabled(bool enabled) {
            std::lock_guard<std::mutex> lock(_mutex);
            _defaults.performance_summary = enabled;
        }

        void put(uint32_t token_id, bool from_draft = false);
        void end();

        bool enabled() const { return _is_enabled; }
        void enable() { _is_enabled = true; }
        void disable() { _is_enabled = false; }
        void wait_streaming();

    private:
        // Resolved callbacks and output settings used as defaults or for the active request.
        struct Settings {
            InfoCallback info;
            TextCallback text;
            std::vector<std::pair<uint32_t, std::string>> preserved_tokens;
            bool performance_summary = true;
        };
        void _begin_locked(RequestOptions options);
        void _wait_for_completion(bool owner);
        // Direct put/end callers are synchronous and use the current defaults.
        const Settings& _settings() const { return _is_streaming ? _request : _defaults; }
        void _on_finalized_text(const std::string& text, bool stream_end = false);
        void _flush_cached_text(bool stream_end = false);
        bool _ends_with_chinese_char(const std::string& text);

        Tokenizer* _tokenizer_ptr;
        std::vector<uint32_t> _cached_token_ids;
        uint32_t _print_len;
        std::atomic<bool> _is_enabled;
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
        bool _completion_queued = true;
        std::exception_ptr _callback_error;
        std::mutex _mutex;
        std::condition_variable_any _cv;

        Settings _defaults;
        Settings _request;
        bool _explicit_request = false;
        bool _muted = false;
        std::atomic<bool> _is_streaming = false;
        std::jthread _pop_thread;

        std::shared_ptr<spdlog::logger> _logger;
};


}
}

#endif
