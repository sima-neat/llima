#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>

#include <fmt/std.h>
#include <nlohmann/json.hpp>

#include "text_streamer.hpp"

namespace simaai {
namespace llima {


TextStreamer::TextStreamer(
    Tokenizer* tokenizer_ptr,
    std::optional<InfoCallback> info_callback,
    std::optional<TextCallback> text_callback
) : _tokenizer_ptr(tokenizer_ptr),
    _print_len(0),
    _is_enabled(true),
    _time_to_first_token(-1),
    // Default info callback: do nothing
    _callback_info(info_callback.value_or([](const std::string&, double) {})),
    // Default text callback: print to console
    _callback_finalize_text(text_callback.value_or(
        [this](const std::string& text, bool stream_end) {
            if (stream_end) {
                std::cout << text << std::endl << std::flush;
            } else {
                std::cout << text << std::flush;
            }
        }
    )),
    _pop_thread(&TextStreamer::pop_forever, this)
{
    auto llima_logger = spdlog::get("llima");
    _logger = llima_logger? llima_logger->clone("STREAM") : spdlog::default_logger();
}


void TextStreamer::push(DecodeCallbackData data) {
    if (!_is_enabled)
        return;

    if (data.type != DecodeCallbackType::TPS) {
        _is_streaming.store(true);
    }

    std::lock_guard<std::mutex> lock(_mutex);
    _queue.push(std::move(data));
    _cv.notify_one();
}


void TextStreamer::pop_forever(std::stop_token thread_stop_token) {
    while (!thread_stop_token.stop_requested()) {
        std::unique_lock<std::mutex> lock(_mutex);
        auto success = _cv.wait(lock, thread_stop_token, [this] { return !_queue.empty(); });

        if (thread_stop_token.stop_requested() || !success)
            return;

        DecodeCallbackData data = std::move(_queue.front());
        _queue.pop();
        
        switch (data.type) {
            case DecodeCallbackType::TTFT:
                put(data.token_id);
                _time_to_first_token = data.duration;
                _callback_info("ttft", data.duration);
                break;
            case DecodeCallbackType::TPS:
                put(data.token_id);
                _time_to_next_token_vec.emplace_back(data.duration);
                _callback_info("tps", 1.0 / data.duration);
                break;
            case DecodeCallbackType::CACHE_FULL:
                end();
                _callback_info("FULL", 0.0);
                _is_streaming.store(false);
                _is_streaming.notify_one();
                _logger->info("Cache full. Please clear history.");
                break;
            case DecodeCallbackType::STOP:
                end();
                _callback_info("END", 0.0);
                _is_streaming.store(false);
                _is_streaming.notify_one();
                break;
            default:
                throw std::runtime_error(
                    fmt::format("Unsupported callback type: {}", static_cast<uint32_t>(data.type))
                );
        }
    }
}


void TextStreamer::put(uint32_t token_id) {
    // Add the new token to the cache and decodes all token ids.
    _cached_token_ids.emplace_back(token_id);
    auto text = _tokenizer_ptr->decode(_cached_token_ids, true);

    std::string printable_text;
    if (text.empty()) {
        // No text. Nothing to be done.
        return;
    } else if (text.back() == '\n') {
        // New line. Flush the cache.
        printable_text = text.substr(_print_len);
        _cached_token_ids.clear();
        _print_len = 0;
    } else if (_ends_with_chinese_char(text)) {
        // CJK charactor or other emoji charactors. Print now.
        printable_text = text.substr(_print_len);
        _print_len += printable_text.length();
    } else {
        // Print until the last space.
        auto last_space_pos = text.rfind(' ');
        if (last_space_pos != std::string::npos && last_space_pos >= _print_len) {
            printable_text = text.substr(_print_len, last_space_pos - _print_len + 1);
            _print_len += printable_text.length();
        }
    }
    _on_finalized_text(printable_text);
}


void TextStreamer::end() {
    std::string printable_text;
    if (_cached_token_ids.empty()) {
        printable_text = "";
    } else {
        auto text = _tokenizer_ptr->decode(_cached_token_ids, true);
        printable_text = text.substr(_print_len);
        _cached_token_ids.clear();
        _print_len = 0;
    }
    _on_finalized_text(printable_text, true);

    // Emit generation stats through the logger. Callers that need structured metrics receive
    // them through the info callback.
    std::vector<std::string> messages;
    messages.emplace_back("");
    if (_time_to_first_token < 0) {
        messages.emplace_back("Number of generated tokens: 0");
    } else if (_time_to_next_token_vec.size()) {
        // Find min/max.
        auto [min_time_it, max_time_it] = std::ranges::minmax_element(_time_to_next_token_vec);
        auto max_tps = static_cast<double>(1) / *min_time_it;
        auto min_tps = static_cast<double>(1) / *max_time_it;

        // Calculate average.
        auto accum_time = std::accumulate(
            _time_to_next_token_vec.begin(), _time_to_next_token_vec.end(), 0.0
        );
        auto avg_tps = static_cast<double>(_time_to_next_token_vec.size()) / accum_time;

        // Find median. Note that the vector is partially sorted after the nth_element.
        size_t mid_pos = _time_to_next_token_vec.size() / 2;
        std::nth_element(
            _time_to_next_token_vec.begin(),
            _time_to_next_token_vec.begin() + mid_pos,
            _time_to_next_token_vec.end()
        );
        auto mid_tps = static_cast<double>(1) / _time_to_next_token_vec[mid_pos];
        auto num_gen_tokens = _time_to_next_token_vec.size() + 1;
        messages.emplace_back(fmt::format("Number of generated tokens: {}", num_gen_tokens));
        messages.emplace_back(fmt::format("TTFT: {:.2f}s", _time_to_first_token));
        messages.emplace_back(
            fmt::format(
                "TPS: max={:.2f}, avg={:.2f}, mid={:.2f}, min={:.2f}", max_tps, avg_tps, mid_tps,
                min_tps
            )
        );
    } else {
        messages.emplace_back("Number of generated tokens: 1");
        messages.emplace_back(fmt::format("TTFT: {:.2f}s", _time_to_first_token));
    }

    for (const auto& message: messages) {
        _logger->info(message);
    }

    // Reset the stats.
    _time_to_first_token = -1;
    _time_to_next_token_vec.clear();
}


void TextStreamer::_on_finalized_text(const std::string& text, bool stream_end) {
    _logger->info("Finalized text: '{}'", text);
    _callback_finalize_text(text, stream_end);
}


bool TextStreamer::_ends_with_chinese_char(const std::string& text) {
    // Empty string is not expected.
    assert(!text.empty());

    // Find the start of the last code point.
    int start_pos = text.size() - 1;
    size_t num_bytes = 1;
    for (start_pos = text.size() - 1; start_pos >= 0; --start_pos, ++num_bytes) {
        if (static_cast<unsigned char>(text[start_pos] & 0xC0) != 0x80)
            break;
    }
    assert(num_bytes <= 4);

    // Construct the code point.
    const unsigned char* data = reinterpret_cast<const unsigned char*>(&text[start_pos]);
    uint32_t cp;
    switch (num_bytes) {
        case 2:
            cp = ((data[0] & 0x1F) << 6) | (data[1] & 0x3F);
            break;
        case 3:
            cp = ((data[0] & 0x0F) << 12) | ((data[1] & 0x3F) << 6) | (data[2] & 0x3F);
            break;
        case 4:
            cp = (
                ((data[0] & 0x07) << 18)
                | ((data[1] & 0x3F) << 12)
                | ((data[2] & 0x3F) << 6)
                | (data[3] & 0x3F)
            );
            break;
        default:
            return false;
    }

    return (
        (cp >= 0x4E00 && cp <= 0x9FFF)
        || (cp >= 0x3400 && cp <= 0x4DBF)
        || (cp >= 0x20000 && cp <= 0x2A6DF)
        || (cp >= 0x2A700 && cp <= 0x2B73F)
        || (cp >= 0x2B740 && cp <= 0x2B81F)
        || (cp >= 0x2B820 && cp <= 0x2CEAF)
        || (cp >= 0xF900 && cp <= 0xFAFF)
        || (cp >= 0x2F800 && cp <= 0x2FA1F)
    );
}
}
}
