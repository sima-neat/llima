#include "language/language_model_state.hpp"

#include <iostream>
#include <stdexcept>

using namespace simaai::llima;

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void check_cancellation() {
    InterruptCounter target{0}, draft{0};
    auto request = [&](std::stop_token token) {
        return RequestState{{{}, 32}, token, RequestState::Interrupt(target),
                            RequestState::Interrupt(draft)};
    };
    std::stop_source first, second;
    auto a = request(first.get_token());
    auto b = request(second.get_token());
    first.request_stop();
    require(a.cancelled() && !b.cancelled(), "A late stop reached the next request");
    require(request(first.get_token()).cancelled(), "A stop issued before admission was lost");
    ++draft;
    require(b.cancelled(), "Stopping the draft did not interrupt its request");
    auto next = request({});
    require(!next.cancelled(), "A legacy interrupt carried over to a new request");
    ++target;
    require(next.cancelled(), "Stopping the target did not interrupt its request");

    TextStreamer stream(nullptr);
    RequestControl control({}, 32, stream);
    auto without_request = [&](auto&& operation) {
        bool rejected = false;
        try { operation(); } catch (const std::logic_error&) { rejected = true; }
        require(rejected, "Execution without a request was accepted");
    };
    without_request([&] { control.limits(); });
    without_request([&] { control.cancelled(); });
    control.active = &next;
    require(&control.limits() == &next.limits && control.cancelled(),
            "Execution did not use its active request");
}

void check_streamer() {
    // The only input token is a preserved marker. It never needs a tokenizer or model.
    TextStreamer stream(nullptr);
    stream.set_performance_summary_enabled(false);
    std::string text;
    bool ended = false;
    TextStreamer::RequestOptions options;
    options.preserved_tokens = PreservedTokens{{42, "<tool>"}};
    options.callbacks.text = [&](const std::string& chunk, bool end, bool) {
        text += chunk;
        ended = end;
    };
    stream.begin_request(std::move(options));
    stream.set_text_callback([](const std::string&, bool, bool) { throw std::runtime_error("last callback"); });
    stream.push(DecodeCallbackType::TTFT, 42, 0.1);
    stream.finish();
    require(ended && text == "<tool>", "A default override changed a running stream");

    stream.begin();
    stream.push(DecodeCallbackType::STOP, 0, 0);
    stream.wait_streaming();
    bool reported = false;
    try {
        stream.finish();
    } catch (const std::runtime_error& error) {
        reported = std::string_view(error.what()) == "last callback";
    }
    require(reported, "The owning request lost its final callback error");
    stream.wait_streaming();
    stream.finish();

    int completions = 0;
    stream.set_text_callback([&](const std::string&, bool end, bool) { if (end) ++completions; });
    TextStreamer::RequestOptions muted;
    muted.streaming = false;
    stream.begin_request(std::move(muted));
    stream.push(DecodeCallbackType::STOP, 0, 0);
    stream.finish();
    require(completions == 0, "A muted request emitted callbacks");
    stream.begin();
    stream.finish();
    require(completions == 1, "A callback error or a mute leaked into the next request");
}
} // namespace

int main() {
    try {
        check_cancellation();
        check_streamer();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Request cancellation and callback lifecycle passed\n";
}
