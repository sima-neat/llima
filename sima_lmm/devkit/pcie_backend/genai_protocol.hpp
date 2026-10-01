// The wire protocol between the host and pcie-genai-backend: tag names and
// JSON payloads. Flow: host -> genai.prompt -> BackendLoop -> LLiMa ->
// EventBridge -> genai.token / genai.metrics / genai.final / genai.error.
// Keep it in sync with the host copy (GenAIProtocol.h / .cpp).
#ifndef _SIMA_LLIMA_PCIE_BACKEND_GENAI_PROTOCOL_
#define _SIMA_LLIMA_PCIE_BACKEND_GENAI_PROTOCOL_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simaai {
namespace llima {
namespace pcie_backend {

// The one folder images come from (on the host's data serve root) and land in
// (under the card's recv root): "pcie-genai/<file>".
inline constexpr const char* kImageFolder = "pcie-genai";

// The eight simaai_svc tags between the host and this backend. They must match
// the host (neat/core/pcie_host/src/genai/GenAIProtocol.h).
inline constexpr const char* kTagPrompt  = "genai.prompt";   // host -> card, JSON
inline constexpr const char* kTagCancel  = "genai.cancel";   // host -> card
inline constexpr const char* kTagToken   = "genai.token";    // card -> host, raw text
inline constexpr const char* kTagMetrics = "genai.metrics";  // card -> host, JSON
inline constexpr const char* kTagFinal   = "genai.final";    // card -> host, ends a run
inline constexpr const char* kTagError   = "genai.error";    // card -> host, ends a run
inline constexpr const char* kTagChat    = "genai.chat";     // host -> card, JSON
inline constexpr const char* kTagReply   = "genai.reply";    // card -> host, JSON

struct PromptRequest {
    std::string id;                          // echoed in final/error so the host can match
    std::string prompt;
    std::optional<std::string> system_prompt;
    std::optional<uint16_t> max_new_tokens;  // nullopt = the model default
    bool enable_thinking = false;
    // VLM images the card pulls over PCIe, in order, named relative to a serve
    // root (e.g. "pcie-genai/h1-1-0.jpg"). Empty = no new image. parse_prompt
    // rejects an absolute path, any ".." component, or an empty string, so a
    // name can never escape the serve root the card pulls from.
    std::vector<std::string> images;
};

// Throws std::invalid_argument on bad JSON or a missing/empty "prompt".
PromptRequest parse_prompt(std::string_view json);
// The "id" of a prompt payload, for error replies when parse_prompt fails; "" if unreadable.
std::string try_read_id(std::string_view json);

// genai.chat: a command on the kept conversation.
enum class ChatOp { Reset, Print };
struct ChatRequest {
    std::string id;
    ChatOp op = ChatOp::Print;
    // Reset only. nullopt = the model's default; "" = none; else that text.
    std::optional<std::string> system_prompt;
    bool enable_thinking = false;
};
// Throws std::invalid_argument on bad JSON or an unknown / missing "op".
ChatRequest parse_chat(std::string_view json);
// A genai.reply payload: {"id","ok","text"}.
std::string encode_reply(const std::string& id, bool ok, const std::string& text);

std::string encode_metric(const std::string& type, double value);
// A genai.token payload: the token's raw text with a per-run sequence number
// in front, so the host can spot a dropped token notification. Format is
// "<seq>\n<raw text>". The host splits on the FIRST newline, so the text may
// itself contain newlines. No JSON: tokens are the hot path, one per token.
std::string encode_token(std::uint64_t seq, const std::string& text);
// history_cleared is written only when true (absent = false).
std::string encode_final(const std::string& id, const std::string& finish_reason,
                         uint32_t generated_tokens, double ttft_s, double tps,
                         bool history_cleared = false);
std::string encode_error(const std::string& id, const std::string& message,
                         bool history_cleared = false);

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai

#endif
