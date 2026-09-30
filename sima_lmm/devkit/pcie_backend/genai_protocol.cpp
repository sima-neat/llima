// Parse genai.prompt and build the card -> host JSON payloads.
// Bad input throws, so BackendLoop can answer with genai.error.
// Output never throws on bad UTF-8 (see dump() below).
#include "genai_protocol.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace simaai {
namespace llima {
namespace pcie_backend {

namespace {
using ordered_json = nlohmann::ordered_json;

// Replace invalid UTF-8 instead of throwing: an exception message or a model
// token can carry stray bytes, and a failed dump would lose the whole event.
std::string dump(const ordered_json& j) {
    return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

// A pulled name must stay inside its serve root: not empty, not absolute, no "..".
void check_image_name(const std::string& name) {
    const std::filesystem::path path(name);
    bool has_dotdot = false;
    for (const auto& part : path) {
        if (part == "..") {
            has_dotdot = true;
            break;
        }
    }
    if (name.empty() || path.is_absolute() || has_dotdot) {
        throw std::invalid_argument("\"images\" names must be relative paths with no \"..\"");
    }
}
}  // namespace

PromptRequest parse_prompt(std::string_view json) {
    const nlohmann::json j = nlohmann::json::parse(json, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        throw std::invalid_argument("genai.prompt payload is not a JSON object");
    }
    if (!j.contains("prompt") || !j["prompt"].is_string() ||
        j["prompt"].get<std::string>().empty()) {
        throw std::invalid_argument("genai.prompt needs a non-empty \"prompt\" string");
    }
    PromptRequest request;
    request.prompt = j["prompt"].get<std::string>();
    if (j.contains("id") && j["id"].is_string()) {
        request.id = j["id"].get<std::string>();
    }
    // Same rule as genai.chat: null (or absent) = the model default; any other
    // non-string is refused, not ignored, so the answer never silently uses a
    // different system prompt than the host asked for.
    if (j.contains("system_prompt") && !j["system_prompt"].is_null()) {
        if (!j["system_prompt"].is_string()) {
            throw std::invalid_argument("\"system_prompt\" must be a string");
        }
        request.system_prompt = j["system_prompt"].get<std::string>();
    }
    if (j.contains("max_new_tokens") && !j["max_new_tokens"].is_null()) {
        const auto& value = j["max_new_tokens"];
        if (!value.is_number_integer() || value.get<long long>() < 0) {
            throw std::invalid_argument("\"max_new_tokens\" must be a non-negative integer");
        }
        // 0 means "use the model default". run_model takes a uint16_t, so clamp.
        const long long n = value.get<long long>();
        if (n > 0) {
            request.max_new_tokens = static_cast<uint16_t>(
                std::min<long long>(n, std::numeric_limits<uint16_t>::max()));
        }
    }
    if (j.contains("enable_thinking") && j["enable_thinking"].is_boolean()) {
        request.enable_thinking = j["enable_thinking"].get<bool>();
    }
    if (j.contains("image")) {
        // An older host sends one "image". Answering without it would be a
        // silent wrong answer, so say which side must be updated.
        throw std::invalid_argument(
            "the host sent the old \"image\" key: update the host pcie-genai (this version sends \"images\")");
    }
    if (j.contains("images") && !j["images"].is_null()) {
        if (!j["images"].is_array()) {
            throw std::invalid_argument("\"images\" must be a list of strings");
        }
        for (const auto& item : j["images"]) {
            if (!item.is_string()) {
                throw std::invalid_argument("\"images\" must be a list of strings");
            }
            const std::string name = item.get<std::string>();
            check_image_name(name);
            request.images.push_back(name);
        }
    }
    return request;
}

ChatRequest parse_chat(std::string_view json) {
    const nlohmann::json j = nlohmann::json::parse(json, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) {
        throw std::invalid_argument("genai.chat payload is not a JSON object");
    }
    ChatRequest request;
    if (j.contains("id") && j["id"].is_string()) request.id = j["id"].get<std::string>();
    const std::string op =
        j.contains("op") && j["op"].is_string() ? j["op"].get<std::string>() : std::string();
    if (op == "print") {
        request.op = ChatOp::Print;
        return request;
    }
    if (op != "reset") {
        throw std::invalid_argument("genai.chat \"op\" must be \"reset\" or \"print\"");
    }
    request.op = ChatOp::Reset;
    if (j.contains("system_prompt") && !j["system_prompt"].is_null()) {
        if (!j["system_prompt"].is_string()) {
            throw std::invalid_argument("\"system_prompt\" must be a string");
        }
        request.system_prompt = j["system_prompt"].get<std::string>();
    }
    if (j.contains("enable_thinking") && j["enable_thinking"].is_boolean()) {
        request.enable_thinking = j["enable_thinking"].get<bool>();
    }
    return request;
}

std::string encode_reply(const std::string& id, bool ok, const std::string& text) {
    ordered_json j;
    j["id"] = id;
    j["ok"] = ok;
    j["text"] = text;
    return dump(j);
}

std::string try_read_id(std::string_view json) {
    const nlohmann::json j = nlohmann::json::parse(json, nullptr, /*allow_exceptions=*/false);
    if (j.is_object() && j.contains("id") && j["id"].is_string()) {
        return j["id"].get<std::string>();
    }
    return "";
}

std::string encode_metric(const std::string& type, double value) {
    ordered_json j;
    j["type"] = type;
    j["value"] = value;
    return dump(j);
}

std::string encode_token(std::uint64_t seq, const std::string& text) {
    return std::to_string(seq) + "\n" + text;
}

std::string encode_final(const std::string& id, const std::string& finish_reason,
                         uint32_t generated_tokens, double ttft_s, double tps,
                         bool history_cleared) {
    ordered_json j;
    j["id"] = id;
    j["finish_reason"] = finish_reason;
    j["generated_tokens"] = generated_tokens;
    j["ttft"] = ttft_s;
    j["tps"] = tps;
    if (history_cleared) j["history_cleared"] = true;
    return dump(j);
}

std::string encode_error(const std::string& id, const std::string& message,
                         bool history_cleared) {
    ordered_json j;
    j["id"] = id;
    j["message"] = message;
    if (history_cleared) j["history_cleared"] = true;
    return dump(j);
}

}  // namespace pcie_backend
}  // namespace llima
}  // namespace simaai
