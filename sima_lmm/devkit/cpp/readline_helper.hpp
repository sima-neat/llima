#ifndef _SIMA_LLIMA_READLINE_HELPER_
#define _SIMA_LLIMA_READLINE_HELPER_

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <readline/readline.h>
#include <readline/history.h>

namespace simaai {
namespace llima {

// Helper class to support bash-completion
struct CompletionNode {
  std::string token;
  std::vector<CompletionNode> children;
  bool expects_path = false;
};

class ReadlineSupport {
public:
  static constexpr size_t DEFAULT_MAX_ENTRIES = 1000;
  static constexpr char* HISTORY_FILENAME = ".llima_history";

  explicit ReadlineSupport(size_t max_entries = DEFAULT_MAX_ENTRIES) : _max_entries(max_entries) {
    _instance = this;
    _path = history_path();

    _grammar = {
        {"add", {{"image", {}, true}}},
        {"clear", {{"image", {}}, {"system", {}}, {"history", {}}}},
        {"set", {{"system", {}}, {"language", {}}, {"lora", {}, true}, {"audio", {}, true}}},
        {"unset", {{"lora", {}}}},
        {"print", {{"history", {}}}},
        {"help", {}},
        {"quit", {}},
    };

    if (std::filesystem::is_regular_file(_path)) {
      read_history(_path.c_str());
    }

    rl_attempted_completion_function = completion_callback;
    rl_attempted_completion_over = 1;
  }

  ~ReadlineSupport() {
    write_history(_path.c_str());
    history_truncate_file(_path.c_str(), _max_entries);
    _instance = nullptr;
  }

  ReadlineSupport(const ReadlineSupport&) = delete;
  ReadlineSupport& operator=(const ReadlineSupport&) = delete;

  static std::optional<std::string> read_line(const std::string& prompt) {
    char* raw = readline(prompt.c_str());
    if (!raw)
      return std::nullopt;
    std::string result(raw);
    free(raw);
    return result;
  }

  void add_to_history(const std::string& line) {
    if (line.empty())
      return;

    HIST_ENTRY* prev = history_get(history_length);
    if (prev && prev->line && line == prev->line)
      return;

    add_history(line.c_str());
    append_history(1, _path.c_str());
  }

private:
  size_t _max_entries;
  std::filesystem::path _path;
  std::vector<CompletionNode> _grammar;

  // Singleton pointer so the static readline callback can reach us.
  static inline ReadlineSupport* _instance = nullptr;

  static inline std::filesystem::path history_path() {
    const char* home = std::getenv("HOME");
    if (!home || !*home)
      home = "."; // Chose the current-dir
    return std::filesystem::path(home) / HISTORY_FILENAME;
  }

  static char** completion_callback(const char* text, int start, int end) {
    (void)end;
    if (!_instance)
      return nullptr;
    return _instance->generate_completions(text, start);
  }

  // Generates auto completion with the gnu-readline backed technique
  char** generate_completions(const char* text, int start) const {
    std::string buffer(rl_line_buffer, start);

    std::vector<std::string> tokens;
    {
      size_t pos = 0;
      while (pos < buffer.size()) {
        pos = buffer.find_first_not_of(' ', pos);
        if (pos == std::string::npos)
          break;
        auto end = buffer.find(' ', pos);
        if (end == std::string::npos)
          end = buffer.size();
        tokens.push_back(buffer.substr(pos, end - pos));
        pos = end;
      }
    }

    const std::vector<CompletionNode>* candidates = &_grammar;
    bool should_complete_path = false;

    for (size_t i = 0; i < tokens.size() && candidates; ++i) {
      const CompletionNode* matched = nullptr;
      for (const auto& node : *candidates) {
        if (node.token == tokens[i]) {
          matched = &node;
          break;
        }
      }
      if (!matched) {
        candidates = nullptr;
        break;
      }
      should_complete_path = matched->expects_path;
      candidates = &matched->children;
    }

    if (should_complete_path || (candidates && candidates->empty() && should_complete_path)) {
      rl_attempted_completion_over = 0; // let readline do filename completion
      return nullptr;
    }

    // Build matches from grammar candidates
    if (!candidates || candidates->empty())
      return nullptr;

    std::string prefix(text);
    std::vector<std::string> matches;
    for (auto& node : *candidates) {
      if (node.token.compare(0, prefix.size(), prefix) == 0) {
        matches.push_back(node.token);
      }
    }

    if (matches.empty())
      return nullptr;

    char** result = static_cast<char**>(malloc(sizeof(char*) * (matches.size() + 2)));
    if (!result)
      return nullptr;

    std::string lcp = matches[0];
    for (size_t i = 1; i < matches.size(); ++i) {
      size_t j = 0;
      while (j < lcp.size() && j < matches[i].size() && lcp[j] == matches[i][j])
        ++j;
      lcp.resize(j);
    }

    result[0] = strdup(lcp.c_str());
    for (size_t i = 0; i < matches.size(); ++i) {
      result[i + 1] = strdup(matches[i].c_str());
    }
    result[matches.size() + 1] = nullptr;

    rl_attempted_completion_over = 1;
    return result;
  }
};

} // namespace llima
} // namespace simaai

#endif // _SIMA_LLIMA_READLINE_HELPER_
