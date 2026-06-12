#include "tokenizer.hpp"

namespace simaai {
namespace llima {

std::unique_ptr<Tokenizer> Tokenizer::from_hf_json(
    const std::filesystem::path tokenizer_json_file_name
) {
    return std::make_unique<HFTokenizer>(tokenizer_json_file_name);
}

std::unique_ptr<Tokenizer> Tokenizer::from_gguf(const std::filesystem::path gguf_file_name) {
    return std::make_unique<GGUFTokenizer>(gguf_file_name);
}

}
}
