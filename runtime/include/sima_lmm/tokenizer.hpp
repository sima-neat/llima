#ifndef _SIMA_LLIMA_TOKENIZER_
#define _SIMA_LLIMA_TOKENIZER_

#include <cstdint>
#include <filesystem>
#include <istream>
#include <limits>
#include <memory>
#include <string>
#include <vector>


namespace simaai {
namespace llima {


// Common text/token conversion interface for Hugging Face and GGUF backends.
// Factories return an owning pointer to the selected implementation.
class Tokenizer {
    public:
        virtual ~Tokenizer() {};

        virtual std::vector<uint32_t> encode(
            const std::string& text, bool add_special_tokens
        ) const = 0;
        virtual std::string decode(
            const std::vector<uint32_t>& ids, bool skip_special_tokens
        ) const = 0;
        virtual std::string id_to_token(uint32_t id) const = 0;
        virtual uint32_t token_to_id(const std::string& token) const = 0;
        virtual std::string get_chat_template() const { return ""; }
        virtual std::string get_bos_token() const { return ""; }
        virtual uint32_t get_bos_token_id() const {
            return std::numeric_limits<uint32_t>::max();
        }
        virtual std::string get_eos_token() const { return ""; }
        virtual uint32_t get_eos_token_id() const {
            return std::numeric_limits<uint32_t>::max();
        }

        static std::unique_ptr<Tokenizer> from_hf_json(
            const std::filesystem::path tokenizer_json_file_name
        );

        // Same as from_hf_json, but reads the tokenizer.json blob from a byte
        // stream (the file seam's open_stream) instead of a path on disk.
        static std::unique_ptr<Tokenizer> from_hf_json(std::istream& tokenizer_json_stream);

        static std::unique_ptr<Tokenizer> from_gguf(const std::filesystem::path gguf_file_name);
};




}
}

#endif
