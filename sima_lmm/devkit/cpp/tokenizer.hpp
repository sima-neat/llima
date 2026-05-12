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

#ifndef _SIMA_LLIMA_TOKENIZER_
#define _SIMA_LLIMA_TOKENIZER_

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <llama.h>
#include <tokenizers_c.h>

namespace simaai {
namespace llima {


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

        static std::unique_ptr<Tokenizer> from_gguf(const std::filesystem::path gguf_file_name);
};


// Reimplement Tokenizer class using tokenizers-cpp's C apis based on 
// https://github.com/mlc-ai/tokenizers-cpp/blob/main/src/huggingface_tokenizer.cc
// to expose the Encode and Decode function with add_special_tokens argument.
class HFTokenizer : public Tokenizer {
    public:
        explicit HFTokenizer(const std::filesystem::path tokenizer_json_file_name) {
            auto size = std::filesystem::file_size(tokenizer_json_file_name);
            std::string blob(size, '\0');
            std::ifstream file(tokenizer_json_file_name, std::ios::binary);
            file.read(blob.data(), size);
            _handle = tokenizers_new_from_str(blob.data(), blob.length());
        }
        HFTokenizer(const HFTokenizer& other) = delete;
        HFTokenizer(HFTokenizer&& other) = delete;
        virtual ~HFTokenizer() {
            if (_handle != nullptr)
                tokenizers_free(_handle);
        }

        virtual std::vector<uint32_t> encode(
            const std::string& text, bool add_special_tokens
        ) const override {
            TokenizerEncodeResult result;
            tokenizers_encode(
                _handle, text.data(), text.length(), static_cast<int>(add_special_tokens), &result
            );
            std::vector<uint32_t> ret(result.token_ids, result.token_ids + result.len);
            tokenizers_free_encode_results(&result, 1);
            return ret;
        }

        virtual std::string decode(
            const std::vector<uint32_t>& ids, bool skip_special_tokens
        ) const override {
            tokenizers_decode(
                _handle, ids.data(), ids.size(), static_cast<int>(skip_special_tokens)
            );
            const char* data;
            size_t len;
            tokenizers_get_decode_str(_handle, &data, &len);
            return std::string(data, len);
        }

        virtual std::string id_to_token(uint32_t id) const override {
            const char* data;
            size_t len;
            tokenizers_id_to_token(_handle, static_cast<uint32_t>(id), &data, &len);
            return std::string(data, len);
        }

        virtual uint32_t token_to_id(const std::string& token) const override {
            int32_t id;
            tokenizers_token_to_id(_handle, token.data(), token.length(), &id);
            if (id < 0) {
                throw std::runtime_error(fmt::format("Unable to find the id for token={}", token));
            }
            return static_cast<uint32_t>(id);
        }

    private:
        TokenizerHandle _handle{nullptr};
};


class GGUFTokenizer : public Tokenizer {
    public:
        explicit GGUFTokenizer(const std::filesystem::path gguf_file_name) {
            llama_backend_init();
            auto params = llama_model_default_params();
            params.vocab_only = true;
            _model = llama_model_load_from_file(gguf_file_name.c_str(), params);
            _vocab = llama_model_get_vocab(_model);
        }
        GGUFTokenizer(const GGUFTokenizer& other) = delete;
        GGUFTokenizer(GGUFTokenizer&& other) = delete;
        virtual ~GGUFTokenizer() {
            llama_model_free(_model);
            llama_backend_free();
        }

        virtual std::vector<uint32_t> encode(
            const std::string& text, bool add_special_tokens
        ) const override {
            // Find the number of tokens.
            const auto num_tokens = -llama_tokenize(
                _vocab, text.c_str(), text.size(), nullptr, 0, add_special_tokens, true
            );
            std::vector<uint32_t> tokens(num_tokens);
            const auto rc = llama_tokenize(
                _vocab, text.c_str(), text.size(), reinterpret_cast<llama_token*>(tokens.data()),
                tokens.size(), add_special_tokens, true
            );
            if (rc < 0) {
                throw std::runtime_error("Failed to tokenize: " + text);
            }
            return tokens;
        }

        virtual std::string decode(
            const std::vector<uint32_t>& ids, bool skip_special_tokens
        ) const override {
            std::string text;
            for (const auto id: ids) {
                char buf[128];
                int n = llama_token_to_piece(_vocab, id, buf, sizeof(buf), 0, !skip_special_tokens);
                if (n < 0) {
                    throw std::runtime_error(
                        "Failed to convert token to piece: " + std::to_string(id)
                    );
                }
                text += std::string(buf, n);
            }
            return text;
        }

        virtual std::string id_to_token(uint32_t id) const override {
            char buf[128];
            int n = llama_token_to_piece(_vocab, id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                throw std::runtime_error("Failed to convert token to piece: " + std::to_string(id));
            }
            return std::string(buf, n);
        }

        virtual uint32_t token_to_id(const std::string& token) const override {
            int n = llama_tokenize(_vocab, token.c_str(), token.size(), nullptr, 0, false, true);
            if (n != -1) {
                throw std::runtime_error("Failed to convert " + token + " to an id");
            }
            int32_t id;
            llama_tokenize(_vocab, token.c_str(), token.size(), &id, 1, false, true);
            if (id < 0) {
                throw std::runtime_error(fmt::format("Unable to find the id for token={}", token));
            }
            return static_cast<uint32_t>(id);
        }

        std::string get_chat_template() const {
            auto chat_template_str_ptr = llama_model_chat_template(_model, nullptr);
            if (!chat_template_str_ptr) {
                throw std::runtime_error("Cannot find the chat template from gguf file.");
            }
            return std::string(chat_template_str_ptr);
        }

        std::string get_bos_token() const {
            return id_to_token(get_bos_token_id());
        }

        uint32_t get_bos_token_id() const {
            return static_cast<uint32_t>(llama_vocab_bos(_vocab));
        }

        std::string get_eos_token() const {
            return id_to_token(get_eos_token_id());
        }

        uint32_t get_eos_token_id() const {
            return static_cast<uint32_t>(llama_vocab_eos(_vocab));
        }

    private:
        llama_model* _model;
        const llama_vocab* _vocab;
};


}
}

#endif
