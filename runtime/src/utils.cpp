#include "utils.hpp"

#include <stdexcept>

#include <openssl/bio.h>
#include <openssl/buffer.h>
#include <openssl/evp.h>

namespace simaai {
namespace llima {


std::optional<std::filesystem::path> sample_image_file_name = std::nullopt;
std::optional<std::filesystem::path> sample_audio_file_name = std::nullopt;

std::vector<uint8_t> base64_decode(std::string_view base64_data) {
    // Decode base64 data.
    std::vector<uint8_t> decoded_data(base64_data.size());

    BIO* bio_b64 = BIO_new(BIO_f_base64());
    BIO_set_flags(bio_b64, BIO_FLAGS_BASE64_NO_NL);
    BIO* bio_mem = BIO_new_mem_buf(base64_data.data(), static_cast<int>(base64_data.size()));
    BIO* bio_chain = BIO_push(bio_b64, bio_mem);
    int decoded_length = BIO_read(
        bio_chain, decoded_data.data(), static_cast<int>(decoded_data.size())
    );
    BIO_free_all(bio_chain);

    if (decoded_length <= 0)
        throw std::runtime_error("Failed to decode base64 data");
    decoded_data.resize(decoded_length);
    return decoded_data;
}

void set_sample_image_file_name(std::filesystem::path file_name) {
    if (std::filesystem::is_regular_file(file_name))
        sample_image_file_name = file_name;
    else
        throw std::runtime_error("Specified image file name does not exist: " + file_name.string());
}

void set_sample_audio_file_name(std::filesystem::path file_name) {
    if (std::filesystem::is_regular_file(file_name))
        sample_audio_file_name = file_name;
    else
        throw std::runtime_error("Specified audio file name does not exist: " + file_name.string());
}


}
}
