#pragma once

#include <cstddef>
#include <cstdint>

namespace simaai::llima {

enum class MediaKind { Image, Audio };

// [begin, end) is a run of soft tokens. The index selects the attachment of that kind.
struct MediaSpan {
    MediaKind kind;
    std::size_t attachment_index;
    uint16_t begin;
    uint16_t end;
};

}  // namespace simaai::llima
