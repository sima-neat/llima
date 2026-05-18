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


#ifndef _SIMA_LLIMA_MLA_BUFFER_
#define _SIMA_LLIMA_MLA_BUFFER_

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <fmt/format.h>

#include "utils.hpp"

typedef struct simaai_memory_t simaai_memory_t;

namespace simaai {
namespace llima {

class MLABuffer {
    public:
        MLABuffer(
            std::string name,
            std::vector<size_t> shape,
            std::string dtype,
            bool align_last_dim
        );
        ~MLABuffer();
        void allocate();
        void try_allocate() { if (!_simaai_mem_ptr) allocate(); }
        void free();
        void clear(bool flush = true);
        void upload(
            const void* data, size_t data_begin = 0, size_t data_size = 0, bool flush = true
        );
        void download(void* data) const;
        uint32_t get_buf_addr_offset(
            const std::optional<std::vector<uint32_t>>& begin = std::nullopt
        ) const;
        uint64_t get_buf_addr(
            const std::optional<std::vector<uint32_t>>& begin = std::nullopt
        ) const;
        uint64_t get_buf_len(
            const std::optional<std::vector<uint32_t>>& shape = std::nullopt
        ) const;
        void flush_cache() const;
        void invalidate_cache() const;

        const std::string& get_name() const { return _name; }
        const std::string& get_dtype() const { return _dtype; }
        const uint8_t get_elem_size() const { return _elem_size; }
        const std::vector<size_t>& get_shape() const { return _shape; }
        size_t get_num_elems() const { return _size / _elem_size; }
        void* get_virtual_addr() const { return _virtual_addr; }
        simaai_memory_t* get_simaai_memory() const { return _simaai_mem_ptr; }

        void print(
            std::ostream& s,
            const std::optional<std::vector<uint32_t>>& override_begin = std::nullopt,
            const std::optional<std::vector<uint32_t>>& override_shape = std::nullopt
        ) const;

    private:
        template<typename T>
        void print_content(
            std::ostream& s, uint32_t level, const T* curr_ptr, const std::vector<uint32_t>& begin,
            const std::vector<uint32_t>& shape
        ) const;

        std::string _name;
        std::vector<size_t> _shape;
        std::string _dtype;
        uint8_t _elem_size;
        bool _align_last_dim;

        size_t _size;
        size_t _size_padded;
        std::vector<int64_t> _stride;
        simaai_memory_t* _simaai_mem_ptr;
        uint64_t _physical_addr;
        void* _virtual_addr;
};


inline std::ostream& operator<<(std::ostream& s, const MLABuffer& buf) {
    buf.print(s);
    return s;
}


class MLABufferSlice {
    public:
        MLABufferSlice(MLABuffer* buf_ptr = nullptr);
        MLABufferSlice(MLABuffer* buf_ptr, std::vector<uint32_t> begins);
        MLABufferSlice(
            MLABuffer* buf_ptr, std::vector<uint32_t> begins, std::vector<uint32_t> shapes
        );
        ~MLABufferSlice() {}

        MLABuffer* get_buf_ptr() const { return _buf_ptr; }
        uint64_t get_buf_addr() const;
        uint64_t get_buf_addr(const std::optional<std::vector<uint32_t>>& begins) const;
        auto get_buf_begins() const { return _begins; }
        auto get_buf_shapes() const { return _shapes; }

        void to_file(const std::filesystem::path& file_name) const;

    private:
        MLABuffer* _buf_ptr;
        std::optional<std::vector<uint32_t>> _begins;
        std::optional<std::vector<uint32_t>> _shapes;
};


inline std::ostream& operator<<(std::ostream& s, const MLABufferSlice& buf) {
    buf.get_buf_ptr()->print(s, buf.get_buf_begins(), buf.get_buf_shapes());
    return s;
}


}
}

#endif
