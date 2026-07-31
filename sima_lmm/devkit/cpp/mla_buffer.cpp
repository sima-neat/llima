#include <fstream>
#include <set>

#include <fmt/ranges.h>
#include <simaai_memory.h>

#include "mla_buffer.hpp"

namespace simaai {
namespace llima {

MLABuffer::MLABuffer(
    std::string name,
    std::vector<size_t> shape,
    std::string dtype,
    bool align_last_dim
) : _name(std::move(name)),
    _shape(std::move(shape)),
    _dtype(std::move(dtype)),
    _align_last_dim(align_last_dim),
    _simaai_mem_ptr(nullptr) {

    if (!_shape.size())
        throw std::runtime_error(
            std::string("Shape needs to have at least 1 dimension") + std::to_string(_shape.size())
        );

    if (_dtype == "bfloat16")
        _elem_size = 2;
    else if (_dtype == "int32")
        _elem_size = 4;
    else if (_dtype == "int8")
        _elem_size = 1;
    else
        throw std::runtime_error(std::string("Dtype to elem size is not defined: ") + _dtype);

    // Compute the total number of bytes required.
    _size = _elem_size;
    for (uint32_t i = 0; i < _shape.size(); ++i) {
        _size *= _shape[i];
    }
    if (!_size)
        throw std::runtime_error(fmt::format("Invalid shape: {}", _shape));
    _stride = std::vector<int64_t>(_shape.size());
    _stride[_shape.size() - 1] = 1;
    if (_align_last_dim) {
        _size_padded = round_up_to_row(_shape.back() * _elem_size);
        for (uint32_t i = 0; i < _shape.size() - 1; ++i) {
            _size_padded *= _shape[i];
        }
        if (_shape.size() > 1) {
            _stride[_shape.size() - 2] = round_up_to(_shape.back(), MLA_ROW_SIZE / _elem_size);
        }
    } else {
        _size_padded = round_up_to_row(_size);
        if (_shape.size() > 1) {
            _stride[_shape.size() - 2] = _shape.back();
        }
    }
    if (_shape.size() > 2) {
        for (int i = _shape.size() - 3; i >= 0; --i) {
            _stride[i] = _stride[i + 1] * _shape[i + 1];
        
        }
    }
}


MLABuffer::~MLABuffer() {
    if (_simaai_mem_ptr) free();
}


void MLABuffer::allocate() {
    // Check if the buffer is already allocated.
    if (_simaai_mem_ptr) {
        throw std::runtime_error("Failed to allocate buffer (" + _name + "): Already allocated");
    }

    // Allocate the buffer.
    _simaai_mem_ptr = simaai_memory_alloc_flags(
        _size_padded, SIMAAI_MEM_TARGET_DMS0, SIMAAI_MEM_FLAG_CACHED
    );
    if (!_simaai_mem_ptr) {
        throw std::runtime_error(
            "Failed to allocate buffer (" + _name + "): " + std::strerror(errno)
        );
    }

    // Map to get the virtual address.
    _virtual_addr = simaai_memory_map(_simaai_mem_ptr);
    if (!_virtual_addr) {
        simaai_memory_free(_simaai_mem_ptr);
        _simaai_mem_ptr = nullptr;
        throw std::runtime_error("Failed to map buffer (" + _name + "): " + std::strerror(errno));
    }

    // Get the physical address.
    _physical_addr = simaai_memory_get_phys(_simaai_mem_ptr);
}

void MLABuffer::free() {
    if (_simaai_mem_ptr == nullptr) return;
    simaai_memory_unmap(_simaai_mem_ptr);
    simaai_memory_free(_simaai_mem_ptr);
    _simaai_mem_ptr = nullptr;
}

void MLABuffer::flush_cache() const {
    simaai_memory_flush_cache(_simaai_mem_ptr);
}

void MLABuffer::invalidate_cache() const {
    simaai_memory_invalidate_cache(_simaai_mem_ptr);
}

void MLABuffer::swap_storage(MLABuffer& other) {
    if (_size_padded != other._size_padded) {
        throw std::invalid_argument("Cannot swap MLA buffers with different allocation sizes");
    }
    std::swap(_simaai_mem_ptr, other._simaai_mem_ptr);
    std::swap(_physical_addr, other._physical_addr);
    std::swap(_virtual_addr, other._virtual_addr);
}

void MLABuffer::clear(bool flush) {
    std::memset(_virtual_addr, 0, _size_padded);
    if (flush)
        flush_cache();
}


void MLABuffer::load_file(const std::filesystem::path& file_name) {
    const size_t file_size = std::filesystem::file_size(file_name);
    if (file_size != _size) {
        throw std::runtime_error(fmt::format(
            "Invalid size for {}: expected {} bytes, got {}",
            file_name.string(), _size, file_size
        ));
    }
    std::ifstream stream(file_name, std::ios::binary);
    load_stream(stream);
}


void MLABuffer::load_stream(std::istream& stream) {
    const size_t row_size = _shape.back() * _elem_size;
    const size_t padded_row_size = round_up_to_row(row_size);
    if (_align_last_dim && row_size != padded_row_size) {
        const size_t num_rows = _size / row_size;
        for (size_t i = 0; i < num_rows; ++i) {
            stream.read(
                reinterpret_cast<char*>(_virtual_addr) + i * padded_row_size,
                row_size
            );
        }
    } else {
        stream.read(reinterpret_cast<char*>(_virtual_addr), _size);
    }
    if (!stream) {
        throw std::runtime_error(fmt::format("Failed to load buffer {}", _name));
    }
    flush_cache();
}


void MLABuffer::upload(const void* data, size_t data_begin, size_t data_size, bool flush) {
    if (_align_last_dim && (_shape.back() % MLA_ROW_SIZE != 0)) {
        assert(data_begin == 0);
        assert(data_size == 0 || data_size == _size);
        uint32_t last_dim = _shape.back() * _elem_size;
        uint32_t last_dim_padded = round_up_to_row(last_dim);
        uint32_t num_last_dims = _size_padded / last_dim_padded;

        for (uint32_t i = 0; i < num_last_dims; ++i) {
            std::memcpy(
                reinterpret_cast<uint8_t*>(_virtual_addr) + i * last_dim_padded,
                reinterpret_cast<const uint8_t*>(data) + i * last_dim,
                last_dim
            );
        }
    } else if (data_size > 0 && data_size < _size) {
        std::memcpy(reinterpret_cast<uint8_t*>(_virtual_addr) + data_begin, data, data_size);
    } else {
        std::memcpy(_virtual_addr, data, _size);
    }
    if (flush)
        flush_cache();
}


void MLABuffer::download(void* data) const {
    invalidate_cache();
    if (_align_last_dim && (_shape.back() % MLA_ROW_SIZE != 0)) {
        uint32_t last_dim = _shape.back() * _elem_size;
        uint32_t last_dim_padded = round_up_to_row(last_dim);
        uint32_t num_last_dims = _size_padded / last_dim_padded;

        std::memset(data, 0, _size);
        for (uint32_t i = 0; i < num_last_dims; ++i) {
            std::memcpy(
                reinterpret_cast<uint8_t*>(data) + i * last_dim,
                reinterpret_cast<const uint8_t*>(_virtual_addr) + i * last_dim_padded,
                last_dim
            );
        }
    } else {
        std::memcpy(data, _virtual_addr, _size);
    }
}


uint32_t MLABuffer::get_buf_addr_offset(const std::optional<std::vector<uint32_t>>& begin) const {
    if (begin.has_value()) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < _shape.size(); ++i) {
            offset = offset * _shape[i] + begin.value()[i];
        }
        return offset * _elem_size;
    } else {
        return 0;
    }
}


uint64_t MLABuffer::get_buf_addr(const std::optional<std::vector<uint32_t>>& begin) const {
    return _physical_addr + get_buf_addr_offset(begin);
}


uint64_t MLABuffer::get_buf_len(const std::optional<std::vector<uint32_t>>& shape) const {
    if (shape.has_value()) {
        uint32_t size = _elem_size;
        for (uint32_t i = 0; i < shape->size(); ++i) {
            size *= shape.value()[i];
        }
        return size;
    } else {
        return _size_padded;
    }
}


void MLABuffer::print(
    std::ostream& s,
    const std::optional<std::vector<uint32_t>>& override_begin,
    const std::optional<std::vector<uint32_t>>& override_shape
) const {
    // Basic info.
    assert(_shape.size() > 0);
    s << _name << " (" << _shape[0];
    for (uint32_t i = 1; i < _shape.size(); ++i) {
        s << ", " << _shape[i];
    }
    s << ") " << _dtype << std::endl;

    std::vector<uint32_t> begin = override_begin.value_or(std::vector<uint32_t>(_shape.size(), 0));
    std::vector<uint32_t> shape = override_shape.value_or(std::vector<uint32_t>(_shape.begin(), _shape.end()));
    assert(begin.size() == _shape.size());
    assert(shape.size() == _shape.size());
    s << "begin = (" << begin[0];
    for (uint32_t i = 1; i < begin.size(); ++i) {
        assert(begin[i] >= 0 && begin[i] < _shape[i]);
        s << ", " << begin[i];
    }
    s << "), shape = (" << shape[0];
    for (uint32_t i = 1; i < shape.size(); ++i) {
        assert(shape[i] > 0 && begin[i] + shape[i]< _shape[i]);
        s << ", " << shape[i];
    }
    s << ")" << std::endl;

    // Invalidate the cache to ensure the content is up to date.
    invalidate_cache();

    // Array content.
    if (_dtype == "int8")
        print_content<int8_t>(s, 0, reinterpret_cast<int8_t*>(_virtual_addr), begin, shape);
    else if (_dtype == "uint8")
        print_content<uint8_t>(s, 0, reinterpret_cast<uint8_t*>(_virtual_addr), begin, shape);
    else if (_dtype == "bfloat16")
        print_content<Eigen::bfloat16>(
            s, 0, reinterpret_cast<Eigen::bfloat16*>(_virtual_addr), begin, shape
        );
    else if (_dtype == "float32")
        print_content<float>(s, 0, reinterpret_cast<float*>(_virtual_addr), begin, shape);
    else if (_dtype == "int32")
        print_content<int32_t>(s, 0, reinterpret_cast<int32_t*>(_virtual_addr), begin, shape);
    else
        throw std::runtime_error(fmt::format("Unsupported print dtype: {}", _dtype));

    // Flush ostream.
    s << std::flush;
}


template <typename T>
void MLABuffer::print_content(
    std::ostream& s, uint32_t level, const T* curr_ptr, const std::vector<uint32_t>& begin,
    const std::vector<uint32_t>& shape
) const {
    const std::string indent(level * 4, ' ');
    std::set<uint32_t> elem_idx_set;
    uint32_t max_print_elems = 4;

    // Array content.
    uint32_t curr_end = begin[level] + shape[level];
    if (level + 1 == _shape.size()) {
        // Last level. Print all elements in one line.
        for (uint32_t i = 0; i < shape[level] && i < max_print_elems; ++i) {
            elem_idx_set.emplace(begin[level] + i);
        }
        if (shape.back() > max_print_elems) {
            for (uint32_t i = 0; i < max_print_elems; ++i) {
                elem_idx_set.emplace(curr_end - i - 1);
            }
        }
        s << indent << "[";
        uint32_t prev_idx = *elem_idx_set.begin();
        for (const auto& idx: elem_idx_set) {
            if (idx > prev_idx + 1) {
                s << " ... ";
            } else {
                s << " ";
            }
            s << curr_ptr[idx];
            prev_idx = idx;
        }
        s << " ]" << std::endl;
    } else {
        for (uint32_t i = 0; i < shape[level] && i < max_print_elems; ++i) {
            elem_idx_set.emplace(begin[level] + i);
        }
        if (shape[level] > max_print_elems) {
            for (uint32_t i = 0; i < max_print_elems; ++i) {
                elem_idx_set.emplace(curr_end - i - 1);
            }
        }
        s << indent << "[" << std::endl;
        uint32_t prev_idx = *elem_idx_set.begin();
        for (const auto& idx: elem_idx_set) {
            if (idx > prev_idx + 1) {
                s << indent << " ... " << std::endl;
            }
            print_content<T>(s, level + 1, curr_ptr + _stride[level] * idx, begin, shape);
            prev_idx = idx;
        }
        s << indent << "]" << std::endl;
    }
}


MLABufferSlice::MLABufferSlice(
    MLABuffer* buf_ptr
) : _buf_ptr(buf_ptr), _begins(std::nullopt), _shapes(std::nullopt) {}


MLABufferSlice::MLABufferSlice(
    MLABuffer* buf_ptr, std::vector<uint32_t> begins
) : _buf_ptr(buf_ptr), _begins(std::move(begins)), _shapes(std::nullopt) {}


MLABufferSlice::MLABufferSlice(
    MLABuffer* buf_ptr, std::vector<uint32_t> begins, std::vector<uint32_t> shapes
) : _buf_ptr(buf_ptr), _begins(std::move(begins)), _shapes(std::move(shapes)) {}


uint64_t MLABufferSlice::get_buf_addr() const {
    if (_buf_ptr) {
        return _buf_ptr->get_buf_addr(_begins);
    } else {
        return 0;
    }
}


uint64_t MLABufferSlice::get_buf_addr(const std::optional<std::vector<uint32_t>>& begins) const {
    if (_buf_ptr) {
        return _buf_ptr->get_buf_addr(begins);
    } else {
        return 0;
    }
}


void MLABufferSlice::to_file(const std::filesystem::path& file_name) const {
    std::filesystem::create_directories(file_name.parent_path());
    std::ofstream out_file(file_name, std::ios::out | std::ios::binary);

    assert(_buf_ptr != nullptr);
    const char* ptr = (
        reinterpret_cast<const char*>(_buf_ptr->get_virtual_addr())
        + _buf_ptr->get_buf_addr_offset(_begins)
    );
    std::streamsize num_bytes = _buf_ptr->get_buf_len(_shapes);

    out_file.write(ptr, num_bytes);
    out_file.close();
}


}
}
