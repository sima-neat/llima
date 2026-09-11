#include <atomic>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <utility>

#include <fmt/ranges.h>
#include <simaai/simaai_heap.h>

#include "mla_buffer.hpp"

namespace simaai {
namespace llima {

namespace {
std::atomic<uint64_t> next_allocation_generation{1};

// B1297's MLA-RT performs partial cache maintenance the same way: the
// cacheable DMS mapping gives userspace a VA and arm64 permits dc cvac/civac
// at EL0 when Linux enables SCTLR_EL1.UCI. Keep these operations local to
// MLABuffer: every buffer allocated by this class comes from the SiMa DMS
// dma-heap and LLiMa serializes CPU access with MLA completion.
#if defined(__aarch64__)
constexpr size_t kA65CacheLineSize = 64;

void clean_cache_range(const void* virtual_address, size_t size) {
    if (!virtual_address || size == 0) return;
    uintptr_t address = reinterpret_cast<uintptr_t>(virtual_address);
    const uintptr_t end = address + size;
    address &= ~static_cast<uintptr_t>(kA65CacheLineSize - 1);
    for (; address < end; address += kA65CacheLineSize) {
        __asm__ __volatile__("dc cvac, %0" : : "r"(address) : "memory");
    }
    __asm__ __volatile__("dsb sy" : : : "memory");
}

void invalidate_cache_range(const void* virtual_address, size_t size) {
    if (!virtual_address || size == 0) return;
    uintptr_t address = reinterpret_cast<uintptr_t>(virtual_address);
    const uintptr_t end = address + size;
    address &= ~static_cast<uintptr_t>(kA65CacheLineSize - 1);
    // civac is available at EL0 and protects unrelated dirty bytes sharing an
    // edge cache line. Ownership discipline guarantees that device-written
    // ranges do not contain intentional dirty CPU data when this is called.
    for (; address < end; address += kA65CacheLineSize) {
        __asm__ __volatile__("dc civac, %0" : : "r"(address) : "memory");
    }
    __asm__ __volatile__("dsb sy" : : : "memory");
}
#endif

size_t checked_mul_size(size_t lhs, size_t rhs, const char* what) {
    if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
        throw std::overflow_error(what);
    }
    return lhs * rhs;
}

size_t checked_round_up_size(size_t value, size_t alignment, const char* what) {
    if (alignment == 0) {
        throw std::invalid_argument("MLA buffer alignment must be non-zero");
    }
    const size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    const size_t increment = alignment - remainder;
    if (value > std::numeric_limits<size_t>::max() - increment) {
        throw std::overflow_error(what);
    }
    return value + increment;
}
}

MLABuffer::MLABuffer(
    std::string name,
    std::vector<size_t> shape,
    std::string dtype,
    bool align_last_dim
) : _name(std::move(name)),
    _shape(std::move(shape)),
    _dtype(std::move(dtype)),
    _align_last_dim(align_last_dim),
    _simaai_dmabuf_ptr(nullptr),
    _virtual_addr(nullptr) {

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

    /*
     * Compute both logical and physically padded sizes with checked
     * arithmetic.  An overflow here would otherwise wrap into a small DMS0
     * allocation and make every later BufferView range check meaningless.
     */
    _size = _elem_size;
    for (const size_t dim : _shape) {
        if (dim == 0) {
            throw std::runtime_error(fmt::format("Invalid zero-sized shape: {}", _shape));
        }
        _size = checked_mul_size(_size, dim, "MLA buffer logical size overflow");
    }
    _stride = std::vector<uint64_t>(_shape.size());
    _stride[_shape.size() - 1] = 1;
    if (_align_last_dim) {
        const size_t logical_row_bytes = checked_mul_size(
            _shape.back(), _elem_size, "MLA buffer row size overflow"
        );
        _size_padded = checked_round_up_size(
            logical_row_bytes, MLA_ROW_SIZE, "MLA buffer padded row size overflow"
        );
        for (uint32_t i = 0; i < _shape.size() - 1; ++i) {
            _size_padded = checked_mul_size(
                _size_padded, _shape[i], "MLA buffer padded size overflow"
            );
        }
        if (_shape.size() > 1) {
            _stride[_shape.size() - 2] = checked_round_up_size(
                logical_row_bytes, MLA_ROW_SIZE, "MLA buffer padded row size overflow"
            ) / _elem_size;
        }
    } else {
        _size_padded = checked_round_up_size(
            _size, MLA_ROW_SIZE, "MLA buffer padded size overflow"
        );
        if (_shape.size() > 1) {
            _stride[_shape.size() - 2] = _shape.back();
        }
    }
    if (_shape.size() > 2) {
        for (int i = _shape.size() - 3; i >= 0; --i) {
            _stride[i] = checked_mul_size(
                _stride[i + 1], _shape[i + 1], "MLA buffer stride overflow"
            );
        }
    }
}


MLABuffer::~MLABuffer() {
    if (_simaai_dmabuf_ptr) free();
}


void MLABuffer::allocate() {
    // Check if the buffer is already allocated.
    if (_simaai_dmabuf_ptr) {
        throw std::runtime_error("Failed to allocate buffer (" + _name + "): Already allocated");
    }

    // Allocate the buffer.
    // B1157 exposes MLA-only memory through /dev/dma_heap/simaai,dms. The
    // legacy DMS0 targets in simaai-memory are no longer allocatable on 3.0.
    // simaai-heap keeps allocation, mapping, address discovery, and cache
    // ownership in one supported API while MLA-RT continues to receive the
    // physical address of each tensor slice.
    simaai_heap_t* heap = simaai_heap_open_type(SIMAAI_HEAP_DMS, nullptr);
    if (!heap) {
        throw std::runtime_error(
            "Failed to open MLA DMS heap for buffer (" + _name + "): " +
            std::strerror(errno)
        );
    }
    _simaai_dmabuf_ptr = simaai_heap_alloc(
        heap, _size_padded, SIMAAI_HEAP_FLAG_DEFAULT
    );
    const int allocation_errno = errno;
    simaai_heap_close(heap);
    if (!_simaai_dmabuf_ptr) {
        throw std::runtime_error(
            "Failed to allocate buffer (" + _name + "): " +
            std::strerror(allocation_errno)
        );
    }

    // Map to get the virtual address.
    _virtual_addr = simaai_dmabuf_map(_simaai_dmabuf_ptr);
    if (!_virtual_addr) {
        const int map_errno = errno;
        simaai_dmabuf_free(_simaai_dmabuf_ptr);
        _simaai_dmabuf_ptr = nullptr;
        throw std::runtime_error(
            "Failed to map buffer (" + _name + "): " + std::strerror(map_errno)
        );
    }

    _physical_addr = simaai_dmabuf_get_phys(_simaai_dmabuf_ptr);
    if (_physical_addr == 0) {
        simaai_dmabuf_free(_simaai_dmabuf_ptr);
        _simaai_dmabuf_ptr = nullptr;
        _virtual_addr = nullptr;
        throw std::runtime_error(
            "Failed to resolve physical address for buffer (" + _name + ")"
        );
    }
    _allocation_generation = next_allocation_generation.fetch_add(
        1, std::memory_order_relaxed
    );
}

void MLABuffer::free() {
    if (_simaai_dmabuf_ptr == nullptr) return;
    simaai_dmabuf_free(_simaai_dmabuf_ptr);
    _simaai_dmabuf_ptr = nullptr;
    _allocation_generation = 0;
    _virtual_addr = nullptr;
    _physical_addr = 0;
}


int MLABuffer::get_dmabuf_fd() const {
    if (!_simaai_dmabuf_ptr) {
        throw std::logic_error("cannot get the dma-buf fd of an unallocated MLA buffer");
    }
    const int fd = simaai_dmabuf_get_fd(_simaai_dmabuf_ptr);
    if (fd < 0) {
        throw std::runtime_error(fmt::format(
            "Failed to get dma-buf fd for MLA buffer {}", _name
        ));
    }
    return fd;
}

void MLABuffer::flush_cache() const {
    flush_cache(0, _size_padded);
}

void MLABuffer::flush_cache(size_t offset, size_t size) const {
    if (!_simaai_dmabuf_ptr)
        throw std::logic_error("cannot flush an unallocated MLA buffer");
    if (offset > _size_padded || size > _size_padded - offset) {
        throw std::out_of_range(fmt::format(
            "Cache flush range [{}, {}) exceeds buffer {} size {}",
            offset, offset + size, _name, _size_padded
        ));
    }
    if (size == 0) return;
#if defined(__aarch64__)
    const auto* address = static_cast<const uint8_t*>(_virtual_addr) + offset;
    clean_cache_range(address, size);
#else
    const int rc = simaai_dmabuf_flush(_simaai_dmabuf_ptr);
    if (rc < 0) {
        throw std::runtime_error(fmt::format(
            "Failed to flush MLA buffer {}: {}", _name, std::strerror(-rc)
        ));
    }
#endif
}

void MLABuffer::invalidate_cache() const {
    invalidate_cache(0, _size_padded);
}

void MLABuffer::invalidate_cache(size_t offset, size_t size) const {
    if (!_simaai_dmabuf_ptr)
        throw std::logic_error("cannot invalidate an unallocated MLA buffer");
    if (offset > _size_padded || size > _size_padded - offset) {
        throw std::out_of_range(fmt::format(
            "Cache invalidate range [{}, {}) exceeds buffer {} size {}",
            offset, offset + size, _name, _size_padded
        ));
    }
    if (size == 0) return;
#if defined(__aarch64__)
    const auto* address = static_cast<const uint8_t*>(_virtual_addr) + offset;
    invalidate_cache_range(address, size);
#else
    const int rc = simaai_dmabuf_invalidate(_simaai_dmabuf_ptr);
    if (rc < 0) {
        throw std::runtime_error(fmt::format(
            "Failed to invalidate MLA buffer {}: {}", _name, std::strerror(-rc)
        ));
    }
#endif
}

void MLABuffer::clear(bool flush) {
    if (!_simaai_dmabuf_ptr || !_virtual_addr) {
        throw std::logic_error("cannot clear an unallocated MLA buffer");
    }
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
    if (!_simaai_dmabuf_ptr || !_virtual_addr) {
        throw std::logic_error("cannot load an unallocated MLA buffer");
    }
    const size_t row_size = checked_mul_size(
        _shape.back(), _elem_size, "MLA stream row size overflow"
    );
    const size_t padded_row_size = checked_round_up_size(
        row_size, MLA_ROW_SIZE, "MLA stream padded row size overflow"
    );
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
    if (!_simaai_dmabuf_ptr || !_virtual_addr) {
        throw std::logic_error("cannot upload to an unallocated MLA buffer");
    }
    if (!data) {
        throw std::invalid_argument("MLA buffer upload source is null");
    }
    if (data_size == 0) {
        if (data_begin != 0) {
            throw std::invalid_argument(
                "full MLA buffer upload must begin at byte zero"
            );
        }
        data_size = _size;
    }
    // Upload a full logical tensor, inserting physical MLA row padding when needed.
    if (data_begin != 0 || data_size != _size) {
        upload_raw(data, data_begin, data_size, flush);
        return;
    }

    const uint32_t last_dim = _shape.back() * _elem_size;
    const uint32_t last_dim_padded = round_up_to_row(last_dim);
    if (_align_last_dim && last_dim != last_dim_padded) {
        uint32_t num_last_dims = _size_padded / last_dim_padded;

        for (uint32_t i = 0; i < num_last_dims; ++i) {
            std::memcpy(
                reinterpret_cast<uint8_t*>(_virtual_addr) + i * last_dim_padded,
                reinterpret_cast<const uint8_t*>(data) + i * last_dim,
                last_dim
            );
        }
    } else {
        std::memcpy(reinterpret_cast<uint8_t*>(_virtual_addr) + data_begin, data, data_size);
    }
    if (flush)
        flush_cache();
}


void MLABuffer::upload_raw(
    const void* data, size_t destination_offset, size_t size, bool flush
) {
    // Copy bytes verbatim into a physical MLA buffer offset.
    if (destination_offset > _size_padded || size > _size_padded - destination_offset) {
        throw std::out_of_range(fmt::format(
            "Upload range [{}, {}) exceeds buffer {} size {}",
            destination_offset, destination_offset + size, _name, _size_padded
        ));
    }
    if (size == 0) return;
    std::memcpy(reinterpret_cast<uint8_t*>(_virtual_addr) + destination_offset, data, size);
    if (flush)
        flush_cache(destination_offset, size);
}


void MLABuffer::download(void* data) const {
    if (!_simaai_dmabuf_ptr || !_virtual_addr) {
        throw std::logic_error("cannot download an unallocated MLA buffer");
    }
    if (!data) {
        throw std::invalid_argument("MLA buffer download destination is null");
    }
    invalidate_cache();
    const size_t logical_row_bytes =
        checked_mul_size(_shape.back(), _elem_size, "MLA download row overflow");
    const size_t physical_row_bytes = checked_round_up_size(
        logical_row_bytes, MLA_ROW_SIZE, "MLA download padded row overflow"
    );
    if (_align_last_dim && logical_row_bytes != physical_row_bytes) {
        const size_t row_count = _size_padded / physical_row_bytes;

        std::memset(data, 0, _size);
        for (size_t i = 0; i < row_count; ++i) {
            std::memcpy(
                reinterpret_cast<uint8_t*>(data) + i * logical_row_bytes,
                reinterpret_cast<const uint8_t*>(_virtual_addr) +
                    i * physical_row_bytes,
                logical_row_bytes
            );
        }
    } else {
        std::memcpy(data, _virtual_addr, _size);
    }
}


uint64_t MLABuffer::get_buf_addr_offset(
    const std::optional<std::vector<uint32_t>>& begin
) const {
    if (!begin.has_value()) {
        return 0;
    }
    if (begin->size() != _shape.size()) {
        throw std::invalid_argument(fmt::format(
            "Buffer {} slice rank {} does not match allocation rank {}",
            _name, begin->size(), _shape.size()
        ));
    }

    /*
     * `_stride` is the physical element stride, including the padded MLA row.
     * The previous logical row-major recurrence silently addressed the wrong
     * row whenever align_last_dim was enabled.
     */
    uint64_t element_offset = 0;
    for (std::size_t i = 0; i < _shape.size(); ++i) {
        if ((*begin)[i] >= _shape[i]) {
            throw std::out_of_range(fmt::format(
                "Buffer {} slice index {} is outside dimension {}",
                _name, (*begin)[i], _shape[i]
            ));
        }
        const uint64_t stride = static_cast<uint64_t>(_stride[i]);
        if ((*begin)[i] != 0 &&
            stride > std::numeric_limits<uint64_t>::max() / (*begin)[i]) {
            throw std::overflow_error("MLA buffer slice offset overflow");
        }
        const uint64_t term = stride * (*begin)[i];
        if (term > std::numeric_limits<uint64_t>::max() - element_offset) {
            throw std::overflow_error("MLA buffer slice offset overflow");
        }
        element_offset += term;
    }
    if (element_offset >
        std::numeric_limits<uint64_t>::max() / _elem_size) {
        throw std::overflow_error("MLA buffer byte offset overflow");
    }
    const uint64_t byte_offset = element_offset * _elem_size;
    if (byte_offset >= _size_padded) {
        throw std::out_of_range("MLA buffer slice starts outside allocation");
    }
    return byte_offset;
}


uint64_t MLABuffer::get_buf_len(const std::optional<std::vector<uint32_t>>& shape) const {
    if (shape.has_value()) {
        if (shape->size() != _shape.size()) {
            throw std::invalid_argument(fmt::format(
                "Buffer {} slice rank {} does not match allocation rank {}",
                _name, shape->size(), _shape.size()
            ));
        }
        for (const uint32_t dim : *shape) {
            if (dim == 0) {
                throw std::invalid_argument("MLA buffer slice has a zero-sized dimension");
            }
        }
        size_t size = checked_mul_size(
            shape->back(), _elem_size, "MLA buffer slice row size overflow"
        );
        if (_align_last_dim) {
            size = checked_round_up_size(
                size, MLA_ROW_SIZE, "MLA buffer slice padded row size overflow"
            );
        }
        for (std::size_t i = 0; i + 1 < shape->size(); ++i) {
            size = checked_mul_size(
                size, (*shape)[i], "MLA buffer slice size overflow"
            );
        }
        if (size > _size_padded) {
            throw std::out_of_range("MLA buffer slice is larger than its allocation");
        }
        return size;
    } else {
        return _size_padded;
    }
}


uint64_t MLABuffer::get_buf_addr(
    const std::optional<std::vector<uint32_t>>& begin
) const {
    if (!_simaai_dmabuf_ptr || _physical_addr == 0) {
        throw std::logic_error("cannot address an unallocated MLA buffer");
    }
    const uint64_t offset = get_buf_addr_offset(begin);
    if (offset > std::numeric_limits<uint64_t>::max() - _physical_addr) {
        throw std::overflow_error("MLA buffer physical address overflow");
    }
    return _physical_addr + offset;
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


void MLABufferSlice::_bind(
    MLABuffer* buf_ptr,
    std::initializer_list<uint32_t> begins
) {
    _buf_ptr = buf_ptr;
    if (!_begins.has_value()) {
        _begins.emplace();
    }
    _begins->assign(begins);
}


uint64_t MLABufferSlice::get_buf_addr() const {
    return _buf_ptr ? _buf_ptr->get_buf_addr(_begins) : 0;
}


uint64_t MLABufferSlice::get_buf_addr(
    const std::optional<std::vector<uint32_t>>& begins
) const {
    return _buf_ptr ? _buf_ptr->get_buf_addr(begins) : 0;
}


uint64_t MLABufferSlice::get_byte_offset() const {
    if (!_buf_ptr) {
        throw std::invalid_argument("MLA buffer slice has no parent");
    }
    return _buf_ptr->get_buf_addr_offset(_begins);
}


void MLABufferSlice::to_file(const std::filesystem::path& file_name) const {
    std::filesystem::create_directories(file_name.parent_path());
    std::ofstream out_file(file_name, std::ios::out | std::ios::binary);

    assert(_buf_ptr != nullptr);
    const uint64_t byte_offset = _buf_ptr->get_buf_addr_offset(_begins);
    const uint64_t num_bytes = _buf_ptr->get_buf_len(_shapes);
    /*
     * Size and offset are individually checked by MLABuffer.  Check their
     * combined envelope as well: a small slice near the end of an allocation
     * must not make a file export read beyond the mapped dma-buf.
     */
    if (byte_offset > _buf_ptr->get_allocation_size() ||
        num_bytes > _buf_ptr->get_allocation_size() - byte_offset) {
        throw std::out_of_range("MLA buffer slice export exceeds allocation");
    }
    if (num_bytes >
        static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::overflow_error("MLA buffer slice is too large for stream I/O");
    }
    _buf_ptr->invalidate_cache(byte_offset, num_bytes);
    const char* ptr = reinterpret_cast<const char*>(_buf_ptr->get_virtual_addr())
        + byte_offset;

    out_file.write(ptr, static_cast<std::streamsize>(num_bytes));
    out_file.close();
}


}
}
