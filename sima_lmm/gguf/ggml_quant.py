#########################################################
# Copyright (C) 2025 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
from enum import IntEnum
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ml_dtypes import bfloat16, int4, uint4
import sima_utils.logging.sima_logger as sima_logger

class QuantizationMode(IntEnum):
    """GGUF quantization Mode.
    """
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    BF16 = 30


class QuantizationSymmetry(IntEnum):
    UNQUANTIZED = -1
    SYMMETRIC = 0
    ASYMMETRIC = 1


@dataclass
class BlockQuantization:
    """ Block quantization configuration.
    """
    qbits: int
    qtype: QuantizationSymmetry
    n_elements: int


DEFAULT_GGML_BLK_SIZE: int = 32
GGML_SUPER_SIZE: int = 256


class GgufBaseQuant(BlockQuantization):
    """GGUF base quantization.
    Type 0 symmetric quantization: 1 block constant (scale)
    Type 1 asymmetric quantization: 2 block constants (scale and min)
    Block constants are saved as FLOAT16 (2 bytes) values.
    """
    mode: QuantizationMode
    n_blk_constants: int
    nbytes_per_blk: int

    def __init__(self, mode: QuantizationMode, qbits: int,
                 qtype: QuantizationSymmetry, n_elements: int):
        super().__init__(qbits=qbits, qtype=qtype, n_elements=n_elements)
        self.mode = mode
        self.n_blk_constants = self.qtype + 1
        self.nbytes_per_blk = (self.n_elements * self.qbits) // 8

    @property
    def blk_buffer_size(self):
        return self.n_blk_constants * 2 + self.nbytes_per_blk


class GgufKQuant(GgufBaseQuant):
    """GGUF super block K quantization.
    Block constants are quantized as int values by scaling only.
    Super block constants of scale and min are saved as FLOAT16 values.
    """
    blk_qbits: int
    n_blocks: int
    n_super_constants: int

    def __init__(self, mode: QuantizationMode, qbits: int,
                 qtype: QuantizationSymmetry, n_elements: int,
                 blk_qbits: int):
        super().__init__(mode, qbits, qtype, n_elements)
        self.blk_qbits = blk_qbits
        self.n_blocks = GGML_SUPER_SIZE // self.n_elements
        self.n_super_constants = self.qtype + 1

    @property
    def super_buffer_size(self):
        n_qw = (self.n_elements * self.qbits) // 8
        n_qb = (self.n_blocks * self.n_blk_constants * self.blk_qbits) // 8
        n_super = self.n_super_constants * 2
        return n_qw + n_qb + n_super


SUPPORTED_GGML_TYPES: dict[str, GgufBaseQuant | GgufKQuant] = {
    "F32": GgufBaseQuant(
        QuantizationMode.F32, 32, QuantizationSymmetry.UNQUANTIZED, 1),
    "F16": GgufBaseQuant(
        QuantizationMode.F16, 16, QuantizationSymmetry.UNQUANTIZED, 1),
    "BF16": GgufBaseQuant(
        QuantizationMode.BF16, 16, QuantizationSymmetry.UNQUANTIZED, 1),
    "Q8_0": GgufBaseQuant(
        QuantizationMode.Q8_0, 8, QuantizationSymmetry.SYMMETRIC, DEFAULT_GGML_BLK_SIZE),
    "Q4_0": GgufBaseQuant(
        QuantizationMode.Q4_0, 4, QuantizationSymmetry.SYMMETRIC, DEFAULT_GGML_BLK_SIZE),
    "Q4_1": GgufBaseQuant(
        QuantizationMode.Q4_1, 4, QuantizationSymmetry.ASYMMETRIC, DEFAULT_GGML_BLK_SIZE),
    "Q5_0": GgufBaseQuant(
        QuantizationMode.Q5_0, 5, QuantizationSymmetry.SYMMETRIC, DEFAULT_GGML_BLK_SIZE),
    "Q5_1": GgufBaseQuant(
        QuantizationMode.Q5_1, 5, QuantizationSymmetry.ASYMMETRIC, DEFAULT_GGML_BLK_SIZE),
    "Q2_K": GgufKQuant(
        QuantizationMode.Q2_K, 2, QuantizationSymmetry.ASYMMETRIC, DEFAULT_GGML_BLK_SIZE // 2, 4),
    "Q3_K": GgufKQuant(
        QuantizationMode.Q3_K, 3, QuantizationSymmetry.SYMMETRIC, DEFAULT_GGML_BLK_SIZE // 2, 6),
    "Q4_K": GgufKQuant(
        QuantizationMode.Q4_K, 4, QuantizationSymmetry.ASYMMETRIC, DEFAULT_GGML_BLK_SIZE, 6),
    "Q5_K": GgufKQuant(
        QuantizationMode.Q5_K, 5, QuantizationSymmetry.ASYMMETRIC, DEFAULT_GGML_BLK_SIZE, 6),
    "Q6_K": GgufKQuant(
        QuantizationMode.Q6_K, 6, QuantizationSymmetry.SYMMETRIC, DEFAULT_GGML_BLK_SIZE // 2, 8),
}

SUPPORTED_GGML_NAMES = {v.mode: name for name, v in SUPPORTED_GGML_TYPES.items()}


def unpack_q8_0(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q8_0 data from a buffer.

    Q8_0 is block quantization with fixed block size of 32 elements.
    Suffix 0 means symmetric quantization.
    Each block consists of a scale in float16 followed by int8.

    Args:
        data: The buffer of raw data for a tensor.

    Returns:
        Unpacked float scales and integer tensor values as new arrays.
        The shape of the scales is (nblocks, 1), and the tensor (nblocks, 32).
    """
    block_size = 2 + 32
    num_blocks = len(data) // block_size
    n_fp16 = block_size // 2
    n_int8 = block_size
    scales = np.frombuffer(
        data, dtype=np.float16).reshape(
            num_blocks, n_fp16)[:, :1].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.int8).reshape(num_blocks, n_int8)[:, 2:]
    return scales, qs.copy()


def unpack_dequantize_q8_0(data: bytes) -> np.ndarray:
    """Q8_0 unpack and dequantization.

    Args:
        data: The buffer of raw data for a tensor.

    Returns:
        Dequantized tensor values with shape (nblocks, 32) in float32.
    """
    scales, qs = unpack_q8_0(data)
    return scales * qs


def unpack_q4(data: bytes, symmetry: QuantizationSymmetry) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q4 data from a buffer.

    Q4 is block quantization with fixed block size of 32 elements.
    Each element in a block is quantized as int4 and packed consectively as int8 bytes.
    Each block consists of block constants in float16 followed by packed [int4|int4].
    For symmetric quantization, block constants are scales only.
    For asymmetric quantization, block constants are scales and min values.

    Args:
        data: The buffer of raw data for a tensor.
        symmetry: The type of symmetric or asymmetric quantization.

    Returns:
        Unpacked block constants in float32 and int4 tensor values as new arrays.
        The shape of the tensor is (nblocks, 32). The shape of block constants is (nblocks, 1)
        for symmetric quantization or (nblocks, 2) for asymmetric quantization.
    """
    num_constants = 1 if symmetry == QuantizationSymmetry.SYMMETRIC else 2
    block_size = num_constants * 2 + 16
    num_blocks = len(data) // block_size
    n_fp16 = block_size // 2
    n_int8 = block_size
    blk_constants = np.frombuffer(data, dtype=np.float16) \
                    .reshape(num_blocks, n_fp16)[:, :num_constants].astype(np.float32)
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, n_int8)[:, num_constants*2:]
    if symmetry == QuantizationSymmetry.SYMMETRIC:
        qs = np.concatenate(
            [(qs & 0xf).astype(np.int8) - 8, (qs >> 4).astype(np.int8) - 8],
            axis=1
        ).astype(int4)
    else:
        qs = np.concatenate(
            [(qs & 0xf).astype(np.uint8), (qs >> 4).astype(np.uint8)],
            axis=1
        ).astype(uint4)
    return blk_constants, qs


def unpack_dequantize_q4(data: bytes, symmetry: QuantizationSymmetry) -> np.ndarray:
    """Q4 unpack and dequantization.

    Args:
        data: The buffer of raw data for a tensor.
        symmetry: The type of symmetric or asymmetric quantization.

    Returns:
        Dequantized tensor values with shape (nblocks, 32) in float32.
    """
    blk_constants, qs = unpack_q4(data, symmetry)
    if blk_constants.shape[-1] == 1:
        return blk_constants * qs
    else:
        return blk_constants[:, 0:1] * qs + blk_constants[:, 1:2]


def unpack_q5(data: bytes, symmetry: QuantizationSymmetry) -> tuple[np.ndarray, np.ndarray]:
    """Unpack Q5 data from a buffer.

    Q5 is block quantization with fixed block size of 32 elements.
    Each element in a block is quantized as int5.
    The lower 4 bits are packed consectively as int8 bytes, so are the high bits.
    Each block consists of block constants in float16.
    For symmetric quantization, block constants are scales only.
    For asymmetric quantization, block constants are scales and min values.

    Args:
        data: The buffer of raw data for a tensor.
        symmetry: The type of symmetric or asymmetric quantization.

    Returns:
        Unpacked block constants in float32 and int8 tensor values as new arrays.
        The shape of the tensor is (nblocks, 32). The shape of block constants is (nblocks, 1)
        for symmetric quantization or (nblocks, 2) for asymmetric quantization.
    """
    num_constants = 1 if symmetry == QuantizationSymmetry.SYMMETRIC else 2
    block_size = num_constants * 2 + 4 + 16
    num_blocks = len(data) // block_size
    n_fp16 = block_size // 2
    n_int8 = block_size
    blk_constants = np.frombuffer(data, dtype=np.float16) \
                    .reshape(num_blocks, n_fp16)[:, :num_constants].astype(np.float32)
    qh = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, n_int8)[:, num_constants*2:num_constants*2+4]
    qs = np.frombuffer(data, dtype=np.uint8).reshape(num_blocks, n_int8)[:, num_constants*2+4:]

    qh = qh.view(np.uint32)
    qh = qh.reshape(
        (num_blocks, 1)) >> np.arange(DEFAULT_GGML_BLK_SIZE).reshape((1, DEFAULT_GGML_BLK_SIZE))
    ql = qs.reshape(
        (num_blocks, -1, 1, 16)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
    qh = (qh & np.uint32(0x01)).astype(np.uint8)
    ql = (ql & np.uint8(0x0F)).reshape((num_blocks, -1))

    qs = (ql | (qh << np.uint8(4))).astype(np.int8)
    if symmetry == QuantizationSymmetry.SYMMETRIC:
        qs = qs - np.int8(16)

    return blk_constants, qs


def unpack_q6_k_quants(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Convert quantized values from Q6_K to int8 values.
    Each block of 256 Q6_K quantized values is stored as 2 arrays.
    The 4 lower bits of each value are packed into one array of uint8 (two values per integer).
    The 2 upper bits of each value are packed into one array of uint8 (four values per integer).
    """
    assert lo.shape[0] == hi.shape[0]
    assert lo.shape[1] == 128
    assert hi.shape[1] == 64

    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(2):
            index1 = 128 * n
            index2 = 64 * n
            lo_e1 = lo[b, index2:index2+32]
            lo_e2 = lo[b, index2 + 32:index2 + 64]
            hi_e = hi[b, 32 * n:32 * n + 32]
            result[b, index1:index1+32]     = ((lo_e1 & 0xf) | ((hi_e & 3)        << 4)).astype(np.int8) - 32
            result[b, index1+32:index1+64]  = ((lo_e2 & 0xf) | (((hi_e >> 2) & 3) << 4)).astype(np.int8) - 32
            result[b, index1+64:index1+96]  = ((lo_e1 >> 4)  | (((hi_e >> 4) & 3) << 4)).astype(np.int8) - 32
            result[b, index1+96:index1+128] = ((lo_e2 >> 4)  | (((hi_e >> 6) & 3) << 4)).astype(np.int8) - 32

    return result


def unpack_q5_k_quants(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Convert quantized values from Q5_K to int8 values.
    Each block of 256 Q5_K quantized values is stored as 2 arrays.
    The 4 lower bits of each value are packed into one array of uint8 (two values per integer).
    The 1 upper bit of each value is packed into one array of uint8 (eight values per integer).
    """
    assert lo.shape[0] == hi.shape[0]
    assert lo.shape[1] == 128
    assert hi.shape[1] == 32

    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(4):
            hi_mask1 = 1 << (2*n)
            hi_mask2 = 2 << (2*n)
            index1 = 64 * n
            index2 = 32 * n
            result[b, index1:index1+32] = \
                (lo[b, index2:index2+32] & 0xf) + np.where(hi[b, :] & hi_mask1, 16, 0)
            result[b, index1 + 32:index1 + 64] = \
                (lo[b, index2:index2+32] >> 4) + np.where(hi[b, :] & hi_mask2, 16, 0)

    return result


def unpack_q5_k_scales(data: np.ndarray) -> np.ndarray:
    """
    Convert quantized scales and mins to separate scale and min arrays.
    The input is an (N, 12) array of int8.  Each 12 integers are unpacked
    to 8 scale values and 8 min values, quantized with 6 bits.
    The return value is an int8 array of shape (N, 2) where index 0 has
    the scales and index 1 has the minimum values.
    """
    assert data.dtype == np.uint8
    assert data.shape[1] == 12
    n_scales = data.shape[0]

    scales = np.zeros((n_scales, 8, 2), dtype=np.int8)
    for block_i in range(n_scales):
        # The first 4 and last 4 elements of a block are stored differently
        for i in range(4):
            scales[block_i, i, 0] = data[block_i, i] & 63
            scales[block_i, i, 1] = data[block_i, i + 4] & 63
        for i in range(4):
            scales[block_i, i + 4, 0] = (data[block_i, i + 8] & 0xf) | ((data[block_i, i + 0] >> 6) << 4)
            scales[block_i, i + 4, 1] = (data[block_i, i + 8] >> 4) | ((data[block_i, i + 4] >> 6) << 4)

    return scales


def unpack_q4_k_quants(lo: np.ndarray) -> np.ndarray:
    """
    Convert quantized values from Q4_K to int8 values.
    Each block of 256 Q4_K quantized values is stored as an array of uint8
    with 2 values per integer.
    """
    assert lo.shape[1] == 128

    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(4):
            hi_mask1 = 1 << (2*n)
            hi_mask2 = 2 << (2*n)
            index1 = 64*n
            index2 = 32*n
            result[b, index1:index1 + 32] = lo[b, index2:index2 + 32] & 0xf
            result[b, index1 + 32:index1 + 64] = lo[b, index2:index2 + 32] >> 4

    return result


def unpack_q3_k_scales(data: np.ndarray) -> np.ndarray:
    """
    Convert quantized scales to a scale array.
    The input is an (N, 12) array of int8.  Each 12 integers are unpacked
    to 16 scale values, quantized with 6 bits.
    The return value is an int8 array of shape (N, 1).
    """
    kmask1 = np.uint32(0x03030303)  # Mask to extract 2 bits
    kmask2 = np.uint32(0x0f0f0f0f)  # Mask to extract 4 bits

    result = np.ndarray((data.shape[0], 16), dtype=np.int8)

    dataq = np.ndarray((4,), dtype=np.uint32)
    for b in range(data.shape[0]):
        # Contents of dataq are dead at this point
        dataq.view(dtype=np.uint8)[0:12] = data[b, :]
        tmp = dataq[2]
        dataq[2] = ((dataq[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        dataq[3] = ((dataq[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        dataq[0] = (dataq[0] & kmask2) | ((tmp & kmask1) << 4)
        dataq[1] = (dataq[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

        # Convert to signed int.  Subtract an offset.
        result[b, :] = dataq.view(dtype=np.int8) - 32

    return result


def unpack_q3_k_quants(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """
    Convert quantized values from Q4_K to int8 values.
    Each block of 256 Q3_K quantized values is stored as two arrays.
    The low bits are stored as 64 int8 values (2 bits per integer).
    The high bits are stored as 32 int8 values (1 bit per integer).
    """
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for k in range(2):
            for j in range(4):
                lo_shift = 2*j
                hi_mask = 1 << (4*k + j)
                index1 = 128 * k + 32 * j
                index2 = 32 * k
                result[b, index1:index1 + 32] = \
                    ((lo[b, index2:index2 + 32] >> lo_shift) & 3) - np.where(hi[b, :] & hi_mask, 0, 4)

    return result


def unpack_q2_k_scales(data: np.ndarray) -> np.ndarray:
    """
    Convert quantized scales to a scale array.
    The input is an (N, 16) array of int8.  Each integer is
    unpacked to a 4-bit scale and 4-bit min value.
    The return value is an int8 array of shape (N, 2).
    """
    result = np.zeros((data.shape[0], 16, 2), dtype=np.int8)
    for b in range(data.shape[0]):
        result[b, :, 0] = data[b, :] & 0xf
        result[b, :, 1] = data[b, :] >> 4

    return result


def unpack_q2_k_quants(lo: np.ndarray) -> np.ndarray:
    """
    Convert quantized values from Q2_K to int8 values.
    Four 2-bit values are packed into an 8-bit integer.
    """
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(2):
            for j in range(4):
                index1 = 128 * n + 32 * j
                index2 = 32 * n
                result[b, index1:index1 + 32] = (lo[b,index2:index2 + 32] >> (2*j)) & 3

    return result


@dataclass
class SuperBlockPartition:
    """Storage partiton for super block quantization of a tensor.
    """
    super_constant_bytes: int
    block_overhead: int
    tensor_bytes_low: int
    tensor_bytes_high: int
    super_block_size: int


def _calculate_k_sections(
    num_constants: int, qs_bits: int, qh_bits: int, b_bits: int, n_blocks: int
) -> SuperBlockPartition:
    """
    Calculate for each super block, the storage space in number of bytes
    for super constants, block overhead, and weight tensor values.

    The number of bits used to quantize the weight tensor may be split into
    lower bits and higher bits, so that they can be packed to align on an integer
    number of bytes. For example, in the case of Q3_k, int3 is split into
    lower 2-bit and higher 1-bit, which are packed separately to fit into byte arrays.

    Args:
        num_constants: Number of float16 constants for super blocks.
        qs_bits: The number of lower bits, if there is split, for tensor quantization.
        qh_bits: The number of higher bits, if there is split, for tensor quantization.
        b_bits: The number of bits used to quantize block constants.
        n_blocks: The number of blocks in a super block.

    Returns:
        A storage partition of super block quantized tensor.
    """
    super_constant_bytes = num_constants * 2
    tensor_bytes_low = GGML_SUPER_SIZE * qs_bits // 8
    tensor_bytes_high = GGML_SUPER_SIZE * qh_bits // 8
    block_overhead = n_blocks * num_constants * b_bits // 8
    super_block_size = super_constant_bytes + block_overhead + tensor_bytes_low + tensor_bytes_high
    return SuperBlockPartition(
        super_constant_bytes,
        block_overhead,
        tensor_bytes_low,
        tensor_bytes_high,
        super_block_size
    )


def _split_k_buffer(ks: SuperBlockPartition, data: bytes, splits: list[int]) -> Sequence[np.ndarray]:
    """
    Extract data from array-of-structs binary format into separate arrays.

    Args:
        ks: Sizes of some fields in the binary struct format
        data: Byte buffer holding an array of structs
        splits: Byte indices where each struct should be split to separate its fields

    Returns:
        The split byte arrays
    """
    n_super = len(data) // ks.super_block_size
    return np.array_split(
        np.frombuffer(data, dtype=np.uint8).reshape(n_super, ks.super_block_size),
        splits, axis=1
    )


def unpack_k_quant(data: bytes, ggml_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack a block of data based on K quantization.

    This function extracts numbers from the binary format into numpy arrays.  The extracted
    numbers correspond one-to-one with numbers in the binary format; this function does not
    do any numeric conversions such as dequantization.

    Args:
        data: The buffer of raw data for a tensor.
        ggml_name: The string name of GGML quantization mode.

    Returns:
        Tuple of the following three arrays.
        Superblock constants of shape (N, 1) or (N, 2) where N is the number of superblocks.
        Block constants of shape (N, B, 1) or (N, B, 2) where B is the blocks per superblock.
        Quantized values of shape (N, B, C) where C is the number of values per block.
    """
    assert "_K" in ggml_name
    cfg = SUPPORTED_GGML_TYPES[ggml_name]
    match ggml_name:
        case "Q2_K":
            """
            typedef struct { // QK_K = 256
                uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
                uint8_t qs[QK_K/4];      // quants
                ggml_fp16_t d;           // super-block scale for quantized scales
                ggml_fp16_t dmin;        // super-block scale for quantized mins
            } block_q2_K;
            """
            ks = _calculate_k_sections(
                cfg.n_super_constants, cfg.qbits, 0, cfg.blk_qbits, cfg.n_blocks
            )
            splits = [
                ks.block_overhead,
                ks.block_overhead + ks.tensor_bytes_low
            ]
            qblock, qs, super_constants = _split_k_buffer(ks, data, splits)
            elements = unpack_q2_k_quants(qs)
            blocks = unpack_q2_k_scales(qblock)
        case "Q3_K":
            """
            typedef struct { // QK_K = 256
                uint8_t hmask[QK_K/8];     // quants - high bit
                uint8_t qs[QK_K/4];        // quants - low 2 bits
                uint8_t scales[12];        // scales, quantized with 6 bits
                ggml_fp16_t d;             // super-block scale
            } block_q3_K;
            """
            ks = _calculate_k_sections(cfg.n_super_constants, 2, 1, cfg.blk_qbits, cfg.n_blocks)
            splits = [
                ks.tensor_bytes_high,
                ks.tensor_bytes_high + ks.tensor_bytes_low,
                ks.tensor_bytes_high + ks.tensor_bytes_low + ks.block_overhead
            ]
            qh, qs, qblock, super_constants = _split_k_buffer(ks, data, splits)
            elements = unpack_q3_k_quants(qs, qh)
            blocks = unpack_q3_k_scales(qblock)
        case "Q4_K":
            """
            typedef struct { // QK_K = 256
                ggml_fp16_t d;             // super-block scale for quantized scales
                ggml_fp16_t dmin;          // super-block scale for quantized mins
                uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
                uint8_t qs[QK_K/2];        // 4--bit quants
            } block_q4_K;
            """
            ks = _calculate_k_sections(
                cfg.n_super_constants, cfg.qbits, 0, cfg.blk_qbits, cfg.n_blocks
            )
            splits = [
                ks.super_constant_bytes,
                ks.super_constant_bytes + ks.block_overhead
            ]
            super_constants, qblock, qs = _split_k_buffer(ks, data, splits)
            elements = unpack_q4_k_quants(qs)
            blocks = unpack_q5_k_scales(qblock)  # Reuse the Q5_K algorithm
        case "Q5_K":
            """
            typedef struct { // QK_K = 256
                ggml_fp16_t d;               // super-block scale for quantized scales
                ggml_fp16_t dmin;            // super-block scale for quantized mins
                uint8_t scales[K_SCALE_SIZE];   // scales and mins, quantized with 6 bits
                uint8_t qh[QK_K/8];          // quants, high bit
                uint8_t qs[QK_K/2];          // quants, low 4 bits
            } block_q5_K;
            """
            ks = _calculate_k_sections(cfg.n_super_constants, 4, 1, cfg.blk_qbits, cfg.n_blocks)
            splits = [
                ks.super_constant_bytes,
                ks.super_constant_bytes + ks.block_overhead,
                ks.super_constant_bytes + ks.block_overhead + ks.tensor_bytes_high
            ]
            super_constants, qblock, qh, qs = _split_k_buffer(ks, data, splits)
            elements = unpack_q5_k_quants(qs, qh)
            blocks = unpack_q5_k_scales(qblock)
        case "Q6_K":
            """
            typedef struct { // QK_K = 256
                uint8_t ql[QK_K/2];      // quants, lower 4 bits
                uint8_t qh[QK_K/4];      // quants, upper 2 bits
                int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
                ggml_fp16_t d;           // super-block scale
            } block_q6_K;
            """
            ks = _calculate_k_sections(cfg.n_super_constants, 4, 2, cfg.blk_qbits, cfg.n_blocks)
            splits = [
                ks.tensor_bytes_low,
                ks.tensor_bytes_low + ks.tensor_bytes_high,
                ks.tensor_bytes_low + ks.tensor_bytes_high + ks.block_overhead
            ]
            qs, qh, qblock, super_constants = _split_k_buffer(ks, data, splits)
            elements = unpack_q6_k_quants(qs, qh)
            blocks = qblock.astype(np.int8)
        case _:
            raise ValueError(f"K-quant {ggml_name} not implemented yet.")

    # Superblock constants consist of 1 or 2 fp16 values per block
    n_super = len(data) // ks.super_block_size
    super_constants = super_constants.view(np.float16).astype(np.float32)
    super_constants = np.reshape(super_constants, (n_super, -1))
    assert super_constants.shape[-1] in (1, 2)

    blocks = np.reshape(blocks, (n_super, cfg.n_blocks, -1))
    assert blocks.shape[-1] == super_constants.shape[-1]

    elements = np.reshape(elements, (n_super, cfg.n_blocks, GGML_SUPER_SIZE // cfg.n_blocks))

    return super_constants, blocks, elements


def convert_to_simple_block_quant(super_constants: np.ndarray, blocks: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks, elements = remove_superblocks(super_constants, blocks, elements)
    blocks = blocks.astype(np.float32, copy=False)
    if blocks.shape[-1] == 2:
        blocks, elements = requantize_symmetric(blocks, elements, 8, np.int8)

    return blocks, elements


def unpack_base_quant(data: bytes, ggml_name: str) -> tuple[np.ndarray | None, np.ndarray]:
    """Unpack for float types and base quantization types.

    Args:
        data: The buffer of raw data for a tensor.
        ggml_name: The string name of GGML quantization mode.

    Returns:
        Unpacked block constants and tensor values.
    """
    cfg = SUPPORTED_GGML_TYPES[ggml_name]
    match ggml_name:
        case "F32":
            s = None
            w = np.copy(np.frombuffer(data, dtype=np.float32))
        case "F16":
            s = None
            w = np.copy(np.frombuffer(data, dtype=np.float16))
        case "BF16":
            s = None
            w = np.copy(np.frombuffer(data, dtype=bfloat16))
        case "Q8_0":
            s, w = unpack_q8_0(data)
        case "Q4_0" | "Q4_1":
            s, w = unpack_q4(data, cfg.qtype)
        case "Q5_0" | "Q5_1":
            s, w = unpack_q5(data, cfg.qtype)
        case _:
            raise ValueError(f"Quantization {ggml_name} not implemented yet.")

    return s, w


def remove_superblocks(
        superblock_constants: np.ndarray,
        block_constants: np.ndarray,
        quantized: np.ndarray
) -> np.ndarray:
    """
    Remove superblocks, producing block quantized data.
    Superblock constants and block constants are combined to produce
    new block constants.

    Args:
        superblock_constants: A float16 array of shape (N, 1) or (N, 2).
            The first column is the scale.  The second column, if present,
            is the minimum value.
        block_constants: A signed integer array of shape (N, M, 1) or (N, M, 2).
            The first column is the scale.  The second column, if present,
            is the minimum value.
        quantized: An integer array of shape (N, M, B) holding quantized values.
    Returns:
        New block constants, as a float16 array of shape (N*M, 1) or (N*M, 2),
        and new quantized values as an integer array of shape (N*M, B).
        The arrays are in the same format as for block quantization such
        as Q4_0 or Q4_1.
    """
    assert len(superblock_constants.shape) == 2
    assert len(block_constants.shape) == 3
    assert len(quantized.shape) == 3
    assert superblock_constants.shape[0] == block_constants.shape[0] == quantized.shape[0]
    assert superblock_constants.shape[1] == block_constants.shape[2]
    assert block_constants.shape[1] == quantized.shape[1]
    assert np.issubdtype(block_constants.dtype, np.signedinteger)

    # Multiply block scale by superblock scale; multiply block min by superblock min
    block_constants = block_constants * np.expand_dims(superblock_constants, 1)
    block_constants = block_constants.astype(np.float16, copy=False)

    # Multiply the "minimum value" field by -1 to put it into the block-quantization format
    if block_constants.shape[2] == 2:
        block_constants[..., -1] *= -1

    # Remove superblocks from the array shape
    block_count = block_constants.shape[0] * block_constants.shape[1]
    return (np.reshape(block_constants, (block_count, block_constants.shape[2])),
            np.reshape(quantized, (block_count, quantized.shape[2])))


def requantize_symmetric(
        block_constants: np.ndarray,
        quantized: np.ndarray,
        output_bits: int,
        new_dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    """
    Requantize from asymmetric to symmetric quantization.
    This is a lossy conversion because the minimum values become quantized.

    Args:
        block_constants: Asymmetric block constants.  A float16 array of shape
            (N, 2).  The first column is the scale and the second column is the
            minimum value.
        quantized: Quantized values.  An integer array of shape (N, B) for some
            block size B.
        output_bits: Number of bits in the output quantized values.  Output
            values will be clipped to this range.
        new_dtype: Dtype of the output.  Must be a signed integer type.
    Returns:
        Tuple of symmetric block constants and quantized values.  These arrays
        have the same format as symmetric block quantization such as Q8_0.
    """
    assert block_constants.ndim == 2
    assert block_constants.dtype == np.float32
    assert quantized.ndim == 2
    assert block_constants.shape[0] == quantized.shape[0]
    assert block_constants.shape[1] == 2
    assert np.issubdtype(new_dtype, np.signedinteger)

    # Convert from asymmetric to symmetric quantization and adjust the scale factor
    # to use more bits.
    # The original quantization approximates real value R using unsigned quantized value Q
    # with scale S and minimum value M:
    #   R = S * Q + M
    # The converted quantization uses signed value Q' and scale S':
    #   R = S' * Q'
    # We choose a scale_factor and then do
    #   S' = S / scale_factor
    #   Q' = Q * scale_factor + round(M/S * scale_factor)
    # All int quantizations use 8 or fewer bits, so int16 precision is enough to avoid overflow.
    output_min = -(1 << (output_bits-1))
    output_max = (1 << (output_bits-1)) - 1

    scales = block_constants[:, 0]
    minima = block_constants[:, 1]

    # For choosing the new scale, the initial scale must not be 0.
    # If scale is 0, we set the initial scale to the minimum value or 1e-4.
    # We can freely choose a new scale because, when scale is 0, the
    # quantized values must also be 0.
    SCALE_EPSILON = np.float16(1e-4)
    alternative_scales = np.where(np.abs(minima) > SCALE_EPSILON, minima, SCALE_EPSILON)
    scales = np.where(scales == 0, alternative_scales, scales)

    # Convert to symmetric quantization and find the value range of each block.
    q_s_minima = minima / scales
    q_s_min = np.min(quantized, axis=1) + q_s_minima
    q_s_max = np.max(quantized, axis=1) + q_s_minima

    # Maximize scale_factor without exceeding the output range
    SCALE_FACTOR_MAX = 16
    scale_factor = SCALE_FACTOR_MAX
    scale_factor = np.minimum(
        scale_factor,
        np.where(q_s_min < 0, output_min / q_s_min, SCALE_FACTOR_MAX)
    )
    np.minimum(
        scale_factor,
        np.where(q_s_max > 0, output_max / q_s_max, SCALE_FACTOR_MAX),
        out=scale_factor
    )
    # When scale factor is greater than or approximately equal to 1,
    # round to an integer so that it introduces no additional rounding error.
    scale_factor = np.where(
        scale_factor > 0.98,
        np.maximum(np.floor(scale_factor), 1),
        scale_factor
    )

    # Requantize weights to the new range
    minimum_value = q_s_minima * scale_factor
    rounded_minimum_value = np.rint(minimum_value).astype(np.int16)
    round_error = np.max(np.abs(rounded_minimum_value - minimum_value))

    new_quantized = np.rint(quantized * np.expand_dims(scale_factor, 1)).astype(np.int16)
    new_quantized += np.expand_dims(rounded_minimum_value, 1)
    clip_quantized = np.clip(new_quantized, output_min, output_max)
    clip_error = np.max(np.abs(clip_quantized - new_quantized))

    new_quantized = clip_quantized.astype(new_dtype)

    # Numerical error should stay within these bounds.
    # Error could be larger if the input minimum value has very large magnitude.
    if clip_error > 0 or round_error > 0.5:
        sima_logger.sima_log_warning(f"Distortion from requantization: rounding {round_error}, clipping {clip_error}")

    s = block_constants[:, 0] / scale_factor
    s = np.expand_dims(s, 1)
    assert s.dtype == np.float32
    
    return (s, new_quantized)
