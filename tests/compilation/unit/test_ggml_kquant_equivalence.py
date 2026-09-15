"""
Bit-exact characterization tests for the GGUF K-quant unpackers.

The reference implementations below are frozen copies of the per-superblock
Python loops that ``sima_lmm.gguf.ggml_quant`` shipped with before the
unpackers were vectorized.  Every test feeds the same random input to the
reference and to the production function and requires identical shape, dtype,
and values.  The inputs are produced the way ``unpack_k_quant`` produces them:
as column slices of a ``(n_superblocks, superblock_bytes)`` ``uint8`` array,
which are non-contiguous views.
"""
import numpy as np
import pytest

from sima_lmm.gguf import ggml_quant


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


# ---------------------------------------------------------------------------
# Frozen reference implementations (do not "improve" these; they are the oracle)
# ---------------------------------------------------------------------------

def _ref_unpack_q6_k_quants(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(2):
            index1 = 128 * n
            index2 = 64 * n
            lo_e1 = lo[b, index2:index2 + 32]
            lo_e2 = lo[b, index2 + 32:index2 + 64]
            hi_e = hi[b, 32 * n:32 * n + 32]
            result[b, index1:index1 + 32] = ((lo_e1 & 0xf) | ((hi_e & 3) << 4)).astype(np.int8) - 32
            result[b, index1 + 32:index1 + 64] = ((lo_e2 & 0xf) | (((hi_e >> 2) & 3) << 4)).astype(np.int8) - 32
            result[b, index1 + 64:index1 + 96] = ((lo_e1 >> 4) | (((hi_e >> 4) & 3) << 4)).astype(np.int8) - 32
            result[b, index1 + 96:index1 + 128] = ((lo_e2 >> 4) | (((hi_e >> 6) & 3) << 4)).astype(np.int8) - 32
    return result


def _ref_unpack_q5_k_quants(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(4):
            hi_mask1 = 1 << (2 * n)
            hi_mask2 = 2 << (2 * n)
            index1 = 64 * n
            index2 = 32 * n
            result[b, index1:index1 + 32] = \
                (lo[b, index2:index2 + 32] & 0xf) + np.where(hi[b, :] & hi_mask1, 16, 0)
            result[b, index1 + 32:index1 + 64] = \
                (lo[b, index2:index2 + 32] >> 4) + np.where(hi[b, :] & hi_mask2, 16, 0)
    return result


def _ref_unpack_q5_k_scales(data: np.ndarray) -> np.ndarray:
    n_scales = data.shape[0]
    scales = np.zeros((n_scales, 8, 2), dtype=np.int8)
    for block_i in range(n_scales):
        for i in range(4):
            scales[block_i, i, 0] = data[block_i, i] & 63
            scales[block_i, i, 1] = data[block_i, i + 4] & 63
        for i in range(4):
            scales[block_i, i + 4, 0] = (data[block_i, i + 8] & 0xf) | ((data[block_i, i + 0] >> 6) << 4)
            scales[block_i, i + 4, 1] = (data[block_i, i + 8] >> 4) | ((data[block_i, i + 4] >> 6) << 4)
    return scales


def _ref_unpack_q4_k_quants(lo: np.ndarray) -> np.ndarray:
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(4):
            index1 = 64 * n
            index2 = 32 * n
            result[b, index1:index1 + 32] = lo[b, index2:index2 + 32] & 0xf
            result[b, index1 + 32:index1 + 64] = lo[b, index2:index2 + 32] >> 4
    return result


def _ref_unpack_q3_k_scales(data: np.ndarray) -> np.ndarray:
    kmask1 = np.uint32(0x03030303)
    kmask2 = np.uint32(0x0f0f0f0f)
    result = np.ndarray((data.shape[0], 16), dtype=np.int8)
    dataq = np.ndarray((4,), dtype=np.uint32)
    for b in range(data.shape[0]):
        dataq.view(dtype=np.uint8)[0:12] = data[b, :]
        tmp = dataq[2]
        dataq[2] = ((dataq[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
        dataq[3] = ((dataq[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
        dataq[0] = (dataq[0] & kmask2) | ((tmp & kmask1) << 4)
        dataq[1] = (dataq[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
        result[b, :] = dataq.view(dtype=np.int8) - 32
    return result


def _ref_unpack_q3_k_quants(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for k in range(2):
            for j in range(4):
                lo_shift = 2 * j
                hi_mask = 1 << (4 * k + j)
                index1 = 128 * k + 32 * j
                index2 = 32 * k
                result[b, index1:index1 + 32] = \
                    ((lo[b, index2:index2 + 32] >> lo_shift) & 3) - np.where(hi[b, :] & hi_mask, 0, 4)
    return result


def _ref_unpack_q2_k_scales(data: np.ndarray) -> np.ndarray:
    result = np.zeros((data.shape[0], 16, 2), dtype=np.int8)
    for b in range(data.shape[0]):
        result[b, :, 0] = data[b, :] & 0xf
        result[b, :, 1] = data[b, :] >> 4
    return result


def _ref_unpack_q2_k_quants(lo: np.ndarray) -> np.ndarray:
    result = np.zeros((lo.shape[0], 256), dtype=np.int8)
    for b in range(lo.shape[0]):
        for n in range(2):
            for j in range(4):
                index1 = 128 * n + 32 * j
                index2 = 32 * n
                result[b, index1:index1 + 32] = (lo[b, index2:index2 + 32] >> (2 * j)) & 3
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Odd superblock count so that no accidental power-of-two reshape hides a bug.
N_SUPERBLOCKS = 37


def _packed_fields(seed: int, *widths: int) -> list[np.ndarray]:
    """
    Produce random ``uint8`` field arrays the way ``_split_k_buffer`` does:
    one contiguous ``(N, sum(widths))`` buffer split column-wise into
    non-contiguous views.
    """
    rng = np.random.default_rng(seed)
    buffer = rng.integers(0, 256, size=(N_SUPERBLOCKS, sum(widths)), dtype=np.uint8)
    splits = np.cumsum(widths)[:-1].tolist()
    fields = np.array_split(buffer, splits, axis=1)
    assert all(not f.flags.c_contiguous for f in fields) or len(fields) == 1
    return fields


def _assert_identical(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q6_k_quants_matches_reference(seed: int):
    lo, hi = _packed_fields(seed, 128, 64)
    _assert_identical(ggml_quant.unpack_q6_k_quants(lo, hi), _ref_unpack_q6_k_quants(lo, hi))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q5_k_quants_matches_reference(seed: int):
    lo, hi = _packed_fields(seed, 128, 32)
    _assert_identical(ggml_quant.unpack_q5_k_quants(lo, hi), _ref_unpack_q5_k_quants(lo, hi))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q5_k_scales_matches_reference(seed: int):
    (data,) = _packed_fields(seed, 12)
    _assert_identical(ggml_quant.unpack_q5_k_scales(data), _ref_unpack_q5_k_scales(data))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q4_k_quants_matches_reference(seed: int):
    (lo,) = _packed_fields(seed, 128)
    _assert_identical(ggml_quant.unpack_q4_k_quants(lo), _ref_unpack_q4_k_quants(lo))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q3_k_scales_matches_reference(seed: int):
    (data,) = _packed_fields(seed, 12)
    _assert_identical(ggml_quant.unpack_q3_k_scales(data), _ref_unpack_q3_k_scales(data))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q3_k_quants_matches_reference(seed: int):
    lo, hi = _packed_fields(seed, 64, 32)
    _assert_identical(ggml_quant.unpack_q3_k_quants(lo, hi), _ref_unpack_q3_k_quants(lo, hi))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q2_k_scales_matches_reference(seed: int):
    (data,) = _packed_fields(seed, 16)
    _assert_identical(ggml_quant.unpack_q2_k_scales(data), _ref_unpack_q2_k_scales(data))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_unpack_q2_k_quants_matches_reference(seed: int):
    (lo,) = _packed_fields(seed, 64)
    _assert_identical(ggml_quant.unpack_q2_k_quants(lo), _ref_unpack_q2_k_quants(lo))


def test_unpackers_accept_a_single_superblock():
    """Shape edge case: N == 1 must not be squeezed away by any reshape."""
    rng = np.random.default_rng(99)
    lo = rng.integers(0, 256, size=(1, 128), dtype=np.uint8)
    hi = rng.integers(0, 256, size=(1, 64), dtype=np.uint8)
    _assert_identical(ggml_quant.unpack_q6_k_quants(lo, hi), _ref_unpack_q6_k_quants(lo, hi))
    _assert_identical(ggml_quant.unpack_q4_k_quants(lo), _ref_unpack_q4_k_quants(lo))
