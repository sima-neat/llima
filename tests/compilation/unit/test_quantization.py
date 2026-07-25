import numpy as np
import pytest

from sima_lmm.gguf.ggml_quant import (
    requantize_symmetric,
    unpack_q2_k_quants,
    unpack_q4_k_quants,
)
from sima_lmm.hf.hf_transformer import (
    CompressedTensorsConfig,
    infer_num_bits_from_packed_shape,
)


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("packed_shape", "original_shape", "expected_bits"),
    [
        ((2, 2), (2, 16), 4),
        ((2, 2), (2, 8), 8),
    ],
)
def test_infer_num_bits_from_packed_shape(
    packed_shape: tuple[int, ...],
    original_shape: tuple[int, ...],
    expected_bits: int,
):
    assert infer_num_bits_from_packed_shape(packed_shape, original_shape) == expected_bits


@pytest.mark.parametrize(
    ("packed_shape", "original_shape", "message"),
    [
        ((0, 2), (2, 16), "Invalid tensor sizes"),
        ((2, 3), (2, 16), "not divisible"),
        ((2, 2), (2, 4), "Unsupported inferred"),
    ],
)
def test_infer_num_bits_rejects_invalid_shapes(
    packed_shape: tuple[int, ...],
    original_shape: tuple[int, ...],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        infer_num_bits_from_packed_shape(packed_shape, original_shape)


def test_compressed_tensors_config_ignores_other_quantization_methods():
    assert (
        CompressedTensorsConfig.from_hf_config(
            {"quantization_config": {"quant_method": "gptq"}}
        )
        is None
    )


def test_compressed_tensors_config_parses_weight_group():
    config = CompressedTensorsConfig.from_hf_config(
        {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "weights": {"group_size": 128, "symmetric": True},
                    }
                },
                "ignore": ["lm_head"],
            }
        }
    )

    assert config == CompressedTensorsConfig(
        symmetric=True,
        group_sizes=[128],
        ignore=["lm_head"],
        targets=["Linear"],
    )


@pytest.mark.parametrize(
    ("quantization_config", "message"),
    [
        (
            {
                "quant_method": "compressed-tensors",
                "config_groups": {},
            },
            "missing config_groups",
        ),
        (
            {
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "group_0": {
                        "weights": {"symmetric": False},
                    }
                },
            },
            "Only symmetric quantization",
        ),
    ],
)
def test_compressed_tensors_config_rejects_unsupported_configuration(
    quantization_config: dict, message: str
):
    with pytest.raises(ValueError, match=message):
        CompressedTensorsConfig.from_hf_config(
            {"quantization_config": quantization_config}
        )


def test_unpack_q2_k_quants_expands_all_bit_planes():
    packed = np.tile(np.array([0b11100100], dtype=np.uint8), (1, 64))

    unpacked = unpack_q2_k_quants(packed)

    assert unpacked.shape == (1, 256)
    np.testing.assert_array_equal(unpacked[0, 0:32], np.full(32, 0, dtype=np.int8))
    np.testing.assert_array_equal(unpacked[0, 32:64], np.full(32, 1, dtype=np.int8))
    np.testing.assert_array_equal(unpacked[0, 64:96], np.full(32, 2, dtype=np.int8))
    np.testing.assert_array_equal(unpacked[0, 96:128], np.full(32, 3, dtype=np.int8))
    np.testing.assert_array_equal(unpacked[0, 128:256], unpacked[0, 0:128])


def test_unpack_q4_k_quants_expands_low_and_high_nibbles():
    packed = np.tile(np.array([0xA3], dtype=np.uint8), (1, 128))

    unpacked = unpack_q4_k_quants(packed)

    assert unpacked.shape == (1, 256)
    for offset in range(0, 256, 64):
        np.testing.assert_array_equal(
            unpacked[0, offset : offset + 32], np.full(32, 3, dtype=np.int8)
        )
        np.testing.assert_array_equal(
            unpacked[0, offset + 32 : offset + 64], np.full(32, 10, dtype=np.int8)
        )


def test_requantize_symmetric_preserves_reconstructed_values():
    constants = np.array([[0.5, -2.0]], dtype=np.float32)
    quantized = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8)
    expected = constants[:, :1] * quantized + constants[:, 1:2]

    scales, requantized = requantize_symmetric(
        constants, quantized, output_bits=8, new_dtype=np.int8
    )

    assert scales.dtype == np.float32
    assert requantized.dtype == np.int8
    np.testing.assert_allclose(scales * requantized, expected, atol=float(scales[0, 0]) / 2)
