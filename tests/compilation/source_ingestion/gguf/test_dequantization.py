from pathlib import Path

import numpy as np
import pytest

from sima_lmm.gguf.gguf_conversion import GgufModel
from tests.compilation.cases import GGUF_FILE_CASES, GgufFileCase
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_source]

BF16_CASE = next(case for case in GGUF_FILE_CASES if case.quantization == "BF16")
QUANTIZED_CASES = tuple(
    case for case in GGUF_FILE_CASES if case.quantization != "BF16"
)


def _dequantize_symmetric(scale: np.ndarray, quants: np.ndarray) -> np.ndarray:
    assert scale.ndim == 2
    assert scale.shape[1] == 1
    shape = quants.shape
    dequantized = scale * quants.reshape((scale.shape[0], -1))
    return dequantized.reshape(shape).astype(np.float32, copy=False)


@pytest.mark.parametrize("case", QUANTIZED_CASES, ids=lambda case: case.id)
def test_all_weights_match_bf16_and_gguf_library(
    model_inputs_path: Path, case: GgufFileCase
):
    bf16_path = require_readable_path(
        model_inputs_path / BF16_CASE.relative_path,
        "BF16 GGUF reference model",
    )
    quantized_path = require_readable_path(
        model_inputs_path / case.relative_path,
        f"{case.quantization} GGUF model",
    )
    bf16_model = GgufModel(bf16_path)
    quantized_model = GgufModel(quantized_path)

    assert case.reference_tolerance is not None
    assert case.library_tolerance is not None
    reference_atol, reference_rtol = case.reference_tolerance
    library_atol, library_rtol = case.library_tolerance

    for name in bf16_model.tensor_info:
        _, library_dequantized = quantized_model.load_weight(
            name, False, force_float=True
        )
        _, bf16_weight = bf16_model.load_weight(name, False)
        scales, quantized_weight = quantized_model.load_weight(name, False)

        assert bf16_weight.shape == quantized_weight.shape, name
        if scales is None:
            llima_dequantized = quantized_weight.astype(np.float32)
        else:
            llima_dequantized = _dequantize_symmetric(scales, quantized_weight)

        np.testing.assert_allclose(
            llima_dequantized,
            bf16_weight.astype(np.float32),
            rtol=reference_rtol,
            atol=reference_atol,
            err_msg=f"{case.quantization} tensor {name} differs from BF16",
        )
        np.testing.assert_allclose(
            llima_dequantized,
            library_dequantized.astype(np.float32),
            rtol=library_rtol,
            atol=library_atol,
            err_msg=(
                f"{case.quantization} tensor {name} differs from GGUF library"
            ),
        )
