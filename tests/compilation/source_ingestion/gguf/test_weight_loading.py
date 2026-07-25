from pathlib import Path

import pytest

from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from tests.compilation.cases import (
    GGUF_FILE_CASES,
    GGUF_HF_REFERENCE_MODEL,
    GgufFileCase,
)
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_source]


@pytest.mark.parametrize("case", GGUF_FILE_CASES, ids=lambda case: case.id)
def test_hf_weight_names_resolve_with_matching_shapes(
    model_inputs_path: Path, case: GgufFileCase
):
    hf_model_path = require_readable_path(
        model_inputs_path / GGUF_HF_REFERENCE_MODEL,
        "Hugging Face GGUF reference model",
    )
    gguf_path = require_readable_path(
        model_inputs_path / case.relative_path,
        f"{case.quantization} GGUF model",
    )
    hf_model = LocalHuggingFaceModel.create_from_directory(
        directory=hf_model_path,
        layer_names=None,
    )
    gguf_model = GgufModel(gguf_path)

    for name in hf_model.weight_map:
        hf_weight = hf_model.load_np_param(name)
        _, gguf_weight = gguf_model.load_weight(name, is_hf_name=True)
        assert hf_weight.shape == gguf_weight.shape, (
            f"GGUF weight mapped from {name} has shape {gguf_weight.shape}; "
            f"expected {hf_weight.shape}"
        )
