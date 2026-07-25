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


@pytest.fixture(scope="module")
def weight_mapping_failures(
    model_inputs_path: Path,
) -> dict[str, str | None]:
    hf_model_path = require_readable_path(
        model_inputs_path / GGUF_HF_REFERENCE_MODEL,
        "Hugging Face GGUF reference model",
    )
    hf_model = LocalHuggingFaceModel.create_from_directory(
        directory=hf_model_path,
        layer_names=None,
    )
    gguf_models = {
        case.quantization: GgufModel(
            require_readable_path(
                model_inputs_path / case.relative_path,
                f"{case.quantization} GGUF model",
            )
        )
        for case in GGUF_FILE_CASES
    }
    failures: dict[str, str | None] = {
        case.quantization: None for case in GGUF_FILE_CASES
    }

    for name in hf_model.weight_map:
        hf_weight = hf_model.load_np_param(name)
        for case in GGUF_FILE_CASES:
            if failures[case.quantization] is not None:
                continue

            try:
                _, gguf_weight = gguf_models[case.quantization].load_weight(
                    name, is_hf_name=True
                )
                assert hf_weight.shape == gguf_weight.shape, (
                    f"GGUF weight mapped from {name} has shape "
                    f"{gguf_weight.shape}; expected {hf_weight.shape}"
                )
            except Exception as exc:
                failures[case.quantization] = (
                    f"{case.quantization} failed at HF weight {name}: "
                    f"{type(exc).__name__}: {exc}"
                )

    return failures


@pytest.mark.parametrize("case", GGUF_FILE_CASES, ids=lambda case: case.id)
def test_hf_weight_names_resolve_with_matching_shapes(
    case: GgufFileCase,
    weight_mapping_failures: dict[str, str | None],
):
    failure = weight_mapping_failures[case.quantization]
    assert failure is None, failure
