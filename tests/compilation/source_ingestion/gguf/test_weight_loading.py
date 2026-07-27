from pathlib import Path

import pytest
from safetensors import safe_open

from sima_lmm.gguf.gguf_conversion import GgufModel
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from tests.compilation.cases import (
    GGUF_FILE_CASES,
    GGUF_HF_REFERENCE_MODEL,
    GgufFileCase,
)
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_source]


def _read_hf_weight_shapes(
    hf_model: LocalHuggingFaceModel,
) -> dict[str, tuple[int, ...]]:
    names_by_shard: dict[str, list[str]] = {}
    for name, shard in hf_model.weight_map.items():
        names_by_shard.setdefault(shard, []).append(name)

    shapes: dict[str, tuple[int, ...]] = {}
    for shard, names in names_by_shard.items():
        with safe_open(hf_model.weights[shard], framework="numpy") as tensors:
            for name in names:
                shapes[name] = tuple(tensors.get_slice(name).get_shape())
    return shapes


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
    hf_weight_shapes = _read_hf_weight_shapes(hf_model)
    failures: dict[str, str | None] = {
        case.quantization: None for case in GGUF_FILE_CASES
    }

    for name, hf_shape in hf_weight_shapes.items():
        for case in GGUF_FILE_CASES:
            if failures[case.quantization] is not None:
                continue

            try:
                gguf_model = gguf_models[case.quantization]
                gguf_name = gguf_model.convert_hf_weight_name(name)
                gguf_shape = tuple(gguf_model.tensor_info[gguf_name]["shape"])
                assert hf_shape == gguf_shape, (
                    f"GGUF weight mapped from {name} has shape "
                    f"{gguf_shape}; expected {hf_shape}"
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
