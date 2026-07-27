import json
from dataclasses import asdict
from pathlib import Path

import pytest
from transformers import AutoConfig

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from sima_lmm.gguf.gguf_conversion import GgufModel
from tests.compilation.cases import CONFIGURATION_CASES, ConfigurationCase
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_config]

REFERENCE_CONFIGS_PATH = Path(__file__).parent / "references"


def _find_implicit_gguf(model_folder: Path) -> Path:
    model_folder = require_readable_path(model_folder, "GGUF model directory")
    for suffix in ("Q4_0.GGUF", "Q8_0.GGUF"):
        matches = sorted(
            path
            for path in model_folder.glob("*.gguf")
            if path.name.upper().endswith(suffix)
        )
        if matches:
            return require_readable_path(matches[0], "GGUF model")
    raise FileNotFoundError(
        f"No Q4_0 or Q8_0 GGUF model found in {model_folder}"
    )


def _normalize_hf_config(config: VlmConfig) -> dict:
    normalized = asdict(config)
    normalized.pop("model_name", None)
    return normalized


def _normalize_gguf_config(config: VlmConfig) -> dict:
    normalized = asdict(config)
    normalized.pop("model_name", None)
    normalized.pop("mm_cfg", None)
    normalized.pop("vm_cfg", None)

    language_config = normalized["lm_cfg"]
    language_config.pop("data_type", None)
    language_config.pop("model_type", None)
    language_config.pop("rms_norm_eps", None)
    return normalized


def _load_reference(case: ConfigurationCase) -> dict:
    reference_path = require_readable_path(
        REFERENCE_CONFIGS_PATH / case.reference_config,
        f"configuration reference for {case.id}",
    )
    return json.loads(reference_path.read_text(encoding="utf-8"))


def _generate_config(case: ConfigurationCase, model_inputs_path: Path) -> VlmConfig:
    model_path = require_readable_path(
        model_inputs_path / case.model_folder,
        f"source model for {case.id}",
    )

    if case.source_format == "hf":
        model_config = AutoConfig.from_pretrained(
            model_path, local_files_only=True
        ).to_dict()
        image_resolution = (
            list(case.image_resolution) if case.image_resolution is not None else None
        )
        return VlmConfig.from_hf_config(
            ModelFormat.FORMAT_HF,
            model_path,
            model_config,
            image_resolution,
        )

    gguf_path = _find_implicit_gguf(model_path)
    gguf_model = GgufModel(gguf_path)
    return VlmConfig.from_hf_config(
        ModelFormat.FORMAT_GGUF,
        gguf_path,
        gguf_model.model_config,
    )


@pytest.mark.parametrize("case", CONFIGURATION_CASES, ids=lambda case: case.id)
def test_configuration_contract(
    model_inputs_path: Path, case: ConfigurationCase
):
    generated = _generate_config(case, model_inputs_path)
    reference = VlmConfig.load(_load_reference(case))

    if case.source_format == "hf":
        actual = _normalize_hf_config(generated)
        expected = _normalize_hf_config(reference)
    else:
        actual = _normalize_gguf_config(generated)
        expected = _normalize_gguf_config(reference)
        actual_epsilon = generated.lm_cfg.rms_norm_eps
        expected_epsilon = reference.lm_cfg.rms_norm_eps
        assert (actual_epsilon is None) == (expected_epsilon is None)
        if actual_epsilon is not None:
            assert actual_epsilon == pytest.approx(expected_epsilon, abs=1e-8)

    assert actual == expected
