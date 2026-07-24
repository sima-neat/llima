"""
ONNX regression tests: generate a fresh ONNX for layer 0 of each supported architecture
and compare its numerical outputs against the stored reference ONNX on the datacenter.

Reference ONNX files are created once via scripts/generate_reference_onnx.py and must
be regenerated whenever the ONNX export logic is intentionally changed.
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import FileGenMode, FileGenPrecision, VisionLanguageModel
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_per_layer_model import LanguagePerLayerModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.vision_model import VisionModel
from tests.conftest import require_readable_path

LAYER_IDX = 0
NUM_TOKENS = 1
MAX_NUM_TOKENS = 1024

# (hf_model_folder, model_type, image_resolution)
_ONNX_REGRESSION_CONFIGS = [
    ("models--meta-llama--Llama-3.2-1B-Instruct",   "pre",    None),
    ("models--meta-llama--Llama-3.2-1B-Instruct",   "cache",  None),
    ("models--meta-llama--Llama-3.2-1B-Instruct",   "post",   None),
    ("models--google--gemma-3-1b-it",               "pre",    None),
    ("models--google--gemma-3-1b-it",               "cache",  None),
    ("models--google--gemma-3-1b-it",               "post",   None),
    ("models--google--gemma-4-E2B-it",              "pre",    [240, 240]),
    ("models--google--gemma-4-E2B-it",              "cache",  [240, 240]),
    ("models--google--gemma-4-E2B-it",              "post",   [240, 240]),
    ("models--google--gemma-4-E2B-it",              "per_layer", [240, 240]),
    ("models--mistralai--Mistral-7B-Instruct-v0.3", "pre",    None),
    ("models--mistralai--Mistral-7B-Instruct-v0.3", "cache",  None),
    ("models--mistralai--Mistral-7B-Instruct-v0.3", "post",   None),
    ("models--microsoft--Phi-3.5-mini-instruct",    "pre",    None),
    ("models--microsoft--Phi-3.5-mini-instruct",    "cache",  None),
    ("models--microsoft--Phi-3.5-mini-instruct",    "post",   None),
    ("models--Qwen--Qwen2.5-0.5B-Instruct",         "pre",    None),
    ("models--Qwen--Qwen2.5-0.5B-Instruct",         "cache",  None),
    ("models--Qwen--Qwen2.5-0.5B-Instruct",         "post",   None),
    ("models--Qwen--Qwen3-0.6B",                    "pre",    None),
    ("models--Qwen--Qwen3-0.6B",                    "cache",  None),
    ("models--Qwen--Qwen3-0.6B",                    "post",   None),
    ("models--LiquidAI--LFM2-350M",                 "conv",   None),
    ("models--stribomon--gemma3-siglip448",         "vision", None),
    ("models--Qwen--Qwen2.5-VL-3B-Instruct",        "vision", [224, 224]),
    ("models--Qwen--Qwen3-VL-2B-Instruct",          "vision", [224, 224]),
    ("models--LiquidAI--LFM2-VL-450M",              "vision", None),
    ("models--google--gemma-4-E2B-it",              "vision", [240, 240]),
]


def _make_inputs(ref_onnx: Path, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Build random feeds matching the reference ONNX input spec."""
    graph = onnx.load(str(ref_onnx), load_external_data=False).graph
    feeds = {}
    for inp in graph.input:
        shape = [max(1, d.dim_value) for d in inp.type.tensor_type.shape.dim]
        feeds[inp.name] = rng.uniform(-1, 1, shape).astype(np.float32)
    return feeds


def _run_onnx(onnx_path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    sess = ort.InferenceSession(str(onnx_path))
    return sess.run([], feeds)


def _setup_model_and_gen_config(
    model_type: str, vlm_model: VisionLanguageModel
) -> tuple[object, dict]:
    if model_type == "pre":
        model_name = f"{vlm_model.model_name}_language_n{NUM_TOKENS}_pre_layer{LAYER_IDX}"
        model = LanguagePreModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS, layer_idx=LAYER_IDX,
        )
        gen_config = {"precision": {LayerID("single_pre", LAYER_IDX): FileGenPrecision.BF16}}
    elif model_type == "cache":
        model_name = f"{vlm_model.model_name}_language_n{NUM_TOKENS}_cache_token{LAYER_IDX}"
        model = LanguageCacheModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS, token_idx=LAYER_IDX,
            logit_softcapping=vlm_model.cfg.lm_cfg.attn_logit_softcapping,
        )
        gen_config = {"precision": {LayerID("single_cache", LAYER_IDX): FileGenPrecision.BF16}}
    elif model_type == "post":
        model_name = f"{vlm_model.model_name}_language_n{NUM_TOKENS}_post_layer{LAYER_IDX}"
        model = LanguagePostModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS, layer_idx=LAYER_IDX,
            final_softcapping=None,
        )
        gen_config = {"precision": {LayerID("single_post", LAYER_IDX): FileGenPrecision.BF16}}
    elif model_type == "per_layer":
        model_name = f"{vlm_model.model_name}_language_n{NUM_TOKENS}_per_layer"
        model = LanguagePerLayerModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS,
        )
        gen_config = {"precision": {LayerID("single_per_layer", 0): FileGenPrecision.BF16}}
    elif model_type == "conv":
        model_name = f"{vlm_model.model_name}_language_n{NUM_TOKENS}_layer{LAYER_IDX}_conv"
        model = LanguageConvModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=NUM_TOKENS, layer_idx=LAYER_IDX,
            final_softcapping=vlm_model.cfg.lm_cfg.final_logit_softcapping,
        )
        gen_config = {"precision": {LayerID("single_conv", LAYER_IDX): FileGenPrecision.BF16}}
    elif model_type == "vision":
        model = VisionModel(
            vlm_model.cfg, vlm_model.vision_model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
        )
        model = model._get_part_model(LAYER_IDX)
        gen_config = {"precision": {LayerID("vision", LAYER_IDX): FileGenPrecision.BF16}}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model, gen_config


@pytest.mark.premerge
@pytest.mark.parametrize(
    "model_folder,model_type,image_resolution",
    _ONNX_REGRESSION_CONFIGS,
    ids=lambda x: x if isinstance(x, str) else str(x),
)
def test_onnx_regression(
    hf_models_path: Path,
    reference_onnx_path: Path,
    model_folder: str,
    model_type: str,
    image_resolution: list[int] | None,
) -> None:
    model_path = require_readable_path(hf_models_path / model_folder)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vlm_model = VisionLanguageModel.from_hf_cache(
            hf_cache_path=model_path,
            model_name=model_path.name,
            onnx_path=tmpdir / "onnx",
            sima_path=tmpdir / "sima",
            max_num_tokens=MAX_NUM_TOKENS,
            system_prompt=None,
            image_resolution=image_resolution,
        )

        model, gen_config = _setup_model_and_gen_config(model_type, vlm_model)

        ref_onnx = require_readable_path(reference_onnx_path / model.onnx_file_name.name, "reference ONNX")

        rng = np.random.default_rng(42)
        feeds = _make_inputs(ref_onnx, rng)

        model.gen_files(
            FileGenMode.SOURCE_TO_ONNX,
            layer_cfg={"precision": next(iter(gen_config["precision"].values()))},
            log_level=logging.WARNING,
            resume=False,
        )

        fresh_outputs = _run_onnx(model.onnx_file_name, feeds)
        ref_outputs = _run_onnx(ref_onnx, feeds)

    assert len(fresh_outputs) == len(ref_outputs), (
        f"Output count mismatch: fresh={len(fresh_outputs)}, ref={len(ref_outputs)}"
    )
    for i, (fresh, ref) in enumerate(zip(fresh_outputs, ref_outputs)):
        assert np.array_equal(fresh, ref), (
            f"Output {i} mismatch for {model_folder}/{model_type}: "
            f"max_diff={np.max(np.abs(fresh - ref)):.4e}"
        )
