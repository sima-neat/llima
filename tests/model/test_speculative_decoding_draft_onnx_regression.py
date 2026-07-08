#########################################################
# Copyright (C) 2026 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
"""
ONNX regression tests for the EAGLE3 speculative decoding draft model: generate a fresh
ONNX for each draft model type and compare its numerical outputs against the stored
reference ONNX on the datacenter.

Reference ONNX files are created once via scripts/generate_reference_draft_onnx.py and
must be regenerated whenever the ONNX export logic is intentionally changed.
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
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel

from tests.model.speculative_decoding_model_setup import load_speculative_test_draft_model
from tests.conftest import require_readable_path

LAYER_IDX = 0
TOKEN_IDX = 2
MAX_NUM_TOKENS = 1024

# (target_model_folder, draft_model_folder, model_type)
_DRAFT_ONNX_REGRESSION_CONFIGS = [
    ("models--meta-llama--Llama-3.1-8B-Instruct", "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge", "pre"),
    ("models--meta-llama--Llama-3.1-8B-Instruct", "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge", "post"),
    ("models--meta-llama--Llama-3.1-8B-Instruct", "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge", "fc"),
    ("models--meta-llama--Llama-3.1-8B-Instruct", "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge", "cache"),
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
    model_type: str, draft_model: VisionLanguageModel
) -> tuple[object, dict]:
    cfg  = draft_model.cfg
    num_tokens = cfg.lm_cfg.speculative_decoding_cfg.speculative_budget

    if model_type == "pre":
        model_name = f"{draft_model.model_name}_language_n{num_tokens}_pre_layer{LAYER_IDX}"
        model = LanguagePreModel(
            cfg, model_name, onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path, hf_model=draft_model.hf_model,
            num_tokens=num_tokens, layer_idx=LAYER_IDX,
        )
        gen_config = {"precision": {LayerID("single_pre", LAYER_IDX): FileGenPrecision.BF16}}
    elif model_type == "post":
        model_name = f"{draft_model.model_name}_language_n{num_tokens}_post_layer{LAYER_IDX}"
        model = LanguagePostModel(
            cfg, model_name, onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path, hf_model=draft_model.hf_model,
            num_tokens=num_tokens, layer_idx=LAYER_IDX,
            final_softcapping=cfg.lm_cfg.final_logit_softcapping,
        )
        gen_config = {"precision": {LayerID("single_post", LAYER_IDX): FileGenPrecision.BF16}}
    elif model_type == "fc":
        model_name = f"{draft_model.model_name}_language_n{num_tokens}_draft_fc"
        model = LanguageDraftFCModel(
            cfg, model_name, onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path, hf_model=draft_model.hf_model,
            num_tokens=num_tokens,
        )
        gen_config = {"precision": {LayerID("single_draft_fc", 0): FileGenPrecision.BF16}}
    elif model_type == "cache":
        model_name = f"{draft_model.model_name}_language_n{num_tokens}_cache_token{TOKEN_IDX}"
        model = LanguageCacheModel(
            cfg, model_name, onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path, hf_model=draft_model.hf_model,
            num_tokens=num_tokens, token_idx=TOKEN_IDX,
            logit_softcapping=cfg.lm_cfg.attn_logit_softcapping,
        )
        gen_config = {"precision": {LayerID("single_cache", TOKEN_IDX): FileGenPrecision.BF16}}
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, gen_config


@pytest.mark.premerge
@pytest.mark.parametrize(
    "target_folder,draft_folder,model_type",
    _DRAFT_ONNX_REGRESSION_CONFIGS,
    ids=[f"{draft}-{t}" for _, draft, t in _DRAFT_ONNX_REGRESSION_CONFIGS],
)
def test_speculative_decoding_draft_onnx_regression(
    hf_models_path: Path,
    reference_draft_onnx_path: Path,
    target_folder: str,
    draft_folder: str,
    model_type: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        draft_model = load_speculative_test_draft_model(
            target_folder, draft_folder, Path(tmpdir), max_num_tokens=MAX_NUM_TOKENS,
            models_path=hf_models_path
        )

        model, gen_config = _setup_model_and_gen_config(model_type, draft_model)

        ref_onnx = require_readable_path(reference_draft_onnx_path / model.onnx_file_name.name, "reference draft ONNX")

        rng = np.random.default_rng(42)
        feeds = _make_inputs(ref_onnx, rng)

        draft_model.gen_files(
            FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
            log_level=logging.WARNING, resume=False,
        )

        fresh_outputs = _run_onnx(model.onnx_file_name, feeds)
        ref_outputs   = _run_onnx(ref_onnx, feeds)

    assert len(fresh_outputs) == len(ref_outputs), (
        f"Output count mismatch: fresh={len(fresh_outputs)}, ref={len(ref_outputs)}"
    )
    for i, (fresh, ref) in enumerate(zip(fresh_outputs, ref_outputs)):
        assert np.array_equal(fresh, ref), (
            f"Output {i} mismatch for {draft_folder}/{model_type}: "
            f"max_diff={np.max(np.abs(fresh - ref)):.4e}"
        )
