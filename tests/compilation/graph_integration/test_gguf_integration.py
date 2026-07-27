import logging
from pathlib import Path

import numpy as np
import pytest

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from tests.compilation.cases import GGUF_GRAPH_CASES, GgufGraphCase
from tests.compilation.helpers.model_factory import load_hf_model
from tests.compilation.helpers.output_comparison import assert_outputs_close


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]

INPUT_SHAPES = {
    "pre": ((1, 1, 1, 1152), (1, 1, 1, 128), (1, 1, 1, 128)),
    "cache": ((1, 4, 1, 256), (1, 1, 3, 256), (1, 1, 3, 256)),
    "post": ((1, 1, 1, 1152), (1, 1, 1, 1024)),
}


def _build_component(case: GgufGraphCase, vlm_model):
    num_tokens = 1
    if case.component == "pre":
        model = LanguagePreModel(
            vlm_model.cfg,
            f"{vlm_model.model_name}_language_n{num_tokens}_pre_layer{case.layer_index}",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=num_tokens,
            layer_idx=case.layer_index,
        )
        layer_id = LayerID("single_pre", case.layer_index)
    elif case.component == "cache":
        model = LanguageCacheModel(
            vlm_model.cfg,
            f"{vlm_model.model_name}_language_n{num_tokens}_cache_token{case.layer_index}",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=num_tokens,
            token_idx=case.layer_index,
            logit_softcapping=vlm_model.cfg.lm_cfg.attn_logit_softcapping,
        )
        layer_id = LayerID("single_cache", case.layer_index)
    elif case.component == "post":
        model = LanguagePostModel(
            vlm_model.cfg,
            f"{vlm_model.model_name}_language_n{num_tokens}_post_layer{case.layer_index}",
            onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path,
            hf_model=vlm_model.hf_model,
            num_tokens=num_tokens,
            layer_idx=case.layer_index,
            final_softcapping=None,
        )
        layer_id = LayerID("single_post", case.layer_index)
    else:
        raise ValueError(f"Unsupported GGUF graph component: {case.component}")

    return model, layer_id


def _generate_and_run(
    case: GgufGraphCase,
    model_inputs_path: Path,
    output_path: Path,
    source_path: str,
    precision: FileGenPrecision,
    inputs: list[np.ndarray],
):
    vlm_model = load_hf_model(source_path, output_path, model_inputs_path)
    model, layer_id = _build_component(case, vlm_model)
    vlm_model.gen_files(
        FileGenMode.SOURCE_TO_QUANT,
        gen_config={"precision": {layer_id: precision}},
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    return model.run_model(EvalMode.SDK, inputs)


@pytest.mark.parametrize("case", GGUF_GRAPH_CASES, ids=lambda case: case.id)
def test_quantized_gguf_graph_matches_reference_source(
    model_inputs_path: Path,
    tmp_path: Path,
    case: GgufGraphCase,
):
    rng = np.random.default_rng(1)
    inputs = [
        rng.uniform(-1.0, 1.0, shape).astype(np.float32)
        for shape in INPUT_SHAPES[case.component]
    ]

    reference_outputs = _generate_and_run(
        case,
        model_inputs_path,
        tmp_path / "reference",
        case.reference_relative_path,
        FileGenPrecision.BF16,
        inputs,
    )
    gguf_outputs = _generate_and_run(
        case,
        model_inputs_path,
        tmp_path / "gguf",
        case.gguf_relative_path,
        FileGenPrecision.A_BF16_W_INT8,
        inputs,
    )

    assert_outputs_close(
        reference_outputs,
        gguf_outputs,
        case.output_tolerances,
        concatenate=case.component == "post",
    )
