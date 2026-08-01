import logging
from pathlib import Path

import numpy as np
import pytest

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from tests.compilation.cases import (
    STANDARD_GRAPH_CASES,
    STANDARD_GRAPH_MODEL,
    GraphPathCase,
)
from tests.compilation.helpers.model_factory import load_hf_model
from tests.compilation.helpers.output_comparison import assert_outputs_close


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]


def _build_component(case: GraphPathCase, vlm_model):
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
        raise ValueError(f"Unsupported graph component: {case.component}")

    return model, {"precision": {layer_id: FileGenPrecision.BF16}}


@pytest.mark.parametrize("case", STANDARD_GRAPH_CASES, ids=lambda case: case.id)
def test_staged_and_direct_generation_are_equivalent(
    model_inputs_path: Path,
    tmp_path: Path,
    case: GraphPathCase,
):
    vlm_model = load_hf_model(
        STANDARD_GRAPH_MODEL,
        tmp_path,
        model_inputs_path,
    )
    model, gen_config = _build_component(case, vlm_model)
    rng = np.random.default_rng(1)
    inputs = [
        rng.uniform(-1.0, 1.0, shape).astype(np.float32)
        for shape in case.input_shapes
    ]

    vlm_model.gen_files(
        FileGenMode.SOURCE_TO_ONNX,
        gen_config=gen_config,
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    vlm_model.gen_files(
        FileGenMode.ONNX_TO_QUANT,
        gen_config=gen_config,
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    staged_outputs = model.run_model(EvalMode.SDK, inputs)

    vlm_model.gen_files(
        FileGenMode.SOURCE_TO_QUANT,
        gen_config=gen_config,
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    direct_outputs = model.run_model(EvalMode.SDK, inputs)

    assert_outputs_close(
        staged_outputs,
        direct_outputs,
        (case.tolerance,),
        concatenate=case.concatenate_outputs,
    )
