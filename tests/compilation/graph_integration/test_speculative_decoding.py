import logging
from pathlib import Path

import numpy as np
import pytest

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from tests.compilation.cases import (
    SPECULATIVE_DRAFT_MODEL,
    SPECULATIVE_GRAPH_CASES,
    SPECULATIVE_TARGET_MODEL,
    SpeculativeGraphCase,
)
from tests.compilation.helpers.model_factory import load_speculative_draft_model
from tests.compilation.helpers.output_comparison import assert_outputs_close


pytestmark = [
    pytest.mark.premerge,
    pytest.mark.compiler_graph_integration,
    pytest.mark.serial,
    pytest.mark.high_memory,
]

NUM_TOKENS = 5
LAYER_INDEX = 0
TOKEN_INDEX = 2


def _build_component(case: SpeculativeGraphCase, draft_model):
    cfg = draft_model.cfg
    attention = cfg.lm_cfg.attn_cfg
    if case.component == "pre":
        model = LanguagePreModel(
            cfg,
            f"{draft_model.model_name}_language_n{NUM_TOKENS}_pre_layer{LAYER_INDEX}",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=NUM_TOKENS,
            layer_idx=LAYER_INDEX,
        )
        layer_id = LayerID("single_pre", LAYER_INDEX)
        shapes = (
            (1, 1, NUM_TOKENS, cfg.lm_cfg.hidden_size),
            (1, 1, NUM_TOKENS, cfg.lm_cfg.hidden_size),
            (1, 1, NUM_TOKENS, attention.head_dim // 2),
            (1, 1, NUM_TOKENS, attention.head_dim // 2),
        )
    elif case.component == "cache":
        context_length = TOKEN_INDEX + 1
        model = LanguageCacheModel(
            cfg,
            f"{draft_model.model_name}_language_n{NUM_TOKENS}_cache_token{TOKEN_INDEX}",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=NUM_TOKENS,
            token_idx=TOKEN_INDEX,
            logit_softcapping=cfg.lm_cfg.attn_logit_softcapping,
        )
        layer_id = LayerID("single_cache", TOKEN_INDEX)
        shapes = (
            (1, 32, NUM_TOKENS, 128),
            (1, 8, context_length, 128),
            (1, 1, NUM_TOKENS, context_length),
            (1, 8, context_length, 128),
        )
    elif case.component == "post":
        model = LanguagePostModel(
            cfg,
            f"{draft_model.model_name}_language_n{NUM_TOKENS}_post_layer{LAYER_INDEX}",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=NUM_TOKENS,
            layer_idx=LAYER_INDEX,
            final_softcapping=cfg.lm_cfg.final_logit_softcapping,
        )
        layer_id = LayerID("single_post", LAYER_INDEX)
        shapes = (
            (1, 1, NUM_TOKENS, 4096),
            (1, 1, NUM_TOKENS, 4096),
        )
    elif case.component == "draft_fc":
        model = LanguageDraftFCModel(
            cfg,
            f"{draft_model.model_name}_language_n{NUM_TOKENS}_draft_fc",
            onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path,
            hf_model=draft_model.hf_model,
            num_tokens=NUM_TOKENS,
        )
        layer_id = LayerID("single_draft_fc", 0)
        shapes = ((1, 1, NUM_TOKENS, cfg.lm_cfg.hidden_size * 3),)
    else:
        raise ValueError(f"Unsupported speculative component: {case.component}")

    return model, {"precision": {layer_id: FileGenPrecision.BF16}}, shapes


@pytest.mark.parametrize(
    "case", SPECULATIVE_GRAPH_CASES, ids=lambda case: case.id
)
def test_speculative_staged_and_direct_generation_are_equivalent(
    model_inputs_path: Path,
    tmp_path: Path,
    case: SpeculativeGraphCase,
):
    draft_model = load_speculative_draft_model(
        SPECULATIVE_TARGET_MODEL,
        SPECULATIVE_DRAFT_MODEL,
        tmp_path,
        model_inputs_path,
    )
    model, gen_config, shapes = _build_component(case, draft_model)
    rng = np.random.default_rng(1)
    inputs = [
        rng.uniform(-1.0, 1.0, shape).astype(np.float32) for shape in shapes
    ]

    draft_model.gen_files(
        FileGenMode.SOURCE_TO_ONNX,
        gen_config=gen_config,
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    draft_model.gen_files(
        FileGenMode.ONNX_TO_QUANT,
        gen_config=gen_config,
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    staged_outputs = model.run_model(EvalMode.SDK, inputs)

    draft_model.gen_files(
        FileGenMode.SOURCE_TO_QUANT,
        gen_config=gen_config,
        num_processes=1,
        log_level=logging.WARNING,
        resume=False,
    )
    direct_outputs = model.run_model(EvalMode.SDK, inputs)

    assert_outputs_close(staged_outputs, direct_outputs, (0.01,))
