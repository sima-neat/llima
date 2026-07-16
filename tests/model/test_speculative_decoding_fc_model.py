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
import logging
import numpy as np
import pytest
import tempfile

from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel

from tests.model.speculative_decoding_model_setup import load_speculative_test_draft_model

TARGET_MODEL_NAME = "models--meta-llama--Llama-3.1-8B-Instruct"
DRAFT_MODEL_NAME  = "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge"

NUM_TOKENS = 5

@pytest.mark.premerge
def test_speculative_decoding_draft_language_fc_model(hf_models_path: Path):
    """
    Create a draft FC fusion model using the ONNX and MODEL_SDK_DIRECT file generation
    paths and verify that both compute identical outputs given the same inputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        draft_model = load_speculative_test_draft_model(
            TARGET_MODEL_NAME, DRAFT_MODEL_NAME, Path(tmpdir), max_num_tokens=1024,
            models_path=hf_models_path
        )

        cfg = draft_model.cfg

        rng  = np.random.default_rng(1)
        ifms = [
            rng.uniform(-5, 5, (1, 1, NUM_TOKENS, cfg.lm_cfg.hidden_size * 3)).astype(np.float32),
        ]

        model_name = f"{draft_model.model_name}_language_n{NUM_TOKENS}_draft_fc"
        model = LanguageDraftFCModel(
            cfg, model_name, onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path, hf_model=draft_model.hf_model,
            num_tokens=NUM_TOKENS,
        )

        gen_config = {"precision": {LayerID("single_draft_fc", 0): FileGenPrecision.BF16}}

        draft_model.gen_files(
            FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False,
        )
        draft_model.gen_files(
            FileGenMode.ONNX_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False,
        )
        out1, = model.run_model(EvalMode.SDK, ifms)

        draft_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False,
        )
        out2, = model.run_model(EvalMode.SDK, ifms)

    assert np.allclose(out1, out2, atol=0.01 * np.max(np.abs(out1))), "output mismatch"
