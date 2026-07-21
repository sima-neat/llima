import logging
import numpy as np
import pytest
import tempfile

from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_cache_model import LanguageCacheModel

from tests.model.speculative_decoding_model_setup import load_speculative_test_draft_model

TARGET_MODEL_NAME = "models--meta-llama--Llama-3.1-8B-Instruct"
DRAFT_MODEL_NAME  = "models--lmsys--SGLang-EAGLE3-Llama-3.1-8B-Instruct-SpecForge"

NUM_TOKENS = 5
TOKEN_IDX  = 2

@pytest.mark.premerge
def test_speculative_decoding_draft_language_cache_model(hf_models_path: Path):
    """
    Create a draft cache model using the ONNX and MODEL_SDK_DIRECT file generation
    paths and verify that both compute identical outputs given the same inputs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        draft_model = load_speculative_test_draft_model(
            TARGET_MODEL_NAME, DRAFT_MODEL_NAME, Path(tmpdir), max_num_tokens=1024,
            models_path=hf_models_path
        )

        cfg = draft_model.cfg

        context_length = TOKEN_IDX + 1  # speculative decoding: all draft tokens attend to same committed context

        rng  = np.random.default_rng(1)
        ifms = [
            rng.uniform(-1, 1, (1, 32, NUM_TOKENS, 128)).astype(np.float32),                # query
            rng.uniform(-1, 1, (1, 8, context_length, 128)).astype(np.float32),             # cached_keys
            rng.uniform(-1, 1, (1, 1, NUM_TOKENS, context_length)).astype(np.float32),      # attn_mask
            rng.uniform(-1, 1, (1, 8, context_length, 128)).astype(np.float32),             # cached_values
        ]

        model_name = f"{draft_model.model_name}_language_n{NUM_TOKENS}_cache_token{TOKEN_IDX}"
        model = LanguageCacheModel(
            cfg, model_name, onnx_path=draft_model.onnx_path,
            sima_path=draft_model.sima_path, hf_model=draft_model.hf_model,
            num_tokens=NUM_TOKENS, token_idx=TOKEN_IDX,
            logit_softcapping=cfg.lm_cfg.attn_logit_softcapping,
        )

        gen_config = {"precision": {LayerID("single_cache", TOKEN_IDX): FileGenPrecision.BF16}}

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
