import logging
import numpy as np
import pytest
import tempfile

from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_cache_model import (
    LanguageCacheModel,
    _get_bmm2_reduction_ranges,
)

from tests.model.model_setup import load_hf_test_model

MODEL_NAME = "models--google--gemma-3-1b-it"
GGUF_MODEL_Q8_0 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q8_0.gguf"

ifm_shapes = [(1, 4, 1, 256), (1, 1, 3, 256), (1, 1, 3, 256)]


@pytest.mark.premerge
@pytest.mark.parametrize("context_length", [1, 1024, 2048])
def test_bmm2_reduction_is_not_split_at_or_below_threshold(context_length: int):
    assert _get_bmm2_reduction_ranges(context_length) == [(0, context_length)]


@pytest.mark.premerge
@pytest.mark.parametrize("context_length", [2049, 4096, 6144, 8192])
def test_bmm2_reduction_uses_contiguous_1k_chunks(context_length: int):
    ranges = _get_bmm2_reduction_ranges(context_length)

    assert ranges[0][0] == 0
    assert ranges[-1][1] == context_length
    assert all(end == next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:]))
    assert all(0 < end - start <= 1024 for start, end in ranges)


@pytest.mark.premerge
def test_language_cache_model(hf_models_path: Path):
    """
    Create a cache model using ONNX and MODEL_SDK_DIRECT file generation methods.
    Verify that both models compute the same output given the same input.
    """

    rng = np.random.default_rng(1)
    ifms = [rng.uniform(-5, 5, ifm_shapes[0]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[1]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[2]).astype(np.float32)]

    # Parameters that determine which cache model to run
    num_tokens = 1
    token_idx = 2

    with tempfile.TemporaryDirectory() as tmpdir:

        vlm_model = load_hf_test_model(MODEL_NAME, Path(tmpdir), 1024, hf_models_path)

        # The model to test
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_cache_token{token_idx}"
        model = LanguageCacheModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, token_idx=token_idx,
            logit_softcapping=vlm_model.cfg.lm_cfg.attn_logit_softcapping
        )

        # Test one layer in BF16 precision
        gen_config = {
            "precision": {LayerID("single_cache", 2): FileGenPrecision.BF16}
        }

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        vlm_model.gen_files(
            FileGenMode.ONNX_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        ofm1, = model.run_model(EvalMode.SDK, ifms)

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        ofm2, = model.run_model(EvalMode.SDK, ifms)

        gguf_vlm_model = load_hf_test_model(GGUF_MODEL_Q8_0, Path(tmpdir), 1024, hf_models_path)

        model_name = f"{gguf_vlm_model.model_name}_language_n{num_tokens}_cache_token{token_idx}"
        model = LanguageCacheModel(
            gguf_vlm_model.cfg, model_name, onnx_path=gguf_vlm_model.onnx_path,
            sima_path=gguf_vlm_model.sima_path, hf_model=gguf_vlm_model.hf_model,
            num_tokens=num_tokens, token_idx=token_idx,
            logit_softcapping=gguf_vlm_model.cfg.lm_cfg.attn_logit_softcapping
        )
        # Change to 8-bit quantized precision for testing GGUF
        gen_config = {
            "precision": {LayerID("single_cache", 2): FileGenPrecision.A_BF16_W_INT8}
        }
        gguf_vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        ofm3, = model.run_model(EvalMode.SDK, ifms)

    # Verify results are equal with tolerance
    assert np.allclose(ofm1, ofm2, atol=0.01 * np.max(np.abs(ofm1)))
    assert np.array_equal(ofm2, ofm3)
