import logging
import numpy as np
import pytest
import tempfile

from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision, VisionLanguageModel
from sima_lmm.model.language_pre_model import LanguagePreModel

from tests.model.model_setup import load_hf_test_model

MODEL_NAME = "models--google--gemma-3-1b-it"

GGUF_MODEL_Q8_0 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q8_0.gguf"
GGUF_MODEL_Q4_0 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_0.gguf"
GGUF_MODEL_Q4_1 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_1.gguf"

ifm_shapes = [(1, 1, 1, 1152), (1, 1, 1, 128), (1, 1, 1, 128)]


@pytest.mark.premerge
def test_language_pre_model(hf_models_path: Path):
    """
    Create a pre-cache model using ONNX and MODEL_SDK_DIRECT file generation methods.
    Verify that both models compute the same output given the same input.
    """

    rng = np.random.default_rng(1)
    ifms = [rng.uniform(-5, 5, ifm_shapes[0]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[1]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[2]).astype(np.float32)]

    # Parameters that determine which cache model to run
    num_tokens = 1
    layer_idx = 2

    with tempfile.TemporaryDirectory() as tmpdir:

        vlm_model = load_hf_test_model(MODEL_NAME, Path(tmpdir), 1024, hf_models_path)

        # The model to test
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_pre_layer{layer_idx}"
        model = LanguagePreModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx
        )

        # Test one layer in BF16 precision
        gen_config = {
            "precision": {LayerID("single_pre", 2): FileGenPrecision.BF16}
        }

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        vlm_model.gen_files(
            FileGenMode.ONNX_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        re1, k1, v1 = model.run_model(EvalMode.SDK, ifms)

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        re2, k2, v2, = model.run_model(EvalMode.SDK, ifms)

    # Verify results are equal with tolerance
    assert np.allclose(re1, re2, atol=0.01 * np.max(np.abs(re1)))
    assert np.allclose(k1, k2, atol=0.01 * np.max(np.abs(k1)))
    assert np.allclose(v1, v2, atol=0.01 * np.max(np.abs(v1)))


tolerances = {
    GGUF_MODEL_Q8_0: (0.02, 0.01, 0.02),
    GGUF_MODEL_Q4_0: (0.09, 0.06, 0.12),
    GGUF_MODEL_Q4_1: (0.12, 0.06, 0.12)
}
@pytest.mark.premerge
@pytest.mark.parametrize(
    "gguf_model", [GGUF_MODEL_Q8_0, GGUF_MODEL_Q4_0, GGUF_MODEL_Q4_1],
    ids=lambda x: f"gguf_type={x}"
)
def test_gguf_language_pre_model(hf_models_path: Path, gguf_model: str):
    """
    Test a pre-cache model created from quantized GGUF model using MODEL_SDK_DIRECT
    file generation method.  Verify its accuracy by comparing its outputs against the
    outputs of a pre-cache model created from HF model.
    """

    rng = np.random.default_rng(1)
    ifms = [rng.uniform(-5, 5, ifm_shapes[0]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[1]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[2]).astype(np.float32)]

    # Parameters that determine which cache model to run
    num_tokens = 1
    layer_idx = 2

    with tempfile.TemporaryDirectory() as tmpdir:

        vlm_model = load_hf_test_model(MODEL_NAME, Path(tmpdir), 1024, hf_models_path)

        # The model to test
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_pre_layer{layer_idx}"
        model = LanguagePreModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx
        )

        # Test one layer
        gen_config = {
            "precision": {LayerID("single_pre", 2): FileGenPrecision.BF16}
        }

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        re1, k1, v1, = model.run_model(EvalMode.SDK, ifms)

        gguf_vlm_model = load_hf_test_model(gguf_model, Path(tmpdir), 1024, hf_models_path)

        model_name = f"{gguf_vlm_model.model_name}_language_n{num_tokens}_pre_layer{layer_idx}"
        model = LanguagePreModel(
            gguf_vlm_model.cfg, model_name, onnx_path=gguf_vlm_model.onnx_path,
            sima_path=gguf_vlm_model.sima_path, hf_model=gguf_vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx
        )
        gen_config = {
            "precision": {LayerID("single_pre", 2): FileGenPrecision.A_BF16_W_INT8}
        }
        gguf_vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        re2, k2, v2, = model.run_model(EvalMode.SDK, ifms)

    # Verify results are equal with tolerance
    assert np.allclose(re1, re2, atol=tolerances[gguf_model][0] * np.max(np.abs(re1)))
    assert np.allclose(k1, k2, atol=tolerances[gguf_model][1] * np.max(np.abs(k1)))
    assert np.allclose(v1, v2, atol=tolerances[gguf_model][2] * np.max(np.abs(v1)))
