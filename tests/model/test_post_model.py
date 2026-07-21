import logging
import numpy as np
import pytest
import tempfile

from pathlib import Path

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision
from sima_lmm.model.language_post_model import LanguagePostModel

from tests.model.model_setup import load_hf_test_model

MODEL_NAME = "models--google--gemma-3-1b-it"
MODEL_NAME2 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-BF16.gguf"

GGUF_MODEL_Q8_0 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q8_0.gguf"
GGUF_MODEL_Q4_0 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_0.gguf"
GGUF_MODEL_Q4_1 = "models--unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_1.gguf"

ifm_shapes = [(1, 1, 1, 1152), (1, 1, 1, 1024)]


@pytest.mark.premerge
@pytest.mark.parametrize("layer_idx", [2, 25], ids=lambda x: f"layer={x}")
def test_language_post_model(hf_models_path: Path, layer_idx: int):
    """
    Create a post-cache model using ONNX and MODEL_SDK_DIRECT file generation methods.
    Verify that both models compute the same output given the same input.
    """

    rng = np.random.default_rng(1)
    ifms = [rng.uniform(-5, 5, ifm_shapes[0]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[1]).astype(np.float32)]

    # Parameters that determine which cache model to run
    num_tokens = 1

    with tempfile.TemporaryDirectory() as tmpdir:

        vlm_model = load_hf_test_model(MODEL_NAME, Path(tmpdir), 1024, hf_models_path)

        # The model to test
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_post_layer{layer_idx}"
        model = LanguagePostModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx, final_softcapping=None
        )
        # Test one layer
        gen_config = {
            "precision": {LayerID("single_post", layer_idx): FileGenPrecision.BF16}
        }

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_ONNX, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        vlm_model.gen_files(
            FileGenMode.ONNX_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        outputs1 = model.run_model(EvalMode.SDK, ifms)

        vlm_model = load_hf_test_model(MODEL_NAME2, Path(tmpdir), 1024, hf_models_path)

        # The model to test
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_post_layer{layer_idx}"
        model = LanguagePostModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx, final_softcapping=None
        )

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        outputs2 = model.run_model(EvalMode.SDK, ifms)

    # Output array contents should be the same.
    # However, outputs may be differently split into multiple arrays.
    output1 = np.concatenate([x.squeeze() for x in outputs1])
    output2 = np.concatenate([x.squeeze() for x in outputs2])

    assert np.allclose(output1, output2, atol=0.01 * np.max(np.abs(output1)))


tolerances = {
    GGUF_MODEL_Q8_0: {2: 0.02, 25: 0.07},
    GGUF_MODEL_Q4_0: {2: 0.06, 25: 0.17},
    GGUF_MODEL_Q4_1: {2: 0.04, 25: 0.15}
}
_model_id = {GGUF_MODEL_Q8_0: "Q8_0", GGUF_MODEL_Q4_0: "Q4_0", GGUF_MODEL_Q4_1: "Q4_1"}


@pytest.mark.premerge
@pytest.mark.parametrize(
    "gguf_model", [GGUF_MODEL_Q8_0, GGUF_MODEL_Q4_0, GGUF_MODEL_Q4_1],
    ids=lambda x: _model_id[x]
)
@pytest.mark.parametrize("layer_idx", [2, 25], ids=lambda x: f"layer={x}")
def test_gguf_language_post_model(hf_models_path: Path, gguf_model: str, layer_idx: int):
    """
    Test a post-cache model created from quantized GGUF model using MODEL_SDK_DIRECT
    file generation method.  Verify its accuracy by comparing its outputs against the
    outputs of a post-cache model created from bfloat16 GGUF model.
    """

    rng = np.random.default_rng(1)
    ifms = [rng.uniform(-5, 5, ifm_shapes[0]).astype(np.float32),
            rng.uniform(-1, 1, ifm_shapes[1]).astype(np.float32)]

    # Parameters that determine which cache model to run
    num_tokens = 1

    with tempfile.TemporaryDirectory() as tmpdir:
        vlm_model = load_hf_test_model(MODEL_NAME2, Path(tmpdir), 1024, hf_models_path)

        # Reference model
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_post_layer{layer_idx}"
        model = LanguagePostModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx, final_softcapping=None
        )
        # Test one layer
        gen_config = {
            "precision": {LayerID("single_post", layer_idx): FileGenPrecision.BF16}
        }

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        ref_outputs = model.run_model(EvalMode.SDK, ifms)

        vlm_model = load_hf_test_model(gguf_model, Path(tmpdir), 1024, hf_models_path)

        # The model to test
        model_name = f"{vlm_model.model_name}_language_n{num_tokens}_post_layer{layer_idx}"
        model = LanguagePostModel(
            vlm_model.cfg, model_name, onnx_path=vlm_model.onnx_path,
            sima_path=vlm_model.sima_path, hf_model=vlm_model.hf_model,
            num_tokens=num_tokens, layer_idx=layer_idx, final_softcapping=None
        )
        gen_config = {
            "precision": {LayerID("single_post", layer_idx): FileGenPrecision.A_BF16_W_INT8}
        }

        vlm_model.gen_files(
            FileGenMode.SOURCE_TO_QUANT, gen_config=gen_config, num_processes=1,
            log_level=logging.DEBUG, resume=False
        )
        outputs = model.run_model(EvalMode.SDK, ifms)

    # Output array contents should be the same.
    # However, outputs may be differently split into multiple arrays.
    output1 = np.concatenate([x.squeeze() for x in ref_outputs])
    output2 = np.concatenate([x.squeeze() for x in outputs])

    tolerance = tolerances[gguf_model][layer_idx]
    assert np.allclose(output1, output2, atol=tolerance * np.max(np.abs(output1)))
