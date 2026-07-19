#########################################################
# Copyright (C) 2025 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
from pathlib import Path
from dataclasses import asdict
import json
import tempfile
import gguf

import pytest
import numpy as np

from sima_lmm.config.vlm_config import (
    GgufFileType, ModelFormat, VlmConfig, _load_json_file, _write_json_file
)
from sima_lmm.gguf.gguf_conversion import (
    DEFAULT_HF_GGUF_WEIGHT_MAP, GgufModel, unpack_gguf_tensor
)
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from tests.conftest import require_readable_path

_MODELS = {
    "Q8_0": "unsloth-gemma-3-1b-it-Q8_0.gguf",
    "Q4_0": "google-gemma-3-1b-it-QAT-Q4_0.gguf",
}

_UNPACK_REF = {
    "BF16": "unsloth-gemma-3-1b-it-BF16.gguf",
    "Q8_0": "unsloth-gemma-3-1b-it-Q8_0.gguf",
    "Q4_0": "unsloth-gemma-3-1b-it-Q4_0.gguf",
    "Q6_K": "unsloth-gemma-3-1b-it-Q6_K.gguf",
    "Q5_K": "unsloth-gemma-3-1b-it-Q5_K_S.gguf",
    "Q4_K": "unsloth-gemma-3-1b-it-Q4_K_S.gguf",
    "Q3_K": "unsloth-gemma-3-1b-it-Q3_K_S.gguf"
}

_GENERAL_FILE_TYPE = {
    "BF16": GgufFileType.BF16,
    "Q8_0": GgufFileType.Q8_0,
    "Q4_0": GgufFileType.Q4_0
}

_QUERIES_ANSWERS = [
    {
        "query": "What is capital city of Argentina?",
        "answer": "Buenos Aires"
    },
]


def _convert_with_default_gguf_map(name: str) -> str:
    for hf_name, gguf_name in DEFAULT_HF_GGUF_WEIGHT_MAP.items():
        if hf_name in name:
            name = name.replace(hf_name, gguf_name)
    return name


@pytest.mark.premerge
def test_attention_bias_weight_map():
    names = {
        "model.layers.0.self_attn.q_proj.bias": "blk.0.attn_q.bias",
        "model.layers.0.self_attn.k_proj.bias": "blk.0.attn_k.bias",
        "model.layers.0.self_attn.v_proj.bias": "blk.0.attn_v.bias",
    }
    for hf_name, gguf_name in names.items():
        assert _convert_with_default_gguf_map(hf_name) == gguf_name


@pytest.mark.premerge
@pytest.mark.parametrize("ggml_type", _MODELS.keys(), ids=lambda x: 'ggml_type=' + str(x))
def test_parser(gguf_models_path: Path, ggml_type: str):
    model_path = require_readable_path(gguf_models_path / _MODELS[ggml_type])
    model = GgufModel(model_path)
    cfg = model.model_config
    assert cfg.get("data_type") == _GENERAL_FILE_TYPE[ggml_type]


# Value of (atol, rtol) for calling np.allclose in test_unpack.
# These thresholds are set for unsloth-gemma-3-1b-it models.
TEST_UNPACK_TOLERANCE = {
    "Q8_0": ((0.01, 0.05), (0.01, 0.02)),
    "Q4_0": ((0.05, 0.18), (0.02, 0.05)),
    "Q6_K": ((0.04, 0.06), (0.02, 0.03)),
    "Q5_K": ((0.04, 0.06), (0.03, 0.04)),
    "Q4_K": ((0.06, 0.10), (0.05, 0.05)),
    "Q3_K": ((0.08, 0.14), (0.08, 0.05))
}


def _dequantize_symmetric(scale: np.ndarray, quants: np.ndarray) -> np.ndarray:
    """
    Dequantize a symmetric quantized tensor.  The quantization of the input
    is the same as what unpack_gguf_tensor returns.

    Args:
        scale: Block scales as an array of shape (M*N, 1).
        quants: Quantized values as an array of shape (M, N*B) where
            B is the quantization block size.

    Returns:
        Dequantized array having the same shape as quants.
    """
    assert scale.ndim == 2
    assert scale.shape[1] == 1
    shape = quants.shape
    dq = scale * quants.reshape((scale.shape[0], -1))
    return dq.reshape(shape).astype(np.float32, copy=False)


@pytest.mark.premerge
def test_unpack_q2_k(gguf_models_path: Path):
    """
    Verify that a tensor in Q2_K format can be dequantized correctly.
    Because the worst-case quantization error is quite large, this test
    checks average similarity with FP instead of worst-case similarity.
    """
    # Load floating-point reference tensor
    data_f = np.load(require_readable_path(gguf_models_path / "testdata_q2_k_float.npy"))
    # Load quantized tensor
    data_q = np.load(require_readable_path(gguf_models_path / "testdata_q2_k_quant.npy"))

    # Create the data structure used by GGUF loader.
    # Fields tensor_type, shape, and data are relevant for this test.
    tensor_q = gguf.ReaderTensor(
        name="blk.0.attn_qkv_weight",
        tensor_type=gguf.GGMLQuantizationType.Q2_K,
        shape=np.array([3072, 9216], dtype=np.uint32),
        n_elements=28311552, n_bytes=9289728, data_offset=0,
        data=data_q, field=gguf.ReaderField(0, "")
    )

    # Convert reference tensor to the right shape
    wf = data_f.reshape((9216, 3072)).astype(np.float32, copy=False)

    # Dequantize using GGUF library
    _, wqf = unpack_gguf_tensor(tensor_q, True)

    # Dequantize using algorithm under test
    scale, wq = unpack_gguf_tensor(tensor_q, False)
    wqdq = _dequantize_symmetric(scale, wq)

    assert np.mean(np.abs(wqdq - wf)) < 0.01, \
        "Dequantized tensor does not match reference tensor"
    assert np.allclose(wqdq, wqf, rtol=0.05, atol=0.1), \
        "Dequantized tensor does not match GGUF library dequantized tensor"


@pytest.mark.premerge
@pytest.mark.parametrize("ggml_type", ["Q8_0", "Q4_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K"], ids=lambda x: 'ggml_type=' + str(x))
def test_unpack(gguf_models_path: Path, ggml_type: str):
    model_f = GgufModel(require_readable_path(gguf_models_path / _UNPACK_REF["BF16"]))
    model_q = GgufModel(require_readable_path(gguf_models_path / _UNPACK_REF[ggml_type]))
    (atol_f, rtol_f), (atol_q, rtol_q) = TEST_UNPACK_TOLERANCE[ggml_type]

    for name in model_f.tensor_info:
        _, wqf = model_q.load_weight(name, False, force_float=True)
        _, wf = model_f.load_weight(name, False)
        bc, wq = model_q.load_weight(name, False)
        assert wf.shape == wq.shape

        if bc is not None:
            wqdq = _dequantize_symmetric(bc, wq)
        else:
            wqdq = wq.astype(np.float32)

        assert np.allclose(wqdq, wf.astype(np.float32), rtol=rtol_f, atol=atol_f), \
            "Dequantized tensor does not match reference tensor"
        assert np.allclose(wqdq, wqf.astype(np.float32), rtol=rtol_q, atol=atol_q), \
            "Dequantized tensor does not match GGUF library dequantized tensor"


@pytest.mark.premerge
@pytest.mark.parametrize("ggml_type", _MODELS.keys(), ids=lambda x: 'ggml_type=' + str(x))
def test_vlm_config(gguf_models_path: Path, ggml_type: str):
    model_path = require_readable_path(gguf_models_path / _MODELS[ggml_type])
    model = GgufModel(model_path)
    model_cfg = model.model_config

    vlm_cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_GGUF, model_path, model_cfg)

    max_num_tokens = 512
    language_group_size = 128
    future_token_mask_size = 128

    vlm_cfg.config_pipeline(
        "Be helpful and polite.", None, max_num_tokens,
        language_group_size, future_token_mask_size
    )
    assert vlm_cfg.pipeline_cfg.input_token_group_offsets == [0, 128, 256, 384]

    tempfilename = (
        f"sima-{vlm_cfg.model_type}-{vlm_cfg.lm_cfg.arch}.json"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        json_vlm = json.dumps(asdict(vlm_cfg), indent=4)
        _write_json_file(json_vlm, Path(f"{tmpdir}/{tempfilename}"))
        sima_dict = _load_json_file(Path(f"{tmpdir}/{tempfilename}"))
        sima_vlm = VlmConfig.load(sima_dict)

    assert sima_vlm == vlm_cfg


@pytest.mark.premerge
@pytest.mark.parametrize("ggml_type", _MODELS.keys(), ids=lambda x: 'ggml_type=' + str(x))
@pytest.mark.parametrize("qna_idx", [0], ids=lambda x: 'qna_idx=' + str(x))
def test_inference(gguf_models_path: Path, ggml_type: str, qna_idx: int):
    model_path = require_readable_path(gguf_models_path / _MODELS[ggml_type])
    model = GgufModel(model_path)
    query = _QUERIES_ANSWERS[qna_idx]["query"]
    out_text = model.execute_llama_cpp(query)
    assert _QUERIES_ANSWERS[qna_idx]["answer"] in out_text


@pytest.mark.premerge
@pytest.mark.parametrize("ggml_type", _UNPACK_REF.keys(), ids=lambda x: 'ggml_type=' + str(x))
def test_load_weights(gguf_models_path: Path, gguf_hf_model_path: Path, ggml_type: str):
    reference_hf_cache = LocalHuggingFaceModel.create_from_directory(
        directory=require_readable_path(gguf_hf_model_path),
        layer_names=None,
    )

    model_path = require_readable_path(gguf_models_path / _UNPACK_REF[ggml_type])
    model = GgufModel(model_path)

    for name in reference_hf_cache.weight_map.keys():
        w_hf = reference_hf_cache.load_np_param(name)
        _, w_gguf = model.load_weight(name, is_hf_name=True)
        assert w_hf.shape == w_gguf.shape, f"Cannot load weight from GGUF: {name}"
