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
"""Embedding quantization tests for LLM, VLM, and Gemma4 per-layer inputs."""

import copy
from pathlib import Path

import numpy as np
import pytest

from afe.ir.defines import NodeName, get_expected_tensor_value
from afe.ir.net import AwesomeNet
from afe.ir.operations import DynamicDequantOp
from afe.ir.sima_ir import SiMaIR
from afe.ir.tensor_type import ScalarType

from sima_lmm.model import EvalMode, VisionLanguageModel
from sima_lmm.model.language_model import LanguageModel
from sima_lmm.model.language_per_layer_model import LanguagePerLayerModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]

GEMMA4_MODEL_FOLDER = "models--google--gemma-4-E2B-it"
QWEN3_VL_MODEL_FOLDER = "models--Qwen--Qwen3-VL-2B-Instruct"


@pytest.fixture(scope="module")
def gemma4_model(model_inputs_path: Path, tmp_path_factory) -> VisionLanguageModel:
    model_path = require_readable_path(model_inputs_path / GEMMA4_MODEL_FOLDER)
    output_path = tmp_path_factory.mktemp("gemma4_embedding_quantization")
    return VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=output_path / "onnx",
        sima_path=output_path / "sima",
        max_num_tokens=1024,
        image_resolution=[240, 240],
        quantize_embeddings=True,
    )


@pytest.fixture(scope="module")
def qwen3_vl_model(model_inputs_path: Path, tmp_path_factory) -> VisionLanguageModel:
    model_path = require_readable_path(model_inputs_path / QWEN3_VL_MODEL_FOLDER)
    output_path = tmp_path_factory.mktemp("qwen3_vl_embedding_quantization")
    return VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=output_path / "onnx",
        sima_path=output_path / "sima",
        max_num_tokens=1024,
        image_resolution=[224, 224],
        quantize_embeddings=True,
    )


@pytest.fixture(params=["gemma4_model", "qwen3_vl_model"], ids=["gemma4", "qwen3-vl"])
def quantized_vlm_model(request) -> VisionLanguageModel:
    return request.getfixturevalue(request.param)


def _language_model(
    gemma4_model: VisionLanguageModel, tmp_path: Path, *, multimodal: bool
) -> LanguageModel:
    cfg = copy.deepcopy(gemma4_model.cfg)
    if not multimodal:
        cfg.vm_cfg = None
        cfg.mm_cfg = None
    return LanguageModel(
        cfg,
        f"gemma4_{'vlm' if multimodal else 'llm'}",
        onnx_path=tmp_path / "onnx",
        sima_path=tmp_path / "sima",
        hf_model=gemma4_model.hf_model,
        vlm_helper=gemma4_model.vlm_helper,
    )


def _all_nodes(net: AwesomeNet):
    for node in net.nodes.values():
        yield node
        if isinstance(node.ir, AwesomeNet):
            yield from _all_nodes(node.ir)


def _input_scalar_type(net: AwesomeNet, name: str) -> ScalarType:
    node = net.nodes[NodeName(name)]
    return get_expected_tensor_value(node.get_type().output).scalar


def _input_names(net: AwesomeNet) -> list[str]:
    return [str(name) for name in net.input_node_names]


def _mla_input_names(net: AwesomeNet) -> list[str]:
    mla_net = net.nodes[NodeName("MLA_0")].ir
    assert isinstance(mla_net, AwesomeNet)
    return [str(name) for name in mla_net.input_node_names]


@pytest.mark.parametrize("embedding_kind", ["llm", "vlm", "per_layer"])
def test_embedding_tensor_quantization(
    gemma4_model: VisionLanguageModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embedding_kind: str,
) -> None:
    multimodal = embedding_kind != "llm"
    language_model = _language_model(gemma4_model, tmp_path, multimodal=multimodal)
    weights = np.array(
        [
            [-1.0, -0.5, 0.0, 0.5],
            [0.25, 0.75, 1.0, -0.25],
            [0.125, -0.75, 0.625, -0.375],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(
        language_model.hf_model,
        "load_np_param",
        lambda _name: weights.copy(),
    )

    if embedding_kind == "per_layer":
        quantized, scale = language_model.get_per_layer_embeddings_tensor()
        embed_scale = language_model.cfg.lm_cfg.hidden_size_per_layer_input ** 0.5
    else:
        quantized, scale = language_model.get_input_embeddings_tensor()
        embed_scale = language_model.cfg.lm_cfg.hidden_size ** 0.5

    assert quantized.dtype == np.int8
    assert quantized.shape == weights.shape
    assert scale is not None
    assert scale.shape == (weights.shape[0], 1)
    np.testing.assert_allclose(
        quantized.astype(np.float32) * scale.astype(np.float32) / 127.0,
        weights * embed_scale,
        atol=float(np.max(scale.astype(np.float32))) / 127.0,
    )


def test_vlm_passes_quantized_embeddings_and_row_scales_to_decode(
    quantized_vlm_model: VisionLanguageModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    quantized_embeddings = np.array([[4, -8], [12, -16]], dtype=np.int8)
    embedding_scales = np.array([[0.25], [0.5]], dtype=np.float32)
    captured = {}
    monkeypatch.setattr(
        quantized_vlm_model.language_model,
        "get_embeddings_tensor",
        lambda: (quantized_embeddings, embedding_scales),
    )

    def capture_decode_embeddings(
        _eval_mode, ifms, *, embeddings_tensor, embedding_scales
    ):
        captured["ifms"] = ifms
        captured["embeddings"] = embeddings_tensor
        captured["scales"] = embedding_scales
        return []

    monkeypatch.setattr(quantized_vlm_model.language_model, "run_model", capture_decode_embeddings)

    quantized_vlm_model.run_model(EvalMode.SDK, [np.array([[0]], dtype=np.int32)])

    np.testing.assert_array_equal(captured["embeddings"], quantized_embeddings)
    np.testing.assert_array_equal(captured["scales"], embedding_scales)
    np.testing.assert_array_equal(captured["ifms"][0], quantized_embeddings[None, None, :1])
    np.testing.assert_array_equal(captured["ifms"][1], embedding_scales[None, None, :1])


def test_non_gemma4_vlm_graph_dequantizes_embedding_rows_on_mla(
    qwen3_vl_model: VisionLanguageModel,
) -> None:
    pre_model = qwen3_vl_model.language_model._get_part_model(
        "pre", num_tokens=1, layer_idx=0
    )
    assert isinstance(pre_model, LanguagePreModel)

    net = pre_model._build_sima_nodes(pre_model._layer_base_name, quantizable=False)

    assert _input_scalar_type(net, "input") == ScalarType.int8
    assert _input_scalar_type(net, "input_scale") == ScalarType.bfloat16
    assert _input_names(net) == ["input", "input_scale", "freq_real", "freq_imag"]
    assert _mla_input_names(net) == [
        "MLA_0/input", "MLA_0/input_scale", "MLA_0/freq_real", "MLA_0/freq_imag"
    ]
    assert any(
        isinstance(node.ir, SiMaIR)
        and isinstance(node.ir.operation, DynamicDequantOp)
        for node in _all_nodes(net)
    )

    post_model = qwen3_vl_model.language_model._get_part_model(
        "post", num_tokens=1, layer_idx=0
    )
    assert isinstance(post_model, LanguagePostModel)
    post_net = post_model._build_sima_nodes(
        post_model._layer_base_name, quantizable=False
    )
    assert _input_names(post_net) == ["input", "input_scale", "self_attn"]
    assert _mla_input_names(post_net) == [
        "MLA_0/input", "MLA_0/input_scale", "MLA_0/self_attn"
    ]


@pytest.mark.parametrize("multimodal", [True, False], ids=["vlm", "llm"])
def test_per_layer_graph_embedding_inputs(
    gemma4_model: VisionLanguageModel,
    tmp_path: Path,
    multimodal: bool,
) -> None:
    language_model = _language_model(gemma4_model, tmp_path, multimodal=multimodal)
    model = LanguagePerLayerModel(
        language_model.cfg,
        f"gemma4_{'vlm' if multimodal else 'llm'}_per_layer",
        onnx_path=language_model.onnx_path,
        sima_path=language_model.sima_path,
        hf_model=language_model.hf_model,
        num_tokens=1,
    )

    net = model._build_sima_nodes(
        language_model.hf_model.language_model_param_base_name,
        quantizable=False,
    )

    assert _input_scalar_type(net, "per_layer_emb_staging") == ScalarType.int8
    assert _input_scalar_type(net, "input") == ScalarType.int8
    assert _input_scalar_type(net, "per_layer_emb_staging_scale") == ScalarType.bfloat16
    assert _input_scalar_type(net, "input_scale") == ScalarType.bfloat16
    assert _input_names(net) == [
        "per_layer_emb_staging",
        "per_layer_emb_staging_scale",
        "input",
        "input_scale",
    ]
    assert _mla_input_names(net) == [
        "MLA_0/per_layer_emb_staging",
        "MLA_0/per_layer_emb_staging_scale",
        "MLA_0/input",
        "MLA_0/input_scale",
    ]
    dequant_nodes = [
        node for node in _all_nodes(net)
        if isinstance(node.ir, SiMaIR)
        and isinstance(node.ir.operation, DynamicDequantOp)
    ]
    assert len(dequant_nodes) == 2
