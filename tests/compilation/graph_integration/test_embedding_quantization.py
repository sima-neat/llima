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
from afe.ir.operations import DequantizationTransformOp
from afe.ir.sima_ir import SiMaIR
from afe.ir.tensor_type import ScalarType

from sima_lmm.config.layer_id import LayerID
from sima_lmm.model import EvalMode, FileGenMode, FileGenPrecision, VisionLanguageModel
from sima_lmm.model.language_model import LanguageModel
from sima_lmm.model.language_per_layer_model import LanguagePerLayerModel
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]

MODEL_FOLDER = "models--google--gemma-4-E2B-it"
NORMAL_SCALE = 0.25
PER_LAYER_SCALE = 0.5


@pytest.fixture(scope="module")
def gemma4_model(model_inputs_path: Path, tmp_path_factory) -> VisionLanguageModel:
    model_path = require_readable_path(model_inputs_path / MODEL_FOLDER)
    output_path = tmp_path_factory.mktemp("gemma4_embedding_quantization")
    return VisionLanguageModel.from_hf_cache(
        hf_cache_path=model_path,
        model_name=model_path.name,
        onnx_path=output_path / "onnx",
        sima_path=output_path / "sima",
        max_num_tokens=128,
        image_resolution=[240, 240],
        quantize_embeddings=True,
    )


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
    np.testing.assert_allclose(
        quantized.astype(np.float32) * scale,
        weights * embed_scale,
        atol=scale / 2,
    )


def test_vlm_passes_dequantized_embeddings_to_decode(
    gemma4_model: VisionLanguageModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    quantized_embeddings = np.array([[4, -8], [12, -16]], dtype=np.int8)
    captured = {}
    monkeypatch.setattr(
        gemma4_model.language_model,
        "get_embeddings_tensor",
        lambda: (quantized_embeddings, 0.25),
    )

    def capture_decode_embeddings(_eval_mode, _ifms, *, embeddings_tensor):
        captured["embeddings"] = embeddings_tensor
        return []

    monkeypatch.setattr(gemma4_model.language_model, "run_model", capture_decode_embeddings)

    gemma4_model.run_model(EvalMode.SDK, [np.array([[0]], dtype=np.int32)])

    np.testing.assert_allclose(
        captured["embeddings"], quantized_embeddings.astype(np.float32) * 0.25
    )


@pytest.mark.parametrize(
    "multimodal,normal_input_dtype,num_dequant_nodes,expected_factors",
    [
        (True, ScalarType.float32, 1, [1 / PER_LAYER_SCALE]),
        (False, ScalarType.int8, 2, [1 / PER_LAYER_SCALE, 1 / NORMAL_SCALE]),
    ],
    ids=["vlm", "llm"],
)
def test_per_layer_graph_embedding_inputs(
    gemma4_model: VisionLanguageModel,
    tmp_path: Path,
    multimodal: bool,
    normal_input_dtype: ScalarType,
    num_dequant_nodes: int,
    expected_factors: list[float],
) -> None:
    language_model = _language_model(gemma4_model, tmp_path, multimodal=multimodal)
    model = LanguagePerLayerModel(
        language_model.cfg,
        f"gemma4_{'vlm' if multimodal else 'llm'}_per_layer",
        onnx_path=language_model.onnx_path,
        sima_path=language_model.sima_path,
        hf_model=language_model.hf_model,
        num_tokens=1,
        embeddings_scale=NORMAL_SCALE,
        per_layer_embeddings_scale=PER_LAYER_SCALE,
    )

    net = model._build_sima_nodes(
        language_model.hf_model.language_model_param_base_name,
        quantizable=True,
    )

    assert _input_scalar_type(net, "per_layer_emb_staging") == ScalarType.int8
    assert _input_scalar_type(net, "input") == normal_input_dtype
    dequant_nodes = [
        node for node in _all_nodes(net)
        if isinstance(node.ir, SiMaIR)
        and isinstance(node.ir.operation, DequantizationTransformOp)
    ]
    assert len(dequant_nodes) == num_dequant_nodes
    factors = sorted(node.ir.get_attrs().channel_params[0][0] for node in dequant_nodes)
    assert factors == sorted(expected_factors)


@pytest.mark.parametrize("multimodal", [True, False], ids=["vlm", "llm"])
def test_per_layer_embedding_scale_wiring(
    gemma4_model: VisionLanguageModel,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    multimodal: bool,
) -> None:
    language_model = _language_model(gemma4_model, tmp_path, multimodal=multimodal)
    language_model._embeddings_scale = NORMAL_SCALE
    language_model._per_layer_embeddings_scale = PER_LAYER_SCALE
    generated_models = []
    monkeypatch.setattr(
        language_model,
        "gen_files_from_model_list",
        lambda model_list, *_args: generated_models.extend(model_list),
    )
    gen_config = {
        "precision": {LayerID("single_per_layer", 0): FileGenPrecision.BF16},
    }

    language_model.gen_files(FileGenMode.SOURCE_TO_FP, gen_config=gen_config)

    assert len(generated_models) == 1
    per_layer_model = generated_models[0][0]
    assert per_layer_model.per_layer_embeddings_scale == PER_LAYER_SCALE
    assert per_layer_model.embeddings_scale == (None if multimodal else NORMAL_SCALE)
