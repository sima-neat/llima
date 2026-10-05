import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from ml_dtypes import int4

from afe.ir.execute import create_node_executor, create_node_quant_executor
from afe.ir.operations import ConvAddActivationOp

from sima_lmm.config.vlm_config import VlmConfig
from sima_lmm.model import EvalMode
from sima_lmm.model.gemma4_vision_model import Gemma4VisionLayerModel
from sima_lmm.model.qwen_vision_model import QwenVisionLayerModel
from sima_lmm.model.vision_model import StandardVisionLayerModel, VisionModel
from sima_lmm.model.model_graph import ModelGraph, activation_dtype


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

REFERENCE_CONFIGS_PATH = (
    Path(__file__).parents[1] / "configuration" / "references"
)


def _load_reference_config(filename: str) -> VlmConfig:
    config = json.loads((REFERENCE_CONFIGS_PATH / filename).read_text())
    return VlmConfig.load(config)


@pytest.mark.parametrize("head_dim", [24, 32])
@pytest.mark.parametrize("precision", ["float32", "int8", "int4"])
def test_standard_vision_head_padding_preserves_attention_and_grouped_weights(head_dim, precision):
    heads, channels = 2, 2 * head_dim
    rng = np.random.default_rng(12)
    params, dequantized = {}, {}
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
        weight = rng.integers(-7, 8, (channels, channels))
        bias = rng.normal(0, 0.02, channels).astype(np.float32)
        if precision == "float32":
            source = weight.astype(np.float32) * 0.005
            dequantized[projection] = source
        else:
            scales = np.full((channels, 2 if precision == "int4" else 1), 0.005, np.float32)
            weight = weight.astype(int4 if precision == "int4" else np.int8)
            source = (scales, weight, 32) if precision == "int4" else (scales, weight)
            dequantized[projection] = weight.astype(np.float32) * 0.005
        params[f"attn.{projection}.weight"] = source
        params[f"attn.{projection}.bias"] = bias
    cfg = SimpleNamespace(vm_cfg=SimpleNamespace(num_attention_heads=heads, hidden_size=channels))
    model = StandardVisionLayerModel(
        cfg, "head_padding", layer_idx=0, include_embeddings=False, include_mm_proj=False
    )
    model.get_hf_param, model.check_hf_param = params.__getitem__, params.__contains__
    shape = (1, 1, 3, channels)
    quantizable = precision == "float32"
    graph = ModelGraph(model, {"x": shape}, quantizable)
    output = model._build_sima_encoder_attention(graph, "attn", graph.inputs["x"])
    net = graph.finish([output])
    convolutions = [
        node for node in net.nodes["MLA_0"].ir.nodes.values()
        if isinstance(node.ir.operation, ConvAddActivationOp)
    ]
    padded = head_dim == 24 and precision != "int4"
    assert len(convolutions) == (10 if head_dim == 24 and not padded else 4)
    if precision == "int4":
        assert all(node.ir.quant_attrs.c_block_size == 32 for node in convolutions[:1])
        np.testing.assert_array_equal(
            convolutions[0].ir.quant_attrs.weight_quant_data.reshape(channels, channels),
            params["attn.q_proj.weight"][1].T,
        )
    x = rng.normal(0, 0.2, shape).astype(activation_dtype(quantizable))
    projections = []
    for name in ("q_proj", "k_proj", "v_proj"):
        values = x.astype(np.float32) @ dequantized[name].T + params[f"attn.{name}.bias"]
        if name == "q_proj":
            values *= head_dim ** -0.5
        projections.append(values.reshape(1, 3, heads, head_dim).transpose(0, 2, 1, 3))
    query, key, value = projections
    scores = query @ key.swapaxes(-1, -2)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    context = (probabilities @ value).transpose(0, 2, 1, 3).reshape(shape)
    expected = context @ dequantized["out_proj"].T + params["attn.out_proj.bias"]
    execute = create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    actual = net.run({"x": x}, node_callable=execute)
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    np.testing.assert_allclose(actual, expected, rtol=3e-6 if quantizable else 0.03,
                               atol=2e-7 if quantizable else 0.002)


@pytest.mark.parametrize(
    ("config_name", "expected_type"),
    [
        ("gemma3_siglip448_vlm_config.json", StandardVisionLayerModel),
        ("lfm2_vl_vlm_config.json", StandardVisionLayerModel),
        ("gemma4_e2b_it_vlm_config.json", Gemma4VisionLayerModel),
        ("qwen2.5_vl_vlm_config.json", QwenVisionLayerModel),
        ("qwen3_vl_vlm_config.json", QwenVisionLayerModel),
    ],
)
def test_vision_parts_put_boundaries_in_first_and_last_layers(
    config_name: str, expected_type: type
):
    config = _load_reference_config(config_name)
    vision_model = VisionModel(config, "test_vision")
    last_layer = config.num_vision_layers - 1

    first = vision_model._get_part_model(0)
    middle = vision_model._get_part_model(last_layer // 2)
    last = vision_model._get_part_model(last_layer)

    assert isinstance(first, expected_type)
    assert first.model_name == "test_vision_layer0"
    assert first.layer_idx == 0
    assert first.include_embeddings
    assert not first.include_mm_proj

    assert isinstance(middle, expected_type)
    assert middle.layer_idx == last_layer // 2
    assert not middle.include_embeddings
    assert not middle.include_mm_proj

    assert isinstance(last, expected_type)
    assert last.model_name == f"test_vision_layer{last_layer}"
    assert last.layer_idx == last_layer
    assert not last.include_embeddings
    assert last.include_mm_proj


def test_vision_part_rejects_out_of_range_layer():
    config = _load_reference_config("gemma4_e2b_it_vlm_config.json")
    vision_model = VisionModel(config, "test_vision")

    with pytest.raises(ValueError, match="outside the valid range"):
        vision_model._get_part_model(config.vm_cfg.num_hidden_layers)


def test_vision_evaluation_chains_layers_and_preserves_deepstack_order():
    config = _load_reference_config("qwen3_vl_vlm_config.json")
    config.vm_cfg.num_hidden_layers = 3
    config.vm_cfg.deepstack_visual_indexes = [1]
    vision_model = VisionModel(config, "test_vision")

    image = object()
    hidden_0 = object()
    hidden_1 = object()
    projection = object()
    scale = object()
    deepstack = object()
    layers = [Mock(), Mock(), Mock()]
    layers[0].run_model.return_value = [hidden_0]
    layers[1].run_model.return_value = [hidden_1, deepstack]
    layers[2].run_model.return_value = [projection, scale]
    vision_model._get_part_model = Mock(side_effect=layers)

    assert vision_model.run_model(EvalMode.SDK, [image]) == [
        projection,
        scale,
        deepstack,
    ]
    layers[0].run_model.assert_called_once_with(EvalMode.SDK, [image])
    layers[1].run_model.assert_called_once_with(EvalMode.SDK, [hidden_0])
    layers[2].run_model.assert_called_once_with(EvalMode.SDK, [hidden_1])


def test_qwen2_layer_uses_its_source_block_and_attention_mode():
    config = _load_reference_config("qwen2.5_vl_vlm_config.json")
    model = VisionModel(config, "test_vision")._get_part_model(7)
    model._onnx_builder = Mock()
    model._onnx_builder.build_conv = Mock(side_effect=AssertionError("unexpected embedding"))
    global_mask = object()
    windowed_mask = object()
    model._prepare_qwen2_static_inputs = Mock(
        return_value=(object(), object(), global_mask, windowed_mask)
    )
    layer_output = object()
    model._build_qwen2_vision_block = Mock(return_value=layer_output)
    model._build_qwen2_merger = Mock(side_effect=AssertionError("unexpected merger"))
    layer_input = object()

    assert model._build_qwen2_vision_model("vision", [layer_input]) is layer_output
    args = model._build_qwen2_vision_block.call_args.args
    assert args[0] == "vision.blocks.7"
    assert args[1] is layer_input
    assert args[2] is global_mask


def test_qwen3_layer_emits_its_deepstack_output_without_final_merger():
    config = _load_reference_config("qwen3_vl_vlm_config.json")
    model = VisionModel(config, "test_vision")._get_part_model(5)
    model._onnx_builder = Mock()
    model._onnx_builder.build_conv = Mock(side_effect=AssertionError("unexpected embedding"))
    model._prepare_qwen3_rotary_tables = Mock(return_value=(object(), object()))
    model._prepare_qwen3_position_embedding = Mock(
        side_effect=AssertionError("unexpected position embedding")
    )
    layer_output = object()
    deepstack_output = object()
    model._build_qwen3_vision_block = Mock(return_value=layer_output)
    model._build_qwen3_deepstack_merger = Mock(return_value=deepstack_output)
    model._build_qwen3_merger = Mock(side_effect=AssertionError("unexpected final merger"))

    assert model._build_qwen3_vision_model("vision", [object()]) == [
        layer_output,
        deepstack_output,
    ]
    assert model._build_qwen3_vision_block.call_args.args[0] == "vision.blocks.5"
    assert model._build_qwen3_deepstack_merger.call_args.args[0] == (
        "vision.deepstack_merger_list.0"
    )


def test_qwen3_direct_layer_emits_its_deepstack_output():
    config = _load_reference_config("qwen3_vl_vlm_config.json")
    model = VisionModel(config, "test_vision")._get_part_model(11)
    builder = Mock()
    model._prepare_sima_qwen3_rotary_tables = Mock(return_value=(object(), object()))
    model._prepare_sima_qwen3_position_embedding = Mock(
        side_effect=AssertionError("unexpected position embedding")
    )
    layer_output = object()
    deepstack_output = object()
    model._build_sima_qwen3_vision_block = Mock(return_value=layer_output)
    model._build_sima_qwen3_deepstack_merger = Mock(return_value=deepstack_output)
    model._build_sima_qwen3_merger = Mock(
        side_effect=AssertionError("unexpected final merger")
    )

    assert model._build_sima_qwen3_vision_model(
        builder, "vision", object(), quantizable=False
    ) == [layer_output, deepstack_output]
    assert model._build_sima_qwen3_vision_block.call_args.args[1] == "vision.blocks.11"
    assert model._build_sima_qwen3_deepstack_merger.call_args.args[1] == (
        "vision.deepstack_merger_list.1"
    )


@pytest.mark.parametrize(
    ("config_name", "layer_idx"),
    [
        ("gemma4_e2b_it_vlm_config.json", 1),
        ("qwen2.5_vl_vlm_config.json", 1),
        ("qwen3_vl_vlm_config.json", 1),
    ],
)
def test_nonfirst_onnx_layer_uses_hidden_state_shapes(
    config_name: str, layer_idx: int
):
    config = _load_reference_config(config_name)
    model = VisionModel(config, "test_vision")._get_part_model(layer_idx)
    model.hf_model = Mock(vision_model_param_base_name="vision")
    builder = Mock()
    builder.input_nodes = [object()]
    builder.get_node_output_name.return_value = "output"
    model.create_onnx_builder = Mock(side_effect=lambda: setattr(model, "_onnx_builder", builder))
    model._build_onnx_nodes = Mock(return_value=[object()])

    model.gen_onnx_files()

    builder.create_input_node.assert_called_once_with(
        "input", (1, config.vm_cfg.hidden_size, 1, config.vm_cfg.seq_len)
    )
    builder.create_output_node.assert_called_once_with(
        "output", (1, config.vm_cfg.hidden_size, 1, config.vm_cfg.seq_len)
    )
