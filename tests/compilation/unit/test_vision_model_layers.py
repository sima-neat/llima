import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from sima_lmm.config.vlm_config import VlmConfig
from sima_lmm.model import EvalMode
from sima_lmm.model.gemma4_vision_model import Gemma4VisionLayerModel
from sima_lmm.model.qwen_vision_model import QwenVisionLayerModel
from sima_lmm.model.vision_model import StandardVisionLayerModel, VisionModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

REFERENCE_CONFIGS_PATH = (
    Path(__file__).parents[1] / "configuration" / "references"
)


def _load_reference_config(filename: str) -> VlmConfig:
    config = json.loads((REFERENCE_CONFIGS_PATH / filename).read_text())
    return VlmConfig.load(config)


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
