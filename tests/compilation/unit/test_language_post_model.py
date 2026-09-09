from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from sima_lmm.config.vlm_config import VlmArchType
from sima_lmm.model.language_post_model import LanguagePostModel

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


class _OnnxBuilder:
    def __init__(self):
        self.ops = []

    def build_conv_from_dense_with_lora(self, *_args, **_kwargs):
        return "attention_projection"

    def create_initializer(self, name, value):
        assert name == "model.layers.0.layer_scalar"
        np.testing.assert_array_equal(value, np.array([[[[0.25]]]], dtype=np.float32))
        return "layer_scalar"

    def build_op(self, name, inputs, op_type, **_kwargs):
        self.ops.append((name, inputs, op_type))
        return name


def _make_post_model() -> LanguagePostModel:
    cfg = SimpleNamespace(
        model_type=VlmArchType.VLM_GEMMA4,
        vm_cfg=None,
        lm_cfg=SimpleNamespace(
            num_hidden_layers=2,
            hidden_size_per_layer_input=0,
            lora_cfg=None,
            mlp_cfg=SimpleNamespace(act="gelu_pytorch_tanh"),
        ),
    )
    model = LanguagePostModel(
        cfg,
        "assistant_language_n1_post_layer0",
        hf_model=SimpleNamespace(language_model_param_base_name="model"),
        num_tokens=1,
        layer_idx=0,
        final_softcapping=None,
    )
    model.check_hf_param = lambda name: name.endswith(
        ("out_proj.weight", "layer_scalar")
    )
    model.get_hf_param = lambda _name: np.array([0.25], dtype=np.float32)
    model._build_onnx_pre_projection_if_needed = lambda node: node
    model.has_ffn_layernorms = lambda _base_name: True
    model._build_rms_norm = lambda name, node: (name, node)
    model._build_onnx_mlp = lambda base_name, nodes: (base_name, nodes)
    return model


def test_onnx_post_applies_layer_scalar_without_per_layer_inputs():
    model = _make_post_model()
    builder = _OnnxBuilder()
    model._onnx_builder = builder

    outputs = model._build_onnx_nodes("model.layers.0", ["input", "attention"])

    assert outputs == ["model.layers.0.layer_scalar_mul"]
    assert builder.ops[-1] == (
        "model.layers.0.layer_scalar_mul",
        ["model.layers.0.add2", "layer_scalar"],
        "Mul",
    )


def test_sima_layer_scalar_uses_checkpoint_value():
    model = _make_post_model()
    builder = Mock()
    builder.create_constant_node.return_value = "layer_scalar"
    builder.create_mul_node.return_value = "scaled"

    output = model._build_sima_layer_scalar_if_needed(
        builder, "model.layers.0", "hidden", quantizable=True
    )

    assert output == "scaled"
    scalar = builder.create_constant_node.call_args.args[0]
    np.testing.assert_array_equal(scalar, np.array([0.25], dtype=np.float32))
    builder.create_mul_node.assert_called_once_with("hidden", "layer_scalar")
