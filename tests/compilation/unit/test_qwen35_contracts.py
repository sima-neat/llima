from types import SimpleNamespace

import numpy as np
import pytest

from sima_lmm.model import language_linear_model
from sima_lmm.model.language_linear_model import LanguageLinearModel
from sima_lmm.model.qwen_vision_model import QwenVisionLayerModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


class _StopGraphBuild(Exception):
    pass


class _FakeNode:
    def __init__(self, name):
        self.name = name


class _FakeSimaBuilder:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.subnet_input_names = []
        self.dynamic_dequant_inputs = None
        self.__class__.instances.append(self)

    def create_placeholder_node(self, name, _tensor_type):
        return _FakeNode(name)

    def begin_subnet(self, inputs):
        self.subnet_input_names = [node.name for node in inputs]

    def create_dynamic_dequant_node(self, input_node, scale_node):
        self.dynamic_dequant_inputs = (input_node.name, scale_node.name)
        return _FakeNode("dequantized_input")


def _linear_model(*, quantize_embeddings: bool, layer_idx: int = 0) -> LanguageLinearModel:
    model = object.__new__(LanguageLinearModel)
    model.num_tokens = 1
    model.layer_idx = layer_idx
    model.cfg = SimpleNamespace(
        pipeline_cfg=SimpleNamespace(quantize_embeddings=quantize_embeddings),
        lm_cfg=SimpleNamespace(
            hidden_size=16,
            linear_attn_cfg=SimpleNamespace(
                conv_kernel_dim=4,
                conv_dim=48,
                num_value_heads=1,
                num_key_heads=1,
                key_head_dim=16,
                value_head_dim=16,
            ),
        ),
    )
    return model


def test_linear_attention_adds_embedding_scale_only_for_quantized_layer_zero(monkeypatch):
    monkeypatch.setattr(language_linear_model, "SimaBuilder", _FakeSimaBuilder)

    def stop_after_input_contract(_self, _builder, _name, input_node):
        assert input_node.name in {"input", "dequantized_input"}
        raise _StopGraphBuild

    monkeypatch.setattr(
        LanguageLinearModel,
        "_build_sima_rms_norm",
        stop_after_input_contract,
    )

    _FakeSimaBuilder.instances.clear()
    with pytest.raises(_StopGraphBuild):
        _linear_model(quantize_embeddings=True)._build_sima_nodes("model.layers.0", False)
    quantized_builder = _FakeSimaBuilder.instances[-1]
    assert quantized_builder.subnet_input_names == [
        "input",
        "input_scale",
        "linear_conv_state",
        "linear_delta_state",
    ]
    assert quantized_builder.dynamic_dequant_inputs == ("input", "input_scale")
    assert _linear_model(quantize_embeddings=True).get_mla_input_tessellate_params() == {}
    assert _linear_model(quantize_embeddings=True).get_mla_output_tessellate_params() == {}

    with pytest.raises(_StopGraphBuild):
        _linear_model(quantize_embeddings=True, layer_idx=1)._build_sima_nodes(
            "model.layers.1", False
        )
    bf16_builder = _FakeSimaBuilder.instances[-1]
    assert bf16_builder.subnet_input_names == [
        "input",
        "linear_conv_state",
        "linear_delta_state",
    ]
    assert bf16_builder.dynamic_dequant_inputs is None


def test_qwen_patch_embedding_preserves_grouped_scales():
    model = object.__new__(QwenVisionLayerModel)
    model.cfg = SimpleNamespace(vm_cfg=SimpleNamespace(hidden_size=8))
    scales = np.arange(24, dtype=np.float32).reshape(8, 3)

    result = model._reshape_qwen_patch_embed_scales(scales)

    assert result is scales
    with pytest.raises(ValueError, match="Qwen patch-embedding scales"):
        model._reshape_qwen_patch_embed_scales(np.ones((7, 3), dtype=np.float32))
