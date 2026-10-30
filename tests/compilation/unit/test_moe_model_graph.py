"""Native MoE interfaces, weight extraction and routing numerics."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from ml_dtypes import bfloat16

from afe.ir.execute import create_node_executor, create_node_quant_executor
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.config.vlm_config import LlmArchType, PipelineConfig
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_moe_router_model import LanguageMoeRouterModel
from sima_lmm.model.language_moe_weightedsum_model import LanguageMoeWeightedSumModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

HIDDEN = 32
EXPERTS = 4
INTERMEDIATE = 32
BASE = "model.layers.0"


def _config(arch, *, norm_topk_prob=False, quantize_embeddings=False):
    attn = SimpleNamespace(
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, attn_output_gate=False,
        get_head_dim=lambda _: 16, get_q_size=lambda _: 32, get_kv_size=lambda _: 16,
    )
    lm = SimpleNamespace(
        arch=arch, num_hidden_layers=2, hidden_size=HIDDEN, layer_types=["full_attention"] * 2,
        attn_cfg=attn, lora_cfg=None, get_lora_rank=lambda *_: pytest.fail("LoRA lookup requires an adapter"),
        rms_norm_eps=1e-5, rms_norm_unit_offset=False,
        mlp_cfg=SimpleNamespace(act="silu", swiglu_limit=7.0 if arch == LlmArchType.GPT_OSS else None),
        get_effective_intermediate_size=lambda _: INTERMEDIATE,
        moe_cfg=SimpleNamespace(num_experts=EXPERTS, num_experts_per_tok=2, norm_topk_prob=norm_topk_prob),
        speculative_decoding_cfg=None, is_kv_shared_layer=lambda _: False,
        rope_cfg=SimpleNamespace(
            get_rope_dimension_count=lambda _: 16,
            rope_scaling=SimpleNamespace(rope_type="default"),
        ),
        lm_head_num_splits=1, lm_head_split_dim=64, draft_vocab_size=0,
        token_cfg=SimpleNamespace(vocab_size=64),
    )
    pipeline = PipelineConfig()
    pipeline.quantize_embeddings = quantize_embeddings
    pipeline.quantize_kv_cache = False
    pipeline.return_logits = True
    return SimpleNamespace(lm_cfg=lm, pipeline_cfg=pipeline, model_type="llm-test", vm_cfg=None)


def _model(cls, cfg, params, path, **kwargs):
    model = cls(cfg, "moe_component", sima_path=path, hf_model=SimpleNamespace(language_model_param_base_name="model"), **kwargs)
    model.get_hf_param = params.__getitem__
    model.check_hf_param = params.__contains__
    return model


def _run(model, values, quantizable):
    with patch.object(ModelGraph, "save", autospec=True) as save:
        model.generate_graph({}, quantizable)
    graph, outputs = save.call_args.args
    net = graph.finish(outputs)
    execute = create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    inputs = {
        name: value.astype(np.int8 if value.dtype == np.int8 else np.float32 if quantizable else bfloat16)
        for name, value in values.items()
    }
    outputs = net.run(inputs, node_callable=execute)
    return list(outputs) if isinstance(outputs, (tuple, list)) else [outputs], net


def _softmax(x):
    values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("arch,norm", [(LlmArchType.GPT_OSS, False), (LlmArchType.OLMOE, False), (LlmArchType.OLMOE, True)])
def test_moe_router_outputs_and_embedding_scale(tmp_path, arch, norm, tokens, quantizable):
    cfg = _config(arch, norm_topk_prob=norm, quantize_embeddings=not quantizable)
    gate_name = "router" if arch == LlmArchType.GPT_OSS else "gate"
    params = {
        f"{BASE}.self_attn.o_proj.weight": np.eye(HIDDEN, dtype=np.float32),
        f"{BASE}.post_attention_layernorm.weight": np.ones(HIDDEN, np.float32),
        f"{BASE}.mlp.{gate_name}.weight": np.zeros((EXPERTS, HIDDEN), np.float32),
        f"{BASE}.mlp.{gate_name}.bias": np.array([-2, -1, 1, 3], np.float32),
    }
    model = _model(LanguageMoeRouterModel, cfg, params, tmp_path, num_tokens=tokens, layer_idx=0)
    values = {
        "input": np.full((1, 1, tokens, HIDDEN), 1, np.float32),
    }
    if not quantizable:
        values["input"] = np.full((1, 1, tokens, HIDDEN), 4, np.int8)
        # Dynamic-dequant scales represent the maximum magnitude, divided by 127.
        values["input_scale"] = np.full((1, 1, tokens, 1), 31.75, np.float32)
    values["self_attn"] = np.full((1, 1, tokens, HIDDEN), 0.5, np.float32)
    outputs, net = _run(model, values, quantizable)
    assert list(net.input_node_names) == list(values)
    weights, indices, residual, norm_hidden = outputs
    assert indices.dtype == np.int32
    np.testing.assert_array_equal(indices, np.broadcast_to([3, 2], (1, 1, tokens, 2)))
    expected = _softmax(params[f"{BASE}.mlp.{gate_name}.bias"])
    expected = expected[[3, 2]] if arch == LlmArchType.OLMOE else _softmax(np.array([3, 1]))
    if norm:
        expected /= expected.sum()
    np.testing.assert_allclose(weights, np.broadcast_to(expected, weights.shape), rtol=0.015, atol=0.001)
    np.testing.assert_array_equal(residual, np.full(residual.shape, 1.5, np.float32))
    np.testing.assert_allclose(norm_hidden, 1.0, atol=0.01)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("bundled", [True, False])
def test_moe_expert_bundled_and_separate_weights(tmp_path, bundled, tokens, quantizable):
    cfg = _config(LlmArchType.GPT_OSS)
    rng = np.random.default_rng(3)
    gate = rng.normal(0, 0.1, (INTERMEDIATE, HIDDEN)).astype(np.float32)
    up = rng.normal(0, 0.1, (INTERMEDIATE, HIDDEN)).astype(np.float32)
    down = rng.normal(0, 0.1, (HIDDEN, INTERMEDIATE)).astype(np.float32)
    gate_bias = np.linspace(-10, 10, INTERMEDIATE, dtype=np.float32)
    up_bias = -gate_bias
    if bundled:
        combined = np.stack([gate.T, up.T], axis=-1).reshape(HIDDEN, 2 * INTERMEDIATE)
        combined_bias = np.stack([gate_bias, up_bias], axis=-1).reshape(-1)
        params = {
            f"{BASE}.mlp.experts.gate_up_proj": np.stack([combined] * EXPERTS),
            f"{BASE}.mlp.experts.gate_up_proj_bias": np.stack([combined_bias] * EXPERTS),
            f"{BASE}.mlp.experts.down_proj": np.stack([down.T] * EXPERTS),
        }
    else:
        prefix = f"{BASE}.mlp.experts.1"
        params = {
            f"{prefix}.gate_proj.weight": gate, f"{prefix}.up_proj.weight": up,
            f"{prefix}.down_proj.weight": down,
            f"{prefix}.gate_proj.bias": gate_bias, f"{prefix}.up_proj.bias": up_bias,
        }
    model = _model(LanguagePostModel, cfg, params, tmp_path, num_tokens=tokens, layer_idx=0, final_softcapping=None, expert_idx=1)
    x = rng.normal(0, 0.1, (1, 1, tokens, HIDDEN)).astype(np.float32)
    routing = np.zeros((1, 1, tokens, EXPERTS), np.float32)
    routing[..., 1] = 0.25
    outputs, _ = _run(model, {"norm_hidden": x, "router": routing}, quantizable)
    g = np.minimum(x @ gate.T + gate_bias, 7)
    u = np.clip(x @ up.T + up_bias, -7, 7)
    expected = ((u + 1) * g / (1 + np.exp(-1.702 * g))) @ down.T * 0.25
    np.testing.assert_allclose(outputs[0], expected, rtol=0.05 if not quantizable else 2e-5, atol=0.03 if not quantizable else 2e-6)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [1, 4])
def test_moe_weighted_sum_interface(tmp_path, tokens, quantizable):
    cfg = _config(LlmArchType.GPT_OSS)
    count = 2 if tokens == 1 else EXPERTS
    model = _model(LanguageMoeWeightedSumModel, cfg, {}, tmp_path, num_tokens=tokens, layer_idx=0, final_softcapping=None)
    values = {f"expert_{i}": np.full((1, 1, tokens, HIDDEN), i + 1, np.float32) for i in range(count)}
    values["residual"] = np.full((1, 1, tokens, HIDDEN), 0.5, np.float32)
    outputs, net = _run(model, values, quantizable)
    assert list(net.input_node_names) == list(values)
    np.testing.assert_array_equal(outputs[0], sum(values.values()))


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [1, 4])
def test_gpt_oss_cache_sinks_and_tessellation(tmp_path, tokens, quantizable):
    cfg = _config(LlmArchType.GPT_OSS)
    model = _model(LanguageCacheModel, cfg, {}, tmp_path, num_tokens=tokens, token_idx=0, logit_softcapping=None)
    values = {
        "input": np.zeros((1, 2, tokens, 16), np.float32),
        "cached_keys": np.zeros((1, 1, tokens, 16), np.float32),
        "sinks": np.zeros((1, 1, tokens, 2), np.float32),
        "cached_values": np.ones((1, 1, tokens, 16), np.float32),
    }
    outputs, net = _run(model, values, quantizable)
    assert list(net.input_node_names) == list(values)
    expected = np.arange(1, tokens + 1, dtype=np.float32) / np.arange(2, tokens + 2)
    np.testing.assert_allclose(outputs[0], np.broadcast_to(expected[None, None, :, None], outputs[0].shape), atol=0.005)
    params = model.get_mla_input_tessellate_params()
    assert params[1].dram_shape[-1] == 16
    assert params[3].dram_shape[-1] == 16
    assert 2 not in params


@pytest.mark.parametrize("quantizable", [True, False])
def test_olmoe_normalizes_qk_before_head_split(tmp_path, quantizable):
    cfg = _config(LlmArchType.OLMOE)
    params = {f"{BASE}.input_layernorm.weight": np.ones(HIDDEN, np.float32)}
    for name, channels in (("q", 32), ("k", 16), ("v", 16)):
        params[f"{BASE}.self_attn.{name}_proj.weight"] = np.eye(HIDDEN, dtype=np.float32)[:channels]
        if name != "v":
            params[f"{BASE}.self_attn.{name}_norm.weight"] = np.ones(channels, np.float32)
    model = _model(LanguagePreModel, cfg, params, tmp_path, num_tokens=1, layer_idx=0)
    x = np.arange(1, HIDDEN + 1, dtype=np.float32).reshape(1, 1, 1, HIDDEN)
    values = {"input": x, "freq_real": np.ones((1, 1, 1, 8), np.float32), "freq_imag": np.zeros((1, 1, 1, 8), np.float32)}
    outputs, _ = _run(model, values, quantizable)
    expected_q = (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-5)).reshape(1, 1, 1, 2, 16).transpose(0, 3, 2, 1, 4).reshape(1, 2, 1, 16) * 0.25
    np.testing.assert_allclose(outputs[0], expected_q, rtol=0.025, atol=0.001)


@pytest.mark.parametrize("quantizable", [True, False])
def test_moe_final_weighted_sum_applies_output_head(tmp_path, quantizable):
    cfg = _config(LlmArchType.GPT_OSS)
    cfg.pipeline_cfg.return_logits = False
    head = np.zeros((64, HIDDEN), np.float32)
    head[17] = 1
    params = {"model.norm.weight": np.ones(HIDDEN, np.float32), "lm_head.weight": head}
    model = _model(LanguageMoeWeightedSumModel, cfg, params, tmp_path, num_tokens=1, layer_idx=1, final_softcapping=None)
    values = {name: np.ones((1, 1, 1, HIDDEN), np.float32) for name in ("expert_0", "expert_1", "residual")}
    outputs, _ = _run(model, values, quantizable)
    assert outputs[0].dtype == np.int32
    np.testing.assert_array_equal(outputs[0], 17)
