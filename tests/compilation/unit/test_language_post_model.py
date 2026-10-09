"""MTP graph contracts and numerical projections, without external checkpoints."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from ml_dtypes import bfloat16

from afe.ir.execute import create_node_executor, create_node_quant_executor
from sima_lmm.config.vlm_config import LlmArchType, PipelineConfig, SpeculativeDecodingMethod
from sima_lmm.config.vlm_config import VlmArchType
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.model_graph import ModelGraph

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _model(cls, tmp_path, tokens, *, draft=True, masked=False):
    rng = np.random.default_rng(251)
    attn = SimpleNamespace(
        num_attention_heads=2, num_key_value_heads=1, attn_output_gate=False,
        get_head_dim=lambda _: 16, get_q_size=lambda _: 32, get_kv_size=lambda _: 16,
    )
    lm = SimpleNamespace(
        num_hidden_layers=1, hidden_size=32, layer_types=["full_attention"],
        hidden_size_per_layer_input=0, assistant_backbone_hidden_size=64,
        assistant_masked_lm_head_enabled=masked, assistant_num_centroids=16,
        speculative_decoding_cfg=SimpleNamespace(
            is_draft=draft, method=SpeculativeDecodingMethod.GEMMA4_MTP,
        ),
        arch=LlmArchType.GEMMA, moe_cfg=None, lora_cfg=None, attn_cfg=attn,
        rms_norm_eps=1e-5, rms_norm_unit_offset=False,
        mlp_cfg=SimpleNamespace(act="silu", swiglu_limit=None),
        rope_cfg=SimpleNamespace(
            get_rope_dimension_count=lambda _: 16,
            rope_scaling=SimpleNamespace(rope_type="default"),
        ),
        is_kv_shared_layer=lambda _: draft, lm_head_num_splits=2,
        lm_head_split_dim=32, draft_vocab_size=0, token_cfg=SimpleNamespace(vocab_size=64),
    )
    cfg = SimpleNamespace(
        model_type=VlmArchType.VLM_GEMMA4, vm_cfg=None, lm_cfg=lm,
        pipeline_cfg=PipelineConfig(quantize_embeddings=True, quantize_kv_cache=False, return_logits=True),
    )
    params = {
        "pre_projection.weight": rng.normal(0, 0.1, (32, 128)).astype(np.float32),
        "post_projection.weight": rng.normal(0, 0.1, (64, 32)).astype(np.float32),
        "masked_embedding.centroids.weight": rng.normal(0, 0.1, (16, 32)).astype(np.float32),
        "lm_head.weight": rng.normal(0, 0.1, (64, 32)).astype(np.float32),
        "model.norm.weight": np.ones(32, np.float32),
        "model.layers.0.layer_scalar": np.array([0.25], np.float32),
    }
    for name in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]:
        params[f"model.layers.0.{name}.weight"] = np.ones(32, np.float32)
    for name in ["q_proj", "o_proj"]:
        params[f"model.layers.0.self_attn.{name}.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
    for name in ["k_proj", "v_proj"]:
        params[f"model.layers.0.self_attn.{name}.weight"] = rng.normal(0, 0.1, (16, 32)).astype(np.float32)
    for name in ["gate_proj", "up_proj", "down_proj"]:
        params[f"model.layers.0.mlp.{name}.weight"] = rng.normal(0, 0.1, (32, 32)).astype(np.float32)
    args = dict(num_tokens=tokens, layer_idx=0)
    if cls is LanguagePostModel:
        args["final_softcapping"] = None
    model = cls(cfg, "mtp_component", sima_path=tmp_path,
                hf_model=SimpleNamespace(language_model_param_base_name="model", is_gguf=False), **args)
    model.get_hf_param = params.__getitem__
    model.check_hf_param = params.__contains__
    return model, params


def _run(model, values, quantizable):
    with patch.object(ModelGraph, "save", autospec=True) as save:
        model.generate_graph({}, quantizable)
    graph, outputs = save.call_args.args
    net = graph.finish(outputs)
    execute = create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    values = {k: v.astype(np.float32 if quantizable else bfloat16) for k, v in values.items()}
    result = net.run(values, node_callable=execute)
    return list(result) if isinstance(result, (list, tuple)) else [result], net


def _norm(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-5))


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [1, 7])
def test_mtp_pre_projects_combined_state_without_eagle_input(tmp_path, tokens, quantizable):
    model, params = _model(LanguagePreModel, tmp_path, tokens)
    x = np.random.default_rng(0).normal(0, 0.1, (1, 1, tokens, 128)).astype(np.float32)
    values = {"input": x, "freq_real": np.ones((1, 1, tokens, 8), np.float32),
              "freq_imag": np.zeros((1, 1, tokens, 8), np.float32)}
    outputs, net = _run(model, values, quantizable)
    assert list(net.input_node_names) == list(values)
    assert len(outputs) == 1  # The assistant shares the target's KV cache.
    expected = _norm(x @ params["pre_projection.weight"].T) @ params["model.layers.0.self_attn.q_proj.weight"].T
    expected = expected.reshape(1, 1, tokens, 2, 16).transpose(0, 3, 2, 1, 4).reshape(1, 2, tokens, 16)
    np.testing.assert_allclose(outputs[0], expected, rtol=0.06 if not quantizable else 2e-5, atol=0.01 if not quantizable else 1e-6)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [1, 7])
@pytest.mark.parametrize("draft,masked", [(True, False), (True, True), (False, False)])
def test_mtp_post_preserves_logits_state_and_layer_scalar(tmp_path, tokens, quantizable, draft, masked):
    model, params = _model(LanguagePostModel, tmp_path, tokens, draft=draft, masked=masked)
    model.cfg.pipeline_cfg.quantize_embeddings = False
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, (1, 1, tokens, 128 if draft else 32)).astype(np.float32)
    attention = rng.normal(0, 0.1, (1, 1, tokens, 32)).astype(np.float32)
    outputs, net = _run(model, {"input": x, "self_attn": attention}, quantizable)
    assert list(net.input_node_names) == ["input", "self_attn"]
    residual = x @ params["pre_projection.weight"].T if draft else x
    hidden = residual + _norm(attention @ params["model.layers.0.self_attn.o_proj.weight"].T)
    normalized = _norm(hidden)
    gate = normalized @ params["model.layers.0.mlp.gate_proj.weight"].T
    up = normalized @ params["model.layers.0.mlp.up_proj.weight"].T
    mlp = (gate / (1 + np.exp(-gate)) * up) @ params["model.layers.0.mlp.down_proj.weight"].T
    final_norm = _norm((hidden + _norm(mlp)) * np.float32(0.25))
    logits = final_norm @ params["lm_head.weight"].T
    expected = [logits[..., :32], logits[..., 32:]]
    if masked:
        expected.append(final_norm @ params["masked_embedding.centroids.weight"].T)
    expected.append(final_norm @ params["post_projection.weight"].T if draft else final_norm)
    assert len(outputs) == len(expected)
    for actual, reference in zip(outputs, expected):
        np.testing.assert_allclose(actual, reference, rtol=0.08 if not quantizable else 3e-5, atol=0.04 if not quantizable else 3e-6)
