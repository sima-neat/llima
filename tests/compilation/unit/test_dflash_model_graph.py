"""DFlash graph contracts and accepted-prefix numerics without model downloads."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from ml_dtypes import bfloat16, int4

from afe.ir.execute import create_node_executor, create_node_quant_executor
from sima_lmm.config.vlm_config import LinearAttentionConfig, LlmArchType, PipelineConfig
from sima_lmm.model.language_dflash_context_model import LanguageDFlashContextModel
from sima_lmm.model.language_dflash_state_resolver_model import LanguageDFlashStateResolverModel
from sima_lmm.model.language_draft_fc_model import LanguageDraftFCModel
from sima_lmm.model.language_linear_model import LanguageLinearModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.model_graph import ModelGraph

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _config(tokens, *, draft=True):
    attn = SimpleNamespace(
        num_attention_heads=2, num_key_value_heads=1, attn_output_gate=False,
        get_head_dim=lambda _: 16, get_q_size=lambda _: 32, get_kv_size=lambda _: 16,
    )
    lm = SimpleNamespace(
        arch=LlmArchType.LLAMA, hidden_size=32, num_hidden_layers=1,
        layer_types=["full_attention"], hidden_size_per_layer_input=0,
        attn_cfg=attn, moe_cfg=None, lora_cfg=None, rms_norm_eps=1e-5,
        rms_norm_unit_offset=False, mlp_cfg=SimpleNamespace(act="silu", swiglu_limit=None),
        speculative_decoding_cfg=SimpleNamespace(
            method="dflash", is_draft=draft, speculative_budget=tokens, target_layer_ids=[0, 1, 2],
        ),
        rope_cfg=SimpleNamespace(
            get_rope_dimension_count=lambda _: 16,
            rope_scaling=SimpleNamespace(rope_type="default"),
        ),
        linear_attn_cfg=LinearAttentionConfig(
            conv_kernel_dim=4, num_key_heads=2, num_value_heads=2, key_head_dim=16, value_head_dim=16,
        ),
        is_kv_shared_layer=lambda _: False,
        lm_head_num_splits=2, lm_head_split_dim=32, token_cfg=SimpleNamespace(vocab_size=64),
    )
    return SimpleNamespace(lm_cfg=lm, vm_cfg=None, model_type="llm-test",
                           pipeline_cfg=PipelineConfig(quantize_embeddings=False, quantize_kv_cache=False))


def _model(cls, tmp_path, tokens):
    args = dict(num_tokens=tokens)
    if cls in (LanguagePreModel, LanguagePostModel, LanguageDFlashContextModel):
        args["layer_idx"] = 0
    if cls is LanguagePostModel:
        args["final_softcapping"] = None
    if cls is LanguageDFlashStateResolverModel:
        args = dict(block_size=tokens)
    model = cls(_config(tokens), "dflash_component", sima_path=tmp_path,
                hf_model=SimpleNamespace(language_model_param_base_name="model", is_gguf=False), **args)
    rng = np.random.default_rng(21)
    params = {name + ".weight": rng.normal(0, 0.05, shape).astype(np.float32) for name, shape in [
        ("fc", (32, 96)), ("layers.0.self_attn.q_proj", (32, 32)),
        ("layers.0.self_attn.k_proj", (16, 32)), ("layers.0.self_attn.v_proj", (16, 32)),
        ("layers.0.self_attn.o_proj", (32, 32)),
        *[("layers.0.mlp." + name, (32, 32)) for name in ("gate_proj", "up_proj", "down_proj")],
    ]}
    for name, channels in [("hidden_norm", 32), ("norm", 32),
                            ("layers.0.input_layernorm", 32), ("layers.0.post_attention_layernorm", 32)]:
        params[name + ".weight"] = np.ones(channels, np.float32)
    model.get_hf_param = params.__getitem__
    model.check_hf_param = params.__contains__
    head = (np.full((64, 2), 0.02, np.float32), rng.integers(-7, 8, (64, 32), dtype=np.int8).astype(int4), 16)
    model.dflash_target_hf_model = SimpleNamespace(
        param_exists=lambda name: name == "lm_head.weight", load_np_param=lambda _: head,
    )
    return model


def _net(model, quantizable):
    with patch.object(ModelGraph, "save", autospec=True) as save:
        model.generate_graph({}, quantizable)
    graph, outputs = save.call_args.args
    return graph.finish(outputs)


def _run(net, values, quantizable):
    executor = create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    values = {name: value.astype(np.float32 if quantizable else bfloat16) for name, value in values.items()}
    result = net.run(values, node_callable=executor)
    return list(result) if isinstance(result, (list, tuple)) else [result]


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [4, 8, 16])
def test_dflash_group_and_resolver_accepted_prefixes(tmp_path, tokens, quantizable):
    model = object.__new__(LanguageLinearModel)
    model.cfg = _config(tokens, draft=False)
    model.num_tokens = tokens
    model.model_name = "group_delta"
    shapes = {name: (1, 2, tokens, 16) for name in ("query", "key", "value")}
    shapes.update(beta=(1, 2, tokens, 1), g=(1, 2, tokens, 1), state=(1, 2, 16, 16))
    graph = ModelGraph(model, shapes, quantizable)
    query, key, value, beta, g, state = graph.inputs.values()
    outputs = model._build_group_delta(
        graph, query, key, query, key, value, beta, g, state, emit_resolver_inputs=True,
    )
    rng = np.random.default_rng(19)
    values = {name: rng.normal(0, 0.02, shape).astype(np.float32) for name, shape in shapes.items()}
    values["beta"] = np.full(shapes["beta"], 0.5, np.float32)
    values["g"] = rng.uniform(-0.2, -0.01, shapes["g"]).astype(np.float32)
    _, s1, keys, v_new, decay = _run(graph.finish(outputs), values, quantizable)
    assert s1.shape == (1, 2, 16, 16)
    assert decay.shape == (1, 2, tokens, tokens)
    resolver = _model(LanguageDFlashStateResolverModel, tmp_path, tokens)
    net = _net(resolver, quantizable)
    assert list(net.input_node_names) == ["linear_delta_state_s1", "key", "v_new", "decay_row"]
    for prefix in range(1, tokens + 1):
        # Runtime selects one row and masks updates beyond the accepted prefix.
        row = decay[:, :, prefix - 1:prefix, :].copy()
        row[:, :, :, prefix:] = 0
        actual = _run(net, {"linear_delta_state_s1": s1, "key": keys,
                            "v_new": v_new, "decay_row": row}, quantizable)[0]
        weights = row.transpose(0, 1, 3, 2).copy()
        weights[:, :, 0] = 0
        expected = s1 * row[:, :, :, :1] + keys.swapaxes(-1, -2) @ (v_new * weights)
        np.testing.assert_allclose(actual, expected, rtol=0.025 if not quantizable else 2e-5,
                                   atol=0.001 if not quantizable else 1e-7)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("tokens", [4, 8, 16])
def test_dflash_draft_graph_contracts(tmp_path, tokens, quantizable):
    rng = np.random.default_rng(1)
    hidden = rng.normal(0, 0.1, (1, 1, tokens, 32)).astype(np.float32)
    values = {"input": hidden, "freq_real": np.ones((1, 1, tokens, 8), np.float32),
              "freq_imag": np.zeros((1, 1, tokens, 8), np.float32)}
    pre = _model(LanguagePreModel, tmp_path, tokens)
    net = _net(pre, quantizable)
    assert list(net.input_node_names) == list(values)  # No EAGLE3 hidden_states input.
    assert [x.shape for x in _run(net, values, quantizable)] == [(1, 2, tokens, 16), (1, 1, tokens, 16), (1, 1, tokens, 16)]
    context = _model(LanguageDFlashContextModel, tmp_path, tokens)
    assert [x.shape for x in _run(_net(context, quantizable), values, quantizable)] == [(1, 1, tokens, 16)] * 2
    fc = _model(LanguageDraftFCModel, tmp_path, tokens)
    fused = _run(_net(fc, quantizable), {f"input_{index}": hidden for index in range(3)}, quantizable)
    assert fused[0].shape == hidden.shape
    post = _model(LanguagePostModel, tmp_path, tokens)
    net = _net(post, quantizable)
    result = _run(net, {"input": hidden, "self_attn": hidden}, quantizable)
    assert [x.shape for x in result] == [(1, 1, tokens - 1, 32)] * 2
    if not quantizable:
        # The paired target's INT4 values and G16 scale groups remain packed.
        subnet = net.nodes["MLA_0"]
        # Weight preservation is checked structurally without relying on generated names.
        int4_convs = [node for node in subnet.ir.nodes.values()
                      if getattr(getattr(node.ir, "quant_attrs", None), "c_block_size", None) == 16]
        assert len(int4_convs) == 2
        weights = post._get_dflash_target_param("lm_head.weight")
        for index, node in enumerate(int4_convs):
            start, stop = index * 32, (index + 1) * 32
            np.testing.assert_array_equal(node.ir.quant_attrs.weight_quant_data.reshape(32, 32),
                                          weights[1][start:stop].T)
            np.testing.assert_array_equal(node.ir.quant_attrs.requant.sc_correction,
                                          weights[0][start:stop].T)


def test_dflash_target_source_does_not_change_moe_factory(tmp_path):
    from sima_lmm.model.language_model import LanguageModel
    from sima_lmm.model.language_moe_weightedsum_model import LanguageMoeWeightedSumModel

    cfg = _config(8, draft=False)
    cfg.lm_cfg.moe_cfg = SimpleNamespace(num_experts=4, num_experts_per_tok=2)
    cfg.lm_cfg.final_logit_softcapping = None
    model = LanguageModel(cfg, "target", sima_path=tmp_path, hf_model=SimpleNamespace())
    assert isinstance(model._get_part_model("moe_weightedsum", 8, layer_idx=0),
                      LanguageMoeWeightedSumModel)
