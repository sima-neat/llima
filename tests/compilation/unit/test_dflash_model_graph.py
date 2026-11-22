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


def _config(tokens, *, draft=True, dflash2=False):
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
            dflash_conv_kernel_size=2 if dflash2 else 0, dflash_conv_group_size=16 if dflash2 else 0,
            dflash_selector_rank=8 if dflash2 else 0, dflash_selector_top_k=4 if dflash2 else 0,
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


def _model(cls, tmp_path, tokens, *, dflash2=False):
    args = dict(num_tokens=tokens)
    if cls in (LanguagePreModel, LanguagePostModel, LanguageDFlashContextModel):
        args["layer_idx"] = 0
    if cls is LanguagePostModel:
        args["final_softcapping"] = None
    if cls is LanguageDFlashStateResolverModel:
        args = dict(block_size=tokens)
    model = cls(_config(tokens, dflash2=dflash2), "dflash_component", sima_path=tmp_path,
                hf_model=SimpleNamespace(language_model_param_base_name="model", is_gguf=False), **args)
    rng = np.random.default_rng(21)
    params = {name + ".weight": rng.normal(0, 0.05, shape).astype(np.float32) for name, shape in [
        ("fc", (32, 96)), ("layers.0.self_attn.q_proj", (32, 32)),
        ("layers.0.self_attn.k_proj", (16, 32)), ("layers.0.self_attn.v_proj", (16, 32)),
        ("layers.0.self_attn.o_proj", (32, 32)),
        *[("layers.0.mlp." + name, (32, 32)) for name in ("gate_proj", "up_proj", "down_proj")],
        # DFlash 2: 2 stages x 2 taps x 2 groups of 16 channels, and a rank-8 selector projection.
        ("layers.0.attention_conv.kernel_projection", (8, 32)),
        ("layers.0.mlp_conv.kernel_projection", (8, 32)),
        ("candidate_selector.hidden_projection", (8, 32)),
    ]}
    for name in ("attention_conv", "mlp_conv"):
        params[f"layers.0.{name}.base_kernel"] = rng.normal(0, 0.5, (2, 2, 32)).astype(np.float32)
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


def _reference_dflash2_conv(hidden, dynamic, base, group_size):
    """NumPy port of z-lab/dflash ``_grouped_dynamic_convolve`` for one (tokens, hidden) block."""
    tokens, hidden_size = hidden.shape
    groups = hidden_size // group_size
    blocks = hidden.reshape(tokens, groups, group_size)
    output = np.zeros_like(blocks)
    for offset in range(base.shape[0]):
        values = blocks if offset == 0 else np.pad(blocks[:-offset], ((offset, 0), (0, 0), (0, 0)))
        output += base[offset].reshape(groups, group_size) * values + dynamic[:, offset, :, None] * values
    return output.reshape(tokens, hidden_size)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("full_block_kernel", [False, True])
@pytest.mark.parametrize("tokens", [4, 8, 16])
def test_dflash2_conv_matches_reference(tmp_path, tokens, full_block_kernel, quantizable):
    from sima_lmm.model.dflash2_conv import DFlash2Conv

    model = _model(LanguagePreModel, tmp_path, tokens, dflash2=True)
    kernel_size = tokens if full_block_kernel else 2
    if full_block_kernel:
        rng = np.random.default_rng(13)
        overrides = {
            "layers.0.attention_conv.base_kernel": rng.normal(0, 0.5, (2, kernel_size, 32)).astype(np.float32),
            "layers.0.attention_conv.kernel_projection.weight":
                rng.normal(0, 0.05, (2 * kernel_size * 2, 32)).astype(np.float32),
        }
        params = model.get_hf_param
        model.get_hf_param = lambda name: overrides[name] if name in overrides else params(name)
    shape = (1, 1, tokens, 32)
    graph = ModelGraph(model, {"input": shape, "sublayer": shape}, quantizable)
    conv = DFlash2Conv(graph, "layers.0.attention_conv", kernel_size=kernel_size, group_size=16)
    prepared = conv.prepare(graph.inputs["input"])
    finished = conv.finish(graph.inputs["sublayer"])
    rng = np.random.default_rng(5)
    values = {name: rng.normal(0, 0.5, shape).astype(np.float32) for name in ("input", "sublayer")}
    actual = _run(graph.finish([prepared, finished]), values, quantizable)

    weight = model.get_hf_param("layers.0.attention_conv.kernel_projection.weight")
    base = model.get_hf_param("layers.0.attention_conv.base_kernel")
    x, y = values["input"][0, 0], values["sublayer"][0, 0]
    dynamic = (x @ weight.T).reshape(tokens, 2, kernel_size, 2)  # (tokens, stage, tap, group)
    expected = [
        _reference_dflash2_conv(x, dynamic[:, 0], base[0], 16),
        _reference_dflash2_conv(y, dynamic[:, 1], base[1], 16),
    ]
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result.reshape(tokens, 32).astype(np.float32), reference,
                                   rtol=0.03 if not quantizable else 1e-5,
                                   atol=0.02 if not quantizable else 1e-6)


@pytest.mark.parametrize("quantizable", [True, False])
def test_dflash2_draft_graph_contracts(tmp_path, quantizable):
    tokens = 8
    rng = np.random.default_rng(3)
    hidden = rng.normal(0, 0.1, (1, 1, tokens, 32)).astype(np.float32)
    values = {"input": hidden, "freq_real": np.ones((1, 1, tokens, 8), np.float32),
              "freq_imag": np.zeros((1, 1, tokens, 8), np.float32)}
    pre = _model(LanguagePreModel, tmp_path, tokens, dflash2=True)
    assert pre.is_dflash2_draft
    pre_net = _net(pre, quantizable)
    assert list(pre_net.input_node_names) == list(values)
    pre_out = _run(pre_net, values, quantizable)
    assert [x.shape for x in pre_out] == [(1, 2, tokens, 16), (1, 1, tokens, 16), (1, 1, tokens, 16)]
    # The convolution changes the query relative to a DFlash draft with the same weights.
    plain = _run(_net(_model(LanguagePreModel, tmp_path, tokens), quantizable), values, quantizable)
    assert not np.allclose(np.asarray(pre_out[0], np.float32), np.asarray(plain[0], np.float32))
    post = _model(LanguagePostModel, tmp_path, tokens, dflash2=True)
    result = _run(_net(post, quantizable), {"input": hidden, "self_attn": hidden}, quantizable)
    # Two lm_head splits plus the rank-8 selector projection of the proposal rows.
    assert [x.shape for x in result] == [(1, 1, tokens - 1, 32)] * 2 + [(1, 1, tokens - 1, 8)]


def test_dflash2_settings_accept_dflash2_and_reject_mixed_configs():
    from sima_lmm.model.vision_language_model import _dflash2_settings

    dflash2 = {"conv_kernel_size": 2, "conv_group_size": 16, "selector_rank": 256, "selector_top_k": 16}
    assert _dflash2_settings({"architectures": ["DFlash2DraftModel"]}, dflash2, 8, 5120, 248320) == dict(
        dflash_conv_kernel_size=2, dflash_conv_group_size=16,
        dflash_selector_rank=256, dflash_selector_top_k=16,
        dflash_output_multiplier=1.0, dflash_final_logit_softcapping=0.0,
    )
    # Logit transforms come from dflash_config first, then the top level, as upstream reads them.
    transformed = _dflash2_settings(
        {"architectures": ["DFlash2DraftModel"], "output_multiplier": 0.5, "final_logit_softcapping": 30.0},
        {**dflash2, "output_multiplier": 2.0}, 8, 5120, 248320,
    )
    assert (transformed["dflash_output_multiplier"], transformed["dflash_final_logit_softcapping"]) == (2.0, 30.0)
    with pytest.raises(ValueError, match="output_multiplier"):
        _dflash2_settings({"architectures": ["DFlash2DraftModel"]}, {**dflash2, "output_multiplier": -1.0}, 8,
                          5120, 248320)
    assert _dflash2_settings({"architectures": ["DFlashDraftModel"]}, {}, 8, 5120, 248320) == {}
    with pytest.raises(ValueError, match="without DFlash2DraftModel"):
        _dflash2_settings({"architectures": ["DFlashDraftModel"]}, dflash2, 8, 5120, 248320)
    with pytest.raises(ValueError, match="missing dflash_config keys"):
        _dflash2_settings({"architectures": ["DFlash2DraftModel"]}, {"conv_kernel_size": 2}, 8, 5120, 248320)
    with pytest.raises(ValueError, match="conv_group_size"):
        _dflash2_settings({"architectures": ["DFlash2DraftModel"]}, {**dflash2, "conv_group_size": 7}, 8, 5120, 248320)
    full_block = _dflash2_settings({"architectures": ["DFlash2DraftModel"]}, {**dflash2, "conv_kernel_size": 8}, 8,
                                   5120, 248320)
    assert full_block["dflash_conv_kernel_size"] == 8
    with pytest.raises(ValueError, match="block size"):
        _dflash2_settings({"architectures": ["DFlash2DraftModel"]}, {**dflash2, "conv_kernel_size": 9}, 8, 5120, 248320)
