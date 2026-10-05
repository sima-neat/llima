from types import SimpleNamespace

import numpy as np
import pytest

from afe.ir.execute import create_node_executor, create_node_quant_executor
from sima_lmm.config.vlm_config import LlmArchType, PipelineConfig
from sima_lmm.model.language_cache_model import (
    LanguageCacheModel,
    _get_bmm2_reduction_ranges,
)
from sima_lmm.model.model_graph import ModelGraph, activation_dtype


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize("context_length", [1, 1024, 2048])
def test_bmm2_reduction_is_not_split_at_or_below_threshold(context_length: int):
    assert _get_bmm2_reduction_ranges(context_length) == [(0, context_length)]


@pytest.mark.parametrize("context_length", [2049, 4096, 6144, 8192])
def test_bmm2_reduction_uses_contiguous_1k_chunks(context_length: int):
    ranges = _get_bmm2_reduction_ranges(context_length)

    assert ranges[0][0] == 0
    assert ranges[-1][1] == context_length
    assert all(
        end == next_start
        for (_, end), (next_start, _) in zip(ranges, ranges[1:])
    )
    assert all(0 < end - start <= 1024 for start, end in ranges)


def test_single_cache_model_when_grouping_is_disabled():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(2048)
    pipeline_cfg.set_group_size(None)
    pipeline_cfg.set_future_token_mask_size(128)
    cfg = SimpleNamespace(
        pipeline_cfg=pipeline_cfg,
        lm_cfg=SimpleNamespace(speculative_decoding_cfg=None),
    )

    model = LanguageCacheModel(
        cfg,
        "single_cache",
        num_tokens=1,
        token_idx=127,
        logit_softcapping=None,
    )

    assert not model._is_group_model
    assert model._cache_mask_size == 128


def test_group_size_one_reuses_single_cache_model():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(2048)
    pipeline_cfg.set_group_size(1)
    pipeline_cfg.set_future_token_mask_size(128)
    cfg = SimpleNamespace(
        pipeline_cfg=pipeline_cfg,
        lm_cfg=SimpleNamespace(speculative_decoding_cfg=None),
    )

    model = LanguageCacheModel(
        cfg,
        "group_cache",
        num_tokens=1,
        token_idx=127,
        logit_softcapping=None,
    )

    assert not model._is_group_model
    assert model._cache_mask_size == 128


@pytest.mark.parametrize("quantizable", [True, False])
def test_gemma2_native_cache_softcapping(monkeypatch, quantizable):
    attention = SimpleNamespace(
        num_attention_heads=2,
        num_key_value_heads=1,
        get_head_dim=lambda _: 16,
        get_q_size=lambda _: 32,
        get_kv_size=lambda _: 16,
    )
    cfg = SimpleNamespace(
        model_type="gemma2",
        lm_cfg=SimpleNamespace(
            attn_cfg=attention,
            speculative_decoding_cfg=None,
            arch=LlmArchType.GEMMA,
            model_type="gemma2",
            attn_logit_softcapping=50.0,
        ),
        pipeline_cfg=SimpleNamespace(
            quantize_kv_cache=False,
            input_token_group_offsets=[],
            input_token_group_size=128,
            get_cache_mask_size=lambda *args, **kwargs: 1,
        ),
    )
    model = LanguageCacheModel(
        cfg, "gemma2_cache", num_tokens=1, token_idx=31, logit_softcapping=50.0
    )
    saved = []
    monkeypatch.setattr(
        ModelGraph, "save", lambda graph, outputs: saved.append(graph.finish(outputs))
    )
    model.generate_graph({}, quantizable=quantizable)
    net = saved[0]
    rng = np.random.default_rng(42)
    inputs = {
        name: rng.normal(0, 0.5, shape).astype(activation_dtype(quantizable))
        for name, shape in (
            ("input", (1, 2, 1, 16)),
            ("cached_keys", (1, 1, 32, 16)),
            ("cached_values", (1, 1, 32, 16)),
        )
    }
    if not quantizable:
        # Uniform attention and integer values give an exact BF16 reference.
        # The separate softcap test covers nonzero scores and BF16 rounding.
        inputs["input"].fill(0)
        inputs["cached_values"] = (np.arange(32 * 16).reshape(1, 1, 32, 16) % 17 - 8).astype(
            activation_dtype(False)
        )
    query, key, value = (inputs[name].astype(np.float32) for name in inputs)
    scores = 50 * np.tanh((query @ key.swapaxes(-1, -2)) / 50)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = (probabilities @ value).transpose(0, 2, 1, 3).reshape(1, 1, 1, 32)
    execute = create_node_executor(False) if quantizable else create_node_quant_executor(False, False)
    actual = net.run(inputs, node_callable=execute)
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    if quantizable:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    else:
        np.testing.assert_array_equal(actual, expected)
