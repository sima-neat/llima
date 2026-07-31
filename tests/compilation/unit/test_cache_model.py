from types import SimpleNamespace

import numpy as np
import pytest

from sima_lmm.config.vlm_config import PipelineConfig
from sima_lmm.model import base as base_model_module
from sima_lmm.model.base import BaseModel, EvalMode
from sima_lmm.model.language_cache_model import (
    LanguageCacheModel,
    _get_bmm2_reduction_ranges,
)
from sima_lmm.model.language_model import (
    LanguageModel,
    ReferenceCachePlan,
    ReferenceRuntimeStep,
    _reference_cache_plan,
    _reference_runtime_steps,
)


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


def test_reference_runtime_uses_group_prefill_then_single_decode():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(2048)
    pipeline_cfg.set_group_size(128)

    assert _reference_runtime_steps(pipeline_cfg, 9, 3) == [
        ReferenceRuntimeStep(128, 0, True, True),
        ReferenceRuntimeStep(1, 9, False, True),
        ReferenceRuntimeStep(1, 10, False, True),
    ]


def test_reference_runtime_only_emits_from_final_prefill_group():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(2048)
    pipeline_cfg.set_group_size(128)

    assert _reference_runtime_steps(pipeline_cfg, 129, 2) == [
        ReferenceRuntimeStep(128, 0, True, False),
        ReferenceRuntimeStep(128, 128, True, True),
        ReferenceRuntimeStep(1, 129, False, True),
    ]


def test_reference_runtime_without_grouping_prefills_one_token_at_a_time():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(1024)
    pipeline_cfg.set_group_size(None)

    assert _reference_runtime_steps(pipeline_cfg, 2, 1) == [
        ReferenceRuntimeStep(1, 0, True, False),
        ReferenceRuntimeStep(1, 1, True, True),
    ]


def test_reference_cache_plan_uses_long_context_bucket():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(8192)
    pipeline_cfg.set_group_size(128)
    pipeline_cfg.set_future_token_mask_size(128)
    attention_cfg = SimpleNamespace(
        sliding_window=None,
        sliding_head_dim=None,
        head_dim=128,
    )

    assert _reference_cache_plan(
        pipeline_cfg, attention_cfg, "full_attention", 7040, 1
    ) == ReferenceCachePlan(
        token_idx_begin=0,
        aligned_context=7168,
        model_token_idx=7167,
        layer_type="full_attention",
        use_future_mask=True,
    )


def test_reference_cache_plan_selects_separate_sliding_cache():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(8192)
    pipeline_cfg.set_group_size(128)
    pipeline_cfg.set_future_token_mask_size(128)
    attention_cfg = SimpleNamespace(
        sliding_window=512,
        sliding_head_dim=64,
        head_dim=128,
    )

    assert _reference_cache_plan(
        pipeline_cfg, attention_cfg, "sliding_attention", 7040, 128
    ) == ReferenceCachePlan(
        token_idx_begin=6656,
        aligned_context=512,
        model_token_idx=384,
        layer_type="sliding_attention",
        use_future_mask=False,
    )


def test_reference_runtime_uses_group_cache_and_single_final_post():
    pipeline_cfg = PipelineConfig()
    pipeline_cfg.set_max_num_tokens(1024)
    pipeline_cfg.set_group_size(128)
    pipeline_cfg.set_future_token_mask_size(128)
    calls = []
    constructions = []

    class FakePart:
        def __init__(self, part, num_tokens, token_idx):
            constructions.append((part, num_tokens, token_idx))
            self.part = part
            self.num_tokens = num_tokens
            self.token_idx = token_idx

        def run_model(self, _eval_mode, _ifms):
            calls.append((self.part, self.num_tokens, self.token_idx))
            if self.part == "pre":
                return [
                    np.zeros((1, 1, self.num_tokens, 2), dtype=np.float32),
                    np.zeros((1, 1, self.num_tokens, 2), dtype=np.float32),
                    np.zeros((1, 1, self.num_tokens, 2), dtype=np.float32),
                ]
            if self.part == "cache":
                return [np.zeros((1, 1, self.num_tokens, 2), dtype=np.float32)]
            return [np.array([[[[7]]]], dtype=np.int32)]

    def get_part_model(part, num_tokens, layer_idx=None, token_idx=None):
        del layer_idx
        return FakePart(part, num_tokens, token_idx)

    attention_cfg = SimpleNamespace(
        swa_enable=False,
        sliding_window=None,
        sliding_head_dim=None,
        head_dim=2,
        num_key_value_heads=1,
        get_head_dim=lambda _layer_type: 2,
    )
    lm_cfg = SimpleNamespace(
        speculative_decoding_cfg=None,
        layer_types=["full_attention"],
        hidden_size_per_layer_input=0,
        attn_cfg=attention_cfg,
        num_hidden_layers=1,
        lm_head_num_splits=1,
        is_kv_shared_layer=lambda _layer_idx: False,
        get_kv_source_layer=lambda layer_idx: layer_idx,
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(
            pipeline_cfg=pipeline_cfg,
            lm_cfg=lm_cfg,
            model_type="llm",
        ),
        vlm_helper=SimpleNamespace(stop_tokens=set()),
        calc_freq_real_imag=lambda _use_swa: (
            np.zeros((1, 1, 1024, 1), dtype=np.float32),
            np.zeros((1, 1, 1024, 1), dtype=np.float32),
        ),
        _get_part_model=get_part_model,
    )

    output = LanguageModel.run_model(
        runtime,
        EvalMode.SDK,
        [np.zeros((1, 1, 9, 2), dtype=np.float32)],
        embeddings_tensor=np.zeros((16, 2), dtype=np.float32),
        max_new_tokens=3,
    )

    assert output.tolist() == [[7, 7, 7]]
    assert calls == [
        ("pre", 128, None),
        ("cache", 128, 0),
        ("post", 1, None),
        ("pre", 1, None),
        ("cache", 1, 127),
        ("post", 1, None),
        ("pre", 1, None),
        ("cache", 1, 127),
        ("post", 1, None),
    ]
    assert constructions == [
        ("pre", 128, None),
        ("cache", 128, 0),
        ("post", 1, None),
        ("pre", 1, None),
        ("cache", 1, 127),
    ]


def test_sdk_reference_model_is_loaded_once(monkeypatch, tmp_path):
    load_calls = []

    class FakeSdkModel:
        _net = SimpleNamespace(input_node_names=["ifm"])

        def execute(self, ifms, *, use_jax):
            assert not use_jax
            return [ifms["ifm"]]

    def load(model_name, model_path, *, include_unquantized_net):
        load_calls.append((model_name, model_path, include_unquantized_net))
        return FakeSdkModel()

    monkeypatch.setattr(base_model_module.SDKModel, "load", load)
    monkeypatch.delenv("SIMA_LMM_SDK_EVAL_USE_JAX", raising=False)
    model = BaseModel(
        cfg=SimpleNamespace(lm_cfg=SimpleNamespace(num_hidden_layers=1)),
        model_name="part",
        sima_path=tmp_path,
    )
    ifm = np.ones((1, 1, 1, 1), dtype=np.float32)

    assert np.array_equal(model._run_model_sdk_model([ifm])[0], ifm)
    assert np.array_equal(model._run_model_sdk_model([ifm])[0], ifm)
    assert load_calls == [("part", tmp_path / "sdk", False)]
