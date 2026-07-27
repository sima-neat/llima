from types import SimpleNamespace

import pytest

from sima_lmm.config.vlm_config import PipelineConfig
from sima_lmm.model.language_cache_model import (
    LanguageCacheModel,
    _get_bmm2_reduction_ranges,
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
