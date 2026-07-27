import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sima_lmm.config.vlm_config import (
    PipelineConfig,
    VlmConfig,
    group_cache_model_indices,
    single_cache_model_indices,
)
from sima_lmm.config.whisper_config import WhisperConfig


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]

REFERENCE_CONFIGS_PATH = (
    Path(__file__).parents[1] / "configuration" / "references"
)


def _load_reference_config(filename: str) -> VlmConfig:
    config = json.loads((REFERENCE_CONFIGS_PATH / filename).read_text())
    return VlmConfig.load(config)


def _layer_indices(config: VlmConfig, part: str) -> list[int]:
    return [layer.part_idx for layer in config.get_layer_ids() if layer.part == part]


def test_whisper_finds_generation_config_in_cache_root(tmp_path):
    snapshot_path = tmp_path / "snapshots" / "revision"
    snapshot_path.mkdir(parents=True)
    generation_config = {
        "lang_to_id": {
            "<|en|>": 50259,
            "<|de|>": 50261,
        }
    }
    (snapshot_path / "generation_config.json").write_text(
        json.dumps(generation_config), encoding="utf-8"
    )

    config = WhisperConfig.from_hf_config(
        str(tmp_path), {"architectures": ["WhisperForConditionalGeneration"]}
    )

    assert config.num_languages == 2


def test_embedding_scale_is_absent_without_embedding_quantization():
    assert PipelineConfig().embeddings_scale is None


def test_default_max_num_tokens():
    assert PipelineConfig().max_num_tokens == 4096


@pytest.mark.parametrize(
    ("max_num_tokens", "expected_long_context_mask"),
    [(1024, None), (2048, None), (3072, 1024), (4096, 1024)],
)
def test_long_context_future_token_mask_resolution(
    max_num_tokens: int, expected_long_context_mask: int | None
):
    config = PipelineConfig()
    config.set_max_num_tokens(max_num_tokens)
    config.set_future_token_mask_size(128)

    assert config.future_token_mask_size == 128
    assert config.long_context_future_token_mask_size == expected_long_context_mask
    assert config.get_cache_mask_size("full_attention", 2048, is_group=False) == 128
    assert config.get_cache_mask_size("full_attention", 2049, is_group=False) == (
        expected_long_context_mask or 128
    )
    assert (
        config.get_cache_mask_size(
            "sliding_attention", max_num_tokens, is_group=False
        )
        == 128
    )


def test_invalid_future_token_mask_is_rejected_for_long_context():
    config = PipelineConfig(max_num_tokens=2048)

    with pytest.raises(ValueError, match="greater than zero"):
        config.set_future_token_mask_size(0)


def test_legacy_pipeline_config_uses_stored_mask_for_all_attention_types():
    config = PipelineConfig(max_num_tokens=2048, future_token_mask_size=128)

    assert config.long_context_future_token_mask_size is None
    assert config.get_cache_mask_size("full_attention", 2048, is_group=False) == 128
    assert (
        config.get_cache_mask_size("sliding_attention", 2048, is_group=False)
        == 128
    )


@pytest.mark.parametrize(
    ("sliding_window", "expected_transition"),
    [(512, 384), (1024, 896)],
)
def test_gemma3_automatic_sliding_cache_transition(
    sliding_window: int, expected_transition: int
):
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = sliding_window
    config.config_pipeline(None, None, 2048, 128, 128)

    assert expected_transition in _layer_indices(config, "group_cache")
    assert _layer_indices(config, "group_sliding_cache") == []


def test_shared_sliding_cache_includes_full_and_sliding_mask_buckets():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = 512
    config.config_pipeline(None, None, 2048, 128, 128)

    assert _layer_indices(config, "group_cache") == list(range(0, 2048, 128))
    assert _layer_indices(config, "single_cache") == list(range(127, 2048, 128))


def test_shared_sliding_cache_transition_does_not_add_execution_offset():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.lm_cfg.attn_cfg.sliding_window = 1000
    config.config_pipeline(None, None, 2048, 128, 128)

    transition = 872
    assert transition not in config.pipeline_cfg.input_token_group_offsets
    assert transition in _layer_indices(config, "group_cache")


def test_gemma4_keeps_separate_sliding_cache_models():
    config = _load_reference_config("gemma4_e2b_it_vlm_config.json")
    config.config_pipeline(None, None, 2048, 128, 128)

    assert _layer_indices(config, "group_cache") == list(range(0, 2048, 128))
    assert _layer_indices(config, "single_cache") == list(range(127, 2048, 128))
    assert _layer_indices(config, "group_sliding_cache") == [0, 128, 256, 384]
    assert _layer_indices(config, "single_sliding_cache") == [127, 255, 383, 511]


def test_non_default_mask_only_buckets_single_cache_models_through_2k():
    config = _load_reference_config("gemma4_e2b_it_vlm_config.json")
    config.config_pipeline(None, None, 2048, 128, 256)

    assert _layer_indices(config, "group_cache") == list(range(0, 2048, 128))
    assert _layer_indices(config, "single_cache") == list(range(255, 2048, 256))
    assert _layer_indices(config, "group_sliding_cache") == [0, 128, 256, 384]
    assert _layer_indices(config, "single_sliding_cache") == [255, 511]


@pytest.mark.parametrize("max_num_tokens", [128, 512, 2049, 2500])
def test_unaligned_max_num_tokens_is_rejected(max_num_tokens: int):
    config = PipelineConfig()

    with pytest.raises(ValueError, match="multiple of 1024"):
        config.set_max_num_tokens(max_num_tokens)


def test_long_context_mask_starts_after_2k():
    config = PipelineConfig()
    config.set_max_num_tokens(8192)
    config.set_group_size(128)
    config.set_future_token_mask_size(256)

    expected_group_indices = [*range(0, 2048, 128), *range(2944, 8065, 1024)]
    expected_single_indices = [
        *range(255, 2048, 256),
        *range(3071, 8192, 1024),
    ]
    assert group_cache_model_indices(config) == expected_group_indices
    assert single_cache_model_indices(config) == expected_single_indices


def test_long_context_mask_supports_groups_larger_than_base_mask():
    config = PipelineConfig()
    config.set_max_num_tokens(8192)
    config.set_group_size(320)
    config.set_future_token_mask_size(128)

    assert group_cache_model_indices(config) == [
        0,
        320,
        640,
        960,
        1280,
        1600,
        2752,
        3776,
        4800,
        5824,
        6848,
        7872,
    ]


def test_group_configuration_is_automatic_and_serializable():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.config_pipeline(None, None, 1024, 128, 128)

    restored = VlmConfig.load(json.loads(json.dumps(asdict(config))))
    assert restored.pipeline_cfg.input_token_group_size == 128
    assert restored.pipeline_cfg.input_token_group_offsets == list(range(0, 1024, 128))


def test_sliding_attention_rejects_group_at_least_as_large_as_window():
    config = _load_reference_config("gemma3_vlm_config.json")

    with pytest.raises(ValueError, match="smaller than sliding_window"):
        config.config_pipeline(None, None, 2048, 1024, 128)
