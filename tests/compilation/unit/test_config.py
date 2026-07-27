import json
from dataclasses import asdict
from pathlib import Path

import pytest

from sima_lmm.config.vlm_config import PipelineConfig, VlmConfig
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

    assert _layer_indices(config, "group_cache") == list(range(0, 1921, 128))
    assert _layer_indices(config, "group_sliding_cache") == [0, 128, 256, 384]


def test_group_configuration_is_automatic_and_serializable():
    config = _load_reference_config("gemma3_vlm_config.json")
    config.config_pipeline(None, None, 512, 128, 128)

    restored = VlmConfig.load(json.loads(json.dumps(asdict(config))))
    assert restored.pipeline_cfg.input_token_group_size == 128
    assert restored.pipeline_cfg.input_token_group_offsets == [0, 128, 256, 384]


def test_sliding_attention_rejects_group_at_least_as_large_as_window():
    config = _load_reference_config("gemma3_vlm_config.json")

    with pytest.raises(ValueError, match="smaller than sliding_window"):
        config.config_pipeline(None, None, 2048, 1024, 128)
