import json

import pytest

from sima_lmm.config.whisper_config import WhisperConfig


pytestmark = pytest.mark.premerge


def test_from_hf_config_finds_generation_config_in_cache_root(tmp_path):
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
