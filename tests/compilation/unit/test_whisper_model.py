import pytest

from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.model.whisper_model import WhisperModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def test_log_probe_reuses_final_decoder_init_model():
    disabled_model = WhisperModel(
        WhisperConfig(decoder_layers=2, log_probe_enabled=False),
        "whisper",
        use_future_token_mask=True,
    )
    enabled_model = WhisperModel(
        WhisperConfig(decoder_layers=2, log_probe_enabled=True),
        "whisper",
        use_future_token_mask=True,
    )

    disabled_final_init = disabled_model._get_part_model("init", layer_idx=1)
    first_init = enabled_model._get_part_model("init", layer_idx=0)
    enabled_final_init = enabled_model._get_part_model("init", layer_idx=1)

    assert not disabled_final_init.enable_log_probe
    assert not first_init.enable_log_probe
    assert enabled_final_init.enable_log_probe
    assert disabled_final_init.model_name == enabled_final_init.model_name
    assert enabled_final_init.model_name == "whisper_decoder_init_layer1"
