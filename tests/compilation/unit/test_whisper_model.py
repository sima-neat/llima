import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import Mock

from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.model.base import FileGenMode
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_init_model import WhisperDecoderInitModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.model.whisper_model import WhisperModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize("model_sdk", [None, True])
def test_whisper_native_generation_is_default(monkeypatch, model_sdk):
    model = WhisperModel(
        WhisperConfig(encoder_layers=1), "whisper", use_future_token_mask=True,
        hf_model=SimpleNamespace(load_all_params=lambda: None, unload_all_params=lambda: None),
    )
    modes = []
    monkeypatch.setattr(model, "gen_devkit_files", lambda **_: modes.append(FileGenMode.DEVKIT))
    monkeypatch.setattr(
        model, "gen_files_from_model_list", lambda _models, mode, *_args: modes.append(mode)
    )
    kwargs = {} if model_sdk is None else {"model_sdk": model_sdk}
    model.gen_files(FileGenMode.ALL, part="encoder", part_idx=0, **kwargs)
    assert modes == [
        FileGenMode.DEVKIT,
        FileGenMode.SOURCE_TO_FP,
        FileGenMode.FP_TO_QUANT,
        FileGenMode.MODEL_SDK_COMPILE,
    ]


def test_whisper_layer_zero_pre_exposes_positioned_residual(monkeypatch):
    model = WhisperDecoderPreModel(
        WhisperConfig(decoder_layers=2),
        "whisper_decoder_n1_pre_layer0",
        num_tokens=1,
        layer_idx=0,
    )
    builder = Mock()
    builder.create_add_node.return_value = "positioned"
    graph = Mock()
    monkeypatch.setattr(
        "sima_lmm.model.whisper_decoder_pre_model.ModelGraph.from_builder", lambda *_: graph
    )
    output_nodes = model._build_sima_nodes(
        builder, ["token_embedding", "position_embedding"], quantizable=True,
    )

    assert len(output_nodes) == 4
    assert (
        output_nodes[WhisperDecoderPreModel.positioned_residual_output_idx]
        == "positioned"
    )
    builder.create_add_node.assert_called_once_with("token_embedding", "position_embedding")


def test_whisper_init_routes_positioned_residual_to_layer_zero_post(monkeypatch):
    model = WhisperDecoderInitModel(
        WhisperConfig(decoder_layers=2),
        "whisper_decoder_init_layer0",
        layer_idx=0,
    )
    builder = Mock()
    graph = Mock()
    graph.parameter.return_value = np.zeros((4, model.cfg.d_model), np.float32)
    graph.constant.return_value = "position_embedding"
    monkeypatch.setattr(
        "sima_lmm.model.whisper_decoder_init_model.ModelGraph.from_builder", lambda *_: graph
    )
    monkeypatch.setattr(
        WhisperDecoderPreModel,
        "_build_sima_nodes",
        lambda self, builder, inputs, quantizable: ["query", "key", "value", "positioned"],
    )
    monkeypatch.setattr(
        WhisperDecoderCacheModel,
        "_build_sima_nodes",
        lambda self, builder, inputs, quantizable: ["self_attention"],
    )
    post_inputs = []

    def build_post(self, builder, inputs, quantizable):
        del self, builder, quantizable
        post_inputs.extend(inputs)
        return ["hidden", "encoder_key", "encoder_value"]

    monkeypatch.setattr(WhisperDecoderPostModel, "_build_sima_nodes", build_post)

    model._build_sima_nodes(
        builder, ["token_embedding", "audio_features"], quantizable=True,
    )

    assert post_inputs[0] == "positioned"


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
