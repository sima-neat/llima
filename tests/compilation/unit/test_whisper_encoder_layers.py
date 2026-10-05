import pytest
from unittest.mock import Mock

from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.model.base import FileGenMode
from sima_lmm.model.whisper_encoder_model import WhisperEncoderModel
from sima_lmm.model.whisper_model import WhisperModel
import sima_lmm.model.whisper_encoder_model as encoder_module


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.fixture
def encoder_builder(monkeypatch):
    builder = Mock()
    graph = Mock()
    graph.project_heads.return_value = ["head"]
    graph.layer_norm.side_effect = lambda name, node, **kwargs: (name, node)
    monkeypatch.setattr(encoder_module.ModelGraph, "from_builder", lambda *_: graph)
    return builder, graph.conv, graph.layer_norm


def test_encoder_part_generates_one_model_per_layer(monkeypatch):
    model = WhisperModel(
        WhisperConfig(encoder_layers=3),
        "whisper",
        use_future_token_mask=True,
    )
    generated_models = []

    def capture_models(model_list, *args):
        del args
        generated_models.extend(part_model for part_model, _ in model_list)

    monkeypatch.setattr(model, "gen_files_from_model_list", capture_models)

    model.gen_files(FileGenMode.MODEL_SDK_COMPILE, part="encoder")

    assert [part_model.model_name for part_model in generated_models] == [
        "whisper_encoder_layer0",
        "whisper_encoder_layer1",
        "whisper_encoder_layer2",
    ]
    assert [part_model.layer_idx for part_model in generated_models] == [0, 1, 2]


def test_encoder_part_idx_generates_only_requested_layer(monkeypatch):
    model = WhisperModel(
        WhisperConfig(encoder_layers=3),
        "whisper",
        use_future_token_mask=True,
    )
    generated_models = []

    def capture_models(model_list, *args):
        del args
        generated_models.extend(part_model for part_model, _ in model_list)

    monkeypatch.setattr(model, "gen_files_from_model_list", capture_models)

    model.gen_files(FileGenMode.MODEL_SDK_COMPILE, part="encoder", part_idx=1)

    assert [part_model.model_name for part_model in generated_models] == [
        "whisper_encoder_layer1"
    ]


def test_encoder_layer_zero_includes_feature_extractor(monkeypatch, encoder_builder):
    model = WhisperEncoderModel(
        WhisperConfig(encoder_layers=3),
        "whisper_encoder_layer0",
        layer_idx=0,
    )
    builder, conv, norm = encoder_builder
    positions = Mock()
    monkeypatch.setattr(encoder_module.ModelGraph.from_builder(model, builder), "parameter", lambda _: positions)
    model._build_sima_nodes(builder, ["mel"], quantizable=True)

    first, second = conv.call_args_list[:2]
    assert first.args[:2] == ("model.encoder.conv1", "mel")
    assert second.args[0] == "model.encoder.conv2"
    assert first.kwargs["stride"] == (1, 1)
    assert second.kwargs["stride"] == (1, 2)
    assert first.kwargs["padding"] == second.kwargs["padding"] == ((0, 0), (1, 1))
    assert "model.encoder.layer_norm" not in [call.args[0] for call in norm.call_args_list]


def test_final_encoder_layer_includes_output_layer_norm(encoder_builder):
    model = WhisperEncoderModel(
        WhisperConfig(encoder_layers=3),
        "whisper_encoder_layer2",
        layer_idx=2,
    )
    builder, conv, norm = encoder_builder
    output = model._build_sima_nodes(builder, ["hidden"], quantizable=True)

    assert output == [norm.call_args.args[:2]]
    assert norm.call_args.args[0] == "model.encoder.layer_norm"
    assert all(call.args[0].startswith("model.encoder.layers.2.") for call in conv.call_args_list)
