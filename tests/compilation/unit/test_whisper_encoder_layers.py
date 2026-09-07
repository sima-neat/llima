import pytest

from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.model.base import FileGenMode
from sima_lmm.model.whisper_encoder_model import WhisperEncoderModel
from sima_lmm.model.whisper_model import WhisperModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


class _EncoderBuilder:
    def build_layer_norm(self, name, input_node):
        return (name, input_node)


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


def test_encoder_layer_zero_includes_feature_extractor(monkeypatch):
    model = WhisperEncoderModel(
        WhisperConfig(encoder_layers=3),
        "whisper_encoder_layer0",
        layer_idx=0,
    )
    model._onnx_builder = _EncoderBuilder()
    calls = []
    monkeypatch.setattr(
        model,
        "_build_feature_extractor",
        lambda base_name, input_node: calls.append((base_name, input_node)) or "features",
    )
    monkeypatch.setattr(
        model,
        "_build_encoder_layer",
        lambda base_name, input_node: (base_name, input_node),
    )

    output = model._build_layer_onnx_nodes("model.encoder", ["mel"])

    assert calls == [("model.encoder", "mel")]
    assert output == [("model.encoder.layers.0", "features")]


def test_final_encoder_layer_includes_output_layer_norm(monkeypatch):
    model = WhisperEncoderModel(
        WhisperConfig(encoder_layers=3),
        "whisper_encoder_layer2",
        layer_idx=2,
    )
    model._onnx_builder = _EncoderBuilder()
    monkeypatch.setattr(
        model,
        "_build_encoder_layer",
        lambda base_name, input_node: (base_name, input_node),
    )

    output = model._build_layer_onnx_nodes("model.encoder", ["hidden"])

    assert output == [
        (
            "model.encoder.layer_norm",
            ("model.encoder.layers.2", "hidden"),
        )
    ]
