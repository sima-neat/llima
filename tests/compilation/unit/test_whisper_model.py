import pytest

from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_init_model import WhisperDecoderInitModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.model.whisper_model import WhisperModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


class _WhisperPreBuilder:
    def build_op(self, name, input_nodes, op_type, **kwargs):
        del input_nodes, op_type, kwargs
        return name

    def build_layer_norm(self, name, input_node):
        del input_node
        return name

    def build_conv(self, name, input_node):
        del input_node
        return name

    def build_split_and_concat(
        self, name, input_node, num_splits, split_axis, concat_axis
    ):
        del input_node, num_splits, split_axis, concat_axis
        return name


def test_whisper_layer_zero_pre_exposes_positioned_residual():
    model = WhisperDecoderPreModel(
        WhisperConfig(decoder_layers=2),
        "whisper_decoder_n1_pre_layer0",
        num_tokens=1,
        layer_idx=0,
    )
    model._onnx_builder = _WhisperPreBuilder()

    output_nodes = model._build_onnx_nodes(
        "model.decoder.layers.0", ["token_embedding", "position_embedding"]
    )

    assert len(output_nodes) == 4
    assert (
        output_nodes[WhisperDecoderPreModel.positioned_residual_output_idx]
        == "model.decoder.layers.0.add_embed"
    )


def test_whisper_init_routes_positioned_residual_to_layer_zero_post(monkeypatch):
    model = WhisperDecoderInitModel(
        WhisperConfig(decoder_layers=2),
        "whisper_decoder_init_layer0",
        layer_idx=0,
    )
    model._onnx_builder = object()
    monkeypatch.setattr(
        WhisperDecoderInitModel,
        "_build_position_embeddings",
        lambda self: "position_embedding",
    )
    monkeypatch.setattr(
        WhisperDecoderPreModel,
        "_build_onnx_nodes",
        lambda self, base_name, input_nodes: ["query", "key", "value", "positioned"],
    )
    monkeypatch.setattr(
        WhisperDecoderCacheModel,
        "_build_onnx_nodes",
        lambda self, base_name, input_nodes: ["self_attention"],
    )
    post_inputs = []

    def build_post(self, base_name, input_nodes):
        del self, base_name
        post_inputs.extend(input_nodes)
        return ["hidden", "encoder_key", "encoder_value"]

    monkeypatch.setattr(WhisperDecoderPostModel, "_build_onnx_nodes", build_post)

    model._build_onnx_nodes(
        "model.decoder.layers.0", ["token_embedding", "audio_features"]
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
