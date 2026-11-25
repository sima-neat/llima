from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from sima_lmm.config.vlm_config import VlmConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.base import BaseModel
from sima_lmm.model.lfm2_audio_model import (
    AudioEncoderSubsamplingModel, AudioProjectionModel, _conformer_relative_shift,
)
from sima_lmm.model.model_graph import ModelGraph, activation_dtype


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _graph(model, quantizable):
    captured = []
    def save(graph, outputs=None, **_kwargs):
        graph.finish(outputs)
        captured.append(graph)
    with patch.object(ModelGraph, "save", save):
        model.generate_graph({}, quantizable)
    return captured[0]


@dataclass
class _RelativeShiftModel(BaseModel):
    length: int
    heads: int = 2

    def generate_graph(self, layer_cfg, quantizable):
        graph = ModelGraph(self, {"scores": (1, self.heads, self.length, 2 * self.length)}, quantizable)
        graph.save([_conformer_relative_shift(graph, graph.inputs["scores"], self.length, self.heads)])


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("length", [32, 33, 48, 51, 64, 188, 376])
def test_audio_relative_shift_preserves_source_layout(tmp_path, quantizable, length):
    source = LocalHuggingFaceModel(tmp_path, {}, {}, {}, {}, {}, params={})
    model = _RelativeShiftModel(VlmConfig(), "relative_shift", hf_model=source, length=length)
    values = np.random.default_rng(39).normal(0, .1, (1, 2, length, 2 * length)).astype(
        activation_dtype(quantizable)
    )
    actual = _graph(model, quantizable).run(scores=values)[0]
    # Transformer-XL's reference shift: flatten away the first T values, then
    # recover rows of width 2*T-1 and retain the first T keys in each row.
    expected = values.reshape(1, 2, 2 * length, length)[:, :, 1:].reshape(
        1, 2, length, 2 * length - 1
    )[..., :length]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("quantizable", [True, False])
def test_audio_projection_keeps_bias_and_output_slice(tmp_path, quantizable):
    rng = np.random.default_rng(42)
    weight = rng.normal(0, .03, (8, 32)).astype(np.float32)
    bias = np.arange(8, dtype=np.float32) * .1
    value = rng.normal(0, .2, (1, 1, 3, 32)).astype(activation_dtype(quantizable))
    source = LocalHuggingFaceModel(tmp_path, {}, {}, {}, {}, {}, params={
        "depth_linear.weight": weight, "depth_linear.bias": bias,
    })
    model = AudioProjectionModel(
        VlmConfig(), "projection", hf_model=source, num_tokens=3, input_dim=32,
        weight_name="depth_linear.weight", output_begin=2, output_end=6,
    )
    actual = _graph(model, quantizable).run(hidden=value)[0]
    expected = value.astype(np.float32) @ weight[2:6].T + bias[2:6]
    np.testing.assert_allclose(actual, expected, rtol=2e-5 if quantizable else .02,
                               atol=2e-6 if quantizable else .004)


@pytest.mark.parametrize("quantizable", [True, False])
@pytest.mark.parametrize("valid_frames", [1, 17, 37, 64])
def test_audio_subsampling_preserves_physical_length_boundaries(tmp_path, quantizable, valid_frames):
    with torch.random.fork_rng():
        torch.manual_seed(81)
        conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16), nn.Conv2d(16, 16, 1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1, groups=16), nn.Conv2d(16, 16, 1), nn.ReLU(),
        ).eval()
        output = nn.Linear(16 * 16, 32).eval()
    params = {f"conformer.pre_encode.conv.{name}": value.numpy()
              for name, value in conv.state_dict().items()}
    params.update({f"conformer.pre_encode.out.{name}": value.numpy()
                   for name, value in output.state_dict().items()})
    source = LocalHuggingFaceModel(tmp_path, {}, {}, {}, {}, {}, params=params)
    model = AudioEncoderSubsamplingModel(
        VlmConfig(), "subsampling", hf_model=source, input_frames=64, channels=16,
    )
    rng = np.random.default_rng(12)
    values = rng.normal(0, .2, (1, valid_frames, 128)).astype(activation_dtype(quantizable))
    padded = np.zeros((1, 1, 64, 128), dtype=values.dtype)
    padded[:, 0, :valid_frames] = values
    masks = {}
    for index, divisor in enumerate((2, 4, 8), start=1):
        mask = np.zeros((1, (64 + divisor - 1) // divisor, 1, 1), dtype=values.dtype)
        mask[:, :(valid_frames + divisor - 1) // divisor] = 1
        masks[f"stage{index}_valid"] = mask
    actual = _graph(model, quantizable).run(features=padded, **masks)[0]
    with torch.no_grad():
        hidden = conv(torch.from_numpy(values.astype(np.float32))[:, None])
        hidden = hidden.transpose(1, 2).reshape(1, (valid_frames + 7) // 8, -1)
        expected = output(hidden).numpy()
    np.testing.assert_allclose(actual[:, 0, :expected.shape[1]], expected,
                               rtol=2e-5 if quantizable else .03, atol=2e-6 if quantizable else .003)
