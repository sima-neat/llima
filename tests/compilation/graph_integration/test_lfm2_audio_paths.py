"""Checkpoint-backed floating reference checks for the audio backbone."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from safetensors import safe_open
from transformers import Lfm2Config
from transformers.models.lfm2.modeling_lfm2 import Lfm2DecoderLayer, Lfm2RotaryEmbedding

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from sima_lmm.hf.lfm2_audio import Lfm2AudioCheckpoint
from sima_lmm.model.language_cache_model import LanguageCacheModel
from sima_lmm.model.language_conv_model import LanguageConvModel
from sima_lmm.model.language_post_model import LanguagePostModel
from sima_lmm.model.language_pre_model import LanguagePreModel
from sima_lmm.model.lfm2_audio_model import AudioFinalConvModel, AudioProjectionModel
from sima_lmm.model.lfm2_audio_compiler import plan_audio_components
from sima_lmm.model.model_graph import ModelGraph
from tests.compilation.cases import LFM2_AUDIO_CASE
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]


def _graph(model):
    captured = []
    def save(graph, outputs=None, **_kwargs):
        graph.finish(outputs)
        captured.append(graph)
    with patch.object(ModelGraph, "save", save):
        model.generate_graph({}, True)
    return captured[0]


@pytest.mark.parametrize("index", [0, 2, 15], ids=["conv-cache", "attention-prefill", "final-hidden"])
def test_audio_backbone_matches_checkpoint(model_inputs_path, tmp_path, index):
    path = require_readable_path(model_inputs_path / LFM2_AUDIO_CASE.model_folder, "LFM2 Audio checkpoint")
    checkpoint = Lfm2AudioCheckpoint.from_directory(path)
    source = checkpoint.language_backbone()
    cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, path, source.config)
    hf = Lfm2Config(**checkpoint.config["lfm"])
    hf._attn_implementation = "eager"
    with torch.device("meta"):
        reference = Lfm2DecoderLayer(hf, index)
    with safe_open(path / "model.safetensors", framework="pt") as weights:
        prefix = f"lfm.layers.{index}."
        params = {name.removeprefix(prefix): weights.get_tensor(name).float()
                  for name in weights.keys() if name.startswith(prefix)}
    reference.load_state_dict(params, assign=True)
    reference.eval()
    rng = np.random.default_rng(17)
    x = rng.normal(0, 0.1, (1, 5, hf.hidden_size)).astype(np.float32)
    common = dict(hf_model=source, sima_path=tmp_path, layer_idx=index)
    if cfg.lm_cfg.layer_types[index] == "conv":
        cache = np.zeros((1, 1, hf.conv_L_cache - 1, hf.hidden_size), np.float32)
        outputs = []
        for start, length in ((0, 4), (4, 1)):
            model = LanguageConvModel(cfg, f"conv_n{length}", num_tokens=length, final_softcapping=None, **common)
            hidden, cache = _graph(model).run(input=x[:, None, start:start + length], conv_cache=cache)
            cache = cache[:, :, -(hf.conv_L_cache - 1):].copy()
            if index == hf.num_hidden_layers - 1:
                final = AudioFinalConvModel(cfg, f"final_n{length}", num_tokens=length, final_softcapping=None, **common)
                hidden = _graph(final).run(input=hidden)[0]
            outputs.append(hidden[:, 0])
        actual = np.concatenate(outputs, axis=1)
        with torch.no_grad():
            expected = reference(torch.from_numpy(x)).numpy()
        if index == hf.num_hidden_layers - 1:
            scale = source.load_np_param("model.embedding_norm.weight").astype(np.float32)
            expected = expected / np.sqrt((expected ** 2).mean(-1, keepdims=True) + hf.norm_eps) * scale
    else:
        length = 4
        x = x[:, :length]
        rotary = Lfm2RotaryEmbedding(hf)
        with torch.no_grad():
            cos, sin = rotary(torch.from_numpy(x), torch.arange(length)[None])
        half = hf.hidden_size // hf.num_attention_heads // 2
        pre = LanguagePreModel(cfg, "pre", num_tokens=length, **common)
        query, key, value = _graph(pre).run(
            input=x[:, None], freq_real=cos.numpy()[:, None, :, :half].copy(),
            freq_imag=sin.numpy()[:, None, :, :half].copy(),
        )
        cache = LanguageCacheModel(cfg, "cache", hf_model=source, num_tokens=length, token_idx=0, logit_softcapping=None)
        attention = _graph(cache).run(input=query, cached_keys=key, cached_values=value)[0]
        post = LanguagePostModel(cfg, "post", num_tokens=length, final_softcapping=None, **common)
        actual = _graph(post).run(input=x[:, None], self_attn=attention)[0][:, 0]
        mask = torch.full((1, 1, length, length), torch.finfo(torch.float32).min).triu(1)
        with torch.no_grad():
            expected = reference(torch.from_numpy(x), position_embeddings=(cos, sin), attention_mask=mask).numpy()
    np.savez(tmp_path / f"audio_layer{index}_reference.npz", actual=actual, expected=expected)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)


def test_audio_depth_head_preserves_source_normalization(model_inputs_path, tmp_path):
    path = require_readable_path(model_inputs_path / LFM2_AUDIO_CASE.model_folder, "LFM2 Audio checkpoint")
    checkpoint = Lfm2AudioCheckpoint.from_directory(path)
    model = next(unit.model for unit in plan_audio_components(checkpoint, tmp_path)
                 if unit.model.model_name == "depth_head0")
    assert isinstance(model, AudioProjectionModel)
    values = np.random.default_rng(28).normal(0, .001, (1, 1, 1, 1024)).astype(np.float32)
    actual = _graph(model).run(hidden=values)[0]
    with safe_open(path / "model.safetensors", framework="pt") as weights:
        scale = weights.get_tensor("depth_embeddings.0.embedding_norm.weight").float()
        projection = weights.get_tensor("depth_embeddings.0.to_logits.weight").float()
    value = torch.from_numpy(values)
    # SharedEmbedding's norm_eps defaults to 1e-5 in the pinned source.
    normalized = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5) * scale
    expected = torch.nn.functional.linear(normalized, projection).numpy()
    np.savez(tmp_path / "audio_depth_head_reference.npz", actual=actual, expected=expected)
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
