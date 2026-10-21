from types import SimpleNamespace
from dataclasses import asdict
import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig
from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder

from sima_lmm.config.vlm_config import AudioModelConfig, AudioPreprocessorConfig, ModelFormat, VlmConfig
from sima_lmm.model import vision_language_model
from sima_lmm.model.gemma4_audio_model import Gemma4AudioLayerModel
from sima_lmm.model.model_graph import ModelGraph
from sima_lmm.preproc.audio_preprocessing import gemma4_audio_features, load_audio


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def test_audio_duration_controls_encoder_capacity_and_input_limit(monkeypatch, tmp_path):
    reference = Path(__file__).parents[1] / "configuration/references/gemma4_e2b_it_vlm_config.json"
    raw_config = json.loads(reference.read_text())
    raw_config["audio_cfg"].update({"output_proj_dims": 1536,
        "attention_context_left": 13, "attention_context_right": 0,
        "sampling_rate": 16000, "num_mel_bins": 128, "frame_length": 320,
        "hop_length": 160, "fft_length": 512})
    config = VlmConfig.load(raw_config)
    monkeypatch.setattr(vision_language_model, "model_file_type", lambda _: ModelFormat.FORMAT_HF)
    monkeypatch.setattr(vision_language_model.LocalHuggingFaceModel, "create_from_directory",
                        lambda **_: SimpleNamespace(config={}))
    monkeypatch.setattr(VlmConfig, "from_hf_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(vision_language_model, "VlmHelper", lambda *args, **kwargs: None)
    model = vision_language_model.VisionLanguageModel.from_hf_cache(
        "audio", tmp_path, tmp_path, 2048, max_audio_duration=10,
    )
    audio = model.cfg.audio_cfg
    assert (audio.max_tokens, audio.num_frames, audio.padded_tokens) == (250, 1000, 252)
    frontend = AudioPreprocessorConfig()
    frontend.set_config(Gemma4AudioFeatureExtractor(), audio)
    assert (frontend.sampling_rate, frontend.num_mel_bins, frontend.frame_length,
            frontend.hop_length, frontend.fft_length) == (16000, 128, 320, 160, 512)
    assert frontend.feature_extractor_type == "Gemma4AudioFeatureExtractor"
    features, mask = gemma4_audio_features(np.zeros(160000, np.float32), audio, frontend)
    assert features.shape == (1, 1, 1000, 128)
    assert np.count_nonzero(mask[0, 0, ::4]) == 250
    with pytest.raises(ValueError, match="compiled token budget"):
        gemma4_audio_features(np.zeros(160001, np.float32), audio, frontend)


@pytest.mark.parametrize("rate", [16000, 48000])
def test_load_audio_preserves_resampler_tail_and_budget(tmp_path, rate):
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((2, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(np.full((rate // 10, 2), 8192, dtype="<i2").tobytes())
    samples = load_audio(path, max_samples=1600)
    assert samples.shape == (1600,)
    np.testing.assert_array_equal(samples, np.full(1600, 0.25, dtype=np.float32))
    with pytest.raises(ValueError, match="compiled token budget"):
        load_audio(path, max_samples=1599)


@pytest.mark.parametrize("samples", [641, 801, 16000, 480000])
def test_gemma_audio_features_and_mask_match_transformers(samples):
    audio = AudioModelConfig()
    extractor = Gemma4AudioFeatureExtractor()
    wave = np.random.default_rng(43).normal(0, 0.1, samples).astype(np.float32)
    frontend = AudioPreprocessorConfig()
    frontend.set_config(extractor, audio)
    frontend = AudioPreprocessorConfig(**json.loads(json.dumps(asdict(frontend))))
    actual, mask = gemma4_audio_features(wave, audio, frontend)
    expected = extractor([wave], sampling_rate=16000)
    features, valid = expected.input_features[0], expected.input_features_mask[0]
    np.testing.assert_allclose(actual[0, 0, :len(features)], features, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(mask[0, 0, :len(valid), 0], valid)
    assert not mask[0, 0, len(valid):].any()


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_gemma_audio_encoder_layers_match_transformers(layer_idx):
    torch.manual_seed(7)
    config = Gemma4AudioConfig(hidden_size=32, num_attention_heads=2,
                               num_hidden_layers=3, subsampling_conv_channels=[8, 4])
    encoder = Gemma4AudioModel(config).eval()
    projector = Gemma4MultimodalEmbedder(config, SimpleNamespace(hidden_size=48)).eval()
    params = {f"model.audio_tower.{name}": value.detach().numpy()
              for name, value in encoder.state_dict().items()}
    params.update({f"model.embed_audio.{name}": value.detach().numpy()
                   for name, value in projector.state_dict().items()})
    audio = AudioModelConfig(hidden_size=32, num_attention_heads=2, num_hidden_layers=3,
                            subsampling_conv_channels=[8, 4], max_tokens=24)
    cfg = SimpleNamespace(audio_cfg=audio, pipeline_cfg=SimpleNamespace(quantize_embeddings=False))
    model = Gemma4AudioLayerModel(cfg, "audio_test", layer_idx=layer_idx)
    model.get_hf_param, model.check_hf_param = params.__getitem__, params.__contains__
    rng = np.random.default_rng(32)
    valid_frames = 73
    frame_mask = np.arange(audio.num_frames) < valid_frames
    mask = frame_mask.astype(np.float32).reshape(1, 1, -1, 1)
    if layer_idx == 0:
        hidden = rng.normal(0, 0.2, (1, 1, audio.num_frames, 8)).astype(np.float32)
        features = torch.from_numpy(hidden[:, 0])
        with torch.no_grad():
            reference, _ = encoder.subsample_conv_projection(features, torch.from_numpy(frame_mask[None]))
    else:
        hidden = rng.normal(0, 0.2, (1, 1, audio.padded_tokens, audio.hidden_size)).astype(np.float32)
        reference = torch.from_numpy(hidden[:, 0])
    valid = torch.from_numpy(frame_mask[::4][None])
    positions = encoder.rel_pos_enc(reference)
    # Construct the exact query-local mask without depending on the global
    # Transformers attention-backend selection.
    qpos, kpos = torch.arange(audio.padded_tokens)[:, None], torch.arange(audio.padded_tokens)[None]
    allowed = (kpos <= qpos) & (kpos >= qpos - 12)
    full_mask = allowed[None, None] & valid[:, None, None]
    blocked_mask = encoder._convert_4d_mask_to_blocked_5d(full_mask)
    with torch.no_grad():
        expected = encoder.layers[layer_idx](reference, blocked_mask, positions)
        if layer_idx == 2:
            expected = projector(encoder.output_proj(expected))
    with patch.object(ModelGraph, "save", autospec=True) as save:
        model.generate_graph({}, quantizable=True)
    graph, outputs = save.call_args.args
    graph.finish(outputs)
    actual = graph.run(hidden=hidden, mask=mask)[0]
    np.testing.assert_allclose(actual[:, 0], expected.numpy(), rtol=2e-4, atol=3e-5)
