"""Unified geometry and image-window execution against the Transformers decoder."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import wave

import numpy as np
import pytest
import torch
from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration

from sima_lmm.config.vlm_config import AudioModelConfig, ModelFormat, VlmConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.base import BaseModel, EvalMode
from sima_lmm.model.language_model import LanguageModel
from sima_lmm.model.model_graph import ModelGraph, tensor_type
from sima_lmm.preproc.gemma4_unified import cache_view, image_windows
from sima_lmm.preproc.gemma4_unified import audio_frames

pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.fixture
def unified(tmp_path):
    torch.manual_seed(249)
    config = Gemma4UnifiedConfig(
        text_config=dict(
            vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
            num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
            head_dim=16, global_head_dim=32, sliding_window=256, attention_k_eq_v=True,
            layer_types=(["sliding_attention"] * 2 + ["full_attention"]) * 2,
            rope_parameters={
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                                   "partial_rotary_factor": 0.5},
            },
        ),
        vision_config=dict(patch_size=4, pooling_kernel_size=2, mm_embed_dim=64,
                           mm_posemb_size=16, output_proj_dims=64),
        image_token_id=60, audio_token_id=61, video_token_id=62,
    )
    config._attn_implementation = "eager"
    reference = Gemma4UnifiedForConditionalGeneration(config).eval()
    for layer in reference.model.language_model.layers:
        layer.layer_scalar.fill_(0.75)
    params = {name: value.detach().numpy() for name, value in reference.state_dict().items()}
    raw = config.to_dict()
    raw.update(architectures=["Gemma4UnifiedForConditionalGeneration"], dtype="float32")
    cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, Path("tiny"), raw)
    assert cfg.vm_cfg.max_soft_tokens == cfg.mm_cfg.mm_tokens_per_image == 384
    cfg.vm_cfg.max_soft_tokens = 16
    cfg.config_pipeline(None, None, 1024, 128, 1)
    cfg.pipeline_cfg.return_logits = True
    source = LocalHuggingFaceModel(tmp_path, {}, raw, {name: "memory" for name in params}, {}, {}, params=params)
    language = LanguageModel(cfg, "tiny", hf_model=source,
                             vlm_helper=SimpleNamespace(stop_tokens=set(range(64))))
    return reference, language, params


@pytest.mark.parametrize("quantize_embeddings,quantize_kv_cache", [
    (False, False), (True, False), (False, True), (True, True),
])
def test_unified_geometry_and_component_contract(unified, quantize_embeddings, quantize_kv_cache):
    _, model, _ = unified
    cfg = model.cfg
    cfg.pipeline_cfg.quantize_embeddings = quantize_embeddings
    cfg.pipeline_cfg.quantize_kv_cache = quantize_kv_cache
    assert cfg.lm_cfg.attn_cfg.get_num_key_value_heads("sliding_attention") == 2
    assert cfg.lm_cfg.attn_cfg.get_num_key_value_heads("full_attention") == 1
    assert cfg.get_prefill_cache_capacity("sliding_attention") == 512
    assert cfg.get_prefill_cache_capacity("full_attention") == 1024
    ids = cfg.get_layer_ids()
    assert [(p.part, p.part_idx) for p in ids if p.part.startswith("group") and "cache" in p.part] == [
        *[("group_cache", at) for at in range(0, 1024, 128)],
        *[("group_sliding_cache", at) for at in range(0, 512, 128)],
    ]
    assert not any("per_layer" in p.part for p in ids)
    assert VlmConfig.load(asdict(cfg)) == cfg
    for kind, index in [("pre", 0), ("post", 0), ("cache", 0),
                        *[("sliding_cache", at) for at in range(0, 512, 128)]]:
        part = model._get_part_model(kind, 128, **({"layer_idx": index} if kind in ("pre", "post") else {"token_idx": index}))
        with patch.object(ModelGraph, "save", autospec=True) as save:
            part.generate_graph({}, quantizable=True)
        graph, outputs = save.call_args.args
        if kind in ("pre", "post"):
            assert ("input_scale" in graph.inputs) == quantize_embeddings
            assert tensor_type(graph.inputs["input"]).scalar.numpy_type() == (np.int8 if quantize_embeddings else np.float32)
            if kind == "pre":
                assert len(outputs) == (5 if quantize_kv_cache else 3)
                assert tensor_type(outputs[1]).scalar.numpy_type() == (np.int8 if quantize_kv_cache else np.float32)
        else:
            names = ["input", "cached_keys"]
            if quantize_kv_cache:
                names.append("cached_keys_scale")
            if kind == "sliding_cache":
                names.append("attn_mask")
            names.append("cached_values")
            if quantize_kv_cache:
                names.append("cached_values_scale")
            assert list(graph.inputs) == names
            assert tensor_type(graph.inputs["cached_keys"]).shape[2] == index + 128
            assert tensor_type(graph.inputs["cached_keys"]).scalar.numpy_type() == (np.int8 if quantize_kv_cache else np.float32)
    with pytest.raises(ValueError, match="grouped prefill"):
        cfg.config_pipeline(None, None, 1024, 1, 1)


def test_image_windows_and_cache_views():
    assert image_windows([(577, 833), (834, 1090)], 128) == [(512, 1152)]
    assert image_windows([(512, 768), (768, 1024)], 128) == [(512, 768), (768, 1024)]
    assert cache_view(0, 128, 4096, 1024, []) == (0, 128)
    assert cache_view(1024, 128, 4096, 1024, [(1151, 1535)]) == (1, 1536)
    assert cache_view(3968, 128, 4096, 1024, []) == (2944, 1152)
    for spans in ([], [(127, 511)], [(1023, 1407), (1408, 1792)], [(3839, 4095)]):
        for query in range(0, 4096, 128):
            begin, capacity = cache_view(query, 128, 4096, 1024, spans)
            assert capacity % 128 == 0 and 128 <= capacity <= 1536
            assert 0 <= begin < begin + capacity <= 4096
            q = np.arange(query, query + 128)[:, None]
            k = np.arange(4096)[None, :]
            allowed = (k <= q) & (k > q - 1024)
            for image_begin, image_end in spans:
                allowed |= (q >= image_begin) & (q < image_end) & (k >= image_begin) & (k < image_end)
            assert not allowed[:, :begin].any() and not allowed[:, begin + capacity:].any()


@pytest.mark.parametrize("length,spans", [
    (129, [(127, 129)]),
    (600, [(511, 527), (528, 544)]),
    (911, [(895, 911)]),
    (1023, [(1007, 1023)]),
    (2177, [(2047, 2063), (2064, 2080)]),
])
def test_n128_image_windows_match_full_decoder(unified, length, spans):
    reference, language, params = unified
    if length > 1024:
        language.cfg.config_pipeline(None, None, 4096, 128, 1)
    hidden = np.random.default_rng(249).normal(0, 0.3, (1, length, 64)).astype(np.float32)
    token_types = torch.zeros((1, length), dtype=torch.int64)
    for begin, end in spans:
        token_types[:, begin:end] = 1
    masks = reference.create_masks_for_generate(
        config=reference.config, inputs_embeds=torch.from_numpy(hidden),
        attention_mask=torch.ones((1, length)), past_key_values=None,
        position_ids=torch.arange(length)[None], mm_token_type_ids=token_types,
        is_first_iteration=True,
    )
    with torch.no_grad():
        expected = reference(inputs_embeds=torch.from_numpy(hidden), mm_token_type_ids=token_types,
                             attention_mask=masks, use_cache=False).logits[:, -1].numpy()
    graphs = {}

    def execute(part, mode, inputs):
        if part.model_name not in graphs:
            with patch.object(ModelGraph, "save", autospec=True) as save:
                part.generate_graph({}, quantizable=True)
            graph, outputs = save.call_args.args
            graph.finish(outputs)
            graphs[part.model_name] = graph
        graph = graphs[part.model_name]
        return graph.run(**{name: np.asarray(value, np.float32)
                            for name, value in zip(graph.inputs, inputs, strict=True)})

    with patch.object(BaseModel, "run_model", execute):
        actual = language.run_model(EvalMode.SDK, [hidden[:, None]],
            embeddings_tensor=params["model.language_model.embed_tokens.weight"], image_spans=spans)
    np.testing.assert_allclose(actual.reshape(1, 64), expected, atol=3e-5, rtol=2e-4)


@pytest.mark.parametrize("samples", [639, 640, 641])
def test_audio_framing_and_graph_match_transformers(unified, tmp_path, samples):
    from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedAudioConfig
    from transformers.models.gemma4_unified.feature_extraction_gemma4_unified import Gemma4UnifiedAudioFeatureExtractor
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedMultimodalEmbedder
    from sima_lmm.model.gemma4_unified_model import Gemma4UnifiedAudioModel

    reference, language, params = unified
    language.cfg.audio_cfg = AudioModelConfig(token_id=61, begin_token_id=62, end_token_id=63)
    module = Gemma4UnifiedMultimodalEmbedder(Gemma4UnifiedAudioConfig(), reference.config.text_config).eval()
    weight_name = "model.embed_audio.embedding_projection.weight"
    params[weight_name] = module.embedding_projection.weight.detach().numpy()
    language.hf_model.weight_map[weight_name] = "memory"
    pcm = np.random.default_rng(249).integers(-32768, 32768, samples, dtype=np.int16)
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(pcm.astype("<i2").tobytes())
    frames = audio_frames(path, language.cfg.audio_cfg)
    expected_frames = Gemma4UnifiedAudioFeatureExtractor()._extract_waveform_features(pcm.astype(np.float32) / 32768)[0]
    np.testing.assert_array_equal(frames, expected_frames)
    part = Gemma4UnifiedAudioModel(language.cfg, "tiny_audio", hf_model=language.hf_model)
    with patch.object(ModelGraph, "save", autospec=True) as save:
        part.generate_graph({}, quantizable=True)
    graph, outputs = save.call_args.args
    graph.finish(outputs)
    padded = np.pad(frames, ((0, 128 - len(frames)), (0, 0)))
    actual = graph.run(waveform=padded[None, None])[0][:, 0, :len(frames)]
    with torch.no_grad():
        expected = module(torch.from_numpy(frames[None])).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    assert VlmConfig.load(asdict(language.cfg)) == language.cfg
    assert any(part.part == "audio" for part in language.cfg.get_layer_ids())
