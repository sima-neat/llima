from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.model_graph import ModelGraph, activation_dtype
from sima_lmm.model.vision_model import StandardVisionLayerModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _config(factor=4):
    return SmolVLMConfig(
        vision_config={"image_size": 32, "patch_size": 4}, scale_factor=factor,
        architectures=["SmolVLMForConditionalGeneration"], dtype="float32",
    ).to_dict()


@pytest.mark.parametrize("factor", [1, 2, 4])
def test_smolvlm_config_preserves_language_and_projected_token_count(factor):
    source = _config(factor)
    config = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, Path("smolvlm"), source)
    assert config.model_type == "vlm-smolvlm"
    assert config.vm_cfg.arch == "siglip"
    assert config.lm_cfg.arch == "llama"
    assert config.mm_cfg.downsample_factor == factor
    assert config.mm_cfg.mm_tokens_per_image == (8 // factor) ** 2
    assert config.mm_cfg.num_layers == 1
    assert config.mm_cfg.hidden_act is None
    assert config.lm_cfg.rope_cfg.rope_scaling.mrope_section is None


def test_smolvlm_keeps_checkpoint_resolution_when_override_is_given():
    config = VlmConfig.from_hf_config(
        ModelFormat.FORMAT_HF, Path("smolvlm"), _config(), [64, 64]
    )
    assert config.vm_cfg.image_size == 32


def test_smolvlm_source_prefixes_do_not_mix_the_two_transformers():
    model = object.__new__(LocalHuggingFaceModel)
    model.weight_map = dict.fromkeys([
        "model.vision_model.embeddings.patch_embedding.weight",
        "model.vision_model.encoder.layers.0.self_attn.q_proj.weight",
        "model.vision_model.post_layernorm.weight",
        "model.connector.modality_projection.proj.weight",
        "model.text_model.embed_tokens.weight",
        "model.text_model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    ], "model.safetensors")
    assert model.vision_model_param_base_name == "model.vision_model"
    assert model.language_model_param_base_name == "model.text_model"


@pytest.mark.parametrize("factor", [3, 4])
@pytest.mark.parametrize("quantizable", [True, False])
def test_smolvlm_connector_matches_upstream_spatial_channel_order(factor, quantizable):
    import torch
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector

    channels, hidden, grid = 32, 32, factor * 2
    config = SimpleNamespace(
        model_type="vlm-smolvlm",
        vm_cfg=SimpleNamespace(num_patches=grid),
        mm_cfg=SimpleNamespace(downsample_factor=factor),
    )
    model = StandardVisionLayerModel(
        config, "connector", layer_idx=0, include_embeddings=False, include_mm_proj=True
    )
    rng = np.random.default_rng(64)
    weight = rng.normal(0, 0.01, (hidden, channels * factor**2)).astype(np.float32)
    params = {"connector.proj.weight": weight}
    model.get_hf_param, model.check_hf_param = params.__getitem__, params.__contains__
    # Distinct spatial/channel values detect transposes hidden by uniform inputs.
    value = rng.normal(0, 0.2, (1, 1, grid * grid, channels)).astype(activation_dtype(quantizable))
    graph = ModelGraph(model, {"hidden": value.shape}, quantizable)
    output = model._build_mm_projector(graph, "connector", graph.inputs["hidden"])
    graph.finish([output])
    actual = graph.run(hidden=value)[0]
    oracle = SmolVLMConnector(SimpleNamespace(
        scale_factor=factor, vision_config=SimpleNamespace(hidden_size=channels),
        text_config=SimpleNamespace(hidden_size=hidden),
    ))
    with torch.no_grad():
        oracle.modality_projection.proj.weight.copy_(torch.from_numpy(weight))
        expected = oracle(torch.from_numpy(value.astype(np.float32)).reshape(1, grid * grid, channels))
    np.testing.assert_allclose(
        actual.reshape(expected.shape), expected.numpy(),
        rtol=2e-5 if quantizable else 0.04, atol=2e-7 if quantizable else 0.002,
    )
