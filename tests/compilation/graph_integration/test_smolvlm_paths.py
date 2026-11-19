"""Check SmolVLM2 preprocessing and vision/projector weights without MLA compilation."""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector, SmolVLMEncoderLayer

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.model_graph import ModelGraph, activation_dtype
from sima_lmm.model.vision_model import VisionModel
from sima_lmm.preproc.vlm_helper import Chat, VlmHelper


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]
MODEL = "SmolVLM2-256M-Video-Instruct"


def test_smolvlm_frames_preserve_aspect_ratio_and_use_one_visual_block(model_inputs_path):
    path = model_inputs_path / f"models--HuggingFaceTB--{MODEL}"
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, path, config.to_dict())
    helper = VlmHelper(cfg, path)
    chat = Chat(helper)
    chat.add_image(Image.new("RGB", (64, 32), "red"))
    chat.add_image(Image.new("RGB", (32, 64), "blue"))
    chat.messages.append({"role": "user", "content": [
        {"type": "text", "text": "Frame from 00:00:"}, {"type": "image"},
        {"type": "text", "text": "Frame from 00:02:"}, {"type": "image"},
        {"type": "text", "text": "Describe the changes."},
    ]})
    _, input_ids, pixels, image_grid = helper.preprocess(chat)
    size = cfg.vm_cfg.image_size
    assert pixels.shape == (2, size, size, 3)
    assert image_grid is None
    np.testing.assert_allclose(pixels[:, size // 2, size // 2], [[1, -1, -1], [-1, -1, 1]])
    # Landscape pads above/below; portrait pads left/right. Neither loses image content.
    np.testing.assert_allclose(pixels[0, 0, size // 2], [-1 / 255] * 3, atol=1e-7)
    np.testing.assert_allclose(pixels[1, size // 2, 0], [-1 / 255] * 3, atol=1e-7)
    np.testing.assert_allclose(pixels[0, size // 2, 0], [1, -1, -1])
    np.testing.assert_allclose(pixels[1, 0, size // 2], [-1, -1, 1])
    decoded = helper.hf_tokenizer.decode(input_ids)
    block = "<fake_token_around_image><global-img>" + "<image>" * cfg.mm_cfg.mm_tokens_per_image
    block += "<fake_token_around_image>"
    assert decoded.count(block) == 2
    assert decoded.count("<image>") == 2 * cfg.mm_cfg.mm_tokens_per_image
    assert "<row_" not in decoded
    assert decoded.index("00:00") < decoded.index(block) < decoded.index("00:02") < decoded.rindex(block)


@pytest.mark.parametrize("quantizable", [True, False])
def test_smolvlm_final_vision_layer_and_projector(quantizable, model_inputs_path, tmp_path):
    path = model_inputs_path / f"models--HuggingFaceTB--{MODEL}"
    hf_config = AutoConfig.from_pretrained(path, local_files_only=True)
    hf_config.vision_config._attn_implementation = "eager"
    cfg = VlmConfig.from_hf_config(ModelFormat.FORMAT_HF, path, hf_config.to_dict())
    cfg.pipeline_cfg.quantize_embeddings = False
    source = LocalHuggingFaceModel.create_from_directory(path)
    model = VisionModel(cfg, MODEL, sima_path=tmp_path, hf_model=source)._get_part_model(
        cfg.num_vision_layers - 1
    )
    shape = (1, 1, cfg.vm_cfg.seq_len, cfg.vm_cfg.hidden_size)
    rng = np.random.default_rng(322)
    hidden = rng.normal(0, 0.2, shape).astype(activation_dtype(quantizable))
    graph = ModelGraph(model, {"hidden": shape}, quantizable)
    projected = model._build_vision_tower(graph, "model.vision_model", graph.inputs["hidden"])
    projected = model._build_mm_projector(graph, "model.connector.modality_projection", projected)
    graph.finish([projected])
    actual = np.asarray(graph.run(hidden=hidden)[0], dtype=np.float32)

    layer = SmolVLMEncoderLayer(hf_config.vision_config).eval()
    layer_base = f"model.vision_model.encoder.layers.{model.layer_idx}."
    layer.load_state_dict({
        key: torch.from_numpy(source.load_np_param(layer_base + key).astype(np.float32))
        for key in layer.state_dict()
    })
    connector = SmolVLMConnector(hf_config).eval()
    connector.load_state_dict({
        key: torch.from_numpy(source.load_np_param("model.connector." + key).astype(np.float32))
        for key in connector.state_dict()
    })
    with torch.no_grad():
        expected = layer(torch.from_numpy(hidden.astype(np.float32))[:, 0], attention_mask=None)
        expected = F.layer_norm(
            expected, (cfg.vm_cfg.hidden_size,),
            torch.from_numpy(source.load_np_param("model.vision_model.post_layernorm.weight").astype(np.float32)),
            torch.from_numpy(source.load_np_param("model.vision_model.post_layernorm.bias").astype(np.float32)),
            cfg.vm_cfg.layer_norm_eps,
        )
        expected = connector(expected).numpy()[:, None]
    assert actual.shape == expected.shape == (1, 1, cfg.mm_cfg.mm_tokens_per_image, cfg.lm_cfg.hidden_size)
    assert np.isfinite(actual).all()
    if quantizable:
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-4)
    else:
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.02
        cosine = np.vdot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))
        assert cosine > 0.999
