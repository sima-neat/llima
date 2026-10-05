"""Execute Whisper SiMa graphs against Hugging Face with deterministic weights."""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from transformers import WhisperConfig as HfWhisperConfig, WhisperModel as HfWhisperModel

from afe.ir.execute import create_node_executor, create_node_quant_executor
from afe.ir.defines import get_expected_tensor_value
from afe.ir.serializer import load_awesomenet
from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.base import FileGenMode, FileGenPrecision
from sima_lmm.model.model_graph import activation_dtype
from sima_lmm.model.whisper_decoder_cache_model import WhisperDecoderCacheModel
from sima_lmm.model.whisper_decoder_init_model import WhisperDecoderInitModel
from sima_lmm.model.whisper_decoder_language_detect_model import WhisperDecoderLanguageDetectModel
from sima_lmm.model.whisper_decoder_post_model import WhisperDecoderPostModel
from sima_lmm.model.whisper_decoder_pre_model import WhisperDecoderPreModel
from sima_lmm.model.whisper_encoder_model import WhisperEncoderModel


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_graph_integration]


@pytest.fixture
def whisper_source(tmp_path, monkeypatch):
    cfg = WhisperConfig(
        d_model=32, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_layers=3, decoder_layers=3, max_source_positions=16,
        max_target_positions=16, num_mel_bins=16, vocab_size=64,
        suppress_tokens=[3, 5], num_languages=3,
    )
    rng = np.random.default_rng(1)
    params = {}

    def weight(name, shape):
        params[f"{name}.weight"] = rng.normal(0, 0.08, shape).astype(np.float32)

    def conv(name, shape, bias=True):
        weight(name, shape)
        if bias:
            params[f"{name}.bias"] = rng.normal(0, 0.02, shape[0]).astype(np.float32)

    def norm(name):
        params[f"{name}.weight"] = rng.uniform(0.8, 1.2, cfg.d_model).astype(np.float32)
        params[f"{name}.bias"] = rng.normal(0, 0.02, cfg.d_model).astype(np.float32)

    conv("model.encoder.conv1", (32, 16, 3))
    conv("model.encoder.conv2", (32, 32, 3))
    weight("model.encoder.embed_positions", (16, 32))
    weight("model.decoder.embed_positions", (16, 32))
    weight("model.decoder.embed_tokens", (64, 32))
    for part in ("encoder", "decoder"):
        norm(f"model.{part}.layer_norm")
        for idx in range(3):
            name = f"model.{part}.layers.{idx}"
            attns = ("self_attn", "encoder_attn") if part == "decoder" else ("self_attn",)
            for attn in attns:
                norm(f"{name}.{attn}_layer_norm")
                for proj in ("q", "k", "v", "out"):
                    conv(f"{name}.{attn}.{proj}_proj", (32, 32), bias=proj != "k")
            norm(f"{name}.final_layer_norm")
            conv(f"{name}.fc1", (64, 32))
            conv(f"{name}.fc2", (32, 64))
    source = LocalHuggingFaceModel(tmp_path, {}, {}, {}, {}, {}, params=params)
    tokenizer = SimpleNamespace(sot=1, all_language_tokens=(9, 10, 11))
    monkeypatch.setattr(
        "sima_lmm.model.whisper_decoder_language_detect_model.get_tokenizer", lambda **_: tokenizer
    )
    monkeypatch.setattr(WhisperDecoderPostModel, "_get_extra_suppress_tokens", lambda _: [6])
    reference = HfWhisperModel(HfWhisperConfig(
        d_model=cfg.d_model, encoder_attention_heads=cfg.encoder_attention_heads,
        decoder_attention_heads=cfg.decoder_attention_heads, encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers, max_source_positions=cfg.max_source_positions,
        max_target_positions=cfg.max_target_positions, num_mel_bins=cfg.num_mel_bins,
        vocab_size=cfg.vocab_size, encoder_ffn_dim=64, decoder_ffn_dim=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        activation_function=cfg.activation_function, attn_implementation="eager",
    )).eval()
    reference.load_state_dict({
        name.removeprefix("model."): torch.from_numpy(value) for name, value in params.items()
    })
    return cfg, source, reference


def _split_heads(value, heads):
    return value.reshape(value.shape[0], value.shape[1], heads, -1).transpose(1, 2)


def _layer_output(output):
    # Transformers versions return either a tensor or a tuple from individual layers.
    return output[0] if isinstance(output, tuple) else output


def _decoder_outputs(model, decoder, hidden):
    if model.layer_idx < model.cfg.decoder_layers - 1:
        return [hidden.unsqueeze(1)]
    logits = F.linear(decoder.layer_norm(hidden), decoder.embed_tokens.weight)
    logits[..., model.cfg.suppress_tokens + [6]] = torch.finfo(torch.float32).min
    outputs = [logits.argmax(-1, keepdim=True).to(torch.int32).unsqueeze(1)]
    if model.enable_log_probe:
        outputs.append(logits.unsqueeze(1))
    return outputs


@torch.no_grad()
def _reference_outputs(model, reference, inputs):
    """Adapt HF modules to the runtime's split graph interfaces and NHWC layout."""
    inputs = {name: torch.from_numpy(value) for name, value in inputs.items()}
    cfg, decoder = model.cfg, reference.decoder
    if isinstance(model, WhisperEncoderModel):
        hidden = inputs["input"][:, 0]
        if model.layer_idx is None:
            hidden = reference.encoder(hidden.transpose(1, 2)).last_hidden_state
        else:
            if model.layer_idx == 0:
                hidden = F.gelu(reference.encoder.conv1(hidden.transpose(1, 2)))
                hidden = F.gelu(reference.encoder.conv2(hidden)).transpose(1, 2)
                hidden = hidden + reference.encoder.embed_positions.weight
            hidden = _layer_output(reference.encoder.layers[model.layer_idx](hidden, None))
            if model.layer_idx == cfg.encoder_layers - 1:
                hidden = reference.encoder.layer_norm(hidden)
        outputs = [hidden.unsqueeze(1)]
    elif isinstance(model, WhisperDecoderPreModel):
        hidden = inputs["input"][:, 0]
        if model.layer_idx == 0:
            hidden = hidden + inputs["embed_positions"][:, 0]
        attn = decoder.layers[model.layer_idx].self_attn
        norm = decoder.layers[model.layer_idx].self_attn_layer_norm(hidden)
        outputs = [
            _split_heads(attn.q_proj(norm) * attn.scaling, cfg.decoder_attention_heads),
            attn.k_proj(norm).unsqueeze(1), attn.v_proj(norm).unsqueeze(1),
        ]
        if model.layer_idx == 0:
            outputs.append(hidden.unsqueeze(1))
    elif isinstance(model, WhisperDecoderCacheModel):
        key, value = [
            _split_heads(inputs[name][:, 0], cfg.decoder_attention_heads)
            for name in ("cached_keys", "cached_values")
        ]
        mask = inputs.get("attn_mask")
        if model.num_tokens > 1:
            mask = torch.full(
                (model.num_tokens, key.shape[2]), torch.finfo(torch.float32).min
            ).triu(diagonal=model.token_idx + 1)
        attn = F.scaled_dot_product_attention(inputs["input"], key, value, attn_mask=mask, scale=1.0)
        outputs = [attn.transpose(1, 2).reshape(1, 1, model.num_tokens, cfg.d_model)]
    elif isinstance(model, WhisperDecoderPostModel):
        layer = decoder.layers[model.layer_idx]
        hidden = inputs["input"][:, 0] + layer.self_attn.out_proj(inputs["self_attn"][:, 0])
        query = _split_heads(
            layer.encoder_attn.q_proj(layer.encoder_attn_layer_norm(hidden)),
            cfg.decoder_attention_heads,
        )
        attn = F.scaled_dot_product_attention(query, inputs["encoder_k_cache"], inputs["encoder_v_cache"])
        attn = attn.transpose(1, 2).reshape(1, model.num_tokens, cfg.d_model)
        hidden = hidden + layer.encoder_attn.out_proj(attn)
        hidden = hidden + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(hidden))))
        outputs = _decoder_outputs(model, decoder, hidden)
    elif isinstance(model, WhisperDecoderInitModel):
        layer = decoder.layers[model.layer_idx]
        hidden = inputs["input"][:, 0]
        audio = inputs["audio_features"][:, 0]
        if model.layer_idx == 0:
            hidden = hidden + decoder.embed_positions.weight[:model.num_tokens]
        norm = layer.self_attn_layer_norm(hidden)
        key, value = layer.self_attn.k_proj(norm), layer.self_attn.v_proj(norm)
        mask = torch.full(
            (model.num_tokens, model.num_tokens), torch.finfo(torch.float32).min
        ).triu(diagonal=1)[None, None]
        hidden = _layer_output(layer(hidden, attention_mask=mask, encoder_hidden_states=audio, use_cache=False))
        if model.layer_idx == cfg.decoder_layers - 1:
            hidden = hidden[:, -1:]
        outputs = _decoder_outputs(model, decoder, hidden)
        outputs.extend([
            key.unsqueeze(1), value.unsqueeze(1),
            _split_heads(layer.encoder_attn.k_proj(audio), cfg.decoder_attention_heads),
            _split_heads(layer.encoder_attn.v_proj(audio), cfg.decoder_attention_heads),
        ])
    elif isinstance(model, WhisperDecoderLanguageDetectModel):
        hidden = decoder(
            input_ids=torch.tensor([[1]]), encoder_hidden_states=inputs["audio_features"][:, 0],
            use_cache=False,
        ).last_hidden_state
        logits = F.linear(hidden, decoder.embed_tokens.weight)
        outputs = [
            logits[..., 9:12].argmax(-1, keepdim=True).to(torch.int32).unsqueeze(1),
            logits.unsqueeze(1),
        ]
    else:
        raise AssertionError(f"Missing Whisper reference for {type(model).__name__}")
    return [output.numpy() for output in outputs]


CASES = [
    pytest.param(WhisperEncoderModel, {"layer_idx": idx}, id=f"encoder-{idx}")
    for idx in (None, 0, 1, 2)
] + [
    pytest.param(WhisperDecoderPreModel, {"layer_idx": idx, "num_tokens": 1}, id=f"pre-{idx}")
    for idx in (0, 1)
] + [
    pytest.param(
        WhisperDecoderCacheModel,
        {"num_tokens": n, "token_idx": idx, "use_future_token_mask": mask},
        id=f"cache-{n}-{idx}-{mask}",
    )
    for n, idx, mask in ((4, 0, False), (4, 3, False), (1, 7, True), (1, 7, False))
] + [
    pytest.param(
        WhisperDecoderPostModel,
        {"layer_idx": idx, "num_tokens": 1, "skip_encoder_kv_proj": True,
         "output_encoder_kv_cache": False, "enable_log_probe": probe},
        id=f"post-{idx}-{probe}",
    )
    for idx, probe in ((1, False), (2, False), (2, True))
] + [
    pytest.param(
        WhisperDecoderInitModel, {"layer_idx": idx, "enable_log_probe": probe},
        id=f"init-{idx}-{probe}",
    )
    for idx, probe in ((0, False), (1, False), (2, False), (2, True))
] + [pytest.param(WhisperDecoderLanguageDetectModel, {}, id="language-detect")]


@pytest.mark.parametrize("model_class,options", CASES)
def test_whisper_sima_matches_huggingface(whisper_source, tmp_path, model_class, options):
    cfg, source, reference = whisper_source
    model = model_class(
        cfg, "whisper", hf_model=source, sima_path=tmp_path / "sima", **options,
    )
    config = {"precision": FileGenPrecision.A_BF16_W_INT8}
    model.gen_files(FileGenMode.SOURCE_TO_FP, layer_cfg=config, log_level=logging.WARNING)
    net = load_awesomenet("whisper.fp32.sima", str(model.sima_model_sdk_path))
    input_shapes = {
        name: get_expected_tensor_value(net.nodes[name].get_type().output).shape
        for name in net.input_node_names
    }
    rng = np.random.default_rng(7)
    inputs = {
        name: rng.normal(0, 0.2, shape).astype(np.float32) for name, shape in input_shapes.items()
    }
    if "attn_mask" in inputs:
        inputs["attn_mask"].fill(0)
        inputs["attn_mask"][..., -3:] = np.finfo(np.float32).min

    expected = _reference_outputs(model, reference, inputs)
    actual = net.run(inputs, node_callable=create_node_executor(False))
    actual = list(actual) if isinstance(actual, (list, tuple)) else [actual]
    assert len(actual) == len(expected)
    for output, reference in zip(actual, expected):
        assert output.shape == reference.shape
        assert output.dtype == reference.dtype
        np.testing.assert_allclose(output, reference, atol=2e-6, rtol=2e-5)

    # Exercise both the INT8 quantizer and direct BF16 graph generation.
    for mode in (FileGenMode.FP_TO_QUANT, FileGenMode.SOURCE_TO_QUANT):
        direct = mode == FileGenMode.SOURCE_TO_QUANT
        config = {"precision": FileGenPrecision.BF16 if direct else FileGenPrecision.A_BF16_W_INT8}
        model.gen_files(mode, layer_cfg=config, log_level=logging.WARNING)
        net = load_awesomenet("whisper.sima", str(model.sima_model_sdk_path))
        quant_inputs = {
            name: value.astype(activation_dtype(not direct)) for name, value in inputs.items()
        }
        actual = net.run(quant_inputs, node_callable=create_node_quant_executor(False, False))
        actual = list(actual) if isinstance(actual, (list, tuple)) else [actual]
        assert len(actual) == len(expected)
        for output, reference in zip(actual, expected):
            assert output.shape == reference.shape
            assert output.dtype == reference.dtype
            if np.issubdtype(reference.dtype, np.integer):
                np.testing.assert_array_equal(output, reference)
            else:
                valid = np.abs(reference) < 1e30
                assert np.all(output[~valid] < -1e30)
                # Bound accumulated quantization error, including a complete encoder/decoder.
                error = np.linalg.norm(output[valid] - reference[valid])
                assert error / max(np.linalg.norm(reference[valid]), 1e-6) < 0.025
