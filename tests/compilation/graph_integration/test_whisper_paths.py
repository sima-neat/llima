"""Compare Whisper's native graphs with ONNX using small, deterministic weights."""

import logging
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest

from afe.ir.execute import create_node_executor, create_node_quant_executor
from afe.ir.serializer import load_awesomenet
from sima_lmm.config.whisper_config import WhisperConfig
from sima_lmm.hf.hf_transformer import LocalHuggingFaceModel
from sima_lmm.model.base import FileGenMode, FileGenPrecision
from sima_lmm.model.sima_builder import activation_dtype
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
    return cfg, source


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
def test_whisper_native_matches_onnx(whisper_source, tmp_path, model_class, options):
    cfg, source = whisper_source
    model = model_class(
        cfg, "whisper", hf_model=source, onnx_path=tmp_path / "onnx",
        sima_path=tmp_path / "sima", **options,
    )
    model.gen_onnx_files()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model.onnx_file_name), session_options)
    input_shapes = {
        node.name: tuple(node.shape[i] for i in (0, 2, 3, 1)) for node in session.get_inputs()
    }
    rng = np.random.default_rng(7)
    inputs = {
        name: rng.normal(0, 0.2, shape).astype(np.float32) for name, shape in input_shapes.items()
    }
    if "attn_mask" in inputs:
        inputs["attn_mask"].fill(0)
        inputs["attn_mask"][..., -3:] = np.finfo(np.float32).min

    expected = [np.transpose(x, (0, 2, 3, 1)) for x in session.run(
        None, {name: np.transpose(value, (0, 3, 1, 2)) for name, value in inputs.items()}
    )]

    config = {"precision": FileGenPrecision.A_BF16_W_INT8}
    model.gen_files(FileGenMode.SOURCE_TO_FP, layer_cfg=config, log_level=logging.WARNING)
    net = load_awesomenet("whisper.fp32.sima", str(model.sima_model_sdk_path))
    actual = net.run(inputs, node_callable=create_node_executor(False))
    actual = list(actual) if isinstance(actual, (list, tuple)) else [actual]
    assert len(actual) == len(expected)
    for output, reference in zip(actual, expected):
        assert output.shape == reference.shape
        assert output.dtype == (np.int32 if reference.dtype == np.int64 else reference.dtype)
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
        if isinstance(model, WhisperDecoderLanguageDetectModel) and not direct:
            imported = replace(model, sima_path=tmp_path / "onnx-sima")
            imported.gen_files(FileGenMode.ONNX_TO_QUANT, layer_cfg=config, log_level=logging.WARNING)
            imported_net = load_awesomenet("whisper.sima", str(imported.sima_model_sdk_path))
            imported_outputs = imported_net.run(
                inputs, node_callable=create_node_quant_executor(False, False)
            )
            for output, reference in zip(actual, imported_outputs):
                np.testing.assert_array_equal(output, reference)
        assert len(actual) == len(expected)
        for output, reference in zip(actual, expected):
            assert output.shape == reference.shape
            assert output.dtype == (np.int32 if reference.dtype == np.int64 else reference.dtype)
            if np.issubdtype(reference.dtype, np.integer):
                np.testing.assert_array_equal(output, reference)
            else:
                valid = np.abs(reference) < 1e30
                assert np.all(output[~valid] < -1e30)
                # Bound accumulated quantization error, including a complete encoder/decoder.
                error = np.linalg.norm(output[valid] - reference[valid])
                assert error / max(np.linalg.norm(reference[valid]), 1e-6) < 0.025
