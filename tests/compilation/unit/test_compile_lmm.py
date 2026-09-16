import inspect
import sys
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest

from sima_lmm.host import compile_lmm


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], (False, True, True, False)),
        (
            ["--onnx", "--no-quantize_embeddings", "--no-quantize_kv_cache"],
            (False, False, False, False),
        ),
        (
            ["--draft_model_path", "draft"],
            (False, True, True, True),
        ),
        (
            [
                "--enable_filter_sharing",
                "--no-quantize_embeddings",
                "--no-quantize_kv_cache",
            ],
            (True, False, False, False),
        ),
    ],
)
def test_memory_optimization_cli_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    options: list[str],
    expected: tuple[bool, bool, bool, bool],
):
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llima-compile",
            str(tmp_path / "model"),
            "-o",
            str(tmp_path / "output"),
            "-j",
            "1",
            *options,
        ],
    )
    monkeypatch.setattr(compile_lmm, "gen_files", lambda *args: calls.append(args))

    compile_lmm.main()

    args = calls[0]
    assert (args[12], args[13], args[14], args[15]) == expected


@pytest.mark.parametrize(
    ("options", "expected_error"),
    [
        (
            ["--onnx"],
            "Pass --no-quantize_embeddings --no-quantize_kv_cache.",
        ),
    ],
)
def test_incompatible_quantization_defaults_report_disable_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    expected_error: str,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llima-compile",
            str(tmp_path / "model"),
            "-o",
            str(tmp_path / "output"),
            "-j",
            "1",
            *options,
        ],
    )

    with pytest.raises(SystemExit):
        compile_lmm.main()

    assert expected_error in capsys.readouterr().err


def test_qwen3tts_uses_one_composite_compiler_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    calls = []
    package = tmp_path / "Qwen3-tts"
    output = tmp_path / "qwen3_model"
    package.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llima-compile",
            str(package),
            "--qwen3tts",
            "-o",
            str(output),
            "-j",
            "1",
        ],
    )
    monkeypatch.setattr(
        compile_lmm,
        "gen_qwen3tts",
        lambda *args: calls.append(args),
    )

    compile_lmm.main()

    assert calls == [
        (package, output, compile_lmm.FileGenMode.ALL, 30, False, 1, None)
    ]


def test_qwen3tts_defaults_to_its_package_model_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    calls = []
    package = tmp_path / "Qwen3-tts"
    package.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["llima-compile", str(package), "--qwen3tts", "-j", "1"],
    )
    monkeypatch.setattr(
        compile_lmm, "gen_qwen3tts", lambda *args: calls.append(args)
    )

    compile_lmm.main()

    assert calls[0][0] == package
    assert calls[0][1] == package / "qwen3_model"


def test_qwen3tts_component_roots_preserve_runner_elf_names(monkeypatch, tmp_path):
    from sima_lmm.model import qwen3tts_model
    from sima_lmm.model.qwen3tts_model import QWEN3TTS_COMPONENTS

    parts = [
        (part.source_directory, part.model_name, part.is_codec_tail)
        for part in QWEN3TTS_COMPONENTS
    ]
    assert parts == [
        ("backbone", "backbone", False),
        ("code_predictor", "code_predictor", False),
        ("codec_decoder", "codec_decoder", False),
        ("codec_decoder", "codec_decoder_tail_full", True),
    ]

    # Exercise the real composite caller against the current gen_files
    # signature: removed/reordered shared arguments must not shift TTS options.
    for directory, _, _ in parts:
        source = tmp_path / "qwen3_components" / directory
        source.mkdir(parents=True, exist_ok=True)
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").touch()
    output = tmp_path / "qwen3_model"
    contract = output / "devkit" / "codec_tail_raw_mla_contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}")
    signature = inspect.signature(compile_lmm.gen_files)
    generate = create_autospec(compile_lmm.gen_files)
    monkeypatch.setattr(compile_lmm, "gen_files", generate)
    monkeypatch.setattr(
        qwen3tts_model, "Qwen3TTSPartModel",
        lambda cfg, name, **kwargs: SimpleNamespace(
            model_name=name, gen_files=lambda *args, **options: None
        ),
    )
    wrapper = object()
    compile_lmm.gen_qwen3tts(
        tmp_path, output, compile_lmm.FileGenMode.ALL, 30, False, 1, wrapper
    )
    assert generate.call_count == len(parts)
    for call, (directory, name, is_tail) in zip(generate.call_args_list, parts, strict=True):
        arguments = signature.bind(*call.args, **call.kwargs).arguments
        assert arguments["model_path"] == tmp_path / "qwen3_components" / directory
        assert arguments["qwen3tts_component_name"] == name
        assert arguments["qwen3tts_codec_tail"] == is_tail
        assert arguments["qwen3tts_tail_wrapper"] is wrapper
        assert arguments["quantize_embeddings"] is True
        assert arguments["quantize_kv_cache"] is True
        assert arguments["return_logits"] is False
        assert arguments["log_level"] == 30


@pytest.mark.parametrize("mode", ["ordinary", "codec", "invalid-scale"])
def test_qwen3tts_layer_scale_survives_unsplit_mlp(monkeypatch, mode):
    from sima_lmm.model import sima_builder
    from sima_lmm.model.language_part_base import LanguagePartBaseModel

    calls = []

    def conv(builder, get_param, check_param, name, node, rank, **kwargs):
        calls.append((name, kwargs))
        return name

    monkeypatch.setattr(sima_builder, "build_conv_from_dense_with_lora", conv)
    monkeypatch.setattr(sima_builder, "build_activation", lambda *args: "activation")
    model = object.__new__(LanguagePartBaseModel)
    model.cfg = SimpleNamespace(lm_cfg=SimpleNamespace(
        arch=(compile_lmm.LlmArchType.QWEN3_TTS_CODEC_DECODER if mode == "codec"
              else compile_lmm.LlmArchType.LLAMA),
        lora_cfg=None, mlp_cfg=SimpleNamespace(act="silu"),
    ))
    model.check_hf_param = lambda name: False
    model.get_hf_param = lambda name: None
    builder = SimpleNamespace(
        create_mul_node=lambda *args: "product",
        create_add_node=lambda left, right: (left, right),
    )
    scale = None if mode == "ordinary" else np.array([2, 3], dtype=np.float32)
    if mode == "invalid-scale":
        with pytest.raises(AssertionError):
            model._build_sima_mlp(builder, "mlp", ["input"], False, output_scale=scale)
        return
    result = model._build_sima_mlp(
        builder, "mlp", ["input", "residual"], False,
        with_residual_add=True, output_scale=scale,
    )
    assert result == ("residual", "mlp.down_proj")
    assert [name for name, _ in calls] == ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    assert all("weight_slice" not in kwargs for _, kwargs in calls)
    if mode == "ordinary":
        assert all(kwargs == {"merged_lora": False} for _, kwargs in calls)
    else:
        kwargs = calls[-1][1]
        np.testing.assert_array_equal(
            kwargs["weight_process_func"](np.ones((2, 4, 1, 1))),
            np.broadcast_to(scale.reshape(2, 1, 1, 1), (2, 4, 1, 1)),
        )
        np.testing.assert_array_equal(kwargs["bias_process_func"](np.ones(2)), scale)


def test_standard_compilation_does_not_import_qwen3tts_compiler(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    qwen3tts_module = "sima_lmm.model.qwen3tts_model"
    monkeypatch.delitem(sys.modules, qwen3tts_module, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llima-compile",
            str(tmp_path / "ordinary-model"),
            "-o",
            str(tmp_path / "output"),
            "-j",
            "1",
        ],
    )
    load_calls = []
    generation_calls = []
    source = object.__new__(compile_lmm.LocalHuggingFaceModel)
    monkeypatch.setattr(
        compile_lmm.LocalHuggingFaceModel, "is_compressed_tensors_model", lambda self: False
    )
    model = SimpleNamespace(
        model_name="ordinary-model",
        hf_model=source,
        cfg=SimpleNamespace(lm_cfg=SimpleNamespace(arch=compile_lmm.LlmArchType.LLAMA)),
        gen_files=lambda mode, **kwargs: generation_calls.append((mode, kwargs)),
    )

    def load_model(**kwargs):
        load_calls.append(kwargs)
        return model

    monkeypatch.setattr(compile_lmm.VisionLanguageModel, "from_hf_cache", load_model)
    monkeypatch.setattr(compile_lmm, "default_configuration", lambda model: {"precision": {}})

    compile_lmm.main()

    assert qwen3tts_module not in sys.modules
    assert load_calls[0]["model_name"] == "ordinary-model"
    assert load_calls[0]["sima_path"] == tmp_path / "output" / "sima_files"
    assert "qwen3tts_package_part" not in load_calls[0]
    assert "qwen3tts_codec_tail" not in load_calls[0]
    assert [mode for mode, _ in generation_calls] == [
        compile_lmm.FileGenMode.DEVKIT,
        compile_lmm.FileGenMode.SOURCE_TO_FP,
        compile_lmm.FileGenMode.FP_TO_QUANT,
        compile_lmm.FileGenMode.MODEL_SDK_COMPILE,
    ]
    assert all("qwen3tts_package_part" not in options for _, options in generation_calls)


def test_qwen3tts_package_guard_preserves_other_architectures_devkit():
    from sima_lmm.model.vision_language_model import VisionLanguageModel

    qwen_arches = {
        compile_lmm.LlmArchType.QWEN3_TTS_TALKER,
        compile_lmm.LlmArchType.QWEN3_TTS_CODE_PREDICTOR,
        compile_lmm.LlmArchType.QWEN3_TTS_CODEC_DECODER,
        compile_lmm.LlmArchType.QWEN3_TTS_CODEC_DECODER_TAIL,
    }
    for arch in compile_lmm.LlmArchType.values():
        model = object.__new__(VisionLanguageModel)
        model.cfg = SimpleNamespace(lm_cfg=SimpleNamespace(arch=arch))
        calls = []
        model.gen_devkit_files = lambda **kwargs: calls.append(kwargs)
        options = {"gen_config": {"precision": {}}}
        if arch in qwen_arches:
            model.gen_files(compile_lmm.FileGenMode.DEVKIT, qwen3tts_package_part=True, **options)
        else:
            with pytest.raises(ValueError, match="requires a Qwen3-TTS architecture"):
                model.gen_files(compile_lmm.FileGenMode.DEVKIT, qwen3tts_package_part=True, **options)
        assert calls == []
        model.gen_files(compile_lmm.FileGenMode.DEVKIT, **options)
        assert calls == [{"precision": {}, "resume": False}]


def test_qwen3tts_package_rejects_non_tts_checkpoint_before_pipeline_setup(monkeypatch, tmp_path):
    from sima_lmm.model import vision_language_model as vlm

    monkeypatch.setattr(vlm, "model_file_type", lambda path: vlm.ModelFormat.FORMAT_HF)
    monkeypatch.setattr(
        vlm.LocalHuggingFaceModel, "create_from_directory",
        lambda **kwargs: SimpleNamespace(config={"model_type": "qwen3"}),
    )
    monkeypatch.setattr(
        vlm.VlmConfig, "from_hf_config",
        lambda *args, **kwargs: SimpleNamespace(
            lm_cfg=SimpleNamespace(arch=compile_lmm.LlmArchType.QWEN)
        ),
    )
    with pytest.raises(ValueError, match="requires a Qwen3-TTS architecture"):
        vlm.VisionLanguageModel.from_hf_cache(
            model_name="qwen3tts-backbone",
            hf_cache_path=tmp_path,
            onnx_path=tmp_path / "onnx",
            sima_path=tmp_path / "sima",
            max_num_tokens=1024,
            qwen3tts_package_part=True,
        )
    with pytest.raises(ValueError, match="requires the codec-decoder component"):
        vlm.VisionLanguageModel.from_hf_cache(
            model_name="qwen3tts-decoder",
            hf_cache_path=tmp_path,
            onnx_path=tmp_path / "onnx",
            sima_path=tmp_path / "sima",
            max_num_tokens=1024,
            qwen3tts_codec_tail=True,
            qwen3tts_package_part=True,
        )
