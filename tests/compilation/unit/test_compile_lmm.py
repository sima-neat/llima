import sys
from types import SimpleNamespace

import pytest

from sima_lmm.host import compile_lmm


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], (False, True, True, True, False)),
        (
            ["--onnx", "--no-quantize_embeddings", "--no-quantize_kv_cache"],
            (False, False, False, True, False),
        ),
        (
            ["--draft_model_path", "draft"],
            (False, True, True, True, True),
        ),
        (
            [
                "--enable_filter_sharing",
                "--no-quantize_embeddings",
                "--no-quantize_kv_cache",
            ],
            (True, False, False, True, False),
        ),
    ],
)
def test_memory_optimization_cli_defaults_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    options: list[str],
    expected: tuple[bool, bool, bool, bool, bool],
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
    assert (args[12], args[13], args[14], args[15], args[16]) == expected


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


def test_qwen3tts_component_roots_preserve_runner_elf_names():
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
