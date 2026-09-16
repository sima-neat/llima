import sys

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
    monkeypatch.setattr(compile_lmm, "gen_files", lambda *args: None)

    compile_lmm.main()

    assert qwen3tts_module not in sys.modules
