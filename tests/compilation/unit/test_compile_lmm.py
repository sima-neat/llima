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
    assert args[12:17] == expected


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
