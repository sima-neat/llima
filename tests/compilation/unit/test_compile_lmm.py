import json
import sys

import pytest

from sima_lmm.host import compile_lmm


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (
            {"model_type": "gemma4_assistant", "architectures": ["Gemma4AssistantForCausalLM"]},
            "gemma4_mtp",
        ),
        (
            {"model_type": "llama", "architectures": ["LlamaForCausalLMEagle3"]},
            "eagle3",
        ),
    ],
)
def test_detect_speculative_method(tmp_path, config: dict, expected: str):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert compile_lmm._detect_speculative_method(tmp_path) == expected


def test_detect_speculative_method_rejects_unknown_draft(tmp_path):
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected a Gemma4 MTP or EAGLE3 draft model"):
        compile_lmm._detect_speculative_method(tmp_path)


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
