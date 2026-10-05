import sys

import pytest

from sima_lmm.host import compile_lmm


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ([], (False, True, True, False)),
        (
            ["--source_to_fp", "--no-quantize_embeddings", "--no-quantize_kv_cache"],
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
    (tmp_path / "model").mkdir()
    (tmp_path / "draft").mkdir()
    monkeypatch.chdir(tmp_path)
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


@pytest.mark.parametrize("jobs", ["0", "-1"])
def test_cli_rejects_nonpositive_jobs(monkeypatch, capsys, jobs):
    monkeypatch.setattr(sys, "argv", ["llima-compile", "model", "-j", jobs])
    with pytest.raises(SystemExit) as error:
        compile_lmm.main()
    assert error.value.code == 2
    assert "--jobs must be a positive integer" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--system_prompt_file", "--chat_template_file"])
def test_cli_reports_unreadable_prompt_files(monkeypatch, tmp_path, capsys, option):
    (tmp_path / "model").mkdir()
    missing = tmp_path / "missing.txt"
    monkeypatch.setattr(
        sys, "argv",
        ["llima-compile", str(tmp_path / "model"), "-o", str(tmp_path / "output"),
         "-j", "1", option, str(missing)],
    )
    with pytest.raises(SystemExit) as error:
        compile_lmm.main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Cannot read" in message
    assert str(missing) in message


@pytest.mark.parametrize("from_files", [False, True])
def test_cli_accepts_system_prompt_with_chat_template(monkeypatch, tmp_path, from_files):
    (tmp_path / "model").mkdir()
    prompt, template = "Be concise. Grüße!", "{{ messages }}"
    if from_files:
        prompt_path, template_path = tmp_path / "prompt.txt", tmp_path / "template.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        template_path.write_text(template, encoding="utf-8")
        options = ["--system_prompt_file", str(prompt_path), "--chat_template_file", str(template_path)]
    else:
        options = ["--system_prompt", prompt, "--chat_template", template]
    monkeypatch.setattr(
        sys, "argv",
        ["llima-compile", str(tmp_path / "model"), "-o", str(tmp_path / "output"),
         "-j", "1", *options],
    )
    calls = []
    monkeypatch.setattr(compile_lmm, "gen_files", lambda *args: calls.append(args))
    compile_lmm.main()
    assert calls[0][7:9] == (prompt, template)
