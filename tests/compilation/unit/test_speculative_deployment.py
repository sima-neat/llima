import json
from pathlib import Path

import pytest

from sima_lmm.devkit.model_manager import resolve_target_and_draft_paths
from sima_lmm.host import deploy_lmm


pytestmark = [pytest.mark.premerge, pytest.mark.compiler_unit]


def _write_config(model_dir: Path, is_draft: bool | None) -> Path:
    (model_dir / "devkit").mkdir(parents=True)
    (model_dir / "elf_files").mkdir()
    spec_cfg = None if is_draft is None else {"is_draft": is_draft}
    (model_dir / "devkit" / "vlm_config.json").write_text(json.dumps({
        "lm_cfg": {"speculative_decoding_cfg": spec_cfg}
    }))
    return model_dir


def _write_compiler_child(parent: Path, name: str, is_draft: bool) -> Path:
    sima_dir = parent / name / "sima_files"
    (sima_dir / "devkit").mkdir(parents=True)
    (sima_dir / "mpk").mkdir()
    (sima_dir / "devkit" / "vlm_config.json").write_text(json.dumps({
        "lm_cfg": {"speculative_decoding_cfg": {"is_draft": is_draft}}
    }))
    return sima_dir


def test_deploy_dispatches_both_speculative_children(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "compiled"
    target = _write_compiler_child(source, "target-model", False)
    draft = _write_compiler_child(source, "draft-model", True)
    destination = tmp_path / "deployed"
    calls = []
    monkeypatch.setattr(
        deploy_lmm,
        "_deploy_sima_files",
        lambda src, dst: calls.append((src, dst)),
    )

    deploy_lmm.deploy(source, destination)

    assert calls == [
        (target, destination / "target-model"),
        (draft, destination / "draft-model"),
    ]


def test_deploy_preserves_normal_destination(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "compiled"
    sima_dir = source / "sima_files"
    (sima_dir / "devkit").mkdir(parents=True)
    destination = tmp_path / "deployed"
    calls = []
    monkeypatch.setattr(
        deploy_lmm,
        "_deploy_sima_files",
        lambda src, dst: calls.append((src, dst)),
    )

    deploy_lmm.deploy(source, destination)

    assert calls == [(sima_dir, destination)]


def test_runtime_resolves_normal_deployed_model(tmp_path: Path) -> None:
    model = _write_config(tmp_path / "model", None)

    assert resolve_target_and_draft_paths(model) == (model, None)


def test_runtime_resolves_speculative_pair(tmp_path: Path) -> None:
    target = _write_config(tmp_path / "target-model", False)
    draft = _write_config(tmp_path / "draft-model", True)

    assert resolve_target_and_draft_paths(tmp_path) == (target, draft)


def test_runtime_rejects_incomplete_speculative_pair(tmp_path: Path) -> None:
    _write_config(tmp_path / "target-model", False)

    with pytest.raises(RuntimeError, match="Invalid model artifact"):
        resolve_target_and_draft_paths(tmp_path)


def test_runtime_rejects_compiler_output_layout(tmp_path: Path) -> None:
    _write_config(tmp_path / "sima_files", None)

    with pytest.raises(RuntimeError, match="Invalid model artifact"):
        resolve_target_and_draft_paths(tmp_path)
