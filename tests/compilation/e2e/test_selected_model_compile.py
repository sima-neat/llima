import hashlib
import json
import subprocess
import tarfile
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from transformers import AutoConfig

from sima_lmm.config.vlm_config import ModelFormat, VlmConfig
from tests.compilation.cases import E2E_ELIGIBLE_MODELS
from tests.compilation.helpers.paths import require_readable_path


pytestmark = [
    pytest.mark.premerge,
    pytest.mark.compiler_e2e,
    pytest.mark.serial,
    pytest.mark.high_memory,
]

COMPILE_CONFIGURATION = """\
def get_layer_configuration(model_properties, layer):
    if layer["is_group"] or layer["index"] != 0:
        return {"compile": False}
    return {"precision": "A_BF16_W_INT4"}
"""

# TODO: Add --compile again once the ARM packages are fixed.
RUN_COMPILE_STAGE = False


def _select_model(candidate_sha: str) -> tuple[str, int]:
    if not E2E_ELIGIBLE_MODELS:
        raise RuntimeError("The E2E eligible-model set is empty.")
    digest = hashlib.sha256(candidate_sha.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    return E2E_ELIGIBLE_MODELS[seed % len(E2E_ELIGIBLE_MODELS)], seed


def _run_stage(
    stage: str,
    command: list[str],
    report: dict,
) -> None:
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    duration = time.monotonic() - started
    report["stages"].append(
        {
            "name": stage,
            "duration_seconds": round(duration, 3),
            "return_code": result.returncode,
        }
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-8000:]
        stdout_tail = result.stdout[-4000:]
        raise AssertionError(
            f"llima-compile {stage} failed with exit code {result.returncode}.\n"
            f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
        )


def _validate_compiled_artifacts(output_path: Path) -> list[dict]:
    mpk_path = require_readable_path(
        output_path / "sima_files" / "mpk",
        "compiled MPK directory",
    )
    archives = sorted(mpk_path.glob("*.tar.gz"))
    if not archives:
        raise FileNotFoundError(f"No compiled MPK archives found in {mpk_path}")

    metadata = []
    for archive in archives:
        if archive.stat().st_size == 0:
            raise AssertionError(f"Compiled MPK is empty: {archive}")
        with tarfile.open(archive, "r:gz") as mpk:
            members = [member for member in mpk.getmembers() if member.isfile()]
        if not members:
            raise AssertionError(f"Compiled MPK contains no files: {archive}")
        if not any(member.name.endswith(".elf") for member in members):
            raise AssertionError(f"Compiled MPK contains no MLA ELF: {archive}")

        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        metadata.append(
            {
                "name": archive.name,
                "size_bytes": archive.stat().st_size,
                "sha256": digest.hexdigest(),
                "file_count": len(members),
            }
        )
    return metadata


def test_selected_model_full_compilation(
    model_inputs_path: Path,
    model_input_provenance: dict,
    candidate_sha: str,
    e2e_report_path: Path,
    tmp_path: Path,
) -> None:
    model_folder, selection_seed = _select_model(candidate_sha)
    output_path = tmp_path / "output"
    compile_config_path = tmp_path / "compile_config.py"
    compile_config_path.write_text(COMPILE_CONFIGURATION, encoding="utf-8")

    report = {
        "candidate_sha": candidate_sha,
        "selection_seed": selection_seed,
        "eligible_models": list(E2E_ELIGIBLE_MODELS),
        "selected_model": model_folder,
        "status": "failed",
        "stages": [],
        "artifacts": [],
    }
    e2e_report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        started = time.monotonic()
        matching_sources = [
            source
            for source in model_input_provenance.get("models", [])
            if source.get("model_folder") == model_folder
        ]
        if len(matching_sources) != 1:
            raise AssertionError(
                f"Expected one provenance entry for {model_folder}, "
                f"found {len(matching_sources)}."
            )
        source = matching_sources[0]
        report["source"] = {
            "repo_id": source["repo_id"],
            "requested_revision": source["requested_revision"],
            "resolved_revision": source["resolved_revision"],
            "payload_format": source["payload_format"],
        }
        model_path = require_readable_path(
            model_inputs_path / model_folder,
            f"selected E2E model {model_folder}",
        )
        report["stages"].append(
            {
                "name": "source_resolution",
                "duration_seconds": round(time.monotonic() - started, 3),
                "return_code": 0,
            }
        )

        started = time.monotonic()
        hf_config = AutoConfig.from_pretrained(
            model_path, local_files_only=True
        ).to_dict()
        llima_config = VlmConfig.from_hf_config(
            ModelFormat.FORMAT_HF,
            model_path,
            hf_config,
        )
        report["configuration"] = {
            "model_type": llima_config.model_type,
            "language_architecture": str(llima_config.lm_cfg.arch),
            "field_count": len(asdict(llima_config)),
        }
        report["stages"].append(
            {
                "name": "configuration",
                "duration_seconds": round(time.monotonic() - started, 3),
                "return_code": 0,
            }
        )

        base_command = [
            "llima-compile",
            "-c",
            str(compile_config_path),
            "-j",
            "4",
            "-o",
            str(output_path),
            str(model_path),
        ]
        _run_stage("onnx", base_command + ["--onnx"], report)
        _run_stage("quantization", base_command + ["--quantize"], report)
        if RUN_COMPILE_STAGE:
            _run_stage("model_compiler", base_command + ["--compile"], report)

            started = time.monotonic()
            report["artifacts"] = _validate_compiled_artifacts(output_path)
            report["stages"].append(
                {
                    "name": "artifact_validation",
                    "duration_seconds": round(time.monotonic() - started, 3),
                    "return_code": 0,
                }
            )
        else:
            report["stages"].append(
                {
                    "name": "model_compiler",
                    "duration_seconds": 0,
                    "return_code": 0,
                    "skipped": True,
                }
            )
        report["status"] = "passed"
    finally:
        e2e_report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
