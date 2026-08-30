"""Qwen3-TTS raw-ELF model-directory support for ``llima run``.

The Qwen3 deployment is self contained: its validated ARM64 executable runs
the raw ELF graphs directly and owns its model buffers. This module recognizes
the downloaded model directory and launches that binary. It deliberately does
not route through the VLM CLI or ModelExecutor.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


_EXECUTABLE = "qwen3tts"
_MANIFEST = "devkit/qwen3_tts_config.json"


@dataclass(frozen=True)
class Qwen3TTSPackage:
    """Validated downloaded Qwen3-TTS model layout."""

    root: Path

    @property
    def executable(self) -> Path:
        return self.root / "runtime" / "bin" / _EXECUTABLE

    @property
    def runtime_lib_dir(self) -> Path:
        return self.root / "runtime" / "lib"

    @property
    def model_dir(self) -> Path:
        return self.root / "qwen3_model"

    @property
    def components_dir(self) -> Path:
        return self.root / "qwen3_components"


def _is_package_root(path: Path) -> bool:
    return (
        (path / "runtime" / "bin" / _EXECUTABLE).is_file()
        and (path / "runtime" / "lib").is_dir()
        and (path / "qwen3_model" / "mpk").is_dir()
        and (path / "qwen3_components").is_dir()
        and (path / _MANIFEST).is_file()
    )


def _package_root_from_directory(path: Path) -> Path | None:
    return path if _is_package_root(path) else None


def is_qwen3tts_package(path: Path) -> bool:
    """Return whether *path* is a downloaded Qwen3-TTS model directory."""
    return _package_root_from_directory(path) is not None


def resolve_qwen3tts_package(path: Path) -> Qwen3TTSPackage:
    """Resolve a downloaded Qwen3-TTS model directory to an executable package."""
    root = _package_root_from_directory(path)
    if root is None:
        raise RuntimeError(f"Not a Qwen3-TTS model directory: {path}")
    return Qwen3TTSPackage(root=root)


def run_qwen3tts(package: Qwen3TTSPackage, args: object) -> int:
    """Run the bundled raw-ELF executable with normalized CLI arguments."""
    output_wav = Path(args.output_wav).expanduser().resolve()
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(package.executable),
        "--model-dir",
        str(package.model_dir),
        "--components-dir",
        str(package.components_dir),
        "--prompt",
        args.prompt,
        "--speaker",
        args.speaker,
        "--language",
        args.language,
        "--seed",
        str(args.seed),
        "--max-frames",
        str(args.max_frames),
        "--prefill-mode",
        "prefix_kv",
        # The bundled raw runtime otherwise enables its legacy endpoint
        # heuristic by default.  Qwen3-TTS package execution is EOS/max-frame
        # terminated only; it must not trim generated frames.
        "--endpoint-disable",
        "--out-wav",
        str(output_wav),
    ]
    command.append("--sample" if args.do_sample else "--no-sample")
    command.append("--subtalker-sample" if args.subtalker_do_sample else "--subtalker-no-sample")

    environment = os.environ.copy()
    existing_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(package.runtime_lib_dir)
        if not existing_library_path
        else f"{package.runtime_lib_dir}:{existing_library_path}"
    )
    print(f"Running Qwen3-TTS raw-ELF package: {package.root}", flush=True)
    subprocess.run(command, env=environment, check=True)
    print(f"WAV written to {output_wav}", flush=True)
    return 0
