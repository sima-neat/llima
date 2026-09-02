"""Qwen3-TTS model-directory support for ``llima run``.

The downloaded directory contains model assets only. The matching ARM64 raw-ELF
runner is supplied by the installed ``sima-lmm-cli`` package.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


_EXECUTABLE = "qwen3tts"
_MANIFEST = "devkit/qwen3_tts_config.json"
_EXECUTABLE_ENV = "SIMA_LMM_QWEN3TTS_EXECUTABLE"


@dataclass(frozen=True)
class Qwen3TTSPackage:
    """Validated downloaded Qwen3-TTS model layout."""

    root: Path

    @property
    def executable(self) -> Path:
        configured = os.environ.get(_EXECUTABLE_ENV)
        if configured:
            return Path(configured)
        resolved = shutil.which(_EXECUTABLE)
        return Path(resolved) if resolved else Path(_EXECUTABLE)

    @property
    def model_dir(self) -> Path:
        return self.root / "qwen3_model"

    @property
    def components_dir(self) -> Path:
        return self.root / "qwen3_components"


def _is_package_root(path: Path) -> bool:
    return (
        (path / "qwen3_model" / "mpk").is_dir()
        and (path / "qwen3_components").is_dir()
        and (path / _MANIFEST).is_file()
    )


def _package_root_from_directory(path: Path) -> Path | None:
    return path if _is_package_root(path) else None


def is_qwen3tts_package(path: Path) -> bool:
    """Return whether *path* is a downloaded Qwen3-TTS model directory."""
    return _package_root_from_directory(path) is not None


def resolve_qwen3tts_package(path: Path) -> Qwen3TTSPackage:
    """Resolve a downloaded Qwen3-TTS model directory and installed runtime."""
    root = _package_root_from_directory(path)
    if root is None:
        raise RuntimeError(f"Not a Qwen3-TTS model directory: {path}")
    package = Qwen3TTSPackage(root=root)
    if not package.executable.is_file():
        raise RuntimeError(
            "Qwen3-TTS runner is not installed. Install sima-lmm-cli "
            f"or set {_EXECUTABLE_ENV} for a development runner."
        )
    return package


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
        "--out-wav",
        str(output_wav),
    ]
    command.append("--sample" if args.do_sample else "--no-sample")
    command.append("--subtalker-sample" if args.subtalker_do_sample else "--subtalker-no-sample")
    if args.endpoint_silence_rms is not None:
        command.extend(("--endpoint-silence-rms", str(args.endpoint_silence_rms)))
    if args.endpoint_silence_frames is not None:
        command.extend(("--endpoint-silence-frames", str(args.endpoint_silence_frames)))
    if args.endpoint_end_pad_frames is not None:
        command.extend(("--endpoint-end-pad-frames", str(args.endpoint_end_pad_frames)))

    print(f"Running Qwen3-TTS model: {package.root}", flush=True)
    environment = os.environ.copy()
    subprocess.run(command, env=environment, check=True)
    print(f"WAV written to {output_wav}", flush=True)
    return 0
