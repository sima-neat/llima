#########################################################
# Copyright (C) 2026 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("llima test data")
    group.addoption(
        "--hf-models-path",
        default=os.environ.get("LLIMA_HF_MODELS_PATH"),
        help="Root directory containing HuggingFace model cache folders. "
             "Can also be set with LLIMA_HF_MODELS_PATH.",
    )
    group.addoption(
        "--gguf-models-path",
        default=os.environ.get("LLIMA_GGUF_MODELS_PATH"),
        help="Root directory containing GGUF test models and GGUF reference arrays. "
             "Can also be set with LLIMA_GGUF_MODELS_PATH.",
    )
    group.addoption(
        "--gguf-hf-model-path",
        default=os.environ.get("LLIMA_GGUF_HF_MODEL_PATH"),
        help="HuggingFace model directory used as reference for GGUF weight tests. "
             "Can also be set with LLIMA_GGUF_HF_MODEL_PATH.",
    )
    group.addoption(
        "--reference-onnx-path",
        default=os.environ.get("LLIMA_REFERENCE_ONNX_PATH"),
        help="Root directory containing reference ONNX files. "
             "Can also be set with LLIMA_REFERENCE_ONNX_PATH.",
    )
    group.addoption(
        "--reference-draft-onnx-path",
        default=os.environ.get("LLIMA_REFERENCE_DRAFT_ONNX_PATH"),
        help="Root directory containing reference draft-model ONNX files. "
             "Can also be set with LLIMA_REFERENCE_DRAFT_ONNX_PATH.",
    )


def _configured_path(request, option_name: str, env_name: str, description: str) -> Path:
    raw_path = request.config.getoption(option_name)
    if not raw_path:
        pytest.skip(f"{description} is not configured; pass {option_name} or set {env_name}")
    return require_readable_path(Path(raw_path).expanduser(), description)


def require_readable_path(path: Path, description: str | None = None) -> Path:
    label = description or str(path)
    try:
        exists = path.exists()
        is_dir = path.is_dir()
    except OSError as exc:
        pytest.skip(f"{label} is not accessible: {exc}")
    if not exists:
        pytest.skip(f"{label} not found: {path}")

    access_mode = os.R_OK | (os.X_OK if is_dir else 0)
    if not os.access(path, access_mode):
        pytest.skip(f"{label} is not readable: {path}")
    return path


@pytest.fixture(scope="session")
def hf_models_path(request) -> Path:
    return _configured_path(request, "--hf-models-path", "LLIMA_HF_MODELS_PATH", "HF model root")


@pytest.fixture(scope="session")
def gguf_models_path(request) -> Path:
    return _configured_path(request, "--gguf-models-path", "LLIMA_GGUF_MODELS_PATH", "GGUF model root")


@pytest.fixture(scope="session")
def gguf_hf_model_path(request) -> Path:
    return _configured_path(
        request, "--gguf-hf-model-path", "LLIMA_GGUF_HF_MODEL_PATH", "GGUF HF reference model"
    )


@pytest.fixture(scope="session")
def reference_onnx_path(request) -> Path:
    return _configured_path(
        request, "--reference-onnx-path", "LLIMA_REFERENCE_ONNX_PATH", "reference ONNX root"
    )


@pytest.fixture(scope="session")
def reference_draft_onnx_path(request) -> Path:
    return _configured_path(
        request,
        "--reference-draft-onnx-path",
        "LLIMA_REFERENCE_DRAFT_ONNX_PATH",
        "reference draft ONNX root",
    )
