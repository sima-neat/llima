import json
import os
from pathlib import Path

import pytest

from tests.compilation.helpers.paths import require_readable_path


def pytest_addoption(parser):
    group = parser.getgroup("llima ONNX regression")
    group.addoption(
        "--onnx-validation-mode",
        choices=("compare", "candidate-only"),
        default=os.environ.get("LLIMA_ONNX_VALIDATION_MODE", "compare"),
    )
    group.addoption(
        "--candidate-onnx-root",
        default=os.environ.get("LLIMA_CANDIDATE_ONNX_ROOT"),
    )
    group.addoption(
        "--candidate-onnx-manifest",
        default=os.environ.get("LLIMA_CANDIDATE_ONNX_MANIFEST"),
    )
    group.addoption(
        "--base-onnx-root",
        default=os.environ.get("LLIMA_BASE_ONNX_ROOT"),
    )
    group.addoption(
        "--base-onnx-manifest",
        default=os.environ.get("LLIMA_BASE_ONNX_MANIFEST"),
    )


def _required_option(request, option: str, description: str) -> Path:
    raw_path = request.config.getoption(option)
    if not raw_path:
        pytest.fail(f"{description} is required; pass {option}.")
    return require_readable_path(Path(raw_path), description)


@pytest.fixture(scope="session")
def onnx_validation_mode(request) -> str:
    return request.config.getoption("--onnx-validation-mode")


@pytest.fixture(scope="session")
def candidate_onnx_root(request) -> Path:
    return _required_option(request, "--candidate-onnx-root", "candidate ONNX root")


@pytest.fixture(scope="session")
def base_onnx_root(request, onnx_validation_mode: str) -> Path | None:
    if onnx_validation_mode == "candidate-only":
        return None
    return _required_option(request, "--base-onnx-root", "base ONNX root")


@pytest.fixture(scope="session")
def candidate_onnx_manifest(request) -> dict:
    path = _required_option(
        request, "--candidate-onnx-manifest", "candidate ONNX manifest"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def base_onnx_manifest(request, onnx_validation_mode: str) -> dict | None:
    if onnx_validation_mode == "candidate-only":
        return None
    path = _required_option(request, "--base-onnx-manifest", "base ONNX manifest")
    return json.loads(path.read_text(encoding="utf-8"))
