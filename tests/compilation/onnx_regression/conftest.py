import json
import os
from pathlib import Path

import pytest

from tests.compilation.helpers.paths import require_readable_path


def pytest_addoption(parser):
    group = parser.getgroup("llima ONNX regression")
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
def candidate_onnx_root(request) -> Path:
    return _required_option(request, "--candidate-onnx-root", "candidate ONNX root")


@pytest.fixture(scope="session")
def base_onnx_root(request) -> Path:
    return _required_option(request, "--base-onnx-root", "base ONNX root")


@pytest.fixture(scope="session")
def candidate_onnx_manifest(request) -> dict:
    path = _required_option(
        request, "--candidate-onnx-manifest", "candidate ONNX manifest"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def base_onnx_manifest(request) -> dict:
    path = _required_option(request, "--base-onnx-manifest", "base ONNX manifest")
    return json.loads(path.read_text(encoding="utf-8"))
