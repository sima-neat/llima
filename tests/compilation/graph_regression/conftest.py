import json
import os
from pathlib import Path

import pytest

from tests.compilation.helpers.paths import require_readable_path


def pytest_addoption(parser):
    group = parser.getgroup("llima native graph regression")
    group.addoption(
        "--graph-validation-mode",
        choices=("compare", "candidate-only"),
        default=os.environ.get("LLIMA_GRAPH_VALIDATION_MODE", "compare"),
    )
    group.addoption(
        "--candidate-graph-root",
        default=os.environ.get("LLIMA_CANDIDATE_GRAPH_ROOT"),
    )
    group.addoption(
        "--candidate-graph-manifest",
        default=os.environ.get("LLIMA_CANDIDATE_GRAPH_MANIFEST"),
    )
    group.addoption(
        "--base-graph-root",
        default=os.environ.get("LLIMA_BASE_GRAPH_ROOT"),
    )
    group.addoption(
        "--base-graph-manifest",
        default=os.environ.get("LLIMA_BASE_GRAPH_MANIFEST"),
    )


def _required_option(request, option: str, description: str) -> Path:
    raw_path = request.config.getoption(option)
    if not raw_path:
        pytest.fail(f"{description} is required; pass {option}.")
    return require_readable_path(Path(raw_path), description)


@pytest.fixture(scope="session")
def graph_validation_mode(request) -> str:
    return request.config.getoption("--graph-validation-mode")


@pytest.fixture(scope="session")
def candidate_graph_root(request) -> Path:
    return _required_option(request, "--candidate-graph-root", "candidate native graph root")


@pytest.fixture(scope="session")
def base_graph_root(request, graph_validation_mode: str) -> Path | None:
    if graph_validation_mode == "candidate-only":
        return None
    return _required_option(request, "--base-graph-root", "base native graph root")


@pytest.fixture(scope="session")
def candidate_graph_manifest(request) -> dict:
    path = _required_option(
        request, "--candidate-graph-manifest", "candidate native graph manifest"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def base_graph_manifest(request, graph_validation_mode: str) -> dict | None:
    if graph_validation_mode == "candidate-only":
        return None
    path = _required_option(request, "--base-graph-manifest", "base native graph manifest")
    return json.loads(path.read_text(encoding="utf-8"))
