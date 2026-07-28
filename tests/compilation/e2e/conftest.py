import json
import os
from pathlib import Path

import pytest

from tests.compilation.helpers.paths import require_readable_path


def pytest_addoption(parser):
    group = parser.getgroup("llima compiler E2E")
    group.addoption(
        "--candidate-sha",
        default=os.environ.get("GITHUB_SHA"),
    )
    group.addoption(
        "--e2e-report",
        default=os.environ.get("LLIMA_E2E_REPORT"),
    )
    group.addoption(
        "--model-input-provenance",
        default=os.environ.get("LLIMA_MODEL_INPUT_PROVENANCE"),
    )


@pytest.fixture(scope="session")
def candidate_sha(request) -> str:
    value = request.config.getoption("--candidate-sha")
    if not value:
        pytest.fail("Candidate SHA is required; pass --candidate-sha or set GITHUB_SHA.")
    return value


@pytest.fixture(scope="session")
def e2e_report_path(request, tmp_path_factory) -> Path:
    value = request.config.getoption("--e2e-report")
    if value:
        return Path(value)
    return tmp_path_factory.getbasetemp() / "compiler-e2e-report.json"


@pytest.fixture(scope="session")
def model_input_provenance(request) -> dict:
    value = request.config.getoption("--model-input-provenance")
    if not value:
        pytest.fail(
            "Model-input provenance is required; pass --model-input-provenance "
            "or set LLIMA_MODEL_INPUT_PROVENANCE."
        )
    path = require_readable_path(Path(value), "model-input provenance")
    return json.loads(path.read_text(encoding="utf-8"))
