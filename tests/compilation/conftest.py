import os
from pathlib import Path

import pytest

from tests.compilation.helpers.paths import require_readable_path


def pytest_addoption(parser):
    group = parser.getgroup("llima compilation test data")
    group.addoption(
        "--model-inputs-path",
        default=os.environ.get("LLIMA_HF_MODELS_PATH"),
        help=(
            "Root directory containing prepared Hugging Face and GGUF model inputs. "
            "Can also be set with LLIMA_HF_MODELS_PATH."
        ),
    )


@pytest.fixture(scope="session")
def model_inputs_path(request) -> Path:
    raw_path = request.config.getoption("--model-inputs-path")
    if not raw_path:
        pytest.fail(
            "Model input root is required; pass --model-inputs-path or set "
            "LLIMA_HF_MODELS_PATH."
        )
    return require_readable_path(Path(raw_path).expanduser(), "model input root")
