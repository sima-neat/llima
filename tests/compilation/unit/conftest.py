import socket

import pytest


_MODEL_PATH_ENVIRONMENT_VARIABLES = (
    "LLIMA_HF_MODELS_PATH",
    "LLIMA_GGUF_MODELS_PATH",
    "LLIMA_GGUF_HF_MODEL_PATH",
)


@pytest.fixture(autouse=True)
def enforce_hermetic_compiler_unit_environment(monkeypatch: pytest.MonkeyPatch):
    """Keep compiler unit tests independent of model caches and network access."""
    for variable in _MODEL_PATH_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def reject_network(*_args, **_kwargs):
        pytest.fail("Fast compiler unit tests must not access the network.")

    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket.socket, "connect", reject_network)
