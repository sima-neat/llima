"""The shared host launcher must support key auth and retain password fallback."""

from types import SimpleNamespace

import pytest
from paramiko.ssh_exception import AuthenticationException

from sima_lmm.mole import modalixboard
from sima_lmm.mole.modalixboard import ModalixBoard


@pytest.mark.parametrize("authentication", ["key", "password", "fallback"])
def test_remote_launcher_authentication_and_owned_cleanup(monkeypatch, authentication):
    connections, prompts = [], []

    class Connection:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            connections.append(self)

        def open(self):
            if authentication == "fallback" and "password" not in self.kwargs["connect_kwargs"]:
                raise AuthenticationException("No authorized key")

        def run(self, command, **kwargs):
            return SimpleNamespace(stdout="1234\n", ok=True)

        def close(self):
            self.closed = True

    monkeypatch.setattr(modalixboard, "Connection", Connection)
    monkeypatch.setattr(modalixboard, "getpass", lambda prompt: prompts.append(prompt) or "test-password")
    board = ModalixBoard(
        address="example.invalid", port=8002, model="model",
        ssh_password="test-password" if authentication == "password" else None,
    )
    probes = iter([False, True])
    monkeypatch.setattr(board, "_port_open", lambda: next(probes))
    monkeypatch.setattr(board, "_server_running", lambda: board._remote_pid is not None)
    monkeypatch.setattr(board, "wait_for_server_stop", lambda **kwargs: True)
    board.start_server()
    assert board._remote_pid == 1234
    if authentication == "fallback":
        assert len(connections) == 2 and len(prompts) == 1
        assert connections[0].closed
    else:
        assert len(connections) == 1 and not prompts
    if authentication == "key":
        assert connections[0].kwargs["connect_kwargs"] == {}
    else:
        assert connections[-1].kwargs["connect_kwargs"] == {"password": "test-password"}
    board.stop_server()
    assert all(connection.closed for connection in connections)
    assert board._connection is None and board._remote_pid is None
