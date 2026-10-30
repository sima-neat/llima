import os
import re
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from getpass import getpass
from pathlib import Path

from fabric import Connection
from loguru import logger


def _is_modalix() -> bool:
    try:
        buildinfo = Path("/etc/buildinfo").read_text()
    except OSError:
        return False
    return re.search(r"^MACHINE\s*=\s*['\"]?modalix['\"]?\s*$", buildinfo, re.MULTILINE) is not None


@dataclass
class ModalixBoard:
    address: str | None
    port: str | int
    name: str = "modalix"
    model: str | None = "/"
    ssh_user: str = "sima"
    ssh_password: str | None = None
    venv_path: str | None = None
    _process: subprocess.Popen | None = field(default=None, init=False, repr=False)
    _connection: Connection | None = field(default=None, init=False, repr=False)
    _remote_pid: int | None = field(default=None, init=False, repr=False)

    @property
    def host(self) -> str:
        if self.address:
            return self.address
        if not _is_modalix():
            raise ValueError(
                "Modalix evaluation from a host requires a remote target (--board_ip). "
                "Omitting --board_ip is only supported on a Modalix DevKit."
            )
        return "127.0.0.1"

    def _port_open(self) -> bool:
        try:
            with socket.create_connection((self.host, int(self.port)), timeout=1):
                return True
        except OSError:
            return False

    def _server_running(self) -> bool:
        if self._process is not None:
            return self._process.poll() is None
        if self._remote_pid is not None:
            return self._connection.run(
                f"kill -0 {self._remote_pid} 2>/dev/null", hide=True, warn=True, timeout=5
            ).ok
        return False

    def start_server(self, log_path: str | Path = "server.log"):
        host = self.host
        if not self.model:
            raise ValueError("Starting a benchmark server requires --board_model.")
        if self._process is not None or self._remote_pid is not None:
            raise RuntimeError("This benchmark run already owns a server.")
        if self._port_open():
            raise RuntimeError(
                f"Port {host}:{self.port} is already in use. "
                "Use --no-board_start_server to attach to an existing benchmark server."
            )
        timeout = int(os.getenv("SIMA_LLIMA_BENCHMARK_START_SERVER_TIMEOUT", 600))
        server_args = ["benchmark-server", str(self.model), "--port", str(self.port)]
        logger.info("Starting benchmark server on {}:{}", host, self.port)
        try:
            if not self.address:
                # MoLE's virtual environment also has a llima entry point; use the installed runtime.
                llima_path = (
                    Path(self.venv_path).expanduser() / "bin" / "llima"
                    if self.venv_path else Path("/usr/bin/llima")
                )
                if not llima_path.is_file():
                    raise FileNotFoundError(
                        f"LLiMa runtime command not found: {llima_path}. "
                        "Install the LLiMa runtime or specify --board_venv_path."
                    )
                log_path = Path(log_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                server_log = str(log_path.resolve())
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.pop("PYTHONHOME", None)
                with log_path.open("w") as log:
                    self._process = subprocess.Popen(
                        [str(llima_path), *server_args],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True,
                    )
            else:
                llima_cmd = "llima"
                if self.venv_path is not None:
                    llima_path = f"{self.venv_path}/bin/llima"
                    llima_cmd = (
                        f'"$HOME"{shlex.quote(llima_path[1:])}'
                        if llima_path.startswith("~/") else shlex.quote(llima_path)
                    )
                if self.ssh_password is None:
                    self.ssh_password = getpass(f"{self.ssh_user}@{host}'s password: ")
                self._connection = Connection(
                    host=host, user=self.ssh_user,
                    connect_kwargs={"password": self.ssh_password}, connect_timeout=10,
                )
                remote_log = f"llima-benchmark-{self.port}-{time.time_ns()}.log"
                server_log = f"{host}:~/{remote_log}"
                result = self._connection.run(
                    f'cd "$HOME" && {{ nohup {llima_cmd} {shlex.join(server_args)} '
                    f'> {shlex.quote(remote_log)} 2>&1 < /dev/null & echo $!; }}',
                    hide=True, timeout=10,
                )
                pid = int(result.stdout.strip())
                if pid <= 1:
                    raise RuntimeError("SSH did not return a valid benchmark server PID")
                self._remote_pid = pid

            logger.info("Benchmark server log: {}", server_log)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not self._server_running():
                    raise RuntimeError(f"Benchmark server exited during startup; see {server_log}")
                if self._port_open():
                    logger.info("Benchmark server is ready")
                    return
                time.sleep(0.2)
            raise RuntimeError(f"Timed out starting benchmark server; see {server_log}")
        except BaseException:
            self.stop_server()
            raise

    def wait_for_server_stop(self, timeout: int = 30) -> bool:
        if self._process is not None:
            try:
                self._process.wait(timeout=timeout)
                return True
            except subprocess.TimeoutExpired:
                return False
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            running = self._server_running() if self._remote_pid is not None else self._port_open()
            if not running:
                return True
            time.sleep(0.2)
        return False

    def stop_server(self):
        try:
            if not self._server_running():
                return
            if self._process is not None:
                self._process.send_signal(signal.SIGINT)
            else:
                self._connection.run(
                    f"kill -INT {self._remote_pid}", hide=True, warn=True, timeout=5
                )
            if not self.wait_for_server_stop():
                logger.warning("Benchmark server did not stop after SIGINT; killing owned process")
                if self._process is not None:
                    self._process.kill()
                    self._process.wait(timeout=5)
                else:
                    self._connection.run(
                        f"kill -KILL {self._remote_pid}", hide=True, warn=True, timeout=5
                    )
        finally:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._remote_pid = None
            self._process = None

    @property
    def tcp_uri(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    @property
    def board_name(self) -> str:
        return f"{self.name}@{self.host}:{self.port}"
