import os
import shlex
import socket
import time
from dataclasses import dataclass, field
from getpass import getpass
from multiprocessing import Process
from pathlib import Path

import invoke.exceptions
from fabric import Connection
from loguru import logger


@dataclass
class ModalixBoard:
    address: str | None
    port: str | int
    name: str = "modalix"
    model: str | None = "/"
    ssh_user: str = "sima"
    ssh_password: str | None = None
    venv_path: str | None = None

    def start_server(self):
        logger.info("Starting ZMQ server")
        # Start the llima benchmark server on the board.
        llima_cmd = "llima"
        if self.venv_path is not None:
            llima_cmd = f"{self.venv_path}/bin/llima"
        server_cmd = (
            f"{shlex.quote(llima_cmd)} benchmark-server "
            f"{shlex.quote(str(self.model))} --port {shlex.quote(str(self.port))}"
        )
        logger.info(f"Starting server on {self.address}:{self.port}")
        logger.debug(server_cmd)
        if self.ssh_password is None:
            self.ssh_password = getpass(f"{self.ssh_user}@{self.address}'s pasword: ")
        conn = Connection(
            host=self.address, user=self.ssh_user, connect_kwargs={"password": self.ssh_password}
        )
        conn.run("pkill -f '[l]lima benchmark-server' 2>/dev/null", warn=True)
        conn.run("pkill --older 60 -f '[l]lima' 2>/dev/null", warn=True)
        time.sleep(2)
        remote_home = shlex.quote(f"/home/{self.ssh_user}")
        conn.run(
            f"cd {remote_home} && nohup {server_cmd} > server.log 2>&1 < /dev/null &",
            disown=True,
        )
        conn.close()

        # Wait until the port is open.
        logger.info("Waiting for the server to be ready to accept requests")
        timeout = int(os.getenv("SIMA_LLIMA_BENCHMARK_START_SERVER_TIMEOUT", 600))
        start_time = time.time()
        is_port_open = False
        while time.time() - start_time < timeout:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((self.address, self.port)) == 0:
                    is_port_open = True
                    break
        if not is_port_open:
            raise RuntimeError("Timeout: failed to start the benchmark server")
        logger.info("Started ZMQ server")

    def wait_for_server_stop(self, timeout: int = 30) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex((self.address, self.port)) != 0:
                    return True
            time.sleep(1)
        return False

    def stop_server(self):
        conn = Connection(
            host=self.address, user=self.ssh_user, connect_kwargs={"password": self.ssh_password}
        )
        conn.run("pkill -f '[l]lima benchmark-server' 2>/dev/null", warn=True)
        conn.close()
        self.wait_for_server_stop(timeout=10)

    @property
    def tcp_uri(self) -> str:
        return f"tcp://{self.address}:{self.port}"

    @property
    def board_name(self) -> str:
        return f"{self.name}@{self.address}:{self.port}"
