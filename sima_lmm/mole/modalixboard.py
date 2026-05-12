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
        # Use nohup to prevent the process from closing with the session.
        server_cmd = f""
        if self.venv_path is not None:
            server_cmd += f"{self.venv_path}/bin/"
        server_cmd += (
            f"llima benchmark-server {self.model} --port {self.port} > server.log 2>&1"
        )
        logger.info(f"Starting server on {self.address}:{self.port}")
        logger.debug(server_cmd)
        if self.ssh_password is None:
            self.ssh_password = getpass(f"{self.ssh_user}@{self.address}'s pasword: ")
        server_cmd += f" <<< '{self.ssh_password}' &"
        conn = Connection(
            host=self.address, user=self.ssh_user, connect_kwargs={"password": self.ssh_password}
        )
        conn.sudo(f"pkill -f '[l]lima' 2>/dev/null <<< '{self.ssh_password}'", warn=True)
        conn.sudo(server_cmd)
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

    def stop_server(self):
        conn = Connection(
            host=self.address, user=self.ssh_user, connect_kwargs={"password": self.ssh_password}
        )
        conn.sudo(f"pkill -f '[l]lima' 2>/dev/null <<< '{self.ssh_password}'", warn=True)
        conn.close()

    @property
    def tcp_uri(self) -> str:
        return f"tcp://{self.address}:{self.port}"

    @property
    def board_name(self) -> str:
        return f"{self.name}@{self.address}:{self.port}"
