import os
import platform
import shutil
import signal
import socket
import struct
import subprocess
from pathlib import Path

import msgpack
import zmq


DEFAULT_MODELS_PATH = Path("/media/nvme/llima/models")
DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-Autoround-a16w4"
SERVER_PUBLIC_KEY = b"xA?#1AE663fk][M)Dd9}x?#nV2Iy!p3^&>kerN.>"


def _text_model_name() -> str:
    model_name = os.environ.get("SIMA_TEST_LLIMA_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    assert model_name
    assert "/" not in model_name
    assert ".." not in model_name
    return model_name


def _unused_tcp_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_zmq_generate_request_and_remote_stop(tmp_path):
    assert platform.machine() == "aarch64", "runtime tests require an ARM64 DevKit"
    subprocess.run(
        ["systemctl", "is-active", "--quiet", "simaai-appcomplex.service"],
        check=True,
    )

    models_path = Path(os.environ.get("LLIMA_MODELS_PATH", DEFAULT_MODELS_PATH))
    model_name = _text_model_name()
    assert (models_path / model_name / "devkit" / "vlm_config.json").is_file()

    llima = shutil.which("llima")
    assert llima is not None, "installed llima executable was not found"
    assert Path(llima).resolve().is_relative_to("/usr")

    port = _unused_tcp_port()
    env = os.environ.copy()
    env["LLIMA_MODELS_PATH"] = str(models_path)
    process = subprocess.Popen(
        [llima, "benchmark-server", model_name, "--port", str(port)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    context = zmq.Context()
    client = context.socket(zmq.REQ)
    client.setsockopt(zmq.LINGER, 0)
    client.setsockopt(zmq.SNDTIMEO, 60_000)
    client.setsockopt(zmq.RCVTIMEO, 60_000)
    client.curve_publickey, client.curve_secretkey = zmq.curve_keypair()
    client.curve_serverkey = SERVER_PUBLIC_KEY
    client.connect(f"tcp://127.0.0.1:{port}")

    output = ""
    try:
        metadata = {
            "type": "generate",
            "tensor_dtype": "uint32",
            "tensor_shape": [1, 1],
            "max_num_tokens": 4,
        }
        client.send_multipart(
            [
                msgpack.packb(metadata, use_bin_type=True),
                struct.pack("=I", 0),
            ]
        )
        response = client.recv_multipart()
        assert len(response) == 2

        response_metadata = msgpack.unpackb(response[0], raw=False)
        assert response_metadata["tensor_dtype"] == "uint32"
        assert response_metadata["tensor_shape"][0] == 1
        generated_tokens = response_metadata["tensor_shape"][1]
        assert generated_tokens > 0
        assert len(response[1]) == generated_tokens * 4
        assert response_metadata["infer_time_ns"] > 0

        client.send(b"stop")
        process.wait(timeout=30)
        output = process.stdout.read() if process.stdout is not None else ""
        assert process.returncode == 0, output
    finally:
        client.close()
        context.term()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                remaining, _ = process.communicate(timeout=30)
                output += remaining
            except subprocess.TimeoutExpired:
                process.kill()
                remaining, _ = process.communicate(timeout=10)
                output += remaining
                raise AssertionError(f"ZMQ server did not stop after SIGINT:\n{output}")
        assert process.returncode == 0, output
