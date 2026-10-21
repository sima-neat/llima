import base64
import json
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


def test_zmq_image_text_generation_and_request_recovery(tmp_path):
    """Run against installed packages/real MLA, including a model without logits."""
    assert platform.machine() == "aarch64", "runtime tests require an ARM64 DevKit"
    models = Path(os.environ.get("LLIMA_MODELS_PATH", DEFAULT_MODELS_PATH))
    model = Path(os.environ.get("SIMA_TEST_LLIMA_VLM_MODEL", "LFM2.5-VL-450M-Autoround-a16w4"))
    if not model.is_absolute():
        model = models / model
    assert (model / "devkit/vlm_config.json").is_file()
    llima = shutil.which("llima")
    assert llima is not None and Path(llima).resolve().is_relative_to("/usr")
    assets = Path(os.environ.get("SIMA_LMM_TEST_ASSETS_PATH", "/usr/share/sima_lmm/assets"))
    image = assets / "sjc.jpg"
    assert image.is_file()
    encoded = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
    port = _unused_tcp_port()
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [llima, "benchmark-server", str(model), "--port", str(port)],
            cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
        )
    context = zmq.Context()
    client = context.socket(zmq.REQ)
    client.linger = 0
    client.rcvtimeo = client.sndtimeo = 600_000
    client.curve_publickey, client.curve_secretkey = zmq.curve_keypair()
    client.curve_serverkey = SERVER_PUBLIC_KEY
    client.connect(f"tcp://127.0.0.1:{port}")

    def request(kind, body=b"", **kwargs):
        client.send_multipart([msgpack.packb({"type": kind, **kwargs}, use_bin_type=True), body])
        response = client.recv_multipart()
        assert len(response) == 2
        metadata = msgpack.unpackb(response[0], raw=False)
        return metadata, response[1]

    try:
        info_metadata, body = request("multimodal_info")
        assert not info_metadata.get("error"), info_metadata
        identity = json.loads(body)
        assert identity["protocol_version"] == 1
        assert len(identity["runtime"]["library_sha256"]) == 64
        assert identity["files"]["devkit/tokenizer.json"]["sha256"]
        assert identity.get("preprocessor_config.json") or identity.get("processor_config.json")
        budget = identity["max_num_tokens"]

        # The unsupported request must not kill the REP server or prevent a later valid request.
        unsupported, _ = request("generate_multimodal", json.dumps({"messages": []}).encode(),
                                 max_num_tokens=budget, max_new_tokens=32)
        assert unsupported["error_type"] == "unsupported"
        malformed, _ = request("generate", b"", tensor_dtype="uint32", tensor_shape=[1, 4])
        assert malformed.get("error")

        for image_count in (1, 2):
            content = [{"type": "text", "text": "Examine these images: "}]
            for _ in range(image_count):
                content.extend([{"type": "image", "image": encoded},
                                {"type": "text", "text": " Describe this scene briefly. "}])
            metadata, answer = request(
                "generate_multimodal",
                json.dumps({"messages": [{"role": "user", "content": content}]}).encode(),
                max_num_tokens=budget, max_new_tokens=32, enable_thinking=False,
            )
            assert not metadata.get("error"), metadata
            assert metadata["tensor_dtype"] == "utf8"
            assert metadata["tensor_shape"] == [len(answer)]
            assert answer.decode().strip()
            assert metadata["input_tokens"] > 0
            assert 0 < metadata["generated_tokens"] <= 32
            assert metadata["finish_reason"] in ("stop", "length")
            assert len(metadata["prompt_sha256"]) == 64
            assert isinstance(metadata["raw_answer"], str)

        client.send(b"stop")
        process.wait(timeout=30)
    finally:
        client.close()
        context.term()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        assert process.returncode == 0, (tmp_path / "server.log").read_text()
