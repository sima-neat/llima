import http.client
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path


DEFAULT_MODELS_PATH = Path("/media/nvme/llima/models")
DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
HOST = "127.0.0.1"
PORT = 9998
QUERY = "What is the capital of Germany? Answer in one sentence."


def _text_model_name() -> str:
    model_name = os.environ.get("SIMA_TEST_LLIMA_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    assert model_name
    assert "/" not in model_name
    assert ".." not in model_name
    return model_name


def _wait_for_server(process: subprocess.Popen[str], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            raise AssertionError(f"llima web exited before accepting requests:\n{output}")
        try:
            with socket.create_connection((HOST, PORT), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    raise AssertionError("llima web did not start listening within 30 seconds")


@contextmanager
def _running_web_server(tmp_path):
    models_path = Path(os.environ.get("LLIMA_MODELS_PATH", DEFAULT_MODELS_PATH))
    model_name = _text_model_name()
    assert (models_path / model_name / "devkit" / "vlm_config.json").is_file()

    llima = shutil.which("llima")
    assert llima is not None, "installed llima executable was not found"
    assert Path(llima).resolve().is_relative_to("/usr")

    env = os.environ.copy()
    env["LLIMA_MODELS_PATH"] = str(models_path)
    process = subprocess.Popen(
        [llima, "run", model_name, "--mode", "web"],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = ""
    try:
        _wait_for_server(process)
        yield
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            remaining, _ = process.communicate(timeout=30)
            output += remaining
        except subprocess.TimeoutExpired:
            process.kill()
            remaining, _ = process.communicate(timeout=10)
            output += remaining
            raise AssertionError(f"llima web did not stop after SIGINT:\n{output}")
        assert process.returncode == 0, output


def _post(path: str, body: bytes, timeout: float = 180):
    connection = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        connection.request(
            "POST",
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


def _chat_body(*, stream: bool, query: str = QUERY) -> bytes:
    return json.dumps(
        {
            "model": "runtime-test",
            "stream": stream,
            "messages": [{"role": "user", "content": query}],
        }
    ).encode()


def _reconstruct_openai_stream(body: bytes) -> tuple[str, bool]:
    content = []
    saw_done = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            saw_done = True
            continue
        chunk = json.loads(payload)
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            content.append(delta["content"])
    return "".join(content), saw_done


def test_openai_http_protocol_and_recovery(tmp_path):
    assert platform.machine() == "aarch64", "runtime tests require an ARM64 DevKit"
    subprocess.run(
        ["systemctl", "is-active", "--quiet", "simaai-appcomplex.service"],
        check=True,
    )

    with _running_web_server(tmp_path):
        malformed_status, _, malformed_body = _post("/v1/chat/completions", b"{")
        assert malformed_status in {400, 500}
        assert json.loads(malformed_body)["error"]

        status, headers, body = _post(
            "/v1/chat/completions", _chat_body(stream=False)
        )
        response = json.loads(body)
        assert status == 200, response
        assert headers["Content-Type"].startswith("application/json")
        assert response["object"] == "chat.completion"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert "capital of Germany is Berlin" in response["choices"][0]["message"]["content"]

        status, headers, body = _post(
            "/v1/chat/completions", _chat_body(stream=True)
        )
        reconstructed, saw_done = _reconstruct_openai_stream(body)
        assert status == 200, body.decode()
        assert headers["Content-Type"].startswith("text/event-stream")
        assert saw_done
        assert "capital of Germany is Berlin" in reconstructed


def test_stop_interrupts_active_http_inference(tmp_path):
    assert platform.machine() == "aarch64", "runtime tests require an ARM64 DevKit"
    first_content = threading.Event()
    stream_finished = threading.Event()
    stream_errors = []

    def consume_long_stream():
        connection = http.client.HTTPConnection(HOST, PORT, timeout=180)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=_chat_body(
                    stream=True,
                    query="Count upward from 1 to 1000, writing every number.",
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            assert response.status == 200
            while line := response.readline():
                if not line.startswith(b"data: "):
                    continue
                payload = line.removeprefix(b"data: ").strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk["choices"][0]["delta"].get("content"):
                    first_content.set()
        except BaseException as exc:
            stream_errors.append(exc)
        finally:
            connection.close()
            stream_finished.set()

    with _running_web_server(tmp_path):
        consumer = threading.Thread(target=consume_long_stream, daemon=True)
        consumer.start()
        assert first_content.wait(timeout=30), "stream produced no content before timeout"

        stop_started = time.monotonic()
        status, _, _ = _post("/stop", b"{}", timeout=30)
        assert status == 200
        assert stream_finished.wait(timeout=30), "stream remained active after /stop"
        assert time.monotonic() - stop_started < 30
        consumer.join(timeout=1)
        assert not consumer.is_alive()
        assert not stream_errors

        status, _, body = _post("/v1/chat/completions", _chat_body(stream=False))
        assert status == 200, body.decode()
        assert "capital of Germany is Berlin" in json.loads(body)["choices"][0]["message"]["content"]
