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

import pytest


DEFAULT_MODELS_PATH = Path("/media/nvme/llima/models")
DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
DEFAULT_REASONING_QWEN_MODEL = "Qwen3-0.6B-GPTQ-a16w4"
DEFAULT_REASONING_GEMMA_MODEL = "Gemma-4-E2B-it-GPTQ-a16w4"
HOST = "127.0.0.1"
PORT = 9998
QUERY = "What is the capital of Germany? Answer in one sentence."
REASONING_QUERY = "Think briefly: what is 2 plus 3? Give the final answer."
REASONING_MARKERS = ("<think>", "</think>", "<|channel>", "<channel|>")
TOOL_MARKERS = ("<tool_call>", "</tool_call>", "<|tool_call>", "<tool_call|>")
TOOL_QUERY = (
    "You must call get_temperature exactly once to check the temperature in "
    "Berlin. Do not answer from memory."
)
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "get_temperature",
        "description": "Get the current outside temperature for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    },
}


def _text_model_name() -> str:
    return _model_name("SIMA_TEST_LLIMA_TEXT_MODEL", DEFAULT_TEXT_MODEL)


def _model_name(env_name: str, default: str) -> str:
    model_name = os.environ.get(env_name, default)
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
    raise AssertionError(
        f"llima web did not start listening within {timeout:g} seconds"
    )


@contextmanager
def _running_web_server(tmp_path, model_name: str | None = None):
    models_path = Path(os.environ.get("LLIMA_MODELS_PATH", DEFAULT_MODELS_PATH))
    model_name = model_name or _text_model_name()
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
        _wait_for_server(process, timeout=60)
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


def _reasoning_chat_body(
    model_name: str,
    *,
    stream: bool,
    enable_thinking: bool | None = None,
    chat_template_thinking: bool | None = None,
) -> bytes:
    payload = {
        "model": model_name,
        "stream": stream,
        "messages": [{"role": "user", "content": REASONING_QUERY}],
    }
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    if chat_template_thinking is not None:
        payload["chat_template_kwargs"] = {
            "enable_thinking": chat_template_thinking,
        }
    return json.dumps(payload).encode()


def _reasoning_tool_chat_body(model_name: str) -> bytes:
    return json.dumps(
        {
            "model": model_name,
            "stream": False,
            "enable_thinking": True,
            "messages": [{"role": "user", "content": TOOL_QUERY}],
            "tools": [TOOL_DEFINITION],
            "tool_choice": "auto",
        }
    ).encode()


def _assert_separated_reasoning(reasoning: str, content: str) -> None:
    assert reasoning.strip(), "reasoning output is empty"
    assert content.strip(), "final content is empty"
    assert "5" in content
    combined = reasoning + content
    for marker in REASONING_MARKERS:
        assert marker not in combined


def _reconstruct_openai_reasoning_stream(body: bytes) -> tuple[str, str, bool]:
    reasoning = []
    content = []
    saw_content = False
    saw_done = False
    for line in body.decode().splitlines():
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            saw_done = True
            continue
        delta = json.loads(payload)["choices"][0]["delta"]
        if delta.get("reasoning_content"):
            assert not saw_content, "reasoning arrived after final content"
            reasoning.append(delta["reasoning_content"])
        if delta.get("content"):
            saw_content = True
            content.append(delta["content"])
    return "".join(reasoning), "".join(content), saw_done


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


@pytest.mark.parametrize(
    ("model_env", "default_model"),
    (
        ("SIMA_TEST_LLIMA_REASONING_QWEN_MODEL", DEFAULT_REASONING_QWEN_MODEL),
        ("SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL", DEFAULT_REASONING_GEMMA_MODEL),
    ),
    ids=("qwen3", "gemma4"),
)
def test_reasoning_http_protocols(tmp_path, model_env, default_model):
    assert platform.machine() == "aarch64", "runtime tests require an ARM64 DevKit"
    model_name = _model_name(model_env, default_model)

    with _running_web_server(tmp_path, model_name):
        status, _, body = _post(
            "/v1/chat/completions",
            _reasoning_chat_body(model_name, stream=False, enable_thinking=True),
        )
        response = json.loads(body)
        assert status == 200, response
        message = response["choices"][0]["message"]
        assert "thinking" not in message
        _assert_separated_reasoning(
            message["reasoning_content"], message["content"]
        )

        status, _, body = _post(
            "/v1/chat/completions", _reasoning_tool_chat_body(model_name)
        )
        response = json.loads(body)
        assert status == 200, response
        choice = response["choices"][0]
        message = choice["message"]
        assert choice["finish_reason"] == "tool_calls"
        assert message["reasoning_content"].strip()
        assert message["content"] is None
        assert len(message["tool_calls"]) == 1
        tool_call = message["tool_calls"][0]
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_temperature"
        arguments = json.loads(tool_call["function"]["arguments"])
        assert arguments == {"city": "Berlin"}
        response_text = body.decode()
        for marker in (*REASONING_MARKERS, *TOOL_MARKERS):
            assert marker not in response_text

        status, headers, body = _post(
            "/v1/chat/completions",
            _reasoning_chat_body(
                model_name,
                stream=True,
                chat_template_thinking=True,
            ),
        )
        reasoning, content, saw_done = _reconstruct_openai_reasoning_stream(body)
        assert status == 200, body.decode()
        assert headers["Content-Type"].startswith("text/event-stream")
        assert saw_done
        _assert_separated_reasoning(reasoning, content)

        status, _, body = _post(
            "/api/chat",
            json.dumps(
                {
                    "model": model_name,
                    "messages": [{"role": "user", "content": REASONING_QUERY}],
                    "think": True,
                    "stream": False,
                }
            ).encode(),
        )
        response = json.loads(body)
        assert status == 200, response
        message = response["message"]
        assert "reasoning_content" not in message
        _assert_separated_reasoning(message["thinking"], message["content"])

        status, _, body = _post(
            "/api/generate",
            json.dumps(
                {
                    "model": model_name,
                    "prompt": REASONING_QUERY,
                    "think": True,
                    "stream": False,
                }
            ).encode(),
        )
        response = json.loads(body)
        assert status == 200, response
        _assert_separated_reasoning(response["thinking"], response["response"])

        status, _, body = _post(
            "/v1/chat/completions",
            _reasoning_chat_body(model_name, stream=False, enable_thinking=False),
        )
        response = json.loads(body)
        assert status == 200, response
        message = response["choices"][0]["message"]
        assert "reasoning_content" not in message
        assert message["content"].strip()
        assert "5" in message["content"]
        for marker in REASONING_MARKERS:
            assert marker not in message["content"]
