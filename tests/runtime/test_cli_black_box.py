import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest


DEFAULT_MODELS_PATH = Path("/media/nvme/llima/models")
DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
QUERY = "What is the capital of Germany? Answer in one sentence."
CMA_TOLERANCE_KB = 128 * 1024


def _text_model_name() -> str:
    model_name = os.environ.get("SIMA_TEST_LLIMA_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    assert model_name
    assert "/" not in model_name
    assert ".." not in model_name
    return model_name


def _daemon_snapshot() -> dict[str, int]:
    pid = int(
        subprocess.run(
            [
                "systemctl",
                "show",
                "--property=MainPID",
                "--value",
                "simaai-appcomplex.service",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout
    )
    assert pid > 0

    fd_result = subprocess.run(
        [
            "sudo",
            "-n",
            "find",
            f"/proc/{pid}/fd",
            "-mindepth",
            "1",
            "-maxdepth",
            "1",
            "-print",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    meminfo = Path("/proc/meminfo").read_text().splitlines()
    cma_free_kb = int(
        next(line for line in meminfo if line.startswith("CmaFree:")).split()[1]
    )
    return {
        "pid": pid,
        "fds": len(fd_result.stdout.splitlines()),
        "threads": len(list(Path(f"/proc/{pid}/task").iterdir())),
        "cma_free_kb": cma_free_kb,
    }


def _assert_resources_return_to_baseline(before: dict[str, int]) -> None:
    tolerance_kb = int(
        os.environ.get("SIMA_TEST_CMA_TOLERANCE_KB", CMA_TOLERANCE_KB)
    )
    deadline = time.monotonic() + 15
    after = _daemon_snapshot()
    while time.monotonic() < deadline:
        if (
            after["pid"] == before["pid"]
            and after["fds"] <= before["fds"]
            and after["threads"] <= before["threads"]
            and after["cma_free_kb"] >= before["cma_free_kb"] - tolerance_kb
        ):
            return
        time.sleep(1)
        after = _daemon_snapshot()

    assert after["pid"] == before["pid"], (before, after)
    assert after["fds"] <= before["fds"], (before, after)
    assert after["threads"] <= before["threads"], (before, after)
    assert after["cma_free_kb"] >= before["cma_free_kb"] - tolerance_kb, (
        before,
        after,
    )


def _run_cli(
    llima: str,
    model_name: str,
    tmp_path: Path,
    env: dict[str, str],
) -> tuple[str, int]:
    tmp_path.mkdir()
    process = subprocess.Popen(
        [llima, "run", model_name, "--mode", "cli"],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=tmp_path,
        env=env,
    )
    try:
        output, _ = process.communicate(input=f"{QUERY}\nquit\n", timeout=180)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        try:
            process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=10)
        raise

    assert process.returncode == 0, output
    assert not Path(f"/proc/{process.pid}").exists(), "llima CLI process leaked"
    return output, process.pid


@pytest.mark.skip(
    reason="Temporarily disabled while mlashmcomplex thread cleanup is investigated"
)
def test_installed_cli_start_query_quit(tmp_path):
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

    env = os.environ.copy()
    env["LLIMA_MODELS_PATH"] = str(models_path)
    daemon_initial = _daemon_snapshot()

    first_output, _ = _run_cli(llima, model_name, tmp_path / "first", env)
    steady_state = _daemon_snapshot()
    assert steady_state["pid"] == daemon_initial["pid"]
    assert steady_state["fds"] <= daemon_initial["fds"] + 1
    assert steady_state["threads"] <= daemon_initial["threads"]

    second_output, _ = _run_cli(llima, model_name, tmp_path / "second", env)
    _assert_resources_return_to_baseline(steady_state)

    for output in (first_output, second_output):
        assert "Setting up environments and loading models" in output
        assert f"Query: {QUERY}" in output
        assert "Assistant:" in output
        assert "capital of Germany is Berlin" in output
        assert ">>> quit" in output
