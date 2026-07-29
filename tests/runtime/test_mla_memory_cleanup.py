import os
import platform
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODELS_PATH = Path("/media/nvme/llima/models")
DEFAULT_TEXT_MODEL = "Qwen2.5-0.5B-Instruct-GPTQ-a16w4"
DEFAULT_REPEAT_COUNT = 4
DEFAULT_RUN_TIMEOUT_SECONDS = 180
DEFAULT_CLEANUP_TIMEOUT_SECONDS = 30
LEAK_BUFFER_SIZE = 0x20000000
LEAK_BUFFER_TARGET = 2

SUMMARY_PATTERN = re.compile(
    r"Total buffers allocated:\s+(\d+)\s+\|"
    r"\s+Total allocated size:\s+(0x[0-9a-fA-F]+)"
)
ALLOCATION_PATTERN = re.compile(
    r"^\|\s+(0x[0-9a-fA-F]+)"
    r"\s+\|\s+(0x[0-9a-fA-F]+)"
    r"\s+\|\s+(\d+)"
    r"\s+\|\s+(0x[0-9a-fA-F]+)"
    r"\s+\|\s+(\d+)"
    r"\s+\|\s+(\d+)\s+\|$"
)


@dataclass(frozen=True)
class MlaAllocation:
    physical_address: int
    parent_address: int
    reference_count: int
    size: int
    target: int
    owner_pid: int


@dataclass(frozen=True)
class MlaMemorySnapshot:
    total_buffers: int
    total_size: int
    allocations: tuple[MlaAllocation, ...]
    raw: str

    @property
    def root_allocations(self) -> tuple[MlaAllocation, ...]:
        return tuple(
            allocation
            for allocation in self.allocations
            if allocation.parent_address == 0
        )


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    assert value > 0, f"{name} must be greater than zero"
    return value


def _text_model_name() -> str:
    model_name = os.environ.get("SIMA_TEST_LLIMA_TEXT_MODEL", DEFAULT_TEXT_MODEL)
    assert model_name
    assert "/" not in model_name
    assert ".." not in model_name
    return model_name


def _daemon_pid() -> int:
    result = subprocess.run(
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
    )
    pid = int(result.stdout)
    assert pid > 0
    return pid


def _read_mla_memory() -> MlaMemorySnapshot:
    result = subprocess.run(
        ["sudo", "-n", "cat", "/dev/simaai-mem"],
        text=True,
        capture_output=True,
        check=True,
    )
    summary = SUMMARY_PATTERN.search(result.stdout)
    assert summary is not None, (
        "Could not parse the /dev/simaai-mem summary:\n" + result.stdout
    )

    allocations = []
    for line in result.stdout.splitlines():
        # The kernel device emits fixed-width records including their trailing
        # NUL terminator, so every record after the summary can start with \0.
        match = ALLOCATION_PATTERN.match(line.lstrip("\0"))
        if match is None:
            continue
        allocations.append(
            MlaAllocation(
                physical_address=int(match.group(1), 16),
                parent_address=int(match.group(2), 16),
                reference_count=int(match.group(3)),
                size=int(match.group(4), 16),
                target=int(match.group(5)),
                owner_pid=int(match.group(6)),
            )
        )

    snapshot = MlaMemorySnapshot(
        total_buffers=int(summary.group(1)),
        total_size=int(summary.group(2), 16),
        allocations=tuple(allocations),
        raw=result.stdout,
    )
    assert len(snapshot.root_allocations) == snapshot.total_buffers, snapshot.raw
    assert (
        sum(allocation.size for allocation in snapshot.root_allocations)
        == snapshot.total_size
    ), snapshot.raw
    return snapshot


def _write_snapshot(path: Path, snapshot: MlaMemorySnapshot) -> None:
    path.write_text(snapshot.raw, encoding="utf-8")


def _leak_buffers(snapshot: MlaMemorySnapshot) -> tuple[MlaAllocation, ...]:
    return tuple(
        allocation
        for allocation in snapshot.root_allocations
        if allocation.size == LEAK_BUFFER_SIZE
        and allocation.target == LEAK_BUFFER_TARGET
    )


def _wait_for_stable_baseline(
    daemon_pid: int,
    timeout_seconds: int,
) -> MlaMemorySnapshot:
    deadline = time.monotonic() + timeout_seconds
    previous = _read_mla_memory()

    while time.monotonic() < deadline:
        time.sleep(1)
        assert _daemon_pid() == daemon_pid, (
            "mlashmcomplex restarted while establishing the MLA memory baseline"
        )
        current = _read_mla_memory()
        if (
            current.total_buffers == previous.total_buffers
            and current.total_size == previous.total_size
        ):
            return current
        previous = current

    raise AssertionError(
        "MLA memory did not reach a stable baseline after restarting "
        f"simaai-appcomplex.service:\n{previous.raw}"
    )


def _wait_for_cleanup(
    baseline: MlaMemorySnapshot,
    daemon_pid: int,
    timeout_seconds: int,
    memory_log: Path,
) -> MlaMemorySnapshot:
    deadline = time.monotonic() + timeout_seconds
    current = _read_mla_memory()

    while time.monotonic() < deadline:
        assert _daemon_pid() == daemon_pid, (
            "mlashmcomplex restarted while waiting for MLA memory cleanup"
        )
        if (
            current.total_buffers == baseline.total_buffers
            and current.total_size == baseline.total_size
            and len(_leak_buffers(current)) <= len(_leak_buffers(baseline))
        ):
            _write_snapshot(memory_log, current)
            return current
        time.sleep(1)
        current = _read_mla_memory()

    _write_snapshot(memory_log, current)
    stale_buffers = _leak_buffers(current)
    raise AssertionError(
        "MLA memory did not return to baseline after llima exited.\n"
        f"baseline: buffers={baseline.total_buffers}, "
        f"size=0x{baseline.total_size:010x}\n"
        f"current:  buffers={current.total_buffers}, "
        f"size=0x{current.total_size:010x}\n"
        f"target-2 512 MiB buffers: {stale_buffers}\n"
        f"current /dev/simaai-mem:\n{current.raw}"
    )


def _run_cli_and_quit(
    llima: str,
    model_name: str,
    run_dir: Path,
    env: dict[str, str],
    timeout_seconds: int,
) -> str:
    run_dir.mkdir(parents=True)
    process = subprocess.Popen(
        [llima, "run", model_name, "--mode", "cli"],
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=run_dir,
        env=env,
    )
    output = ""
    try:
        output, _ = process.communicate(input="quit\n", timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        try:
            remaining, _ = process.communicate(timeout=30)
            output += remaining
        except subprocess.TimeoutExpired:
            process.kill()
            remaining, _ = process.communicate(timeout=10)
            output += remaining
        raise AssertionError(
            f"llima did not exit within {timeout_seconds} seconds:\n{output}"
        )
    finally:
        (run_dir / "llima.log").write_text(output, encoding="utf-8")

    assert process.returncode == 0, output
    assert not Path(f"/proc/{process.pid}").exists(), "llima CLI process leaked"
    assert "Setting up environments and loading models" in output
    assert ">>> quit" in output
    return output


def test_repeated_cli_quit_releases_mla_memory(tmp_path):
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

    repeat_count = _positive_int_from_env(
        "SIMA_TEST_LLIMA_MEMORY_REPEAT_COUNT", DEFAULT_REPEAT_COUNT
    )
    run_timeout = _positive_int_from_env(
        "SIMA_TEST_LLIMA_MEMORY_RUN_TIMEOUT_SECONDS",
        DEFAULT_RUN_TIMEOUT_SECONDS,
    )
    cleanup_timeout = _positive_int_from_env(
        "SIMA_TEST_LLIMA_MEMORY_CLEANUP_TIMEOUT_SECONDS",
        DEFAULT_CLEANUP_TIMEOUT_SECONDS,
    )
    configured_log_dir = os.environ.get("SIMA_TEST_LLIMA_MEMORY_LOG_DIR")
    log_dir = Path(configured_log_dir) if configured_log_dir else tmp_path
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["LLIMA_MODELS_PATH"] = str(models_path)
    daemon_pid = _daemon_pid()
    baseline = _wait_for_stable_baseline(daemon_pid, cleanup_timeout)
    _write_snapshot(log_dir / "baseline-memory.txt", baseline)

    for iteration in range(1, repeat_count + 1):
        run_dir = log_dir / f"iteration-{iteration:02d}"
        _run_cli_and_quit(
            llima,
            model_name,
            run_dir,
            env,
            run_timeout,
        )
        _wait_for_cleanup(
            baseline,
            daemon_pid,
            cleanup_timeout,
            run_dir / "memory-after-cleanup.txt",
        )

    final = _read_mla_memory()
    _write_snapshot(log_dir / "final-memory.txt", final)
    assert _daemon_pid() == daemon_pid
    assert final.total_buffers == baseline.total_buffers, final.raw
    assert final.total_size == baseline.total_size, final.raw
    assert len(_leak_buffers(final)) <= len(_leak_buffers(baseline)), final.raw
