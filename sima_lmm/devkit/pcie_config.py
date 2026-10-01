"""Read PCIe transfer settings from the pep-daemon config.

Kept free of MLA / cpp_ext imports so it can be unit-tested on any host.
recv_root_from_pep_conf() resolves the board's default receive directory the
same way the daemon does, so `llima run --pcie` can default `--pcie-recv-root`
instead of making the caller repeat a value that the daemon already owns.
claim_recv_root() takes the card-wide lock on that directory.
"""
import fcntl
import os
from pathlib import Path
from typing import Optional

PEP_DAEMON_CONF = "/etc/simaai/simaai-pep-daemon.conf"

def checked_recv_root(given: Optional[str], conf_path: str = PEP_DAEMON_CONF) -> str:
    """The receive directory for `llima run --pcie`, or raise ValueError.

    given = --pcie-recv-root. The daemon writes every pulled file under its
    default-recv folder, so a given folder must be that same folder (compared
    after resolving symlinks); otherwise the first model read would fail with
    an unclear "file not found". With no given folder, default-recv is used.
    """
    from_conf = recv_root_from_pep_conf(conf_path)
    if given is None:
        if from_conf is None:
            raise ValueError(
                "--pcie could not determine the receive directory. Pass "
                "--pcie-recv-root, or set 'default-recv' (and its [recv] path) "
                f"in {conf_path}."
            )
        return from_conf
    if from_conf is not None and os.path.realpath(given) != os.path.realpath(from_conf):
        raise ValueError(
            f"--pcie-recv-root {given} is not the pep daemon's default-recv folder "
            f"{from_conf}; the daemon writes the pulled files there. Drop the flag, "
            "or pass that folder."
        )
    return given


# The card-wide claim on the pep recv root. pcie-genai-backend takes the same
# lock (QueueOwnership on <run dir>/recv-root.pid, lifecycle.cpp), so only one
# PCIe model user runs per card: they all stage the same file names
# (devkit/vlm_config.json, elf_files/...) in the same recv root.
PCIE_RUN_DIR_ENV = "SIMA_NEAT_PCIE_RUN_DIR"
PCIE_RUN_DIR = "/run/sima-neat/pcie"
RECV_ROOT_PID = "recv-root.pid"


class RecvRootBusyError(RuntimeError):
    """Another PCIe model user on this card holds the recv-root lock."""


class RecvRootClaim:
    """Holds the recv-root lock until close() or process exit.

    The lock is an flock() on "<run dir>/recv-root.pid.lock"; the kernel drops
    it when the process dies, so a crash leaves nothing stale. The pid file
    next to it only says who holds the lock, for the error message.
    """

    def __init__(self, fd: int, pid_path: Path):
        self._fd = fd
        self._pid_path = pid_path

    def close(self) -> None:
        if self._fd < 0:
            return
        try:
            if self._pid_path.read_text().strip() == str(os.getpid()):
                self._pid_path.unlink()
        except (OSError, ValueError):
            pass
        os.close(self._fd)
        self._fd = -1


def claim_recv_root(run_dir: Optional[str] = None) -> RecvRootClaim:
    """Take the card-wide recv-root lock, or raise RecvRootBusyError.

    Raises OSError if the run dir cannot be made or the lock file cannot be
    opened.
    """
    run = Path(run_dir or os.environ.get(PCIE_RUN_DIR_ENV) or PCIE_RUN_DIR)
    run.mkdir(parents=True, exist_ok=True)
    pid_path = run / RECV_ROOT_PID
    lock_path = run / (RECV_ROOT_PID + ".lock")
    # O_RDONLY is enough for flock, and works on a lock file another user made.
    fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        # Named only if alive: a new holder may not have replaced a dead
        # holder's pid file yet.
        holder = ""
        try:
            pid = int(pid_path.read_text().strip())
            if pid <= 0:
                raise ValueError(pid)  # kill(0 or -1) would hit a whole group
            try:
                os.kill(pid, 0)
            except PermissionError:
                pass  # alive, another user's process
            holder = f" (pid {pid})"
        except (OSError, ValueError):
            pass
        raise RecvRootBusyError(f"{lock_path} is locked by another process{holder}") from None
    except BaseException:
        os.close(fd)
        raise
    # Who holds the lock, for the busy message. Best effort: the lock is the claim.
    try:
        pid_path.unlink(missing_ok=True)
        pid_path.write_text(f"{os.getpid()}\n")
    except OSError:
        pass
    return RecvRootClaim(fd, pid_path)


def recv_root_from_pep_conf(conf_path: str = PEP_DAEMON_CONF) -> Optional[str]:
    """Resolve the board's default receive directory from the pep-daemon config.

    The daemon writes an inbound transfer that names no root into `default-recv`,
    whose name is mapped to an absolute path by the `[recv]` section — e.g.
    `default-recv = recv5g` with `[recv] recv5g = /tmp/pcie-recv` resolves to
    "/tmp/pcie-recv". That resolved path is what `--pcie-recv-root` must equal, so
    the daemon and llima agree on where a pulled file lands.

    The config puts `default-recv` at top level (before any section), so a plain
    INI parser will not do; this walks it by hand. Returns None when the file is
    absent/unreadable or the mapping cannot be resolved, leaving the caller to
    fall back to an explicit flag with a clear error.
    """
    try:
        lines = Path(conf_path).read_text().splitlines()
    except OSError:
        return None

    default_recv: Optional[str] = None
    recv_roots: dict[str, str] = {}
    section: Optional[str] = None
    for raw in lines:
        # Comments are whole-line ('#...'); paths never contain '#'.
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if section is None and key == "default-recv":
            default_recv = value
        elif section == "recv":
            recv_roots[key] = value

    if default_recv is None:
        return None
    return recv_roots.get(default_recv)
