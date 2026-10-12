"""Remote rsync destination checks shared by the deploy utilities."""

import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Exit codes of the remote pre-flight script. ssh itself exits with 255 when
# the connection or authentication fails.
_SSH_CONNECTION_FAILED = 255
_EXIT_MKDIR_FAILED = 3
_EXIT_NOT_WRITABLE = 4
_EXIT_REQUIRED_DIR_MISSING = 5


@dataclass(frozen=True)
class RemoteDestination:
    """An rsync remote-shell destination split into its host and path parts."""
    host: str
    path: str

    def child_path(self, name: str) -> str:
        """Return the remote path of a child directory."""
        return f"{self.path.rstrip('/')}/{name}" if self.path else name

    def join(self, name: str) -> str:
        """Return the rsync destination string for a child directory."""
        return f"{self.host}:{self.child_path(name)}"

    def __str__(self) -> str:
        return f"{self.host}:{self.path}"


def parse_remote_destination(dst: str) -> RemoteDestination | None:
    """
    Apply rsync's rule for remote-shell destinations: a ':' before the first '/'.
    Returns None for local paths and for rsync daemon destinations
    (host::module and rsync://), which do not use a remote shell.
    """
    colon = dst.find(":")
    if colon <= 0:
        return None
    slash = dst.find("/")
    if slash != -1 and slash < colon:
        return None
    if dst.startswith("rsync://") or dst[colon + 1:colon + 2] == ":":
        return None
    return RemoteDestination(dst[:colon], dst[colon + 1:])


def _remote_path_arg(path: str) -> str:
    """Quote a remote path for the remote shell while keeping '~' expansion."""
    if not path:
        return "."
    if path == "~":
        return '"$HOME"'
    if path.startswith("~/"):
        return '"$HOME"/' + shlex.quote(path[2:])
    return shlex.quote(path)


def local_size_kb(paths: list[Path]) -> int:
    """Total size of the regular files under paths, in KiB."""
    total = 0
    for path in paths:
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total // 1024


class RemoteSession:
    """
    One SSH connection to a remote rsync destination, used first for the
    pre-flight check and then by rsync.

    With the default ssh, the pre-flight opens a master connection that rsync
    reuses, so a password or host-key prompt happens only once, before any
    slow local work. A user-provided remote shell (--rsh or RSYNC_RSH) is used
    unchanged for both steps, without connection sharing.
    """

    def __init__(self, host: str, rsh: str | None = None):
        self.host = host
        self._control_dir = None
        if rsh:
            self._base = shlex.split(rsh)
            self._master_opts = []
        else:
            self._control_dir = tempfile.mkdtemp(prefix="llima-deploy-")
            self._base = ["ssh", "-o", f"ControlPath={self._control_dir}/%C"]
            self._master_opts = ["-o", "ControlMaster=auto", "-o", "ControlPersist=3600"]

    @property
    def rsync_rsh(self) -> str:
        """Value for rsync's -e option."""
        return shlex.join(self._base)

    def preflight(self, path: str, required_dir: str | None = None) -> int | None:
        """
        Create path on the remote host and check that it is writable.
        If required_dir is given, it must already exist on the remote host.
        Returns the free space at path in KiB, or None if it cannot be read.
        Raises RuntimeError with an actionable message on failure.
        """
        path_arg = _remote_path_arg(path)
        script = []
        if required_dir is not None:
            script.append(
                f"test -d {_remote_path_arg(required_dir)} || exit {_EXIT_REQUIRED_DIR_MISSING}"
            )
        script += [
            f"mkdir -p {path_arg} || exit {_EXIT_MKDIR_FAILED}",
            f"test -w {path_arg} || exit {_EXIT_NOT_WRITABLE}",
            f"df -Pk {path_arg} | awk 'NR == 2 {{ print $4 }}'",
        ]
        cmd = [*self._base, *self._master_opts, self.host, "\n".join(script)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        detail = result.stderr.strip().splitlines()
        detail = f" ({detail[-1]})" if detail else ""
        if result.returncode == _SSH_CONNECTION_FAILED:
            raise RuntimeError(
                f"Cannot reach {self.host}{detail}. Check the address, network, and SSH "
                "credentials; nothing was extracted or copied."
            )
        if result.returncode == _EXIT_REQUIRED_DIR_MISSING:
            raise RuntimeError(
                f"{required_dir} does not exist on {self.host}; nothing was copied."
            )
        if result.returncode == _EXIT_MKDIR_FAILED:
            raise RuntimeError(
                f"Cannot create destination {path} on {self.host}{detail}; "
                "nothing was extracted or copied."
            )
        if result.returncode == _EXIT_NOT_WRITABLE:
            raise RuntimeError(
                f"Destination {path} on {self.host} is not writable; is the storage mounted? "
                "Nothing was extracted or copied."
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Pre-flight check of {self.host}:{path} failed with exit code "
                f"{result.returncode}{detail}; nothing was extracted or copied."
            )

        free_kb = result.stdout.strip().splitlines()
        return int(free_kb[-1]) if free_kb and free_kb[-1].isdigit() else None

    def close(self) -> None:
        """Close the shared master connection, if one was opened."""
        if self._control_dir is None:
            return
        subprocess.run(
            [*self._base, "-O", "exit", self.host],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        shutil.rmtree(self._control_dir, ignore_errors=True)
        self._control_dir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def report_free_space(destination: str, free_kb: int | None, needed_kb: int) -> None:
    """Print the pre-flight result and warn when space looks insufficient."""
    if free_kb is None:
        print(f"Destination {destination} OK.")
        return
    print(f"Destination {destination} OK ({free_kb / 1024**2:.1f} GB free).")
    if free_kb < needed_kb:
        print(
            f"Warning: the deployment may need up to {needed_kb / 1024**2:.1f} GB, "
            f"but only {free_kb / 1024**2:.1f} GB is free at {destination}."
        )
