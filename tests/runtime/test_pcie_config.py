"""Unit tests for pcie_config: recv_root_from_pep_conf and claim_recv_root.

Pure-Python (no MLA / cpp_ext), so it runs on the host as well as the board.
Proves the board's default receive directory is derived from the pep-daemon
config exactly the way the daemon resolves it: take `default-recv`, then map
that name through the `[recv]` section to an absolute path. Section scoping
matters — a same-named key in `[serve]` must never be picked up.
"""
import os
import signal
import textwrap
import time

from sima_lmm.devkit.pcie_config import (
    RecvRootBusyError,
    checked_recv_root,
    claim_recv_root,
    recv_root_from_pep_conf,
)

# A trimmed copy of the shape /etc/simaai/simaai-pep-daemon.conf ships with:
# top-level keys before any section, comments, extra spaces, [recv]/[serve].
SAMPLE = textwrap.dedent(
    """\
    # SiMa.ai SoC PCIe endpoint management daemon.
    #socket = /run/simaai-pep-daemon.sock

    default-recv  = recv5g
    default-serve = data

    max-clients  = 64

    [recv]
    recv5g = /tmp/pcie-recv
    tmp  = /tmp
    data = /data

    [serve]
    data = /data
    logs = /var/log
    """
)


def _write(tmp_path, text):
    p = tmp_path / "simaai-pep-daemon.conf"
    p.write_text(text)
    return str(p)


def test_resolves_default_recv_through_recv_section(tmp_path):
    assert recv_root_from_pep_conf(_write(tmp_path, SAMPLE)) == "/tmp/pcie-recv"


def test_section_scoped_never_reads_serve(tmp_path):
    # default-recv names "data"; the [recv] "data" (/data) must win over the
    # identically named [serve] entry.
    text = SAMPLE.replace("default-recv  = recv5g", "default-recv  = data")
    assert recv_root_from_pep_conf(_write(tmp_path, text)) == "/data"


def test_missing_file_returns_none(tmp_path):
    assert recv_root_from_pep_conf(str(tmp_path / "does-not-exist.conf")) is None


def test_no_default_recv_returns_none(tmp_path):
    text = "\n".join(
        line for line in SAMPLE.splitlines() if not line.startswith("default-recv")
    )
    assert recv_root_from_pep_conf(_write(tmp_path, text)) is None


def test_default_recv_name_absent_from_recv_section_returns_none(tmp_path):
    text = SAMPLE.replace("recv5g = /tmp/pcie-recv\n", "")
    assert recv_root_from_pep_conf(_write(tmp_path, text)) is None


def test_checked_recv_root(tmp_path):
    conf = _write(tmp_path, SAMPLE)  # default-recv -> /tmp/pcie-recv
    assert checked_recv_root(None, conf) == "/tmp/pcie-recv"
    assert checked_recv_root("/tmp/pcie-recv", conf) == "/tmp/pcie-recv"
    assert checked_recv_root("/tmp/pcie-recv/", conf) == "/tmp/pcie-recv/"  # same folder
    try:
        checked_recv_root("/data/other", conf)
        raise AssertionError("a folder that is not default-recv must be refused")
    except ValueError as e:
        assert "not the pep daemon's default-recv folder /tmp/pcie-recv" in str(e), str(e)
    missing = str(tmp_path / "missing.conf")
    assert checked_recv_root("/data/any", missing) == "/data/any"  # no config: trust the flag
    try:
        checked_recv_root(None, missing)
        raise AssertionError("no flag and no config must be refused")
    except ValueError as e:
        assert "--pcie-recv-root" in str(e)


def test_recv_root_claim_is_exclusive_and_released(tmp_path):
    # Same lock file as pcie-genai-backend's card claim (recv-root.pid.lock).
    first = claim_recv_root(str(tmp_path))
    assert (tmp_path / "recv-root.pid.lock").exists()
    assert (tmp_path / "recv-root.pid").read_text().strip() == str(os.getpid())
    # flock is per open file, so a second claim in this process is refused too.
    try:
        claim_recv_root(str(tmp_path))
        raise AssertionError("a second claim must be refused while the first is held")
    except RecvRootBusyError as e:
        assert f"pid {os.getpid()}" in str(e), str(e)
    first.close()
    assert not (tmp_path / "recv-root.pid").exists()
    # The lock file stays (deleting it would let two holders lock two files).
    assert (tmp_path / "recv-root.pid.lock").exists()
    claim_recv_root(str(tmp_path)).close()


def test_recv_root_busy_message_skips_a_dead_pid(tmp_path):
    # The holder may not have replaced a dead holder's pid file yet: a dead
    # pid must not be named as the holder.
    first = claim_recv_root(str(tmp_path))
    child = os.fork()
    if child == 0:
        os._exit(0)
    os.waitpid(child, 0)
    (tmp_path / "recv-root.pid").write_text(f"{child}\n")
    try:
        claim_recv_root(str(tmp_path))
        raise AssertionError("the held lock must be refused")
    except RecvRootBusyError as e:
        assert "pid" not in str(e).split("locked by")[-1], str(e)
    first.close()


def test_recv_root_claim_is_refused_while_another_process_holds_it(tmp_path):
    # A child takes the lock and waits; the parent must be refused, then get it
    # once the child exits (the kernel drops the lock with the process).
    r, w = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(r)
        claim = claim_recv_root(str(tmp_path))
        os.write(w, b"x")
        time.sleep(30)
        claim.close()
        os._exit(0)
    os.close(w)
    assert os.read(r, 1) == b"x"
    try:
        try:
            claim_recv_root(str(tmp_path))
            raise AssertionError("a lock held by another process must be refused")
        except RecvRootBusyError as e:
            assert f"pid {child}" in str(e), str(e)
    finally:
        os.kill(child, signal.SIGKILL)
        os.waitpid(child, 0)
    claim_recv_root(str(tmp_path)).close()


if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
