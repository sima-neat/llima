"""Unit tests for pcie_config.recv_root_from_pep_conf.

Pure-Python (no MLA / cpp_ext), so it runs on the host as well as the board.
Proves the board's default receive directory is derived from the pep-daemon
config exactly the way the daemon resolves it: take `default-recv`, then map
that name through the `[recv]` section to an absolute path. Section scoping
matters — a same-named key in `[serve]` must never be picked up.
"""
import textwrap

from sima_lmm.devkit.pcie_config import recv_root_from_pep_conf

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
