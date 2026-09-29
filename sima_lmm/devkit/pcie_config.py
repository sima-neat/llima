"""Read PCIe transfer settings from the pep-daemon config.

Kept free of MLA / cpp_ext imports so it can be unit-tested on any host. The
sole entry point resolves the board's default receive directory the same way
the daemon does, so `llima run --pcie` can default `--pcie-recv-root` instead
of making the caller repeat a value that the daemon already owns.
"""
from pathlib import Path
from typing import Optional

PEP_DAEMON_CONF = "/etc/simaai/simaai-pep-daemon.conf"


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
