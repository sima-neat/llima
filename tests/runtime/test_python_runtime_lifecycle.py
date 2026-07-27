import logging
import platform
import subprocess
from pathlib import Path


def test_installed_python_binding_dispatcher_lifecycle(tmp_path, monkeypatch):
    assert platform.machine() == "aarch64", "runtime tests require an ARM64 DevKit"
    subprocess.run(
        ["systemctl", "is-active", "--quiet", "simaai-appcomplex.service"],
        check=True,
    )

    from sima_lmm.devkit import cpp_ext
    from sima_lmm.devkit.utils import connect, disconnect

    extension_path = Path(cpp_ext.__file__).resolve()
    assert extension_path.is_file()
    assert extension_path.is_relative_to("/usr/lib/python3/dist-packages/sima_lmm")

    monkeypatch.chdir(tmp_path)
    connected = False
    try:
        connect(logging.INFO)
        connected = True
    finally:
        if connected:
            disconnect()

    assert (tmp_path / "run.cpp.log").is_file()
