import subprocess
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restart_appcomplex_before_runtime_test():
    restart_script = Path(__file__).with_name("restart_appcomplex_before_test.sh")
    subprocess.run(["bash", str(restart_script), "true"], check=True)
