import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.premerge


def test_cmake_package_compatibility_uses_platform_version() -> None:
    manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert manifest["package-version"] != manifest["platform-version"]
    assert "print(data['platform-version'])" in cmake
    assert re.search(
        r"write_basic_package_version_file\(\s*"
        r'"[^"]*SimaLMMConfigVersion\.cmake"\s*'
        r"VERSION \$\{SIMA_LMM_PLATFORM_VERSION\}",
        cmake,
    )
