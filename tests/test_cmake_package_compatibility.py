import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.premerge


def test_cmake_package_reports_llima_release_version() -> None:
    manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert manifest["package-version"] != manifest["platform-version"]
    assert "print(data['package-version'])" in cmake
    assert "print(data['platform-version'])" not in cmake
    assert re.search(
        r"write_basic_package_version_file\(\s*"
        r'"[^"]*SimaLMMConfigVersion\.cmake"\s*'
        r"VERSION \$\{PROJECT_VERSION\}",
        cmake,
    )
