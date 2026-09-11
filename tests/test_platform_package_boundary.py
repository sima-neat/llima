"""Tests for LLiMa's direct platform runtime dependency boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.premerge


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_runtime_uses_platform_libraries_without_neat_internals() -> None:
    cmake = read("CMakeLists.txt")
    build = read("build.sh")
    installer = read("tools/install_llima.sh")
    manifest = json.loads(read("deps/manifest.json"))

    assert "find_package(NeatInternals" not in cmake
    assert "NeatInternals::" not in cmake
    assert "simaai-mlart-modalix, " in cmake
    assert "simaai-heap (>= 3.0~), " in cmake

    assert "NEAT_INTERNALS" not in build
    assert "neat-runtime" not in installer
    assert "neat-gst-plugins" not in installer
    assert "neat-ev74-firmware" not in installer
    assert "internals" not in manifest


def test_build_checks_direct_platform_development_files() -> None:
    build = read("build.sh")

    assert "simaai-mlart-modalix-dev:arm64" in build
    assert "simaai-heap-dev:arm64" in build
    assert "/usr/include/simaai/gst-api.h" in build
    assert "/usr/include/simaai/simaai_heap.h" in build
    assert "/usr/lib/aarch64-linux-gnu/libMLArt.so" in build
    assert "/usr/lib/aarch64-linux-gnu/libsimaai_heap.so" in build


def test_installer_keeps_modalix_3_platform_check() -> None:
    installer = read("tools/install_llima.sh")
    manifest = json.loads(read("deps/manifest.json"))
    workflow = read(".github/workflows/vulcan-ci.yml")

    assert manifest["platform-version"] == "3.0.0"
    assert manifest["sysroot-version"].endswith("-1297")
    assert '["sysroot-version"]' in workflow
    assert 'sysroot update "${sysroot_version}"' in workflow
    assert "MACHINE=modalix" in installer
    assert 'actual="$(read_devkit_platform_version' in installer
    assert 'if [[ "${actual}" != "${expected}" ]]' in installer
    assert "Refusing to install before modifying apt packages" in installer
