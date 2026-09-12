"""Tests for LLiMa's direct platform runtime dependency boundary."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


class PlatformPackageBoundaryTest(unittest.TestCase):
    def test_runtime_uses_platform_libraries_without_neat_internals(self) -> None:
        cmake = read("CMakeLists.txt")
        build = read("build.sh")
        installer = read("tools/install_llima.sh")
        manifest = json.loads(read("deps/manifest.json"))

        self.assertNotIn("find_package(NeatInternals", cmake)
        self.assertNotIn("NeatInternals::", cmake)
        self.assertIn("simaai-mlart-modalix, ", cmake)
        self.assertIn("simaai-heap (>= 3.0~), ", cmake)
        self.assertNotIn("NEAT_INTERNALS", build)
        self.assertNotIn("neat-runtime", installer)
        self.assertNotIn("neat-gst-plugins", installer)
        self.assertNotIn("neat-ev74-firmware", installer)
        self.assertNotIn("internals", manifest)

    def test_build_checks_direct_platform_development_files(self) -> None:
        build = read("build.sh")

        self.assertIn("simaai-mlart-modalix-dev:arm64", build)
        self.assertIn("simaai-heap-dev:arm64", build)
        self.assertIn("/usr/include/simaai/gst-api.h", build)
        self.assertIn("/usr/include/simaai/simaai_heap.h", build)
        self.assertIn("/usr/lib/aarch64-linux-gnu/libMLArt.so", build)
        self.assertIn("/usr/lib/aarch64-linux-gnu/libsimaai_heap.so", build)

    def test_build_bootstrap_uses_modalix_3_library_abis(self) -> None:
        build = read("build.sh")

        for package in (
            "libopencv-flann410:arm64",
            "libopencv-dnn410:arm64",
            "libopencv-features2d410:arm64",
            "libopencv-objdetect410:arm64",
            "libopencv-video410:arm64",
            "libfmt10:arm64",
            "libspdlog1.15:arm64",
            "libcpp-httplib0.18:arm64",
        ):
            self.assertIn(package, build)

        for stale_abi in (
            "406:arm64",
            "libfmt9:arm64",
            "libspdlog1.10:arm64",
            "libcpp-httplib0.11:arm64",
        ):
            self.assertNotIn(stale_abi, build)

    def test_installer_keeps_modalix_3_platform_check(self) -> None:
        installer = read("tools/install_llima.sh")
        manifest = json.loads(read("deps/manifest.json"))
        workflow = read(".github/workflows/vulcan-ci.yml")

        self.assertEqual(manifest["platform-version"], "3.0.0")
        self.assertRegex(
            manifest["sysroot-version"],
            r"^3[.]0[.]0~git[0-9]{12}[.][0-9a-f]{7,40}-[0-9]+$",
        )
        self.assertIn('["sysroot-version"]', workflow)
        self.assertIn(
            'if [[ "${sdk_platform_version}" != "${sysroot_version}" ]]', workflow
        )
        self.assertIn(
            'setup-sdk-sysroot.sh "${sysroot_version}" "${SDK_PKG_LIST:-}"',
            workflow,
        )
        self.assertIn('"Platform Revision = ${sysroot_version}"', workflow)
        self.assertNotIn('sysroot update "${sysroot_version}"', workflow)
        self.assertIn("MACHINE=modalix", installer)
        self.assertIn('actual="$(read_devkit_platform_version', installer)
        self.assertIn('if [[ "${actual}" != "${expected}" ]]', installer)
        self.assertIn("Refusing to install before modifying apt packages", installer)
