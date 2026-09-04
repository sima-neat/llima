"""Tests for the LLiMa-to-Internals package boundary."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:
    pytestmark = ()
else:
    pytestmark = pytest.mark.premerge

ROOT = Path(__file__).resolve().parents[1]


def cmake() -> str:
    return (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def build_script() -> str:
    return (ROOT / "build.sh").read_text(encoding="utf-8")


def shell_function(name: str) -> str:
    text = build_script()
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 2]


def run_sync(
    artifact: dict[str, str] | str | None,
    consumer_base: str = "2.1.3",
    enabled: str = "ON",
    update_status: int = 0,
    sdk_platform_version: str | None = "2.1.3",
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_dir = root / "artifact"
        artifact_dir.mkdir()
        if artifact is not None:
            contents = json.dumps(artifact) if isinstance(artifact, dict) else artifact
            (artifact_dir / "internals-manifest.json").write_text(
                contents, encoding="utf-8"
            )
        consumer = root / "manifest.json"
        consumer.write_text(
            json.dumps({"platform-version": consumer_base}), encoding="utf-8"
        )
        sdk_release = root / "sdk-release"
        if sdk_platform_version is not None:
            sdk_release.write_text(
                f"Platform Version = {sdk_platform_version}\n", encoding="utf-8"
            )
        log = root / "sysroot.log"
        script = f"""
set -euo pipefail
id() {{ echo 0; }}
sysroot() {{
  printf '%s\n' "$*" >> {shlex.quote(str(log))}
  [[ "$1" != update ]] || return {update_status}
}}
{shell_function("run_as_root")}
{shell_function("sync_sysroot_from_internals_manifest")}
ELXR_SDK=ON
ELXR_SDK_RELEASE_FILE={shlex.quote(str(sdk_release))}
NEAT_SYNC_SYSROOT={shlex.quote(enabled)}
NEAT_INTERNALS_MANIFEST={shlex.quote(str(consumer))}
sync_sysroot_from_internals_manifest {shlex.quote(str(artifact_dir))}
"""
        result = subprocess.run(
            ["bash", "-c", script], check=False, text=True, capture_output=True
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls


def run_targeted_sysroot_payload_install() -> tuple[
    subprocess.CompletedProcess[str], int, int, int, list[str]
]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        sysroot = root / "sysroot"
        untouched = sysroot / "existing" / "private.dat"
        untouched.parent.mkdir(parents=True)
        untouched.write_text("existing\n", encoding="utf-8")
        untouched.chmod(0o600)
        existing_libdir = sysroot / "usr" / "lib"
        existing_libdir.mkdir(parents=True)
        existing_libdir.chmod(0o710)
        root_log = root / "root.log"

        script = f"""
set -euo pipefail
apt-get() {{
  [[ "$1" == "download" ]]
  shift
  local package
  for package in "$@"; do
    touch "${{package%%:*}}_1.0_arm64.deb"
  done
}}
dpkg-deb() {{
  [[ "$1" == "-x" ]]
  local deb="$2"
  local payload_root="$3"
  local name
  name="$(basename "${{deb}}" .deb)"
  mkdir -p "${{payload_root}}/usr/lib/aarch64-linux-gnu"
  printf 'payload\n' > "${{payload_root}}/usr/lib/aarch64-linux-gnu/${{name}}.so"
  chmod 600 "${{payload_root}}/usr/lib/aarch64-linux-gnu/${{name}}.so"
}}
run_as_root() {{
  printf '%s\n' "$1" >> {shlex.quote(str(root_log))}
  "$@"
}}
{shell_function("install_sdk_sysroot_package_payloads")}
install_sdk_sysroot_package_payloads {shlex.quote(str(sysroot))} libexample:arm64
"""
        result = subprocess.run(
            ["bash", "-c", script], check=False, text=True, capture_output=True
        )
        installed = sysroot / "usr/lib/aarch64-linux-gnu/libexample_1.0_arm64.so"
        untouched_mode = untouched.stat().st_mode & 0o777
        existing_libdir_mode = existing_libdir.stat().st_mode & 0o777
        installed_mode = installed.stat().st_mode & 0o777 if installed.exists() else 0
        root_calls = root_log.read_text(encoding="utf-8").splitlines()
        return (
            result,
            untouched_mode,
            existing_libdir_mode,
            installed_mode,
            root_calls,
        )


class InternalsSysrootSyncTest(unittest.TestCase):
    def test_syncs_the_exact_receipt(self) -> None:
        base = "2.1.3"
        receipt = f"{base}~pre9999"
        result, calls = run_sync({"sysroot-version": receipt}, base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [f"update {receipt}", "status"])

    def test_uses_a_matching_stable_sdk_sysroot(self) -> None:
        base = "2.1.3"
        result, calls = run_sync({"sysroot-version": base}, base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "Using stable SDK sysroot 2.1.3 without updating it.", result.stdout
        )
        self.assertEqual(calls, [])

    def test_rejects_a_stable_receipt_without_a_matching_sdk(self) -> None:
        base = "2.1.3"
        for sdk_platform_version, actual in (("2.1.2", "2.1.2"), (None, "unknown")):
            with self.subTest(sdk_platform_version=sdk_platform_version):
                result, calls = run_sync(
                    {"sysroot-version": base},
                    base,
                    sdk_platform_version=sdk_platform_version,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"SDK platform {actual} does not match required stable platform {base}",
                    result.stderr,
                )
                self.assertEqual(calls, [])

    def test_sync_is_off_by_default(self) -> None:
        result, calls = run_sync(None, enabled="OFF")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_empty_receipt_keeps_the_existing_sysroot(self) -> None:
        result, calls = run_sync({"sysroot-version": ""})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_rejects_invalid_artifact_receipts(self) -> None:
        base = "2.1.3"
        receipt = f"{base}~pre9999"
        cases = (
            (None, base, "missing internals-manifest.json"),
            ("{", base, "Cannot read Internals build receipt"),
            ({}, base, "Cannot read Internals build receipt"),
            (
                {"sysroot-version": "latest"},
                base,
                "Cannot read Internals build receipt",
            ),
            (
                {"sysroot-version": "|latest"},
                base,
                "Cannot read Internals build receipt",
            ),
            (
                {"sysroot-version": "\nmalformed"},
                base,
                "Cannot read Internals build receipt",
            ),
            (
                {"sysroot-version": receipt},
                "2.1.4",
                "Cannot read Internals build receipt",
            ),
        )
        for artifact, consumer_base, message in cases:
            with self.subTest(message=message):
                result, calls = run_sync(artifact, consumer_base)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(calls, [])

    def test_failed_update_stops_the_sync(self) -> None:
        base = "2.1.3"
        result, calls = run_sync(
            {"sysroot-version": f"{base}~pre9999"}, base, update_status=23
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, [f"update {base}~pre9999"])


class SdkSysrootPackageInstallTest(unittest.TestCase):
    def test_normalizes_only_the_new_package_payload(self) -> None:
        result, untouched_mode, libdir_mode, installed_mode, root_calls = (
            run_targeted_sysroot_payload_install()
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(untouched_mode, 0o600)
        self.assertEqual(libdir_mode, 0o710)
        self.assertEqual(installed_mode, 0o644)
        self.assertEqual(root_calls, ["cp"])


def test_internals_is_located_without_a_derived_version() -> None:
    assert "find_package(NeatInternals CONFIG REQUIRED)" in cmake()


def test_no_manually_constructed_internals_version_ranges() -> None:
    text = cmake()
    for removed in (
        "SIMA_LMM_INTERNALS_PACKAGE_MIN_VERSION",
        "SIMA_LMM_INTERNALS_PACKAGE_MAX_VERSION",
        "SIMA_LMM_MEMORY_LIB_VERSION",
        "SIMA_LMM_MEMORY_LIB_DEPENDENCY",
    ):
        assert removed not in text, removed

    assert "neat-runtime, " in text
    assert "neat-internals-dev, " in text
    assert "simaai-memory-lib, " in text
    assert "simaai-memory-lib (= " not in text
    assert "neat-runtime (>=" not in text
    assert "neat-runtime (<<" not in text
    assert "neat-internals-dev (>=" not in text
    assert "neat-internals-dev (<<" not in text


def test_memory_is_not_looked_up_in_the_internals_artifact() -> None:
    text = build_script()
    assert "resolve_neat_internals_memory_version" not in text
    assert "SIMA_LMM_MEMORY_LIB_VERSION" not in text
    assert "simaai-memory-lib" not in text


def test_every_delivered_internals_package_is_installed_and_forwarded() -> None:
    text = build_script()
    assert "deb_pattern_groups" not in text
    assert "internals_patterns" not in text
    assert (
        'mapfile -t debs < <(find "${NEAT_INTERNALS_DEB_DIR}" -maxdepth 1 '
        "-type f -name '*.deb' | sort)" in text
    )
    assert (
        'mapfile -t internals_debs < <(find "${NEAT_INTERNALS_DEB_DIR}" '
        "-maxdepth 1 -type f -name '*.deb' | sort)" in text
    )


def test_selected_internals_payload_is_validated_before_sysroot_overlay() -> None:
    text = build_script()
    validate = text.index(
        'validate_neat_internals_payload "${payload_root}" "${archive_name}"'
    )
    overlay = text.index('run_as_root dpkg-deb -x "${deb}" "${sysroot}"', validate)

    assert 'payload_root="$(mktemp -d ' in text
    assert "NeatInternals/NeatInternalsConfig.cmake" in text
    assert "NeatInternals/NeatInternalsTargets.cmake" in text
    assert validate < overlay


def test_vulcan_build_uses_the_internals_sysroot_receipt() -> None:
    text = build_script()
    workflow = (ROOT / ".github/workflows/vulcan-ci.yml").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
    sync = text.index('sync_sysroot_from_internals_manifest "${extract_dir}"')
    extract = text.index('dpkg-deb -x "${deb}" "${payload_root}"', sync)

    assert '[[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]] || return 0' in text
    assert '-e NEAT_SYNC_SYSROOT="ON"' in workflow
    assert "internals-manifest.json" in text
    assert '(?:~pre[0-9]+)?' in text
    assert 'sysroot update "${receipt}"' in text
    assert "Using stable SDK sysroot" in text
    assert "Internals artifact is missing internals-manifest.json" in text
    assert "invalid sysroot-version" in text
    assert "platform-version does not match the Internals receipt" in text
    assert "sysroot-version" not in manifest
    assert not re.search(r"\b[0-9]+(?:\.[0-9]+){2}~pre[0-9]+\b", text)
    assert not re.search(r"\b[0-9]+(?:\.[0-9]+){2}~pre[0-9]+\b", workflow)
    assert sync < extract


def test_llima_package_fallback_does_not_run_the_sdk_overlay() -> None:
    text = build_script()
    package_start = text.index("ensure_sdk_sysroot_packages() {")
    package_end = text.index("\n}\n", package_start)
    package_function = text[package_start:package_end]

    assert "https://github.com/sima-neat/sdk/issues/164" in package_function
    assert "install-sysroot-overlay.sh" not in package_function
    assert 'install_sdk_sysroot_package_payloads "${sysroot}"' in package_function
    assert 'chmod -R a+rX "${payload_root}"' in text
    assert 'chmod -R a+rX "${sysroot}"' not in text
    assert 'run_as_root cp -R "${payload_root}/." "${sysroot}/"' in text
    assert 'cp -a "${payload_root}/."' not in text


def test_llima_has_no_recovery_or_libcamera_branch() -> None:
    text = build_script()
    assert "recovery" not in text.lower()
    assert "libcamera" not in text.lower()


def test_manifest_still_selects_only_environment_and_release() -> None:
    manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
    assert "platform-package-version" not in manifest


def test_ci_has_no_bundled_memory_policy() -> None:
    workflow = (ROOT / ".github/workflows/vulcan-ci.yml").read_text(encoding="utf-8")
    assert "memory_deb" not in workflow
    assert "bundled memory" not in workflow.lower()


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, pattern
    functions = [
        value
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    ]
    tests.addTests(unittest.FunctionTestCase(function) for function in functions)
    return tests
