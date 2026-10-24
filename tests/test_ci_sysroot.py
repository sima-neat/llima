"""Hermetic policy, exact inventory and provisioning refusal contracts."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/ci/platform_policy.py"
spec = importlib.util.spec_from_file_location("llima_platform_policy", SCRIPT)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
FIRMWARE = "3.0.0_master_B2372"
RECEIPT = "3.0.0~pre2372"


class PlatformPolicyContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def manifest(self, **changes):
        value = {"sdk": {"channel": "develop"}, "platform-version": "3.0.0", "elxr_version": FIRMWARE}
        value.update(changes)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value))
        return path

    def inventory(self, rows):
        path = self.root / "var/lib/sima-sdk/sysroot-packages.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rows)
        return path

    def test_current_declaration_requires_matching_derived_receipt(self):
        policy.validate_request(ROOT / "deps/manifest.json", RECEIPT)
        self.assertEqual(policy.declaration(ROOT / "deps/manifest.json"), FIRMWARE)

    def test_invalid_or_dual_declarations_are_refused(self):
        for changes in [
            {"sysroot-version": RECEIPT}, {"elxr_version": None},
            {"elxr_version": "3.0.0_master_B0"}, {"elxr_version": "3.0.0_master_B2372\n"},
            {"elxr_version": "3.0.0_release_B2372"}, {"platform-version": "2.1.3"},
            {"sdk": {"channel": "daily"}}, {"sdk": None},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy.declaration(self.manifest(**changes))

    def test_nonobject_manifest_is_refused(self):
        path = self.root / "manifest.json"
        for value in (None, [], "unexpected"):
            path.write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "JSON object"):
                policy.declaration(path)

    def test_legacy_only_declaration_is_refused(self):
        path = self.manifest(**{"sysroot-version": RECEIPT})
        data = json.loads(path.read_text())
        del data["elxr_version"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "legacy or dual"):
            policy.declaration(path)

    def test_mismatched_or_malformed_receipts_are_refused(self):
        for version in ["", "3.0.0~pre2263", "3.0.0~pre02372", "2.1.3~pre2372",
            "3.0.0~git202610140145.e5ece50-2372", "3.0.0~pre2372;touch /tmp/unwanted", "3.0.0~pre2372\n"]:
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "does not match"):
                policy.validate_request(self.manifest(), version)

    def test_develop_receipt_must_match_branch_base_and_build(self):
        path = self.manifest(elxr_version="3.0.0_develop_B2372")
        policy.validate_request(path, "3.0.0~git202610140145.e5ece50-2372")
        for version in (RECEIPT, "3.0.0~git202610140145.e5ece50-2371", "3.0.0~git202610140145.e5ece50-02372"):
            with self.subTest(version=version), self.assertRaises(ValueError):
                policy.validate_request(path, version)

    def test_narrow_bootstrap_patch_changes_only_retired_package(self):
        path = self.root / "setup-sdk-sysroot.sh"
        before = "prefix\n  a65apps-dev appcomplex-dev simaai-gst-plugins-dev simaai-heap-dev\nsuffix\n"
        path.write_text(before)
        policy.patch_setup(path)
        self.assertEqual(path.read_text(), before.replace("appcomplex-dev ", ""))
        with self.assertRaisesRegex(ValueError, "unreviewed compatibility patch"):
            policy.patch_setup(path)

    def test_drifted_sdk_package_list_is_not_patched(self):
        path = self.root / "setup-sdk-sysroot.sh"
        for count in (0, 2):
            before = "a65apps-dev appcomplex-dev simaai-gst-plugins-dev\n" * count
            path.write_text(before)
            with self.subTest(count=count), self.assertRaises(ValueError):
                policy.patch_setup(path)
            self.assertEqual(path.read_text(), before)

    def test_bootstrap_release_copy_preserves_image_identity(self):
        source, output = self.root / "sdk-release", self.root / "provisioning-release"
        before = ("Platform Base = 3.0.0\nPlatform Version = old-image-revision\n"
            "Platform Channel = daily\nPlatform Repository = https://debian.neat.sima.ai/daily\nOther = unchanged\n")
        source.write_text(before)
        policy.write_release(source, output)
        self.assertEqual(source.read_text(), before)
        for field in ("Platform Version = old-image-revision", "Platform Channel = pre-release",
            "Platform Repository = https://debian.neat.sima.ai/pre-release", "Other = unchanged"):
            self.assertIn(field, output.read_text())

    def test_unsupported_or_ambiguous_sdk_image_receipt_is_refused(self):
        source, output = self.root / "release", self.root / "output"
        for text in [
            "Platform Base = 2.1.3\nPlatform Channel = daily\nPlatform Repository = repo\n",
            "Platform Base = 3.0.0\nPlatform Channel = daily\n",
            "Platform Base = 3.0.0\nPlatform Base = 3.0.0\nPlatform Channel = daily\nPlatform Repository = repo\n",
            "Platform Base = 3.0.0\nPlatform Channel = daily\nPlatform Channel = daily\nPlatform Repository = repo\n",
        ]:
            source.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                policy.write_release(source, output)
            self.assertFalse(output.exists())

    def test_inventory_requires_one_exact_platform_package(self):
        for architecture in ("arm64", "all"):
            self.inventory(f"simaai-palette-modalix\t{architecture}\t{RECEIPT}\t/usr/share\n")
            policy.verify_inventory(self.root, RECEIPT)

    def test_incomplete_wrong_or_duplicate_platform_inventory_is_refused(self):
        for rows in ["", "simaai-palette-modalix\tarm64\t3.0.0~pre2263\t/usr/share\n",
            f"simaai-palette-modalix\tamd64\t{RECEIPT}\t/usr/share\n",
            f"simaai-palette-modalix\tarm64\t{RECEIPT}\n",
            f"simaai-palette-modalix\tarm64\t{RECEIPT}\t/usr/share\n" * 2]:
            self.inventory(rows)
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "Unexpected platform inventory"):
                policy.verify_inventory(self.root, RECEIPT)

    def test_actual_firmware_can_come_from_either_metadata_file(self):
        paths = self.root / "build", self.root / "buildinfo"
        paths[1].write_text(f'SIMA_BUILD_VERSION="{FIRMWARE}" # actual image\n')
        policy.verify_firmware(FIRMWARE, paths)
        paths[0].write_text(f"SIMA_BUILD_VERSION={FIRMWARE}\n")
        policy.verify_firmware(FIRMWARE, paths)

    def test_missing_or_mismatched_actual_firmware_refuses_execution(self):
        path = self.root / "build"
        for actual in ("3.0.0_master_B2263", "3.0.0_develop_B2372", "unknown", ""):
            path.write_text(f"SIMA_BUILD_VERSION={actual}\n")
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                policy.verify_firmware(FIRMWARE, (path,))
        path.unlink()
        with self.assertRaises(ValueError):
            policy.verify_firmware(FIRMWARE, (path,))

    def test_conflicting_firmware_metadata_is_refused(self):
        paths = self.root / "build", self.root / "buildinfo"
        paths[0].write_text(f"SIMA_BUILD_VERSION={FIRMWARE}\n")
        paths[1].write_text("SIMA_BUILD_VERSION=3.0.0_develop_B2372\n")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            policy.verify_firmware(FIRMWARE, paths)

    def test_daily_provisioning_forwards_exact_receipt_and_propagates_failure(self):
        receipt = "3.0.0~git202610140145.e5ece50-2372"
        path = self.manifest(elxr_version="3.0.0_develop_B2372")
        self.inventory(f"simaai-palette-modalix\tarm64\t{receipt}\t/usr/share\n")
        invocation = self.root / "invocation"
        updater = self.root / "sysroot"
        for exit_code in (0, 17):
            invocation.unlink(missing_ok=True)
            updater.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$INVOCATION"\n'
                               f'if [ "$1" = update ]; then exit {exit_code}; fi\n')
            updater.chmod(0o755)
            result = subprocess.run(["bash", str(ROOT / "scripts/ci/prepare_sysroot.sh"), receipt, str(path)],
                env={**os.environ, "PATH": f"{self.root}:{os.environ['PATH']}", "INVOCATION": str(invocation),
                     "SYSROOT": str(self.root), "LLIMA_DISPOSABLE_CI": "1"}, capture_output=True, text=True)
            self.assertEqual(result.returncode, exit_code, result.stderr)
            self.assertEqual(invocation.read_text().splitlines(), ["update", receipt] + ([] if exit_code else ["status"]))

    def test_provisioning_requires_explicit_disposable_ci_admission(self):
        invocation = self.root / "invocation"
        updater = self.root / "sysroot"
        updater.write_text('#!/bin/sh\ntouch "$INVOCATION"\n')
        updater.chmod(0o755)
        result = subprocess.run(["bash", str(ROOT / "scripts/ci/prepare_sysroot.sh"), RECEIPT],
            env={**os.environ, "PATH": f"{self.root}:{os.environ['PATH']}", "INVOCATION": str(invocation),
                 "LLIMA_DISPOSABLE_CI": ""}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("disposable CI", result.stderr)
        self.assertFalse(invocation.exists())

    def test_mismatched_request_refuses_before_any_provisioner(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/ci/prepare_sysroot.sh"), "3.0.0~pre2263"],
            env={**os.environ, "LLIMA_DISPOSABLE_CI": "1"}, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match firmware declaration", result.stderr)

    def test_policy_cli_refuses_missing_resolution(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "request"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--version", result.stderr)


if __name__ == "__main__":
    unittest.main()
