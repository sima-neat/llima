from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def shell_function(name: str) -> str:
    text = (ROOT / "build.sh").read_text(encoding="utf-8")
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 2]


def run_sync(
    artifact: dict[str, str] | str | None,
    consumer_base: str = "2.1.3",
    enabled: str = "ON",
    update_status: int = 0,
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
        log = root / "sysroot.log"
        script = f"""
set -e
id() {{ echo 0; }}
sysroot() {{
  printf '%s\n' "$*" >> {shlex.quote(str(log))}
  [[ "$1" != update ]] || return {update_status}
}}
{shell_function("run_as_root")}
{shell_function("sync_sysroot_from_internals_manifest")}
ELXR_SDK=ON
NEAT_SYNC_SYSROOT={shlex.quote(enabled)}
NEAT_INTERNALS_MANIFEST={shlex.quote(str(consumer))}
sync_sysroot_from_internals_manifest {shlex.quote(str(artifact_dir))}
"""
        result = subprocess.run(
            ["bash", "-c", script], check=False, text=True, capture_output=True
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls


class InternalsSysrootSyncTest(unittest.TestCase):
    def test_syncs_the_exact_receipt(self) -> None:
        base = "2.1.3"
        receipt = f"{base}~pre9999"
        result, calls = run_sync({"sysroot-version": receipt}, base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [f"update {receipt}", "status"])

    def test_sync_is_off_by_default(self) -> None:
        result, calls = run_sync(None, enabled="OFF")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_rejects_invalid_artifact_receipts(self) -> None:
        base = "2.1.3"
        receipt = f"{base}~pre9999"
        cases = (
            (None, base, "missing internals-manifest.json"),
            ("{", base, "Cannot read Internals build receipt"),
            ({"sysroot-version": "latest"}, base, "invalid platform receipt"),
            ({"sysroot-version": receipt}, "2.1.4", "but LLiMa declares"),
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


if __name__ == "__main__":
    unittest.main()
