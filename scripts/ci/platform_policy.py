#!/usr/bin/env python3
"""Fail-closed firmware declaration and derived sysroot checks for LLiMa CI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex

ROOT = Path(__file__).resolve().parents[2]
FIRMWARE = re.compile(r"([0-9]+(?:\.[0-9]+){2})_(master|develop)_B([1-9][0-9]*)")
DAILY = re.compile(r"([0-9]+(?:\.[0-9]+){2})~git[0-9]{12}\.[0-9a-f]{7,40}-([1-9][0-9]*)")


def declaration(manifest: Path) -> str:
    data = json.loads(manifest.read_text())
    if not isinstance(data, dict):
        raise ValueError("Platform manifest must be a JSON object")
    if "sysroot-version" in data:
        raise ValueError("Use elxr_version only; legacy or dual sysroot-version declarations are not accepted")
    version = data.get("elxr_version")
    match = FIRMWARE.fullmatch(version) if isinstance(version, str) else None
    if not match or match[1] != data.get("platform-version"):
        raise ValueError("elxr_version must match platform-version and use X.Y.Z_(master|develop)_BN")
    sdk = data.get("sdk")
    if not isinstance(sdk, dict) or sdk.get("channel") not in {"develop", "official"}:
        raise ValueError("Unsupported SDK toolchain channel")
    return version


def validate_request(manifest: Path, version: str) -> None:
    firmware = declaration(manifest)
    base, channel, build = FIRMWARE.fullmatch(firmware).groups()
    if channel == "master":
        valid = version == f"{base}~pre{build}"
    else:
        receipt = DAILY.fullmatch(version)
        valid = receipt is not None and receipt[1] == base and receipt[2] == build
    if not valid:
        raise ValueError(f"Resolved sysroot {version!r} does not match firmware declaration {firmware}")


def patch_setup(path: Path) -> None:
    text = path.read_text()
    old = "a65apps-dev appcomplex-dev simaai-gst-plugins-dev"
    if text.count(old) != 1:
        raise ValueError("Pinned SDK package list changed; refusing an unreviewed compatibility patch")
    path.write_text(text.replace(old, "a65apps-dev simaai-gst-plugins-dev"))


def write_release(source: Path, output: Path) -> None:
    text = source.read_text()
    if re.findall(r"^Platform Base\s*=\s*(\S+)\s*$", text, re.M) != ["3.0.0"]:
        raise ValueError("Master provisioning requires exactly one Agate 3.0 SDK platform base")
    for field, value in {
        "Platform Channel": "pre-release",
        "Platform Repository": "https://debian.neat.sima.ai/pre-release",
    }.items():
        text, count = re.subn(rf"^{re.escape(field)}\s*=.*$", f"{field} = {value}", text, flags=re.M)
        if count != 1:
            raise ValueError(f"Expected exactly one {field} in SDK receipt")
    output.write_text(text)


def verify_inventory(root: Path, version: str) -> None:
    inventory = root / "var/lib/sima-sdk/sysroot-packages.tsv"
    rows = [line.split("\t") for line in inventory.read_text().splitlines()]
    platform = [row for row in rows if row and row[0] == "simaai-palette-modalix"]
    if (len(platform) != 1 or len(platform[0]) != 4
            or platform[0][1] not in {"arm64", "all"} or platform[0][2] != version):
        raise ValueError(f"Unexpected platform inventory for {version}: {platform}")


def verify_firmware(expected: str, paths: tuple[Path, ...]) -> None:
    if not FIRMWARE.fullmatch(expected):
        raise ValueError("Missing or invalid expected DevKit firmware")
    versions = []
    for path in paths:
        if path.is_file():
            for raw in re.findall(r"^\s*SIMA_BUILD_VERSION\s*=\s*([^\n]+)$", path.read_text(), re.M):
                fields = shlex.split(raw, comments=True)
                if len(fields) != 1 or not FIRMWARE.fullmatch(fields[0]):
                    raise ValueError(f"Invalid firmware identity in {path}")
                versions.append(fields[0])
    if not versions or any(actual != expected for actual in versions):
        raise ValueError(f"DevKit firmware mismatch: expected {expected}; observed {versions or '<missing>'}")
    print(f"Verified DevKit firmware: {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    request = sub.add_parser("request")
    request.add_argument("--manifest", type=Path, default=ROOT / "deps/manifest.json")
    request.add_argument("--version", required=True)
    patch = sub.add_parser("patch-setup")
    patch.add_argument("path", type=Path)
    release = sub.add_parser("release")
    release.add_argument("source", type=Path)
    release.add_argument("output", type=Path)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--sysroot", type=Path, required=True)
    inventory.add_argument("--version", required=True)
    firmware = sub.add_parser("firmware")
    firmware.add_argument("--manifest", type=Path, default=ROOT / "deps/manifest.json")
    firmware.add_argument("--expected", required=True)
    args = parser.parse_args()
    try:
        if args.command == "request":
            validate_request(args.manifest, args.version)
        elif args.command == "patch-setup":
            patch_setup(args.path)
        elif args.command == "release":
            write_release(args.source, args.output)
        elif args.command == "inventory":
            verify_inventory(args.sysroot, args.version)
        else:
            if declaration(args.manifest) != args.expected:
                raise ValueError("Runner resolver firmware differs from the source declaration")
            verify_firmware(args.expected, (Path("/etc/build"), Path("/etc/buildinfo")))
    except (OSError, ValueError) as error:
        parser.exit(2, f"Platform admission refused: {error}\n")


if __name__ == "__main__":
    main()
