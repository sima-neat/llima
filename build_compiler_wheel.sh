#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LLIMA_COMPILER_WHEEL_PYTHON:-python3.12}"
BUILD_TOOLS_VENV="${LLIMA_WHEEL_BUILD_TOOLS_VENV:-${ROOT_DIR}/build/wheel-tools-venv}"
OUTPUT_DIR="${LLIMA_COMPILER_WHEEL_OUTPUT_DIR:-${ROOT_DIR}/dist/compiler}"
BUILD_DIR="${LLIMA_COMPILER_WHEEL_BUILD_DIR:-${ROOT_DIR}/build/compiler-wheel/{wheel_tag}}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
INSTALLER_SOURCE="${ROOT_DIR}/tools/install_compiler.sh"

# shellcheck source=tools/ensure_wheel_build_env.sh
source "${ROOT_DIR}/tools/ensure_wheel_build_env.sh"
ensure_wheel_build_env "${PYTHON_BIN}" "${BUILD_TOOLS_VENV}"
PYTHON_BIN="${WHEEL_BUILD_PYTHON}"

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "ERROR: sha256sum is required." >&2
  exit 1
fi

if ! command -v "${SIMA_CLI_BIN}" >/dev/null 2>&1; then
  echo "ERROR: sima-cli is required to generate dist/compiler/metadata.json." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# This directory is owned by this helper. Remove only artifacts that this helper
# generated so a wheel from an older source revision cannot be selected.
find "${OUTPUT_DIR}" -maxdepth 1 -type f \
  \( -name 'sima_lmm-*.whl' -o -name 'sima_lmm-*.whl.sha256' \
     -o -name 'install_compiler.sh' -o -name 'manifest.json' \
     -o -name 'metadata.json' \) \
  -delete

echo "[compiler-wheel] Building the pure-Python LLiMa wheel"
echo "[compiler-wheel] Python: ${PYTHON_BIN}"
echo "[compiler-wheel] Output: ${OUTPUT_DIR}"

DEBIAN_PACKAGE_VERSION="$(bash "${ROOT_DIR}/tools/compute_package_version.sh")"
WHEEL_VERSION="${LLIMA_WHEEL_VERSION:-}"
if [[ -z "${WHEEL_VERSION}" ]]; then
  WHEEL_VERSION="$(
    "${PYTHON_BIN}" - "${DEBIAN_PACKAGE_VERSION}" <<'PY'
import re
import sys

from packaging.version import InvalidVersion, Version

debian_version = sys.argv[1]
if "+" in debian_version:
    public, local = debian_version.split("+", 1)
    local = re.sub(r"[^a-z0-9]+", ".", local.lower()).strip(".")
    if not local:
        raise SystemExit(
            f"ERROR: Debian package version has no usable wheel suffix: {debian_version}"
        )
    candidate = f"{public}+{local}"
else:
    candidate = debian_version

try:
    print(Version(candidate))
except InvalidVersion as error:
    raise SystemExit(f"ERROR: Invalid wheel version {candidate!r}: {error}") from error
PY
  )"
fi

echo "[compiler-wheel] Version: ${WHEEL_VERSION}"

LLIMA_WHEEL_VERSION="${WHEEL_VERSION}" "${PYTHON_BIN}" -m build \
  --wheel \
  --outdir "${OUTPUT_DIR}" \
  -Cbuild-dir="${BUILD_DIR}" \
  -Cwheel.cmake=false \
  "${ROOT_DIR}"

mapfile -t WHEELS < <(
  find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'sima_lmm-*.whl' -print | sort
)

if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "ERROR: Expected exactly one sima_lmm wheel in ${OUTPUT_DIR}; found ${#WHEELS[@]}." >&2
  printf '  %s\n' "${WHEELS[@]}" >&2
  exit 1
fi

WHEEL_PATH="${WHEELS[0]}"
WHEEL_NAME="$(basename "${WHEEL_PATH}")"

if [[ "${WHEEL_NAME}" != sima_lmm-*-py3-none-any.whl ]]; then
  echo "ERROR: Expected a py3-none-any wheel; built ${WHEEL_NAME}." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${WHEEL_PATH}" <<'PY'
import configparser
import sys
import zipfile
from pathlib import Path

wheel_path = Path(sys.argv[1])

with zipfile.ZipFile(wheel_path) as wheel:
    names = wheel.namelist()

    native_files = [
        name for name in names
        if name.endswith((".so", ".dylib", ".dll", ".pyd", ".a"))
        or ".so." in name
    ]
    cpp_sources = [
        name for name in names
        if name.startswith("sima_lmm/devkit/cpp/")
    ]
    if native_files or cpp_sources:
        details = "\n".join(f"  {name}" for name in native_files + cpp_sources)
        raise SystemExit(
            "ERROR: Compiler wheel contains native runtime content:\n" + details
        )

    wheel_metadata = [name for name in names if name.endswith(".dist-info/WHEEL")]
    if len(wheel_metadata) != 1:
        raise SystemExit(
            f"ERROR: Expected one WHEEL metadata file; found {len(wheel_metadata)}."
        )
    metadata = wheel.read(wheel_metadata[0]).decode("utf-8")
    if "Tag: py3-none-any" not in metadata.splitlines():
        raise SystemExit("ERROR: Wheel metadata does not declare Tag: py3-none-any.")

    entry_points = [
        name for name in names if name.endswith(".dist-info/entry_points.txt")
    ]
    if len(entry_points) != 1:
        raise SystemExit(
            "ERROR: Expected one entry_points.txt file; "
            f"found {len(entry_points)}."
        )
    parser = configparser.ConfigParser()
    parser.read_string(wheel.read(entry_points[0]).decode("utf-8"))
    actual_entry_point = parser.get(
        "console_scripts", "llima-compile", fallback=""
    ).strip()
    expected_entry_point = "sima_lmm.host.compile_lmm:main"
    if actual_entry_point != expected_entry_point:
        raise SystemExit(
            "ERROR: llima-compile entry point is missing or invalid: "
            f"{actual_entry_point!r}."
        )
PY

BUILT_WHEEL_VERSION="$("${PYTHON_BIN}" - "${WHEEL_PATH}" <<'PY'
import email
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata_name = next(
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    )
    metadata = email.message_from_bytes(wheel.read(metadata_name))
print(metadata["Version"])
PY
)"

if [[ "${BUILT_WHEEL_VERSION}" != "${WHEEL_VERSION}" ]]; then
  echo "ERROR: Built wheel version ${BUILT_WHEEL_VERSION} does not match requested version ${WHEEL_VERSION}." >&2
  exit 1
fi

(
  cd "${OUTPUT_DIR}"
  sha256sum "${WHEEL_NAME}" > "${WHEEL_NAME}.sha256"
)

SOURCE_REPOSITORY="${GITHUB_REPOSITORY:-sima-neat/llima}"
SOURCE_REF="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
if [[ -z "${SOURCE_REF}" ]]; then
  SOURCE_REF="$(git -C "${ROOT_DIR}" branch --show-current 2>/dev/null || true)"
fi
SOURCE_REF="${SOURCE_REF:-detached}"
SOURCE_SHA="${GITHUB_SHA:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"

"${PYTHON_BIN}" - \
  "${OUTPUT_DIR}/manifest.json" \
  "${WHEEL_NAME}" \
  "${WHEEL_VERSION}" \
  "${SOURCE_REPOSITORY}" \
  "${SOURCE_REF}" \
  "${SOURCE_SHA}" \
  "${GITHUB_RUN_ID:-}" \
  "${GITHUB_RUN_ATTEMPT:-}" <<'PY'
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    manifest_path,
    wheel_name,
    wheel_version,
    repository,
    ref,
    commit,
    workflow_run_id,
    workflow_run_attempt,
) = sys.argv[1:]

wheel_path = Path(manifest_path).parent / wheel_name
manifest = {
    "schema-version": 1,
    "artifact": "llima-compiler-wheel",
    "package": "sima-lmm",
    "version": wheel_version,
    "wheel": {
        "filename": wheel_name,
        "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
        "size-bytes": wheel_path.stat().st_size,
        "tag": "py3-none-any",
    },
    "source": {
        "repository": repository,
        "ref": ref,
        "commit": commit,
    },
    "build": {
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "built-at-utc": datetime.now(timezone.utc).isoformat(),
        "workflow-run-id": workflow_run_id,
        "workflow-run-attempt": workflow_run_attempt,
    },
}
Path(manifest_path).write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

install -m 0755 "${INSTALLER_SOURCE}" "${OUTPUT_DIR}/install_compiler.sh"

SIMA_CLI_CHECK_FOR_UPDATE=0 "${SIMA_CLI_BIN}" packages build "${OUTPUT_DIR}" \
  --name "gh:sima-neat/llima" \
  --version "${WHEEL_VERSION}" \
  --description "Pure-Python LLiMa compiler wheel" \
  --install-script install_compiler.sh \
  --download-compatible-files-only \
  --host-platform linux

"${PYTHON_BIN}" - "${OUTPUT_DIR}/metadata.json" "${OUTPUT_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata.update(
    {
        "artifact": {
            "type": "python-wheel",
            "profile": "compiler",
            "wheel": manifest["wheel"]["filename"],
        },
        "repository": manifest["source"]["repository"],
        "branch": manifest["source"]["ref"],
        "commit": manifest["source"]["commit"],
        "published-artifact": "compiler",
    }
)
metadata.setdefault("installation", {})["post-message"] = (
    "[bold]LLiMa compiler wheel installed successfully.[/bold]\n"
)
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

echo "[compiler-wheel] Built: ${WHEEL_PATH}"
echo "[compiler-wheel] Checksum: ${WHEEL_PATH}.sha256"
echo "[compiler-wheel] Installer: ${OUTPUT_DIR}/install_compiler.sh"
echo "[compiler-wheel] Manifest: ${OUTPUT_DIR}/manifest.json"
echo "[compiler-wheel] Metadata: ${OUTPUT_DIR}/metadata.json"
