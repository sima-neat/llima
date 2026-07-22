#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${LLIMA_MOLE_PACKAGE_PYTHON:-python3.12}"
BUILD_TOOLS_VENV="${LLIMA_WHEEL_BUILD_TOOLS_VENV:-${ROOT_DIR}/build/wheel-tools-venv}"
SIMA_CLI_BIN="${SIMA_CLI_BIN:-sima-cli}"
COMPILER_DIR="${LLIMA_COMPILER_WHEEL_OUTPUT_DIR:-${ROOT_DIR}/dist/compiler}"
OUTPUT_DIR="${LLIMA_MOLE_PACKAGE_OUTPUT_DIR:-${ROOT_DIR}/dist/mole}"
INSTALLER_SOURCE="${ROOT_DIR}/tools/install_mole.sh"

# shellcheck source=tools/ensure_wheel_build_env.sh
source "${ROOT_DIR}/tools/ensure_wheel_build_env.sh"
ensure_wheel_build_env "${PYTHON_BIN}" "${BUILD_TOOLS_VENV}"
PYTHON_BIN="${WHEEL_BUILD_PYTHON}"

for command in "${SIMA_CLI_BIN}" sha256sum; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "ERROR: Required command not found: ${command}" >&2
    exit 1
  fi
done

mapfile -t SOURCE_WHEELS < <(
  find "${COMPILER_DIR}" -maxdepth 1 -type f -name 'sima_lmm-*.whl' -print | sort
)
if [[ "${#SOURCE_WHEELS[@]}" -ne 1 ]]; then
  echo "ERROR: Expected one compiler wheel in ${COMPILER_DIR}; found ${#SOURCE_WHEELS[@]}." >&2
  echo "       Run ./build_compiler_wheel.sh first." >&2
  exit 1
fi

SOURCE_WHEEL="${SOURCE_WHEELS[0]}"
WHEEL_NAME="$(basename "${SOURCE_WHEEL}")"
SOURCE_CHECKSUM="${SOURCE_WHEEL}.sha256"
if [[ ! -f "${SOURCE_CHECKSUM}" ]]; then
  echo "ERROR: Missing compiler wheel checksum: ${SOURCE_CHECKSUM}" >&2
  exit 1
fi
(
  cd "${COMPILER_DIR}"
  sha256sum -c "${WHEEL_NAME}.sha256"
)

"${PYTHON_BIN}" - "${SOURCE_WHEEL}" <<'PY'
import configparser
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    entry_points_name = next(
        name for name in wheel.namelist()
        if name.endswith(".dist-info/entry_points.txt")
    )
    parser = configparser.ConfigParser()
    parser.read_string(wheel.read(entry_points_name).decode("utf-8"))
    actual = parser.get("console_scripts", "llima-benchmark", fallback="").strip()
    expected = "sima_lmm.host.benchmark:main"
    if actual != expected:
        raise SystemExit(f"ERROR: Invalid llima-benchmark entry point: {actual!r}")

    metadata_name = next(
        name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
    )
    metadata = wheel.read(metadata_name).decode("utf-8")
    sdk_ext_lines = [
        line for line in metadata.splitlines()
        if line.startswith("Requires-Dist:") and 'extra == "sdk-ext"' in line
    ]
    if not sdk_ext_lines:
        raise SystemExit("ERROR: Wheel does not define sdk_ext dependencies.")
    if any("sima-frontend" in line.lower() for line in sdk_ext_lines):
        raise SystemExit("ERROR: sdk_ext unexpectedly depends on sima-frontend.")
PY

WHEEL_VERSION="$("${PYTHON_BIN}" - "${SOURCE_WHEEL}" <<'PY'
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

mkdir -p "${OUTPUT_DIR}"
find "${OUTPUT_DIR}" -maxdepth 1 -type f \
  \( -name 'sima_lmm-*.whl' -o -name 'sima_lmm-*.whl.sha256' \
     -o -name 'install_mole.sh' -o -name 'metadata.json' \) \
  -delete

cp "${SOURCE_WHEEL}" "${OUTPUT_DIR}/${WHEEL_NAME}"
cp "${SOURCE_CHECKSUM}" "${OUTPUT_DIR}/${WHEEL_NAME}.sha256"
install -m 0755 "${INSTALLER_SOURCE}" "${OUTPUT_DIR}/install_mole.sh"

SIMA_CLI_CHECK_FOR_UPDATE=0 "${SIMA_CLI_BIN}" packages build "${OUTPUT_DIR}" \
  --name "gh:sima-neat/llima/mole" \
  --version "${WHEEL_VERSION}" \
  --description "MoLE - Modalix Language Model Evaluator" \
  --install-script 'bash ./install_mole.sh' \
  --download-compatible-files-only \
  --host-platform linux

SOURCE_REPOSITORY="${GITHUB_REPOSITORY:-sima-neat/llima}"
SOURCE_REF="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
if [[ -z "${SOURCE_REF}" ]]; then
  SOURCE_REF="$(git -C "${ROOT_DIR}" branch --show-current 2>/dev/null || true)"
fi
SOURCE_REF="${SOURCE_REF:-detached}"
SOURCE_SHA="${GITHUB_SHA:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"

"${PYTHON_BIN}" - \
  "${OUTPUT_DIR}/metadata.json" \
  "${WHEEL_NAME}" \
  "${SOURCE_REPOSITORY}" \
  "${SOURCE_REF}" \
  "${SOURCE_SHA}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

metadata_path = Path(sys.argv[1])
wheel_name, repository, ref, commit = sys.argv[2:]
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata.update(
    {
        "artifact": {
            "type": "python-wheel",
            "repository": "llima",
            "package_path": "mole",
            "profile": "mole-sdk-ext",
            "wheel": wheel_name,
        },
        "repository": repository,
        "branch": ref,
        "commit": commit,
        "commit_folder": commit[:12],
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
)
metadata.setdefault("installation", {})["post-message"] = (
    "[bold]MoLE installed successfully.[/bold]\n"
)
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY

echo "[mole-package] Wheel: ${OUTPUT_DIR}/${WHEEL_NAME}"
echo "[mole-package] Checksum: ${OUTPUT_DIR}/${WHEEL_NAME}.sha256"
echo "[mole-package] Installer: ${OUTPUT_DIR}/install_mole.sh"
echo "[mole-package] Metadata: ${OUTPUT_DIR}/metadata.json"
