#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SIMA_LLIMA_COMPILER_PYTHON:-python}"

usage() {
  cat <<EOF
Usage: $(basename "$0")

Replace sima_lmm in the active Model Compiler virtual environment with the
verified wheel packaged beside this installer. Existing Model Compiler
dependencies are preserved.

Environment:
  SIMA_LLIMA_COMPILER_PYTHON  Active Model Compiler Python (default: python)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  "")
    ;;
  *)
    echo "ERROR: Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if ! command -v sha256sum >/dev/null 2>&1; then
  echo "ERROR: sha256sum is required." >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: Activate the Model Compiler virtual environment before installing LLiMa." >&2
  exit 1
fi

"${PYTHON_BIN}" - "${VIRTUAL_ENV}" <<'PY'
import importlib.util
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
actual = Path(sys.prefix).resolve()
if actual != expected:
    raise SystemExit(
        f"ERROR: Active Python prefix {actual} does not match VIRTUAL_ENV {expected}."
    )
if importlib.util.find_spec("afe") is None:
    raise SystemExit(
        "ERROR: The active virtual environment does not contain the Model Compiler afe package."
    )
print(f"[compiler-install] Model Compiler Python: {sys.executable}")
PY

mapfile -t WHEELS < <(
  find "${SCRIPT_DIR}" -maxdepth 1 -type f -name 'sima_lmm-*.whl' -print | sort
)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "ERROR: Expected exactly one sima_lmm wheel beside this installer; found ${#WHEELS[@]}." >&2
  printf '  %s\n' "${WHEELS[@]}" >&2
  exit 1
fi

WHEEL_PATH="${WHEELS[0]}"
WHEEL_NAME="$(basename "${WHEEL_PATH}")"
CHECKSUM_PATH="${WHEEL_PATH}.sha256"
if [[ ! -f "${CHECKSUM_PATH}" ]]; then
  echo "ERROR: Missing wheel checksum: ${CHECKSUM_PATH}" >&2
  exit 1
fi
(
  cd "${SCRIPT_DIR}"
  sha256sum -c "${WHEEL_NAME}.sha256"
)

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata

try:
    dist = metadata.distribution("sima-lmm")
except metadata.PackageNotFoundError:
    print("[compiler-install] No existing sima_lmm distribution is installed.")
else:
    print(f"[compiler-install] Replacing {dist.metadata['Name']} {dist.version}")
PY

"${PYTHON_BIN}" -m pip uninstall -y sima_lmm
"${PYTHON_BIN}" -m pip install --no-deps "${WHEEL_PATH}"

"${PYTHON_BIN}" - "${VIRTUAL_ENV}" <<'PY'
import importlib.metadata as metadata
import sys
from pathlib import Path

import sima_lmm

venv = Path(sys.argv[1]).resolve()
module_path = Path(sima_lmm.__file__).resolve()
if venv not in module_path.parents:
    raise SystemExit(
        f"ERROR: sima_lmm was imported from {module_path}, outside {venv}."
    )
print(f"[compiler-install] Installed sima-lmm {metadata.version('sima-lmm')}")
print(f"[compiler-install] Import path: {module_path}")
PY

LLIMA_COMPILE="${VIRTUAL_ENV}/bin/llima-compile"
if [[ ! -x "${LLIMA_COMPILE}" ]]; then
  echo "ERROR: llima-compile was not installed in ${VIRTUAL_ENV}/bin." >&2
  exit 1
fi
"${LLIMA_COMPILE}" --help >/dev/null

echo "[compiler-install] LLiMa compiler wheel installed successfully."
