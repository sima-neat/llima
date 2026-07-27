#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${SIMA_MOLE_PYTHON:-python3.12}"
DEFAULT_VENV_PATH="${HOME}/sima-mole-venv"
VENV_PATH="${SIMA_MOLE_VENV_PATH:-}"
ASSUME_DEFAULT=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--venv <path>] [--yes]

Install MoLE and its sdk_ext dependencies into a Python virtual environment.

Options:
  --venv <path>  Virtual environment path (default: ${DEFAULT_VENV_PATH})
  --yes          Use the default path without prompting
  -h, --help     Show this help

Environment:
  SIMA_MOLE_VENV_PATH  Virtual environment path
  SIMA_MOLE_PYTHON     Python interpreter used to create it (default: python3.12)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="${2:-}"
      if [[ -z "${VENV_PATH}" ]]; then
        echo "ERROR: --venv requires a path." >&2
        exit 2
      fi
      shift 2
      ;;
    --yes)
      ASSUME_DEFAULT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python interpreter not found: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${VENV_PATH}" ]]; then
  if [[ "${ASSUME_DEFAULT}" -eq 0 && -t 0 ]]; then
    read -r -p "MoLE virtual environment [${DEFAULT_VENV_PATH}]: " VENV_PATH
  fi
  VENV_PATH="${VENV_PATH:-${DEFAULT_VENV_PATH}}"
fi

VENV_PATH="$("${PYTHON_BIN}" - "${VENV_PATH}" <<'PY'
import sys
from pathlib import Path

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
)"

mapfile -t WHEELS < <(
  find "${SCRIPT_DIR}" -maxdepth 1 -type f -name 'sima_lmm-*.whl' -print | sort
)
if [[ "${#WHEELS[@]}" -ne 1 ]]; then
  echo "ERROR: Expected exactly one sima_lmm wheel beside this installer; found ${#WHEELS[@]}." >&2
  printf '  %s\n' "${WHEELS[@]}" >&2
  exit 1
fi
WHEEL_PATH="${WHEELS[0]}"

if [[ -e "${VENV_PATH}" && ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
  echo "ERROR: Refusing to use an existing non-venv path: ${VENV_PATH}" >&2
  exit 1
fi

if [[ ! -f "${VENV_PATH}/pyvenv.cfg" ]]; then
  echo "[mole] Creating virtual environment: ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
  echo "[mole] Reusing virtual environment: ${VENV_PATH}"
fi

VENV_PYTHON="${VENV_PATH}/bin/python"
"${VENV_PYTHON}" -m pip install --upgrade pip
"${VENV_PYTHON}" -m pip install --upgrade "${WHEEL_PATH}[sdk_ext]"
# A branch wheel can retain the same public version as an older installation.
# Reinstall only LLiMa itself without disturbing the resolved MoLE dependencies.
"${VENV_PYTHON}" -m pip install --force-reinstall --no-deps "${WHEEL_PATH}"
"${VENV_PATH}/bin/llima-benchmark" --help >/dev/null

echo "[mole] Installed successfully."
echo "[mole] Activate with: source ${VENV_PATH}/bin/activate"
