#!/usr/bin/env bash

# Source this helper, then call ensure_wheel_build_env <python> <venv>.
# The selected interpreter is returned in WHEEL_BUILD_PYTHON.
ensure_wheel_build_env() {
  local base_python="$1"
  local build_tools_venv="$2"

  if ! command -v "${base_python}" >/dev/null 2>&1; then
    echo "ERROR: Python interpreter not found: ${base_python}" >&2
    return 1
  fi

  if [[ ! -x "${build_tools_venv}/bin/python" ]]; then
    echo "[wheel-build] Creating Python build environment: ${build_tools_venv}"
    if ! "${base_python}" -m venv "${build_tools_venv}"; then
      echo "ERROR: Could not create the wheel build-tools virtual environment." >&2
      echo "       Ensure the Python venv module is installed (for example, python3.12-venv)." >&2
      return 1
    fi
  else
    echo "[wheel-build] Reusing Python build environment: ${build_tools_venv}"
  fi

  WHEEL_BUILD_PYTHON="${build_tools_venv}/bin/python"

  if ! "${WHEEL_BUILD_PYTHON}" -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
    echo "ERROR: ${build_tools_venv} is not a Python 3.12 virtual environment." >&2
    echo "       Remove it or set LLIMA_WHEEL_BUILD_TOOLS_VENV to a Python 3.12 environment." >&2
    return 1
  fi

  if ! "${WHEEL_BUILD_PYTHON}" -m pip --version >/dev/null 2>&1; then
    "${WHEEL_BUILD_PYTHON}" -m ensurepip --upgrade
  fi
  if ! "${WHEEL_BUILD_PYTHON}" -c \
    'import importlib.util; raise SystemExit(importlib.util.find_spec("build.__main__") is None)' \
    >/dev/null 2>&1; then
    echo "[wheel-build] Installing Python package 'build'"
    "${WHEEL_BUILD_PYTHON}" -m pip install build
  fi
}
