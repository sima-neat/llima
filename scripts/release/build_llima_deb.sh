#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
BUILD_DIR="${LLIMA_DEB_BUILD_DIR:-$ROOT_DIR/build-deb}"
BUILD_JOBS="${LLIMA_DEB_BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
ARCH=arm64

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [--] [extra cmake configure args...]

Options:
  --build-dir <dir>   CMake build directory (default: llima/build-deb)
  --jobs <count>      Parallel build jobs (default: nproc; env: LLIMA_DEB_BUILD_JOBS)
  --clean             Remove stale sima-lmm*.deb outputs before packaging
  --all               Build all sima-lmm binary packages (default)
  --core              Package only sima-lmm-core
  --dev               Package only sima-lmm-dev
  --cli               Package only sima-lmm-cli
  --package <name>    Package one component; repeatable. Accepts core, dev, cli,
                      or the full Debian package name.
  -h, --help          Show this help

Extra arguments after -- are passed to cmake configure.
EOF
}

DO_CLEAN=0
EXTRA_CMAKE_ARGS=()
COMPONENTS=()

version_from_version_in() {
  local key
  local value
  local parts=()
  for key in major minor patch; do
    value="$(awk -F: -v key="$key" '$1 == key { gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2 }' "$ROOT_DIR/VERSION.in")"
    if [ -z "$value" ]; then
      echo "ERROR: Unable to parse $key from $ROOT_DIR/VERSION.in" >&2
      exit 1
    fi
    parts+=("$value")
  done
  printf '%s.%s.%s\n' "${parts[0]}" "${parts[1]}" "${parts[2]}"
}

add_component() {
  case "$1" in
    core|sima-lmm-core)
      COMPONENTS+=("core")
      ;;
    dev|sima-lmm-dev)
      COMPONENTS+=("dev")
      ;;
    cli|sima-lmm-cli)
      COMPONENTS+=("cli")
      ;;
    all)
      COMPONENTS=()
      ;;
    *)
      echo "ERROR: Unknown package selector: $1" >&2
      echo "       Expected core, dev, cli, sima-lmm-core, sima-lmm-dev, or sima-lmm-cli." >&2
      exit 2
      ;;
  esac
}

check_local_build_tools() {
  local tool
  for tool in cmake cpack python3; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "ERROR: $tool is required" >&2
      exit 1
    fi
  done
}

detect_build_jobs() {
  if [ -n "$BUILD_JOBS" ]; then
    return
  fi

  if command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
  else
    BUILD_JOBS=8
  fi
}

ensure_python_build_env() {
  if [ ! -x "$BUILD_VENV/bin/python" ]; then
    echo "[build] Creating Python build environment: $BUILD_VENV"
    python3 -m venv "$BUILD_VENV"
  fi

  if ! "$BUILD_VENV/bin/python" -m nanobind --cmake_dir >/dev/null 2>&1; then
    echo "[build] Installing Python build requirements into $BUILD_VENV"
    if ! "$BUILD_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
      "$BUILD_VENV/bin/python" -m ensurepip --upgrade
    fi
    "$BUILD_VENV/bin/python" -m pip install --upgrade pip
    "$BUILD_VENV/bin/python" -m pip install "nanobind>=2.12.0" pyyaml
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --build-dir)
      BUILD_DIR="${2:-}"
      shift 2
      ;;
    --jobs)
      BUILD_JOBS="${2:-}"
      shift 2
      ;;
    --clean)
      DO_CLEAN=1
      shift
      ;;
    --all)
      COMPONENTS=()
      shift
      ;;
    --core)
      add_component core
      shift
      ;;
    --dev)
      add_component dev
      shift
      ;;
    --cli)
      add_component cli
      shift
      ;;
    --package)
      add_component "${2:-}"
      shift 2
      ;;
    --)
      shift
      EXTRA_CMAKE_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_CMAKE_ARGS+=("$1")
      shift
      ;;
  esac
done

BUILD_VENV="$BUILD_DIR/.deb-build-venv"

detect_build_jobs

if ! [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --jobs must be a positive integer" >&2
  exit 2
fi

if [ "${#COMPONENTS[@]}" -gt 0 ]; then
  mapfile -t COMPONENTS < <(printf '%s\n' "${COMPONENTS[@]}" | sort -u)
fi

check_local_build_tools
ensure_python_build_env

LLIMA_VERSION="$(version_from_version_in)"
MULTIARCH="$(dpkg-architecture -a"$ARCH" -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
if [ -z "$MULTIARCH" ]; then
  MULTIARCH="aarch64-linux-gnu"
fi

if [ "$DO_CLEAN" -eq 1 ]; then
  rm -f "$ROOT_DIR"/sima-lmm*.deb
fi

echo "[build] Configuring sima-lmm $LLIMA_VERSION for arch=$ARCH"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_INSTALL_LIBDIR="lib/$MULTIARCH" \
  -DPython_EXECUTABLE="$BUILD_VENV/bin/python" \
  -DSIMA_LMM_BUILD_PYTHON=ON \
  -DSIMA_LMM_INSTALL_PYTHON_PACKAGE=ON \
  -DSIMA_LMM_PYTHON_EXTENSION_INSTALL_DIR="lib/python3/dist-packages/sima_lmm/devkit" \
  -DSIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR="lib/python3/dist-packages/sima_lmm" \
  -DCPACK_DEBIAN_PACKAGE_ARCHITECTURE="$ARCH" \
  "${EXTRA_CMAKE_ARGS[@]}"

echo "[build] Building sima-lmm targets with $BUILD_JOBS parallel job(s)"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

CPACK_ARGS=(
  --config "$BUILD_DIR/CPackConfig.cmake"
  -D "CPACK_PACKAGE_DIRECTORY=$ROOT_DIR"
)

if [ "${#COMPONENTS[@]}" -eq 0 ]; then
  echo "[build] Packaging all sima-lmm Debian components with CPack"
else
  component_list="$(
    IFS=';'
    printf '%s' "${COMPONENTS[*]}"
  )"
  echo "[build] Packaging selected sima-lmm Debian components with CPack: ${COMPONENTS[*]}"
  CPACK_ARGS+=(-D "CPACK_COMPONENTS_ALL=$component_list")
fi

cpack "${CPACK_ARGS[@]}"

echo "[build] Generated packages:"
if [ "${#COMPONENTS[@]}" -eq 0 ]; then
  find "$ROOT_DIR" -maxdepth 1 -type f -name 'sima-lmm*.deb' -printf '  %p\n' | sort
else
  for component in "${COMPONENTS[@]}"; do
    find "$ROOT_DIR" -maxdepth 1 -type f -name "sima-lmm-${component}_*.deb" -printf '  %p\n'
    find "$ROOT_DIR" -maxdepth 1 -type f -name "sima-lmm-${component}-*.deb" -printf '  %p\n'
  done | sort -u
fi
