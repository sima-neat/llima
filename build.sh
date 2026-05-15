#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${LLIMA_DEB_BUILD_DIR:-$ROOT_DIR/build-deb}"
BUILD_JOBS="${LLIMA_DEB_BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
ARCH=arm64

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [--] [extra cmake configure args...]

Options:
  --install-deps-only
                      Install host build dependencies, then exit
  --build-dir <dir>   CMake build directory (default: llima/build-deb)
  --jobs <count>      Parallel build jobs (default: nproc; env: LLIMA_DEB_BUILD_JOBS)
  --clean             Remove the build directory and stale sima-lmm*.deb outputs
  --all               Build all sima-lmm binary packages and create dist archive (default)
  --no-dist           Skip dist tarball creation
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
INSTALL_DEPS_ONLY=0
SKIP_DIST=0
EXTRA_CMAKE_ARGS=()
COMPONENTS=()

has_cmake_define() {
  local name="$1"
  local arg
  for arg in "${EXTRA_CMAKE_ARGS[@]}"; do
    if [[ "${arg}" == "-D${name}="* ]]; then
      return 0
    fi
  done
  return 1
}

running_in_neat_sdk() {
  if [[ -f /etc/sdk-release ]]; then
    return 0
  fi
  if [[ -n "${SYSROOT:-}" && -d "${SYSROOT}" ]]; then
    return 0
  fi
  return 1
}

apply_default_sdk_toolchain() {
  local toolchain_file="${ROOT_DIR}/toolchain-sima.cmake"

  if ! running_in_neat_sdk; then
    return 0
  fi
  if has_cmake_define "CMAKE_TOOLCHAIN_FILE"; then
    return 0
  fi
  if [[ ! -f "${toolchain_file}" ]]; then
    echo "ERROR: NEAT SDK detected but toolchain file is missing: ${toolchain_file}" >&2
    exit 1
  fi

  echo "[build] NEAT SDK detected; defaulting to ${toolchain_file}"
  EXTRA_CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=${toolchain_file}")
}

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
  for tool in cmake cpack git python3 dpkg-architecture; do
    if ! command -v "$tool" >/dev/null 2>&1; then
      echo "ERROR: $tool is required" >&2
      exit 1
    fi
  done
}

run_as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return $?
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return $?
  fi
  echo "ERROR: Root permissions are required and sudo is not available." >&2
  exit 1
}

install_deps() {
  run_as_root apt-get update
  run_as_root apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    dpkg-dev
}

ensure_git_submodules() {
  local path
  local missing=0

  if [[ ! -f "$ROOT_DIR/.gitmodules" ]]; then
    return
  fi

  if git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[build] Updating git submodules"
    git -C "$ROOT_DIR" submodule update --init --recursive
  else
    echo "[build] Source tree is not a git checkout; checking submodule directories"
  fi

  while read -r path; do
    if [[ -z "$path" ]]; then
      continue
    fi
    if [[ ! -d "$ROOT_DIR/$path" ]] ||
       [[ -z "$(find "$ROOT_DIR/$path" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
      echo "ERROR: Missing git submodule content: $path" >&2
      missing=1
    fi
  done < <(git -C "$ROOT_DIR" config --file "$ROOT_DIR/.gitmodules" --get-regexp path | awk '{print $2}')

  if [[ "$missing" -ne 0 ]]; then
    echo "ERROR: Required third-party sources are missing." >&2
    echo "       Run: git -C \"$ROOT_DIR\" submodule update --init --recursive" >&2
    exit 1
  fi
}

ensure_sdk_sysroot_packages() {
  local sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  local overlay_script="/usr/local/bin/install-sysroot-overlay.sh"
  local packages=(
    libopencv-flann406:arm64
    libopencv-dnn406:arm64
    libopencv-features2d406:arm64
    libopencv-objdetect406:arm64
    libopencv-video406:arm64
    libssl-dev:arm64
    libpgm-dev:arm64
  )

  if ! running_in_neat_sdk; then
    return
  fi
  if [[ "${LLIMA_SKIP_SYSROOT_OVERLAY:-0}" == "1" ]]; then
    echo "[build] Skipping SDK sysroot package overlay"
    return
  fi
  if [[ ! -x "${overlay_script}" ]]; then
    echo "ERROR: SDK sysroot overlay installer not found: ${overlay_script}" >&2
    exit 1
  fi

  echo "[build] Installing llima SDK sysroot package overlay"
  run_as_root "${overlay_script}" "${sysroot}" "${packages[@]}"
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

ensure_writable_cargo_home() {
  local cargo_home="${LLIMA_CARGO_HOME:-${CARGO_HOME:-}}"

  if [[ -z "${cargo_home}" || ! -w "${cargo_home}" ]]; then
    cargo_home="${BUILD_DIR}/.cargo-home"
  fi

  mkdir -p "${cargo_home}"
  export CARGO_HOME="${cargo_home}"
  echo "[build] Using Cargo home: ${CARGO_HOME}"
}

package_dist_archive() {
  if [ "${#COMPONENTS[@]}" -gt 0 ]; then
    echo "[build] Skipping dist archive because a subset of components was selected"
    return
  fi

  local version archive_name latest_name
  version="$1"
  archive_name="${ARCHIVE_NAME:-sima-llima-${version}.tar.gz}"
  latest_name="${LATEST_NAME:-}"

  local required_debs=(
    "sima-lmm-${version}-Linux-cli.deb"
    "sima-lmm-${version}-Linux-core.deb"
    "sima-lmm-${version}-Linux-dev.deb"
  )

  mkdir -p "${ROOT_DIR}/dist"

  for deb in "${required_debs[@]}"; do
    if [[ ! -f "${ROOT_DIR}/${deb}" ]]; then
      echo "Expected Debian package not found: ${ROOT_DIR}/${deb}" >&2
      find "${ROOT_DIR}" -maxdepth 1 -type f -name 'sima-lmm*.deb' -printf '  %p\n' | sort >&2
      return 1
    fi
  done

  tar -C "${ROOT_DIR}" -czf "${ROOT_DIR}/dist/${archive_name}" "${required_debs[@]}"
  sha256sum "${ROOT_DIR}/dist/${archive_name}" | awk '{print $1}' > "${ROOT_DIR}/dist/${archive_name}.sha256"
  ls -lh "${ROOT_DIR}/dist/${archive_name}"
  cat "${ROOT_DIR}/dist/${archive_name}.sha256"

  if [[ -n "${latest_name}" ]]; then
    cp "${ROOT_DIR}/dist/${archive_name}" "${ROOT_DIR}/dist/${latest_name}"
    sha256sum "${ROOT_DIR}/dist/${latest_name}" | awk '{print $1}' > "${ROOT_DIR}/dist/${latest_name}.sha256"
    ls -lh "${ROOT_DIR}/dist/${latest_name}"
    cat "${ROOT_DIR}/dist/${latest_name}.sha256"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-deps-only)
      INSTALL_DEPS_ONLY=1
      shift
      ;;
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
    --no-dist)
      SKIP_DIST=1
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

if [ "$INSTALL_DEPS_ONLY" -eq 1 ]; then
  install_deps
  exit 0
fi

detect_build_jobs

if ! [[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: --jobs must be a positive integer" >&2
  exit 2
fi

if [ "${#COMPONENTS[@]}" -gt 0 ]; then
  mapfile -t COMPONENTS < <(printf '%s\n' "${COMPONENTS[@]}" | sort -u)
fi

check_local_build_tools
ensure_git_submodules
apply_default_sdk_toolchain
ensure_sdk_sysroot_packages

LLIMA_VERSION="$(version_from_version_in)"
MULTIARCH="$(dpkg-architecture -a"$ARCH" -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
if [ -z "$MULTIARCH" ]; then
  MULTIARCH="aarch64-linux-gnu"
fi

if [ "$DO_CLEAN" -eq 1 ]; then
  echo "[build] Removing build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  rm -rf "$ROOT_DIR/dist"
  rm -f "$ROOT_DIR"/sima-lmm*.deb
fi

ensure_writable_cargo_home
ensure_python_build_env

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

if [ "$SKIP_DIST" -eq 0 ]; then
  package_dist_archive "$LLIMA_VERSION"
fi
