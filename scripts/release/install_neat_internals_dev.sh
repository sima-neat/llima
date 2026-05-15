#!/usr/bin/env bash
set -euo pipefail

INTERNALS_URL="${NEAT_INTERNALS_ARCHIVE_URL:-https://artifacts.sima-neat.com/internals/sima-neat-internals-beta_changes-latest.tar.gz}"
SYSROOT="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
if [[ -n "${LLIMA_INTERNALS_WORK_DIR:-}" ]]; then
  WORK_DIR="${LLIMA_INTERNALS_WORK_DIR}"
  CLEAN_WORK_DIR=0
else
  WORK_DIR="$(mktemp -d /tmp/llima-neat-internals.XXXXXX)"
  CLEAN_WORK_DIR=1
fi
ARCHIVE_PATH="${WORK_DIR}/$(basename "${INTERNALS_URL}")"

log() {
  printf '[install_neat_internals_dev] %s\n' "$*"
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
  echo "Root permissions are required to write to ${SYSROOT}; sudo is not available." >&2
  exit 1
}

if [[ ! -d "${SYSROOT}" ]]; then
  echo "SYSROOT does not exist: ${SYSROOT}" >&2
  exit 1
fi

cleanup() {
  if [[ "${CLEAN_WORK_DIR}" == "1" ]]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

for tool in curl tar find dpkg-deb; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Required tool not found: ${tool}" >&2
    exit 1
  fi
done

if [[ "${CLEAN_WORK_DIR}" == "0" ]]; then
  rm -rf "${WORK_DIR}"
fi
mkdir -p "${WORK_DIR}"

log "Downloading NEAT internals archive:"
log "  ${INTERNALS_URL}"
curl -fsSL "${INTERNALS_URL}" -o "${ARCHIVE_PATH}"

log "Extracting NEAT internals packages"
tar -xzf "${ARCHIVE_PATH}" \
  --wildcards \
  -C "${WORK_DIR}" \
  'neat-runtime_*_arm64.deb' \
  'neat-gst-plugins_*_arm64.deb' \
  'neat-internals-dev_*_arm64.deb'

deb_patterns=(
  'neat-runtime_*_arm64.deb'
  'neat-gst-plugins_*_arm64.deb'
  'neat-internals-dev_*_arm64.deb'
)

for pattern in "${deb_patterns[@]}"; do
  deb="$(find "${WORK_DIR}" -maxdepth 1 -type f -name "${pattern}" | sort | head -n 1)"
  if [[ -z "${deb}" ]]; then
    echo "No ${pattern} found in ${ARCHIVE_PATH}" >&2
    exit 1
  fi

  log "Installing ${deb} into sysroot:"
  log "  ${SYSROOT}"
  run_as_root dpkg-deb -x "${deb}" "${SYSROOT}"
done

config_dir="${SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/NeatInternals"
dispatcher_header="${SYSROOT}/usr/include/dispatcher.h"
profiler_lib="${SYSROOT}/usr/lib/aarch64-linux-gnu/neat/runtime/libsimaaineatprofiler.so"
tensorbuffer_plugin="${SYSROOT}/usr/lib/aarch64-linux-gnu/neat/gst-plugins/libgstneattensorbuffer.so"

if [[ ! -d "${config_dir}" ]]; then
  echo "NeatInternals CMake package not found after install: ${config_dir}" >&2
  exit 1
fi
if [[ ! -f "${dispatcher_header}" ]]; then
  echo "dispatcher.h not found after install: ${dispatcher_header}" >&2
  exit 1
fi
if [[ ! -f "${profiler_lib}" ]]; then
  echo "NEAT runtime library not found after install: ${profiler_lib}" >&2
  exit 1
fi
if [[ ! -f "${tensorbuffer_plugin}" ]]; then
  echo "NEAT GStreamer plugin not found after install: ${tensorbuffer_plugin}" >&2
  exit 1
fi

log "Installed NEAT internals packages successfully."
find "${config_dir}" -type f | sort
