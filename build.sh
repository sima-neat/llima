#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${LLIMA_DEB_BUILD_DIR:-$ROOT_DIR/build-deb}"
BUILD_JOBS="${LLIMA_DEB_BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
NEAT_INTERNALS_BASE_URL="${NEAT_INTERNALS_BASE_URL:-https://artifacts.sima-neat.com/internals}"
NEAT_INTERNALS_ARCHIVE_URL="${NEAT_INTERNALS_ARCHIVE_URL:-}"
NEAT_INTERNALS_SOURCE="${NEAT_INTERNALS_SOURCE:-vulcan}"
NEAT_INTERNALS_VULCAN_REPOSITORY="${NEAT_INTERNALS_VULCAN_REPOSITORY:-internals}"
NEAT_INTERNALS_SNAP_POLICY="${NEAT_INTERNALS_SNAP_POLICY:-ON}"
NEAT_INTERNALS_MANIFEST="${NEAT_INTERNALS_MANIFEST:-${ROOT_DIR}/deps/manifest.json}"
NEAT_INTERNALS_RESOLVED_REF=""
NEAT_VULCAN_ENV="${NEAT_VULCAN_ENV:-dev}"
NEAT_VULCAN_BASE_URL="${NEAT_VULCAN_BASE_URL:-}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
ARCH=arm64
ELXR_SDK=OFF
ELXR_SDK_VERSION=""
ELXR_VERSION=""

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

detect_elxr_sdk() {
  ELXR_SDK=OFF
  ELXR_SDK_VERSION=""
  ELXR_VERSION=""

  if [[ -f "${ELXR_SDK_RELEASE_FILE}" ]]; then
    ELXR_SDK_VERSION="$(sed -n 's/^SDK Version[[:space:]]*=[[:space:]]*//p' "${ELXR_SDK_RELEASE_FILE}" | head -n1)"
    ELXR_VERSION="$(sed -n 's/^eLXr Version[[:space:]]*=[[:space:]]*//p' "${ELXR_SDK_RELEASE_FILE}" | head -n1)"
    if [[ -n "${ELXR_SDK_VERSION}" && -n "${ELXR_VERSION}" ]]; then
      ELXR_SDK=ON
      return
    fi
  fi

  if [[ -n "${SYSROOT:-}" && -d "${SYSROOT}" ]]; then
    ELXR_SDK=ON
  fi

  echo "[build] eLxr SDK mode: ${ELXR_SDK}"
  if [[ "${ELXR_SDK}" == "ON" ]]; then
    if [[ -n "${ELXR_SDK_VERSION}" || -n "${ELXR_VERSION}" ]]; then
      echo "[build]   SDK Version : ${ELXR_SDK_VERSION:-unknown}"
      echo "[build]   eLXr Version: ${ELXR_VERSION:-unknown}"
    fi
    echo "[build]   SYSROOT     : ${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  fi
}

apply_default_sdk_toolchain() {
  local toolchain_file="${ROOT_DIR}/toolchain-sima.cmake"

  if [[ "${ELXR_SDK}" != "ON" ]]; then
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
  for tool in cmake cpack git python3 dpkg-architecture dpkg-deb tar; do
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

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${out}"
    return $?
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "${out}" "${url}"
    return $?
  fi

  echo "ERROR: curl or wget is required to download build artifacts." >&2
  return 1
}

compute_sha256() {
  local path="$1"

  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${path}" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${path}" | awk '{print $1}'
    return 0
  fi

  echo "ERROR: sha256sum or shasum is required to verify build artifacts." >&2
  return 1
}

extract_json_string() {
  local key="$1"
  local file="$2"
  sed -n "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" "${file}" | head -n1
}

manifest_has_json_key() {
  local key="$1"
  local file="$2"
  python3 - "${key}" "${file}" <<'PY'
import json
import sys
from pathlib import Path

key = sys.argv[1]
manifest_path = Path(sys.argv[2])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
raise SystemExit(0 if key in data else 1)
PY
}

manifest_dependency_spec() {
  local key="$1"
  local file="$2"
  python3 - "${key}" "${file}" <<'PY'
import json
import sys
from pathlib import Path

key = sys.argv[1]
manifest_path = Path(sys.argv[2])
data = json.loads(manifest_path.read_text(encoding="utf-8"))
if key not in data:
    raise SystemExit(f"ERROR: {manifest_path} must define '{key}'.")

value = data[key]
if isinstance(value, str):
    print("__SNAP__" if not value.strip() else value.strip())
    raise SystemExit(0)

if isinstance(value, dict):
    policy = str(value.get("policy", "")).strip().lower()
    if policy == "snap":
        print("__SNAP__")
        raise SystemExit(0)
    if policy:
        raise SystemExit(f"ERROR: unsupported {key}.policy in {manifest_path}: {policy!r}")

    spec = str(value.get("spec", "")).strip()
    branch = str(value.get("branch", value.get("ref", ""))).strip()
    if branch:
        print(f"{branch}:{spec or 'latest'}")
        raise SystemExit(0)

raise SystemExit(
    f"ERROR: {manifest_path} field '{key}' must be a string, "
    "or an object with {'policy':'snap'} or {'branch':'...', 'spec':'...'}."
)
PY
}

sanitize_branch_key() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' |
    sed -E 's#[^a-z0-9._-]+#-#g; s/^-+//; s/-+$//'
}

current_branch_name() {
  if [[ -n "${GITHUB_HEAD_REF:-}" ]]; then
    printf '%s\n' "${GITHUB_HEAD_REF}"
    return 0
  fi
  if [[ -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 &&
     git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${ROOT_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null
    return 0
  fi
  printf '\n'
}

internals_checksum_available() {
  local ref="$1"
  local probe_path
  probe_path="$(mktemp /tmp/llima-neat-internals-probe.XXXXXX)"
  if download_file "${NEAT_INTERNALS_BASE_URL}/sima-neat-internals-${ref}.tar.gz.sha256" \
      "${probe_path}" >/dev/null 2>&1; then
    rm -f "${probe_path}"
    return 0
  fi
  rm -f "${probe_path}"
  return 1
}

resolve_neat_internals_ref() {
  if [[ ! -f "${NEAT_INTERNALS_MANIFEST}" ]]; then
    echo "ERROR: Missing manifest: ${NEAT_INTERNALS_MANIFEST}" >&2
    return 1
  fi

  if ! manifest_has_json_key "internals" "${NEAT_INTERNALS_MANIFEST}"; then
    echo "ERROR: ${NEAT_INTERNALS_MANIFEST} must define an internals dependency." >&2
    return 1
  fi

  local manifest_ref
  if ! manifest_ref="$(manifest_dependency_spec "internals" "${NEAT_INTERNALS_MANIFEST}")"; then
    return 1
  fi
  if [[ "${manifest_ref}" != "__SNAP__" ]]; then
    case "${manifest_ref}" in
      *:*)
        printf '%s\n' "${manifest_ref}"
        ;;
      *-latest)
        printf '%s:latest\n' "${manifest_ref%-latest}"
        ;;
      *)
        printf '%s\n' "${manifest_ref}"
        ;;
    esac
    return 0
  fi

  local branch branch_key
  branch="$(current_branch_name)"
  branch_key="$(sanitize_branch_key "${branch}")"
  if [[ -n "${branch_key}" && "${branch_key}" != "head" ]]; then
    printf '%s\n' "${branch_key}:latest"
    return 0
  fi

  echo "Could not determine current branch for internals snap; using develop:latest." >&2
  printf '%s\n' "develop:latest"
}

resolve_neat_internals_archive_ref() {
  if [[ ! -f "${NEAT_INTERNALS_MANIFEST}" ]]; then
    echo "ERROR: Missing manifest: ${NEAT_INTERNALS_MANIFEST}" >&2
    return 1
  fi

  if ! manifest_has_json_key "internals" "${NEAT_INTERNALS_MANIFEST}"; then
    echo "ERROR: ${NEAT_INTERNALS_MANIFEST} must define an internals dependency." >&2
    return 1
  fi

  local manifest_ref
  if ! manifest_ref="$(manifest_dependency_spec "internals" "${NEAT_INTERNALS_MANIFEST}")"; then
    return 1
  fi
  if [[ "${manifest_ref}" != "__SNAP__" ]]; then
    printf '%s\n' "${manifest_ref/:/-}"
    return 0
  fi

  local branch branch_key candidate
  branch="$(current_branch_name)"
  branch_key="$(sanitize_branch_key "${branch}")"
  if [[ -n "${branch_key}" && "${branch_key}" != "head" ]]; then
    candidate="${branch_key}-latest"
    if internals_checksum_available "${candidate}"; then
      echo "Resolved empty internals manifest to matching branch artifact: ${candidate}" >&2
      printf '%s\n' "${candidate}"
      return 0
    fi
    echo "No internals artifact found for branch '${branch}' (${candidate}); using develop-latest." >&2
  else
    echo "Could not determine current branch for internals snap; using develop-latest." >&2
  fi

  printf '%s\n' "develop-latest"
}

resolve_neat_internals_archive_url() {
  if [[ -n "${NEAT_INTERNALS_ARCHIVE_URL}" ]]; then
    printf '%s\n' "${NEAT_INTERNALS_ARCHIVE_URL}"
    return
  fi

  local internals_ref
  if ! internals_ref="$(resolve_neat_internals_archive_ref)"; then
    return 1
  fi
  NEAT_INTERNALS_RESOLVED_REF="${internals_ref}"
  printf '%s/sima-neat-internals-%s.tar.gz\n' "${NEAT_INTERNALS_BASE_URL}" "${internals_ref}"
}

require_sima_cli_vulcan_install() {
  if ! command -v sima-cli >/dev/null 2>&1; then
    echo "ERROR: sima-cli is required for Vulcan internals artifact access." >&2
    exit 1
  fi
  if ! sima-cli vulcan install --help >/dev/null 2>&1; then
    echo "ERROR: sima-cli with Vulcan install support is required." >&2
    exit 1
  fi
}

fetch_neat_internals_vulcan_artifacts() {
  local internals_ref="$1"
  local output_dir="$2"

  require_sima_cli_vulcan_install

  local -a base_args=(
    vulcan
    --env "${NEAT_VULCAN_ENV}"
  )
  if [[ -n "${NEAT_VULCAN_BASE_URL}" ]]; then
    base_args+=(--base-url "${NEAT_VULCAN_BASE_URL}")
  fi

  local resolve_output resolved_ref
  if ! resolve_output="$(sima-cli "${base_args[@]}" install "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" --json)"; then
    if [[ "${NEAT_INTERNALS_SNAP_POLICY}" != "ON" || "${internals_ref}" == "develop:latest" ]]; then
      echo "ERROR: Failed to resolve internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi

    echo "No internals Vulcan artifact found for '${internals_ref}'; retrying develop:latest." >&2
    internals_ref="develop:latest"
    if ! resolve_output="$(sima-cli "${base_args[@]}" install "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" --json)"; then
      echo "ERROR: Failed to resolve fallback internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi
  fi

  resolved_ref="$(python3 - <<'PY' "${resolve_output}"
import json
import sys

text = sys.argv[1]
start = text.find("{")
if start < 0:
    raise SystemExit("missing JSON object in sima-cli vulcan install --json output")
payload = json.loads(text[start:])
ref = str(payload.get("ref", "")).strip()
spec = str(payload.get("resolved_spec", "")).strip()
if not ref or not spec:
    raise SystemExit("sima-cli vulcan install --json did not return ref and resolved_spec")
print(f"{ref}:{spec}")
PY
)"
  NEAT_INTERNALS_RESOLVED_REF="${resolved_ref}"

  local -a install_args=(
    "${base_args[@]}"
    install
    -d "${output_dir}"
    "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}"
  )

  echo "[build] Fetching NEAT internals packages from Vulcan:"
  echo "[build]   ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}"
  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  if ! sima-cli "${install_args[@]}"; then
    echo "ERROR: Failed to fetch internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}" >&2
    exit 1
  fi
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

  if [[ "${ELXR_SDK}" != "ON" ]]; then
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

  ensure_sdk_sysroot_header_package "${sysroot}" "libeigen3-dev" "Eigen" \
    "${sysroot}/usr/include/eigen3/unsupported/Eigen/CXX11/Tensor" \
    "${sysroot}/usr/share/eigen3/cmake/Eigen3Config.cmake"
  ensure_sdk_sysroot_header_package "${sysroot}" "nlohmann-json3-dev" "nlohmann_json" \
    "${sysroot}/usr/include/nlohmann/json.hpp" \
    "${sysroot}/usr/share/cmake/nlohmann_json/nlohmann_jsonConfig.cmake"
  ensure_sdk_sysroot_header_package "${sysroot}" "libfmt-dev:arm64" "fmt" \
    "${sysroot}/usr/include/fmt/core.h" \
    "${sysroot}/usr/lib/aarch64-linux-gnu/cmake/fmt/fmt-config.cmake" \
    "${sysroot}/usr/lib/aarch64-linux-gnu/cmake/fmt/fmtConfig.cmake"
  ensure_sdk_sysroot_header_package "${sysroot}" "libspdlog-dev:arm64" "spdlog" \
    "${sysroot}/usr/include/spdlog/spdlog.h" \
    "${sysroot}/usr/lib/aarch64-linux-gnu/cmake/spdlog/spdlogConfig.cmake"
  ensure_sdk_sysroot_header_package "${sysroot}" "libbrotli-dev:arm64" "brotli" \
    "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlicommon.pc" \
    "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlidec.pc" \
    "${sysroot}/usr/lib/aarch64-linux-gnu/pkgconfig/libbrotlienc.pc"
  ensure_sdk_sysroot_header_package "${sysroot}" "libcpp-httplib-dev:arm64" "cpp-httplib" \
    "${sysroot}/usr/include/httplib.h"

  local libdir="${sysroot}/usr/lib/aarch64-linux-gnu"
  local packages=()

  path_exists_any "${libdir}/libopencv_flann.so.406*" ||
    packages+=(libopencv-flann406:arm64)
  path_exists_any "${libdir}/libopencv_dnn.so.406*" ||
    packages+=(libopencv-dnn406:arm64)
  path_exists_any "${libdir}/libopencv_features2d.so.406*" ||
    packages+=(libopencv-features2d406:arm64)
  path_exists_any "${libdir}/libopencv_objdetect.so.406*" ||
    packages+=(libopencv-objdetect406:arm64)
  path_exists_any "${libdir}/libopencv_video.so.406*" ||
    packages+=(libopencv-video406:arm64)
  if [[ ! -f "${sysroot}/usr/include/openssl/ssl.h" ||
        ! -e "${libdir}/libcrypto.so" ]]; then
    packages+=(libssl-dev:arm64)
  fi
  path_exists_any "${libdir}/libfmt.so.9.1.0" ||
    packages+=(libfmt9:arm64)
  path_exists_any "${libdir}/libspdlog.so.1.10.0" ||
    packages+=(libspdlog1.10:arm64)
  path_exists_any "${libdir}/libcpp-httplib.so.0.11*" ||
    packages+=(libcpp-httplib0.11:arm64)
  path_exists_any "${libdir}/libpgm*.so" "${libdir}/libpgm*.so.*" ||
    packages+=(libpgm-dev:arm64)

  if [[ "${#packages[@]}" -eq 0 ]]; then
    echo "[build] llima SDK sysroot package overlay already present"
    return
  fi

  echo "[build] Installing missing llima SDK sysroot package overlay payloads"
  run_as_root "${overlay_script}" "${sysroot}" "${packages[@]}"
}

ensure_sdk_sysroot_header_package() {
  local sysroot="$1"
  local package="$2"
  local label="$3"
  shift 3

  if path_exists_any "$@"; then
    return
  fi

  if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "ERROR: apt-get and dpkg-deb are required to install ${package} into the SDK sysroot." >&2
    exit 1
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/llima-sysroot-header.XXXXXX)"

  echo "[build] Installing SDK sysroot header package: ${package} (${label})"
  (
    cd "${tmp_dir}"
    apt-get download "${package}"
  )

  local deb
  local package_deb_name
  package_deb_name="${package%%:*}"
  deb="$(find "${tmp_dir}" -maxdepth 1 -type f -name "${package_deb_name}_*.deb" | sort | head -n 1)"
  if [[ -z "${deb}" ]]; then
    echo "ERROR: Failed to download ${package}." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  run_as_root dpkg-deb -x "${deb}" "${sysroot}"
  rm -rf "${tmp_dir}"
}

path_exists_any() {
  local pattern
  for pattern in "$@"; do
    if compgen -G "${pattern}" >/dev/null; then
      return 0
    fi
  done
  return 1
}

ensure_neat_internals() {
  local sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  local tmp_dir
  tmp_dir="$(mktemp -d /tmp/llima-neat-internals.XXXXXX)"
  local extract_dir="${tmp_dir}/extract"
  local archive_name="Vulcan internals artifact"

  if [[ -n "${NEAT_INTERNALS_ARCHIVE_URL}" || "${NEAT_INTERNALS_SOURCE}" == "archive" ]]; then
    local archive_url
    if ! archive_url="$(resolve_neat_internals_archive_url)"; then
      exit 1
    fi
    archive_name="$(basename "${archive_url}")"

    if [[ -z "${archive_url}" ]]; then
      echo "ERROR: NEAT_INTERNALS_ARCHIVE_URL resolved to an empty URL." >&2
      exit 1
    fi

    local archive_path="${tmp_dir}/${archive_name}"
    local checksum_path="${archive_path}.sha256"

    echo "[build] Downloading NEAT internals artifact:"
    echo "[build]   ${archive_url}"
    download_file "${archive_url}" "${archive_path}"
    download_file "${archive_url}.sha256" "${checksum_path}"

    local expected_sha actual_sha
    expected_sha="$(awk '{print $1}' "${checksum_path}" | tr -d '[:space:]' | head -n1)"
    if [[ -z "${expected_sha}" || ! "${expected_sha}" =~ ^[0-9a-fA-F]{64}$ ]]; then
      echo "ERROR: Invalid sha256 content in ${archive_url}.sha256" >&2
      exit 1
    fi
    actual_sha="$(compute_sha256 "${archive_path}")"
    if [[ "${actual_sha}" != "${expected_sha}" ]]; then
      echo "ERROR: sha256 mismatch for ${archive_name}" >&2
      echo "  expected: ${expected_sha}" >&2
      echo "  actual  : ${actual_sha}" >&2
      exit 1
    fi

    mkdir -p "${extract_dir}"
    tar -xzf "${archive_path}" -C "${extract_dir}"
  else
    local internals_ref
    if ! internals_ref="$(resolve_neat_internals_ref)"; then
      exit 1
    fi
    extract_dir="${tmp_dir}/package"
    fetch_neat_internals_vulcan_artifacts "${internals_ref}" "${extract_dir}"
  fi

  local deb_pattern_groups=(
    'neat-common_*_all.deb simaai-common_*_all.deb'
    'neat-runtime_*_arm64.deb'
    'neat-gst-plugins_*_arm64.deb'
    'neat-internals-dev_*_arm64.deb'
    'neat-appcomplex_*_arm64.deb appcomplex_*_arm64.deb'
  )
  local debs=()
  local pattern_group pattern deb
  for pattern_group in "${deb_pattern_groups[@]}"; do
    deb=""
    for pattern in ${pattern_group}; do
      deb="$(find "${extract_dir}" -maxdepth 3 -type f -name "${pattern}" | sort | head -n 1)"
      if [[ -n "${deb}" ]]; then
        break
      fi
    done
    if [[ -z "${deb}" ]]; then
      echo "ERROR: No matching deb found in ${archive_name}; expected one of: ${pattern_group}" >&2
      exit 1
    fi
    debs+=("${deb}")
  done

  if [[ "${ELXR_SDK}" == "ON" ]]; then
    if [[ ! -d "${sysroot}" ]]; then
      echo "ERROR: SYSROOT does not exist: ${sysroot}" >&2
      exit 1
    fi
    echo "[build] Installing NEAT internals deb payloads into SDK sysroot:"
    echo "[build]   ${sysroot}"
    for deb in "${debs[@]}"; do
      echo "[build]   $(basename "${deb}")"
      run_as_root dpkg-deb -x "${deb}" "${sysroot}"
    done
  else
    if ! command -v apt >/dev/null 2>&1; then
      echo "ERROR: apt is required to install NEAT internals deb packages outside SDK mode." >&2
      exit 1
    fi
    echo "[build] Installing NEAT internals deb packages into host system"
    run_as_root apt install -y --allow-downgrades "${debs[@]}"
  fi

  local config_dir dispatcher_header profiler_lib tensorbuffer_plugin
  if [[ "${ELXR_SDK}" == "ON" ]]; then
    config_dir="${sysroot}/usr/lib/aarch64-linux-gnu/cmake/NeatInternals"
    dispatcher_header="${sysroot}/usr/include/dispatcher.h"
    profiler_lib="${sysroot}/usr/lib/aarch64-linux-gnu/neat/runtime/libsimaaineatprofiler.so"
    tensorbuffer_plugin="${sysroot}/usr/lib/aarch64-linux-gnu/neat/gst-plugins/libgstneattensorbuffer.so"
  else
    config_dir="/usr/lib/aarch64-linux-gnu/cmake/NeatInternals"
    dispatcher_header="/usr/include/dispatcher.h"
    profiler_lib="/usr/lib/aarch64-linux-gnu/neat/runtime/libsimaaineatprofiler.so"
    tensorbuffer_plugin="/usr/lib/aarch64-linux-gnu/neat/gst-plugins/libgstneattensorbuffer.so"
  fi

  if [[ ! -d "${config_dir}" ]]; then
    echo "ERROR: NeatInternals CMake package not found after install: ${config_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${dispatcher_header}" ]]; then
    echo "ERROR: dispatcher.h not found after install: ${dispatcher_header}" >&2
    exit 1
  fi
  if [[ ! -f "${profiler_lib}" ]]; then
    echo "ERROR: NEAT runtime library not found after install: ${profiler_lib}" >&2
    exit 1
  fi
  if [[ ! -f "${tensorbuffer_plugin}" ]]; then
    echo "ERROR: NEAT GStreamer plugin not found after install: ${tensorbuffer_plugin}" >&2
    exit 1
  fi

  rm -rf "${tmp_dir}"
  echo "[build] NEAT internals are ready."
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
detect_elxr_sdk
ensure_neat_internals
apply_default_sdk_toolchain
ensure_sdk_sysroot_packages

LLIMA_VERSION="$(version_from_version_in)"
MULTIARCH="$(dpkg-architecture -a"$ARCH" -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
if [ -z "$MULTIARCH" ]; then
  MULTIARCH="aarch64-linux-gnu"
fi
PYTHON_ABI_TAG="cpython-311"
PYTHON_TARGET_SOABI="${PYTHON_ABI_TAG}-${MULTIARCH}"

if [ "$DO_CLEAN" -eq 1 ]; then
  echo "[build] Removing build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  rm -rf "$ROOT_DIR/dist"
  rm -f "$ROOT_DIR"/sima-lmm*.deb
fi

ensure_writable_cargo_home
ensure_python_build_env
CMAKE_SOABI_ARGS=()
if [[ "${ELXR_SDK}" == "ON" ]]; then
  CMAKE_SOABI_ARGS+=("-DSKBUILD_SOABI=cpython-311-${MULTIARCH}")
fi

echo "[build] Configuring sima-lmm $LLIMA_VERSION for arch=$ARCH"
echo "[build] Python extension SOABI: $PYTHON_TARGET_SOABI"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_INSTALL_LIBDIR="lib/$MULTIARCH" \
  -DPython_EXECUTABLE="$BUILD_VENV/bin/python" \
  -DSKBUILD_SOABI="$PYTHON_TARGET_SOABI" \
  -DSIMA_LMM_BUILD_PYTHON=ON \
  -DSIMA_LMM_INSTALL_PYTHON_PACKAGE=ON \
  -DSIMA_LMM_PYTHON_EXTENSION_INSTALL_DIR="lib/python3/dist-packages/sima_lmm/devkit" \
  -DSIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR="lib/python3/dist-packages/sima_lmm" \
  -DCPACK_DEBIAN_PACKAGE_ARCHITECTURE="$ARCH" \
  "${CMAKE_SOABI_ARGS[@]}" \
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
