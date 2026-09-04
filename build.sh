#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${LLIMA_DEB_BUILD_DIR:-$ROOT_DIR/build-deb}"
BUILD_JOBS="${LLIMA_DEB_BUILD_JOBS:-${CMAKE_BUILD_PARALLEL_LEVEL:-}}"
NEAT_INTERNALS_VULCAN_REPOSITORY="${NEAT_INTERNALS_VULCAN_REPOSITORY:-internals}"
NEAT_INTERNALS_SNAP_POLICY="${NEAT_INTERNALS_SNAP_POLICY:-ON}"
NEAT_INTERNALS_MANIFEST="${NEAT_INTERNALS_MANIFEST:-${ROOT_DIR}/deps/manifest.json}"
NEAT_INTERNALS_PACKAGE_DIR="${NEAT_INTERNALS_PACKAGE_DIR:-}"
NEAT_INTERNALS_RESOLVED_REF="${NEAT_INTERNALS_RESOLVED_REF:-}"
NEAT_INTERNALS_DEB_DIR="${NEAT_INTERNALS_DEB_DIR:-}"
NEAT_INTERNALS_RESOLVED_MANIFEST="${NEAT_INTERNALS_RESOLVED_MANIFEST:-}"
NEAT_VULCAN_ENV="${NEAT_VULCAN_ENV:-prod}"
NEAT_VULCAN_BASE_URL="${NEAT_VULCAN_BASE_URL:-}"
LLIMA_INSTALL_SCRIPT="install_llima.sh"
LLIMA_INSTALL_MANIFEST="llima-install-manifest.txt"
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
  --all               Build all packages, DevKit runtime tests, and publishable layouts (default)
  --no-dist           Skip publishable artifact layout creation
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
BUILD_RUNTIME_TESTS=ON
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

compute_package_version() {
  if [[ -n "${LLIMA_PACKAGE_VERSION:-}" ]]; then
    printf '%s\n' "${LLIMA_PACKAGE_VERSION}"
    return 0
  fi
  bash "${ROOT_DIR}/tools/compute_package_version.sh"
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

current_exact_tag() {
  if [[ "${GITHUB_REF_TYPE:-}" == "tag" && -n "${GITHUB_REF_NAME:-}" ]]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return 0
  fi
  if command -v git >/dev/null 2>&1 &&
     git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "${ROOT_DIR}" describe --tags --exact-match HEAD 2>/dev/null || true
    return 0
  fi
  printf '\n'
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

  local branch tag
  tag="$(current_exact_tag)"
  if [[ -n "${tag}" ]]; then
    printf '%s\n' "${tag}:latest"
    return 0
  fi

  branch="$(current_branch_name)"
  if [[ -n "${branch}" && "${branch}" != "HEAD" ]]; then
    printf '%s\n' "${branch}:latest"
    return 0
  fi

  echo "Could not determine current branch for internals snap; using develop:latest." >&2
  printf '%s\n' "develop:latest"
}

require_sima_cli_neat_install() {
  if ! command -v sima-cli >/dev/null 2>&1; then
    echo "ERROR: sima-cli is required for Vulcan internals artifact access." >&2
    exit 1
  fi
  if ! SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli neat install --help >/dev/null 2>&1; then
    echo "ERROR: sima-cli with Neat artifact install support is required." >&2
    exit 1
  fi
}

fetch_neat_internals_vulcan_artifacts() {
  local internals_ref="$1"
  local output_dir="$2"

  require_sima_cli_neat_install

  local -a base_args=(
    neat
    install
    --env "${NEAT_VULCAN_ENV}"
  )
  if [[ -n "${NEAT_VULCAN_BASE_URL}" ]]; then
    base_args+=(--base-url "${NEAT_VULCAN_BASE_URL}")
  fi

  local exact_tag resolve_output resolved_ref
  if ! resolve_output="$(SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli "${base_args[@]}" "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" --json)"; then
    exact_tag="$(current_exact_tag)"
    if [[ -n "${exact_tag}" && "${internals_ref}" == "${exact_tag}:latest" ]]; then
      echo "ERROR: Failed to resolve exact tag-snap internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi
    if [[ "${NEAT_INTERNALS_SNAP_POLICY}" != "ON" || "${internals_ref}" == "develop:latest" ]]; then
      echo "ERROR: Failed to resolve internals Vulcan artifact: ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" >&2
      exit 1
    fi

    echo "No internals Vulcan artifact found for '${internals_ref}'; retrying develop:latest." >&2
    internals_ref="develop:latest"
    if ! resolve_output="$(SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli "${base_args[@]}" "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${internals_ref}" --json)"; then
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
    raise SystemExit("missing JSON object in sima-cli neat install --json output")
payload = json.loads(text[start:])
ref = str(payload.get("ref", "")).strip()
spec = str(payload.get("resolved_spec", "")).strip()
if not ref or not spec:
    raise SystemExit("sima-cli neat install --json did not return ref and resolved_spec")
print(f"{ref}:{spec}")
PY
)"
  NEAT_INTERNALS_RESOLVED_REF="${resolved_ref}"

  local -a install_args=(
    "${base_args[@]}"
    -d "${output_dir}"
    "${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}"
  )

  echo "[build] Fetching NEAT internals packages from Vulcan:"
  echo "[build]   ${NEAT_INTERNALS_VULCAN_REPOSITORY}@${resolved_ref}"
  rm -rf "${output_dir}"
  mkdir -p "${output_dir}"
  if ! SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli "${install_args[@]}"; then
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
    echo "[build] Synchronizing git submodule URLs"
    git -C "$ROOT_DIR" submodule sync --recursive
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

  if [[ "${ELXR_SDK}" != "ON" ]]; then
    return
  fi
  if [[ "${LLIMA_SKIP_SYSROOT_OVERLAY:-0}" == "1" ]]; then
    echo "[build] Skipping SDK sysroot package overlay"
    return
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
    packages+=(libpgm-5.3-0:arm64 libpgm-dev:arm64)

  if [[ "${#packages[@]}" -eq 0 ]]; then
    echo "[build] llima SDK sysroot package overlay already present"
    return
  fi

  # The SDK should eventually provide this complete dependency contract:
  # https://github.com/sima-neat/sdk/issues/164
  # Until then, install only the missing package payloads. The SDK overlay
  # installer performs whole-sysroot repair work that is unnecessary here.
  echo "[build] Installing missing LLiMa SDK sysroot package payloads"
  install_sdk_sysroot_package_payloads "${sysroot}" "${packages[@]}"

  if ! path_exists_any "${libdir}/libopencv_flann.so.406*" ||
     ! path_exists_any "${libdir}/libopencv_dnn.so.406*" ||
     ! path_exists_any "${libdir}/libopencv_features2d.so.406*" ||
     ! path_exists_any "${libdir}/libopencv_objdetect.so.406*" ||
     ! path_exists_any "${libdir}/libopencv_video.so.406*" ||
     [[ ! -f "${sysroot}/usr/include/openssl/ssl.h" ]] ||
     [[ ! -e "${libdir}/libcrypto.so" ]] ||
     ! path_exists_any "${libdir}/libfmt.so.9.1.0" ||
     ! path_exists_any "${libdir}/libspdlog.so.1.10.0" ||
     ! path_exists_any "${libdir}/libcpp-httplib.so.0.11*" ||
     ! path_exists_any "${libdir}/libpgm*.so" "${libdir}/libpgm*.so.*"; then
    echo "ERROR: LLiMa SDK sysroot package payloads are incomplete after install." >&2
    exit 1
  fi
}

install_sdk_sysroot_package_payloads() {
  local sysroot="$1"
  shift

  if [[ "$#" -eq 0 ]]; then
    return
  fi
  if ! command -v apt-get >/dev/null 2>&1 || ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "ERROR: apt-get and dpkg-deb are required to install packages into the SDK sysroot." >&2
    exit 1
  fi

  local tmp_dir
  local payload_root
  tmp_dir="$(mktemp -d /tmp/llima-sysroot-packages.XXXXXX)"
  payload_root="${tmp_dir}/payload"
  mkdir -p "${payload_root}"

  if ! (
    cd "${tmp_dir}"
    apt-get download "$@"
  ); then
    echo "ERROR: Failed to download SDK sysroot package payloads." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  local -a debs=()
  local deb
  while IFS= read -r deb; do
    debs+=("${deb}")
  done < <(find "${tmp_dir}" -maxdepth 1 -type f -name '*.deb' | sort)
  if [[ "${#debs[@]}" -eq 0 ]]; then
    echo "ERROR: SDK sysroot package download produced no Debian packages." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  for deb in "${debs[@]}"; do
    echo "[build]   $(basename "${deb}")"
    if ! dpkg-deb -x "${deb}" "${payload_root}"; then
      echo "ERROR: Failed to extract $(basename "${deb}")." >&2
      rm -rf "${tmp_dir}"
      exit 1
    fi
  done

  # Normalize only the payload being added. Do not traverse or chmod the
  # existing SDK sysroot. Copy without archive/preserve semantics so a
  # non-root staging user's ownership and directory attributes are not
  # applied to shared sysroot paths. Use a known umask so the privileged copy
  # cannot remove the read/traverse access established on the payload.
  chmod -R a+rX "${payload_root}"
  if ! run_as_root bash -c 'umask 022; cp -R "$1/." "$2/"' \
    bash "${payload_root}" "${sysroot}"; then
    echo "ERROR: Failed to install package payloads into SYSROOT=${sysroot}." >&2
    rm -rf "${tmp_dir}"
    exit 1
  fi

  rm -rf "${tmp_dir}"
}

ensure_sdk_sysroot_header_package() {
  local sysroot="$1"
  local package="$2"
  local label="$3"
  shift 3

  if path_exists_any "$@"; then
    return
  fi

  echo "[build] Installing SDK sysroot header package: ${package} (${label})"
  install_sdk_sysroot_package_payloads "${sysroot}" "${package}"
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

validate_neat_internals_payload() {
  local root="${1%/}"
  local source_name="$2"
  local path
  local -a missing=()
  local -a required=(
    "/usr/lib/aarch64-linux-gnu/cmake/NeatInternals/NeatInternalsConfig.cmake"
    "/usr/lib/aarch64-linux-gnu/cmake/NeatInternals/NeatInternalsTargets.cmake"
    "/usr/include/dispatcherfactory.hh"
    "/usr/include/dispatcherbase.hh"
    "/usr/lib/aarch64-linux-gnu/neat/runtime/libneatdispatchercore.so"
  )

  for path in "${required[@]}"; do
    [[ -e "${root}${path}" ]] || missing+=("${path}")
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    return 0
  fi

  echo "ERROR: ${source_name} is incomplete." >&2
  echo "Missing:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  return 1
}

sync_sysroot_from_internals_manifest() {
  local artifact_dir="$1"
  [[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]] || return 0

  if [[ "${ELXR_SDK}" != "ON" ]]; then
    echo "ERROR: NEAT_SYNC_SYSROOT requires an eLxr SDK." >&2
    exit 1
  fi

  local artifact_manifest="${artifact_dir}/internals-manifest.json"
  if [[ ! -f "${artifact_manifest}" ]]; then
    echo "ERROR: Internals artifact is missing internals-manifest.json." >&2
    exit 1
  fi

  local receipt
  if ! receipt="$(python3 -c '
import json, re, sys
artifact = json.load(open(sys.argv[1], encoding="utf-8"))
consumer = json.load(open(sys.argv[2], encoding="utf-8"))
receipt = artifact["sysroot-version"]
consumer_base = consumer["platform-version"]
if not isinstance(receipt, str) or (
    receipt
    and not re.fullmatch(r"[0-9]+(?:[.][0-9]+){2}(?:~pre[0-9]+)?", receipt)
):
    raise ValueError("invalid sysroot-version")
if receipt and consumer_base != receipt.split("~pre", 1)[0]:
    raise ValueError("platform-version does not match the Internals receipt")
print(receipt)
' "${artifact_manifest}" "${NEAT_INTERNALS_MANIFEST}")"; then
    echo "ERROR: Cannot read Internals build receipt." >&2
    exit 1
  fi
  if [[ -z "${receipt}" ]]; then
    echo "LLiMa is using the existing SDK sysroot."
    return 0
  fi

  if [[ "${receipt}" == *"~pre"* ]]; then
    echo "[build] Updating SDK sysroot to Internals receipt ${receipt}"
    run_as_root sysroot update "${receipt}"
    sysroot status
    return 0
  fi

  local sdk_platform_version
  sdk_platform_version="$(sed -nE \
    's/^Platform Version[[:space:]]*=[[:space:]]*([^[:space:]]+).*$/\1/p' \
    "${ELXR_SDK_RELEASE_FILE}" 2>/dev/null | head -n1 || true)"
  if [[ "${sdk_platform_version}" != "${receipt}" ]]; then
    echo "ERROR: SDK platform ${sdk_platform_version:-unknown} does not match required stable platform ${receipt}." >&2
    exit 1
  fi
  echo "[build] Using stable SDK sysroot ${sdk_platform_version} without updating it."
}

ensure_neat_internals() {
  local sysroot="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  local tmp_dir
  tmp_dir=""
  local extract_dir
  local payload_root
  local archive_name="Vulcan internals artifact"
  local -a all_debs=()

  if [[ -n "${NEAT_INTERNALS_PACKAGE_DIR}" ]]; then
    if [[ ! -d "${NEAT_INTERNALS_PACKAGE_DIR}" ]]; then
      echo "ERROR: NEAT_INTERNALS_PACKAGE_DIR does not exist: ${NEAT_INTERNALS_PACKAGE_DIR}" >&2
      exit 1
    fi
    extract_dir="${NEAT_INTERNALS_PACKAGE_DIR}"
    archive_name="local internals package directory"
    echo "[build] Using local NEAT internals packages: ${extract_dir}"
  else
    tmp_dir="$(mktemp -d /tmp/llima-neat-internals.XXXXXX)"
    extract_dir="${tmp_dir}/package"
    local internals_ref
    if ! internals_ref="$(resolve_neat_internals_ref)"; then
      exit 1
    fi
    fetch_neat_internals_vulcan_artifacts "${internals_ref}" "${extract_dir}"
  fi

  mapfile -t all_debs < <(find "${extract_dir}" -type f -name '*.deb' | sort)
  if [[ "${#all_debs[@]}" -eq 0 ]]; then
    echo "ERROR: ${archive_name} did not contain any Debian packages." >&2
    [[ -z "${tmp_dir}" ]] || rm -rf "${tmp_dir}"
    exit 1
  fi

  sync_sysroot_from_internals_manifest "${extract_dir}"

  mkdir -p "${NEAT_INTERNALS_DEB_DIR}"
  rm -f "${NEAT_INTERNALS_DEB_DIR}"/*.deb
  local source_deb cached_deb
  for source_deb in "${all_debs[@]}"; do
    cached_deb="${NEAT_INTERNALS_DEB_DIR}/$(basename "${source_deb}")"
    if [[ -e "${cached_deb}" ]]; then
      echo "ERROR: Duplicate Internals package basename: $(basename "${source_deb}")" >&2
      [[ -z "${tmp_dir}" ]] || rm -rf "${tmp_dir}"
      exit 1
    fi
    cp -f "${source_deb}" "${cached_deb}"
  done
  echo "[build] Cached ${#all_debs[@]} NEAT internals Debian package(s):"
  find "${NEAT_INTERNALS_DEB_DIR}" -maxdepth 1 -type f -name '*.deb' -printf '[build]   %f\n' | sort

  local debs=()
  local deb
  mapfile -t debs < <(find "${NEAT_INTERNALS_DEB_DIR}" -maxdepth 1 -type f -name '*.deb' | sort)

  if [[ "${ELXR_SDK}" == "ON" ]]; then
    if [[ ! -d "${sysroot}" ]]; then
      echo "ERROR: SYSROOT does not exist: ${sysroot}" >&2
      exit 1
    fi

    payload_root="$(mktemp -d /tmp/llima-neat-internals-payload.XXXXXX)"
    for deb in "${debs[@]}"; do
      dpkg-deb -x "${deb}" "${payload_root}"
    done
    if ! validate_neat_internals_payload "${payload_root}" "${archive_name}"; then
      rm -rf "${payload_root}"
      [[ -z "${tmp_dir}" ]] || rm -rf "${tmp_dir}"
      exit 1
    fi
    rm -rf "${payload_root}"

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
    validate_neat_internals_payload "" "installed NEAT internals packages"
  fi

  if [[ -n "${tmp_dir}" ]]; then
    rm -rf "${tmp_dir}"
  fi
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

write_resolved_deps_manifest() {
  mkdir -p "$(dirname "${NEAT_INTERNALS_RESOLVED_MANIFEST}")"
  python3 - \
    "${NEAT_INTERNALS_MANIFEST}" \
    "${NEAT_INTERNALS_RESOLVED_REF}" \
    "${NEAT_INTERNALS_RESOLVED_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
resolved_ref = sys.argv[2].strip()
output_path = Path(sys.argv[3])

manifest = json.loads(source_path.read_text(encoding="utf-8"))
if resolved_ref:
    manifest["internals"] = resolved_ref
output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
  echo "[build] Resolved dependency manifest: ${NEAT_INTERNALS_RESOLVED_MANIFEST}"
}

required_llima_debs() {
  local version="$1"
  printf '%s\n' \
    "sima-lmm-${version}-Linux-cli.deb" \
    "sima-lmm-${version}-Linux-core.deb" \
    "sima-lmm-${version}-Linux-dev.deb"
}

verify_required_llima_debs() {
  local version="$1"
  local directory="$2"
  local deb
  while IFS= read -r deb; do
    if [[ ! -f "${directory}/${deb}" ]]; then
      echo "Expected Debian package not found: ${directory}/${deb}" >&2
      find "${directory}" -maxdepth 1 -type f -name 'sima-lmm*.deb' -printf '  %p\n' | sort >&2
      return 1
    fi
  done < <(required_llima_debs "${version}")
}

write_install_manifest() {
  local manifest_path="${ROOT_DIR}/dist/${LLIMA_INSTALL_MANIFEST}"
  local file
  {
    echo "# Generated by build.sh. The installer only consumes files listed here."
    echo "# Keep this file next to ${LLIMA_INSTALL_SCRIPT}."
  } > "${manifest_path}"

  while IFS= read -r file; do
    basename "${file}" >> "${manifest_path}"
  done < <(find "${ROOT_DIR}/dist" -maxdepth 1 -type f -name '*.deb' | sort)
  echo "[build] Install manifest: ${manifest_path}"
}

stage_package_artifacts() {
  if [ "${#COMPONENTS[@]}" -gt 0 ]; then
    echo "[build] Skipping publishable artifact layouts because a subset of components was selected"
    return
  fi

  if [[ -n "${NEAT_INTERNALS_PACKAGE_DIR}" &&
        ! "${NEAT_INTERNALS_RESOLVED_REF}" =~ ^[^:[:space:]]+:[^:[:space:]]+$ ]]; then
    echo "ERROR: NEAT_INTERNALS_RESOLVED_REF must be an exact branch:spec reference when NEAT_INTERNALS_PACKAGE_DIR is used for a publishable artifact." >&2
    return 1
  fi

  local version="$1"
  local deb
  local -a internals_debs=()

  verify_required_llima_debs "${version}" "${ROOT_DIR}/dist"
  mapfile -t internals_debs < <(find "${NEAT_INTERNALS_DEB_DIR}" -maxdepth 1 -type f -name '*.deb' | sort)
  if [[ "${#internals_debs[@]}" -eq 0 ]]; then
    echo "ERROR: No cached Internals Debian packages found in ${NEAT_INTERNALS_DEB_DIR}." >&2
    return 1
  fi

  find "${ROOT_DIR}/dist" -maxdepth 1 -type f -name '*.deb' ! -name 'sima-lmm-*.deb' -delete
  rm -f \
    "${ROOT_DIR}/dist/${LLIMA_INSTALL_SCRIPT}" \
    "${ROOT_DIR}/dist/${LLIMA_INSTALL_MANIFEST}" \
    "${ROOT_DIR}/dist/resolved-deps-manifest.json" \
    "${ROOT_DIR}/dist/metadata.json"
  rm -rf "${ROOT_DIR}/dist/debs"

  for deb in "${internals_debs[@]}"; do
    cp -f "${deb}" "${ROOT_DIR}/dist/$(basename "${deb}")"
  done
  install -m 0755 "${ROOT_DIR}/tools/${LLIMA_INSTALL_SCRIPT}" "${ROOT_DIR}/dist/${LLIMA_INSTALL_SCRIPT}"
  install -m 0644 "${NEAT_INTERNALS_RESOLVED_MANIFEST}" "${ROOT_DIR}/dist/resolved-deps-manifest.json"
  write_install_manifest

  mkdir -p "${ROOT_DIR}/dist/debs"
  while IFS= read -r deb; do
    cp -f "${ROOT_DIR}/dist/${deb}" "${ROOT_DIR}/dist/debs/${deb}"
  done < <(required_llima_debs "${version}")

  echo "[build] Staged installable root bundle with ${#internals_debs[@]} Internals package(s)."
  echo "[build] Staged download-only Debian profile: dist/debs/"
}

package_dist_archive() {
  if [ "${#COMPONENTS[@]}" -gt 0 ]; then
    echo "[build] Skipping dist archive because a subset of components was selected"
    return
  fi

  local version archive_name latest_name output_dir
  local -a required_debs=()
  version="$1"
  archive_name="${ARCHIVE_NAME:-sima-llima-${version}.tar.gz}"
  latest_name="${LATEST_NAME:-}"
  output_dir="${ROOT_DIR}/dist/debs"

  verify_required_llima_debs "${version}" "${output_dir}"
  mapfile -t required_debs < <(required_llima_debs "${version}")

  tar -C "${output_dir}" -czf "${output_dir}/${archive_name}" "${required_debs[@]}"
  sha256sum "${output_dir}/${archive_name}" | awk '{print $1}' > "${output_dir}/${archive_name}.sha256"
  ls -lh "${output_dir}/${archive_name}"

  if [[ -n "${latest_name}" ]]; then
    cp "${output_dir}/${archive_name}" "${output_dir}/${latest_name}"
    sha256sum "${output_dir}/${latest_name}" | awk '{print $1}' > "${output_dir}/${latest_name}.sha256"
  fi
}

build_extras_archive() {
  if [[ "${BUILD_RUNTIME_TESTS}" != "ON" ]]; then
    echo "[build] Skipping runtime test extras archive because tests are disabled"
    return
  fi
  if [[ "${SKIP_DIST}" -eq 1 ]]; then
    echo "[build] Skipping runtime test extras archive (--no-dist)"
    return
  fi

  local stage_root install_prefix archive_path
  stage_root="$(mktemp -d /tmp/sima-lmm-extras-stage.XXXXXX)"
  install_prefix="${stage_root}/prefix"
  archive_path="${ROOT_DIR}/dist/sima-lmm-${LLIMA_VERSION}-Linux-extras.tar.gz"

  (
    trap 'rm -rf "${stage_root}"' EXIT
    mkdir -p "${install_prefix}"
    cmake --install "${BUILD_DIR}" --component extras --prefix "${install_prefix}"

    if [[ ! -x "${install_prefix}/lib/sima-lmm/tests/sima_lmm_dispatcher_lifecycle_test" ]]; then
      echo "ERROR: Runtime test extras are missing the dispatcher lifecycle executable." >&2
      exit 1
    fi
    if [[ ! -x "${install_prefix}/lib/sima-lmm/tests/sima_lmm_text_generation_test" ]]; then
      echo "ERROR: Runtime test extras are missing the text generation executable." >&2
      exit 1
    fi
    if [[ ! -x "${install_prefix}/lib/sima-lmm/tests/sima_lmm_vision_generation_test" ]]; then
      echo "ERROR: Runtime test extras are missing the vision generation executable." >&2
      exit 1
    fi
    if [[ ! -x "${install_prefix}/lib/sima-lmm/tests/sima_lmm_asr_transcription_test" ]]; then
      echo "ERROR: Runtime test extras are missing the ASR transcription executable." >&2
      exit 1
    fi
    if [[ ! -x "${install_prefix}/lib/sima-lmm/tests/sima_lmm_tool_call_parser_test" ]]; then
      echo "ERROR: Runtime test extras are missing the tool-call parser executable." >&2
      exit 1
    fi
    if [[ ! -f "${install_prefix}/lib/sima-lmm/tests/CTestTestfile.cmake" ]]; then
      echo "ERROR: Runtime test extras are missing CTestTestfile.cmake." >&2
      exit 1
    fi

    find "${ROOT_DIR}/dist" -maxdepth 1 -type f \
      -name 'sima-lmm-*-Linux-extras.tar.gz' -delete
    tar -C "${install_prefix}" -czf "${archive_path}" .
  )

  echo "[build] Built runtime test extras archive: ${archive_path}"
}

read_package_compatibility_args() {
  local -n out_ref="$1"
  local compatibility_args
  if ! compatibility_args="$(python3 - "${NEAT_INTERNALS_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
platform_version = str(manifest.get("platform-version", "")).strip()
if not platform_version:
    raise SystemExit(f"Missing or empty platform-version in {manifest_path}")

for argument in ("--board-platform", f"modalix@{platform_version}"):
    print(argument)
PY
  )"; then
    echo "ERROR: Failed to read package compatibility from ${NEAT_INTERNALS_MANIFEST}." >&2
    return 1
  fi
  mapfile -t out_ref <<< "${compatibility_args}"
}

add_package_source_metadata() {
  local metadata_path="$1"
  local package_path="$2"
  local profile="$3"
  local source_repository source_ref source_sha
  source_repository="${GITHUB_REPOSITORY:-sima-neat/llima}"
  source_ref="${GITHUB_HEAD_REF:-${GITHUB_REF_NAME:-}}"
  if [[ -z "${source_ref}" ]]; then
    source_ref="$(git -C "${ROOT_DIR}" branch --show-current 2>/dev/null || true)"
  fi
  source_ref="${source_ref:-detached}"
  source_sha="${GITHUB_SHA:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"

  python3 - "${metadata_path}" "${package_path}" "${profile}" \
    "${source_repository}" "${source_ref}" "${source_sha}" \
    "${NEAT_INTERNALS_RESOLVED_REF}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

metadata_path = Path(sys.argv[1])
package_path, profile, repository, ref, commit, internals_ref = sys.argv[2:]
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
metadata.update(
    {
        "artifact": {
            "type": "debian-packages",
            "repository": "llima",
            "package_path": package_path,
            "profile": profile,
            "internals_ref": internals_ref,
        },
        "repository": repository,
        "branch": ref,
        "commit": commit,
        "commit_folder": commit[:12],
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
)
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY
}

generate_package_metadata() {
  if [ "${#COMPONENTS[@]}" -gt 0 ]; then
    echo "[build] Skipping package metadata because a subset of components was selected"
    return
  fi

  local version root_package_dir debs_package_dir file deb
  local -a package_compatibility_args=()
  version="$1"
  root_package_dir="$(mktemp -d /tmp/llima-root-package-metadata.XXXXXX)"
  debs_package_dir="$(mktemp -d /tmp/llima-debs-package-metadata.XXXXXX)"

  if ! command -v sima-cli >/dev/null 2>&1; then
    echo "ERROR: sima-cli is required to generate package metadata." >&2
    return 1
  fi
  if ! sima-cli packages build --help >/dev/null 2>&1; then
    echo "ERROR: sima-cli packages build is required to generate package metadata." >&2
    return 1
  fi
  read_package_compatibility_args package_compatibility_args

  while IFS= read -r file; do
    cp -f "${file}" "${root_package_dir}/"
  done < <(find "${ROOT_DIR}/dist" -maxdepth 1 -type f ! -name 'metadata.json' | sort)

  echo "[build] Building installable root package metadata: dist/metadata.json"
  SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli packages build "${root_package_dir}" \
    --name "gh:sima-neat/llima" \
    --version "${version}" \
    --description "Installable SiMa.ai LLiMa runtime and exact Internals dependencies" \
    --install-script "bash ./${LLIMA_INSTALL_SCRIPT}" \
    "${package_compatibility_args[@]}"

  cp -f "${root_package_dir}/metadata.json" "${ROOT_DIR}/dist/metadata.json"
  add_package_source_metadata "${ROOT_DIR}/dist/metadata.json" "." "devkit-install"

  while IFS= read -r deb; do
    cp -f "${ROOT_DIR}/dist/debs/${deb}" "${debs_package_dir}/${deb}"
  done < <(required_llima_debs "${version}")

  echo "[build] Building download-only Debian profile metadata: dist/debs/metadata.json"
  SIMA_CLI_CHECK_FOR_UPDATE=0 sima-cli packages build "${debs_package_dir}" \
    --name "gh:sima-neat/llima/debs" \
    --version "${version}" \
    --description "Download-only SiMa.ai LLiMa Debian packages" \
    --install-script 'echo "LLiMa Debian packages downloaded."'

  cp -f "${debs_package_dir}/metadata.json" "${ROOT_DIR}/dist/debs/metadata.json"
  add_package_source_metadata "${ROOT_DIR}/dist/debs/metadata.json" "debs" "download-only"

  rm -rf "${root_package_dir}" "${debs_package_dir}"
  ls -lh "${ROOT_DIR}/dist/metadata.json" "${ROOT_DIR}/dist/debs/metadata.json"
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
      BUILD_RUNTIME_TESTS=ON
      shift
      ;;
    --no-dist)
      SKIP_DIST=1
      shift
      ;;
    --core)
      add_component core
      BUILD_RUNTIME_TESTS=OFF
      shift
      ;;
    --dev)
      add_component dev
      BUILD_RUNTIME_TESTS=OFF
      shift
      ;;
    --cli)
      add_component cli
      BUILD_RUNTIME_TESTS=OFF
      shift
      ;;
    --package)
      add_component "${2:-}"
      BUILD_RUNTIME_TESTS=OFF
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

NEAT_INTERNALS_DEB_DIR="${NEAT_INTERNALS_DEB_DIR:-${BUILD_DIR}/internals-debs}"
NEAT_INTERNALS_RESOLVED_MANIFEST="${NEAT_INTERNALS_RESOLVED_MANIFEST:-${BUILD_DIR}/resolved-deps-manifest.json}"
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

if [ "$DO_CLEAN" -eq 1 ]; then
  echo "[build] Removing build directory: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
  rm -rf "$ROOT_DIR/dist"
  rm -f "$ROOT_DIR"/sima-lmm*.deb
fi

check_local_build_tools
ensure_git_submodules
detect_elxr_sdk
ensure_neat_internals
write_resolved_deps_manifest
apply_default_sdk_toolchain
ensure_sdk_sysroot_packages

LLIMA_VERSION="$(compute_package_version)"
LLIMA_PROJECT_VERSION="${LLIMA_VERSION%%+*}"
MULTIARCH="$(dpkg-architecture -a"$ARCH" -qDEB_HOST_MULTIARCH 2>/dev/null || true)"
if [ -z "$MULTIARCH" ]; then
  MULTIARCH="aarch64-linux-gnu"
fi
PYTHON_ABI_TAG="cpython-311"
PYTHON_TARGET_SOABI="${PYTHON_ABI_TAG}-${MULTIARCH}"

ensure_writable_cargo_home
ensure_python_build_env
CMAKE_SOABI_ARGS=()
CMAKE_PYTHON_ARGS=("-DPython_EXECUTABLE=$BUILD_VENV/bin/python")
if [[ "${ELXR_SDK}" == "ON" ]]; then
  CMAKE_SOABI_ARGS+=("-DSKBUILD_SOABI=cpython-311-${MULTIARCH}")
  SDK_SYSROOT="${SYSROOT:-/opt/toolchain/aarch64/modalix}"
  SDK_PYTHON_EXECUTABLE="${SDK_SYSROOT}/usr/bin/python3"
  SDK_PYTHON_INCLUDE_DIR="${SDK_SYSROOT}/usr/include/python3.11"
  SDK_PYTHON_LIBRARY="${SDK_SYSROOT}/usr/lib/${MULTIARCH}/libpython3.11.so"
  if [[ ! -x "${SDK_PYTHON_EXECUTABLE}" ]]; then
    echo "ERROR: SDK Python executable not found: ${SDK_PYTHON_EXECUTABLE}" >&2
    exit 1
  fi
  if [[ ! -f "${SDK_PYTHON_INCLUDE_DIR}/Python.h" ]]; then
    echo "ERROR: SDK Python headers not found: ${SDK_PYTHON_INCLUDE_DIR}/Python.h" >&2
    exit 1
  fi
  if [[ ! -f "${SDK_PYTHON_LIBRARY}" ]]; then
    echo "ERROR: SDK Python library not found: ${SDK_PYTHON_LIBRARY}" >&2
    exit 1
  fi
  nanobind_cmake_dir="$("$BUILD_VENV/bin/python" -m nanobind --cmake_dir)"
  export PKG_CONFIG="${PKG_CONFIG:-/usr/bin/pkg-config}"
  export PKG_CONFIG_EXECUTABLE="${PKG_CONFIG_EXECUTABLE:-${PKG_CONFIG}}"
  export PKG_CONFIG_SYSROOT_DIR="${PKG_CONFIG_SYSROOT_DIR:-${SDK_SYSROOT}}"
  export PKG_CONFIG_LIBDIR="${PKG_CONFIG_LIBDIR:-${SDK_SYSROOT}/usr/lib/${MULTIARCH}/pkgconfig:${SDK_SYSROOT}/usr/lib/pkgconfig:${SDK_SYSROOT}/usr/share/pkgconfig}"
  CMAKE_PYTHON_ARGS=(
    "-DPython_EXECUTABLE=$BUILD_VENV/bin/python"
    "-DPython_INCLUDE_DIR=${SDK_PYTHON_INCLUDE_DIR}"
    "-DPython_LIBRARY=${SDK_PYTHON_LIBRARY}"
    "-DPKG_CONFIG_EXECUTABLE=${PKG_CONFIG_EXECUTABLE}"
    "-Dnanobind_ROOT=${nanobind_cmake_dir}"
  )
fi

echo "[build] Configuring sima-lmm $LLIMA_VERSION for arch=$ARCH"
echo "[build] Python extension SOABI: $PYTHON_TARGET_SOABI"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_INSTALL_PREFIX=/usr \
  -DCMAKE_INSTALL_LIBDIR="lib/$MULTIARCH" \
  -DSKBUILD_SOABI="$PYTHON_TARGET_SOABI" \
  -DSIMA_LMM_BUILD_PYTHON=ON \
  -DSIMA_LMM_BUILD_TESTS="$BUILD_RUNTIME_TESTS" \
  -DSIMA_LMM_INSTALL_PYTHON_PACKAGE=ON \
  -DSIMA_LMM_PYTHON_EXTENSION_INSTALL_DIR="lib/python3/dist-packages/sima_lmm/devkit" \
  -DSIMA_LMM_PYTHON_PACKAGE_INSTALL_DIR="lib/python3/dist-packages/sima_lmm" \
  -DSKBUILD_PROJECT_VERSION="$LLIMA_PROJECT_VERSION" \
  -DSIMA_LMM_PACKAGE_VERSION="$LLIMA_VERSION" \
  -DCPACK_DEBIAN_PACKAGE_ARCHITECTURE="$ARCH" \
  "${CMAKE_SOABI_ARGS[@]}" \
  "${CMAKE_PYTHON_ARGS[@]}" \
  "${EXTRA_CMAKE_ARGS[@]}"

echo "[build] Building sima-lmm targets with $BUILD_JOBS parallel job(s)"
cmake --build "$BUILD_DIR" --parallel "$BUILD_JOBS"

mkdir -p "$ROOT_DIR/dist"
rm -f "$ROOT_DIR"/sima-lmm*.deb
rm -f "$ROOT_DIR/dist"/sima-lmm*.deb

CPACK_ARGS=(
  --config "$BUILD_DIR/CPackConfig.cmake"
  -D "CPACK_PACKAGE_DIRECTORY=$ROOT_DIR/dist"
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
  find "$ROOT_DIR/dist" -maxdepth 1 -type f -name 'sima-lmm*.deb' -printf '  %p\n' | sort
else
  for component in "${COMPONENTS[@]}"; do
    find "$ROOT_DIR/dist" -maxdepth 1 -type f -name "sima-lmm-${component}_*.deb" -printf '  %p\n'
    find "$ROOT_DIR/dist" -maxdepth 1 -type f -name "sima-lmm-${component}-*.deb" -printf '  %p\n'
  done | sort -u
fi

if [ "$SKIP_DIST" -eq 0 ]; then
  build_extras_archive
  stage_package_artifacts "$LLIMA_VERSION"
  package_dist_archive "$LLIMA_VERSION"
  generate_package_metadata "$LLIMA_VERSION"
fi
