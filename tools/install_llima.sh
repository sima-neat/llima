#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
INSTALL_MANIFEST="${LLIMA_INSTALL_MANIFEST:-llima-install-manifest.txt}"
LLIMA_PACKAGE_MANIFEST="${LLIMA_PACKAGE_MANIFEST:-llima-package-manifest.json}"
LLIMA_BUILDINFO_FILE="${LLIMA_BUILDINFO_FILE:-/etc/buildinfo}"
ELXR_SDK_RELEASE_FILE="${ELXR_SDK_RELEASE_FILE:-/etc/sdk-release}"
LLIMA_INSTALLER_SKIP_PLATFORM_CHECK="${LLIMA_INSTALLER_SKIP_PLATFORM_CHECK:-OFF}"
SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"

log() {
  printf '[install_llima] %s\n' "$*"
}

resolve_package_manifest_path() {
  if [[ "${LLIMA_PACKAGE_MANIFEST}" == /* ]]; then
    printf '%s\n' "${LLIMA_PACKAGE_MANIFEST}"
    return 0
  fi
  printf '%s\n' "${SCRIPT_DIR}/${LLIMA_PACKAGE_MANIFEST}"
}

read_manifest_platform_version() {
  local manifest_path="$1"
  python3 - "${manifest_path}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
try:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
except FileNotFoundError:
    raise SystemExit(f"missing package manifest: {manifest_path}")
except json.JSONDecodeError as exc:
    raise SystemExit(f"invalid package manifest JSON: {manifest_path}: {exc}")

version = str(data.get("platform-version", "")).strip()
if not version:
    raise SystemExit(f"missing or empty platform-version in package manifest: {manifest_path}")
print(version.split("+", 1)[0])
PY
}

read_devkit_platform_version() {
  local buildinfo_file="$1"
  awk -F'=' '
    $1 ~ /^[[:space:]]*DISTRO_VERSION[[:space:]]*$/ {
      value=$2
      sub(/^[[:space:]]+/, "", value)
      sub(/[[:space:]]+$/, "", value)
      print value
      exit
    }
  ' "${buildinfo_file}" 2>/dev/null || true
}

ensure_platform_compatible() {
  if [[ "${LLIMA_INSTALLER_SKIP_PLATFORM_CHECK}" == "ON" ]]; then
    log "LLIMA_INSTALLER_SKIP_PLATFORM_CHECK=ON; skipping platform compatibility check."
    return 0
  fi
  if [[ -f "${ELXR_SDK_RELEASE_FILE}" ]]; then
    echo "The root LLiMa artifact installs a Modalix DevKit; it must not be run in an eLxr SDK container." >&2
    exit 1
  fi
  if [[ "$(dpkg --print-architecture)" != "arm64" ]]; then
    echo "The root LLiMa artifact requires an arm64 Modalix DevKit." >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to read the LLiMa package manifest before install." >&2
    exit 1
  fi

  local manifest_path expected actual
  manifest_path="$(resolve_package_manifest_path)"
  if ! expected="$(read_manifest_platform_version "${manifest_path}")"; then
    echo "Unable to verify LLiMa package platform compatibility." >&2
    exit 1
  fi

  if [[ ! -f "${LLIMA_BUILDINFO_FILE}" ]]; then
    echo "Cannot verify Modalix DevKit compatibility: missing ${LLIMA_BUILDINFO_FILE}." >&2
    echo "This installer only supports Modalix DevKit targets." >&2
    exit 1
  fi
  if ! grep -qE '^MACHINE[[:space:]]*=[[:space:]]*modalix' "${LLIMA_BUILDINFO_FILE}"; then
    echo "Cannot verify Modalix DevKit compatibility: ${LLIMA_BUILDINFO_FILE} does not report MACHINE=modalix." >&2
    exit 1
  fi

  actual="$(read_devkit_platform_version "${LLIMA_BUILDINFO_FILE}")"
  if [[ -z "${actual}" ]]; then
    echo "Cannot verify platform compatibility: DISTRO_VERSION is missing in ${LLIMA_BUILDINFO_FILE}." >&2
    exit 1
  fi
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Incompatible platform version for this LLiMa package." >&2
    echo "  Package platform-version: ${expected} (${manifest_path})" >&2
    echo "  Detected DISTRO_VERSION: ${actual} (${LLIMA_BUILDINFO_FILE})" >&2
    echo "Refusing to install before modifying apt packages." >&2
    exit 1
  fi

  log "Platform compatibility verified: ${actual}"
}

run_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo is required to install LLiMa packages." >&2
    exit 1
  fi
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return
  fi

  local password="${SUDO_PASSWORD:-${DEFAULT_SUDO_PASSWORD}}"
  if printf '%s\n' "${password}" | sudo -S -v >/dev/null 2>&1; then
    printf '%s\n' "${password}" | sudo -S "$@"
    return
  fi

  echo "Unable to authenticate with sudo." >&2
  exit 1
}

for command_name in apt-get dpkg dpkg-deb dpkg-query; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "${command_name} is required to install LLiMa." >&2
    exit 1
  fi
done

ensure_platform_compatible

manifest_path="${SCRIPT_DIR}/${INSTALL_MANIFEST}"
if [[ ! -f "${manifest_path}" ]]; then
  echo "LLiMa install manifest not found: ${manifest_path}" >&2
  exit 1
fi

declare -A seen_files=()
declare -A llima_debs=()
declare -A llima_versions=()
debs=()

while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%%#*}"
  line="${line%$'\r'}"
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [[ -n "${line}" ]] || continue
  if [[ "${line}" != "$(basename "${line}")" || "${line}" != *.deb ]]; then
    echo "Invalid entry in ${INSTALL_MANIFEST}: ${line}" >&2
    exit 1
  fi
  if [[ -n "${seen_files["${line}"]+x}" ]]; then
    echo "Duplicate entry in ${INSTALL_MANIFEST}: ${line}" >&2
    exit 1
  fi

  deb_path="${SCRIPT_DIR}/${line}"
  if [[ ! -f "${deb_path}" ]]; then
    echo "Install manifest references a missing package: ${line}" >&2
    exit 1
  fi
  seen_files["${line}"]=1
  debs+=("${deb_path}")

  package="$(dpkg-deb -f "${deb_path}" Package 2>/dev/null || true)"
  version="$(dpkg-deb -f "${deb_path}" Version 2>/dev/null || true)"
  if [[ -z "${package}" || -z "${version}" ]]; then
    echo "Unable to read package identity from ${line}." >&2
    exit 1
  fi
  case "${package}" in
    sima-lmm-core|sima-lmm-cli|sima-lmm-dev)
      if [[ -n "${llima_debs["${package}"]+x}" ]]; then
        echo "Install bundle contains more than one ${package} package." >&2
        exit 1
      fi
      llima_debs["${package}"]="${deb_path}"
      llima_versions["${package}"]="${version}"
      ;;
    *)
      echo "Install bundle contains unexpected package ${package}." >&2
      exit 1
      ;;
  esac
done < "${manifest_path}"

if [[ "${#debs[@]}" -eq 0 ]]; then
  echo "The LLiMa install manifest does not contain any Debian packages." >&2
  exit 1
fi

expected_version=""
for package in sima-lmm-core sima-lmm-cli sima-lmm-dev; do
  if [[ -z "${llima_debs["${package}"]:-}" ]]; then
    echo "Install bundle is missing ${package}." >&2
    exit 1
  fi
  version="${llima_versions["${package}"]}"
  if [[ -z "${version}" ]]; then
    echo "Unable to read the version of ${package}." >&2
    exit 1
  fi
  if [[ -z "${expected_version}" ]]; then
    expected_version="${version}"
  elif [[ "${version}" != "${expected_version}" ]]; then
    echo "Bundled LLiMa package versions do not match." >&2
    exit 1
  fi
done

log "Validated ${#debs[@]} Debian package(s); LLiMa version ${expected_version}."

log "Refreshing APT package indexes."
run_sudo apt-get update

simulate_output="$(mktemp /tmp/install-llima-apt-simulate.XXXXXX)"
trap 'rm -f "${simulate_output}"' EXIT
if ! apt-get install --simulate --reinstall --allow-downgrades "${debs[@]}" >"${simulate_output}" 2>&1; then
  cat "${simulate_output}" >&2
  echo "APT cannot satisfy the bundled LLiMa package transaction." >&2
  exit 1
fi
mapfile -t removed_packages < <(awk '$1 == "Remv" {print $2}' "${simulate_output}")
if [[ "${#removed_packages[@]}" -gt 0 ]]; then
  for package in "${removed_packages[@]}"; do
    cat "${simulate_output}" >&2
    echo "Refusing to install because APT would remove ${package}." >&2
    exit 1
  done
fi

log "Installing LLiMa packages."
run_sudo apt-get install -y --reinstall --allow-downgrades "${debs[@]}"

for deb_path in "${debs[@]}"; do
  package="$(dpkg-deb -f "${deb_path}" Package)"
  bundled_version="$(dpkg-deb -f "${deb_path}" Version)"
  installed_version="$(dpkg-query -W -f='${Version}' "${package}" 2>/dev/null || true)"
  if [[ "${installed_version}" != "${bundled_version}" ]]; then
    echo "Installed ${package} version ${installed_version:-<missing>} does not match bundled version ${bundled_version}." >&2
    exit 1
  fi
done

if ! command -v llima >/dev/null 2>&1; then
  echo "LLiMa installation completed, but the llima command is unavailable." >&2
  exit 1
fi
llima --help >/dev/null
log "LLiMa ${expected_version} installed successfully."
