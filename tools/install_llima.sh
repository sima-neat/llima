#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
INSTALL_MANIFEST="${LLIMA_INSTALL_MANIFEST:-llima-install-manifest.txt}"
SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"

log() {
  printf '[install_llima] %s\n' "$*"
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

if [[ "${LLIMA_INSTALLER_SKIP_PLATFORM_CHECK:-OFF}" != "ON" ]]; then
  if [[ -f /etc/sdk-release ]]; then
    echo "The root LLiMa artifact installs a Modalix DevKit; it must not be run in an eLxr SDK container." >&2
    exit 1
  fi
  if [[ "$(dpkg --print-architecture)" != "arm64" ]]; then
    echo "The root LLiMa artifact requires an arm64 Modalix DevKit." >&2
    exit 1
  fi
fi

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
  case "${package}" in
    sima-lmm-core|sima-lmm-cli|sima-lmm-dev)
      if [[ -n "${llima_debs["${package}"]+x}" ]]; then
        echo "Install bundle contains more than one ${package} package." >&2
        exit 1
      fi
      llima_debs["${package}"]="${deb_path}"
      llima_versions["${package}"]="${version}"
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
    if [[ "${package%%:*}" != "sima-neat" && "${package%%:*}" != "sima-neat-dev" ]]; then
      cat "${simulate_output}" >&2
      echo "Refusing to install because APT would remove ${package}." >&2
      exit 1
    fi
  done
  log "Removing incompatible Neat packages: ${removed_packages[*]}"
fi

log "Installing bundled Internals and LLiMa packages."
run_sudo apt-get install -y --reinstall --allow-downgrades \
  -o Dpkg::Options::=--force-overwrite \
  "${debs[@]}"

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
if [[ "${#removed_packages[@]}" -gt 0 ]]; then
  echo "WARNING: Removed incompatible Neat packages: ${removed_packages[*]}. Reinstall a Core package compatible with the bundled Internals before using Neat." >&2
fi
