#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
INSTALL_MANIFEST="${LLIMA_INSTALL_MANIFEST:-llima-install-manifest.txt}"
SUDO_PASSWORD="${SUDO_PASSWORD:-${DEVKIT_PASSWORD:-}}"
DEFAULT_SUDO_PASSWORD="${DEFAULT_SUDO_PASSWORD:-edgeai}"
LLIMA_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD="${LLIMA_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD:-ON}"

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

stop_board_runtime_before_install() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  log "Stopping NEAT runtime services before package replacement."
  local svc
  for svc in \
      simaai-pipeline-manager.service \
      simaai-appcomplex.service \
      rctd.service \
      encoder.service \
      decoder.service \
      simaai-log.service; do
    if systemctl cat "${svc}" >/dev/null 2>&1; then
      run_sudo systemctl stop "${svc}" >/dev/null 2>&1 || true
      run_sudo systemctl reset-failed "${svc}" >/dev/null 2>&1 || true
    fi
  done

  if [[ -x /usr/libexec/simaai-appcomplex/clean-stale-mlashmcomplex ]]; then
    run_sudo /usr/libexec/simaai-appcomplex/clean-stale-mlashmcomplex || true
  else
    run_sudo pkill -TERM -x mlashmcomplex >/dev/null 2>&1 || true
    sleep 0.5
    run_sudo pkill -KILL -x mlashmcomplex >/dev/null 2>&1 || true
  fi

  run_sudo rm -f /tmp/mlactrl /dev/shm/mlashmdata
}

activate_board_runtime_after_install() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  # These files are recreated by simaai-appcomplex.service. Remove stale IPC
  # before the post-install MLA init/reset path so clients cannot observe an
  # old dispatcher lifetime after package replacement.
  run_sudo rm -f /tmp/mlactrl /dev/shm/mlashmdata
  # Package configuration intentionally does not restart services. Reload
  # systemd here so the owned maintenance window starts services from the unit
  # files that were just unpacked.
  run_sudo systemctl daemon-reload || true

  if [[ "${LLIMA_INSTALLER_ACTIVATE_FIRMWARE_ON_BOARD}" == "ON" &&
        -x /usr/libexec/sima-neat-firmware/install.sh ]]; then
    log "Activating staged EV74 firmware and resetting runtime state."
    run_sudo /usr/libexec/sima-neat-firmware/install.sh --activate
  else
    log "EV74 firmware activation skipped; starting simaai-appcomplex.service directly."
    if systemctl cat simaai-appcomplex.service >/dev/null 2>&1; then
      run_sudo systemctl restart simaai-appcomplex.service || true
    fi
  fi
}

verify_board_runtime_services() {
  local service="simaai-appcomplex.service"

  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  if ! systemctl list-unit-files "${service}" --no-legend 2>/dev/null | grep -q "^${service}[[:space:]]"; then
    return 0
  fi

  # Debian service start failures can be non-fatal during package installation,
  # but LLiMa cannot run without the MLA shared-memory dispatcher.
  if ! systemctl is-active --quiet "${service}"; then
    log "${service} is not active after package install; attempting to start it once."
    run_sudo systemctl start "${service}" || true
    sleep 1
  fi

  if ! systemctl is-active --quiet "${service}"; then
    echo "${service} is not active after LLiMa package installation." >&2
    run_sudo systemctl --no-pager --full status "${service}" >&2 || true
    run_sudo journalctl -u "${service}" --no-pager -n 80 >&2 || true
    run_sudo bash -c 'for f in /sys/class/remoteproc/remoteproc*/name /sys/class/remoteproc/remoteproc*/state; do [ -e "$f" ] && printf "%s: " "$f" && cat "$f"; done' >&2 || true
    exit 1
  fi

  log "Verified ${service} is active."
}

restart_board_codec_services() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  local -a services=()
  local service
  for service in encoder.service decoder.service; do
    if systemctl list-unit-files "${service}" --no-legend 2>/dev/null | grep -q "^${service}[[:space:]]"; then
      services+=("${service}")
    fi
  done

  if [[ "${#services[@]}" -eq 0 ]]; then
    return 0
  fi

  log "Restarting codec services after package replacement."
  run_sudo systemctl daemon-reload || true
  run_sudo systemctl enable "${services[@]}" || true
  if ! run_sudo systemctl restart "${services[@]}"; then
    echo "Failed to restart codec services after LLiMa package installation." >&2
    run_sudo systemctl --no-pager --full status "${services[@]}" >&2 || true
    run_sudo journalctl -u encoder.service -u decoder.service --no-pager -n 80 >&2 || true
    exit 1
  fi
}

verify_board_codec_services() {
  if ! command -v systemctl >/dev/null 2>&1; then
    return 0
  fi

  local service
  for service in encoder.service decoder.service; do
    if ! systemctl list-unit-files "${service}" --no-legend 2>/dev/null | grep -q "^${service}[[:space:]]"; then
      continue
    fi

    if ! systemctl is-active --quiet "${service}"; then
      log "${service} is not active after package install; attempting to start it once."
      run_sudo systemctl start "${service}" || true
      sleep 1
    fi

    if ! systemctl is-active --quiet "${service}"; then
      echo "${service} is not active after LLiMa package installation." >&2
      run_sudo systemctl --no-pager --full status "${service}" >&2 || true
      run_sudo journalctl -u "${service}" --no-pager -n 80 >&2 || true
      exit 1
    fi

    log "Verified ${service} is active."
  done
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
stop_board_runtime_before_install
run_sudo apt-get install -y --reinstall --allow-downgrades \
  -o Dpkg::Options::=--force-overwrite \
  "${debs[@]}"
activate_board_runtime_after_install
restart_board_codec_services
verify_board_codec_services
verify_board_runtime_services

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
