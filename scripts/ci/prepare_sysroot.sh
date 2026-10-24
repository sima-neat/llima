#!/usr/bin/env bash
# Provision only a disposable CI SDK container. Keep shared SDK images unchanged.
set -euo pipefail
version="${1:?Usage: prepare_sysroot.sh EXACT_PLATFORM_VERSION [MANIFEST]}"
ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${2:-${ci_dir}/../../deps/manifest.json}"
python3 "$ci_dir/platform_policy.py" request --manifest "$manifest" --version "$version"
[[ "${LLIMA_DISPOSABLE_CI:-}" == 1 ]] || {
  echo 'Provisioning is restricted to an explicitly disposable CI SDK container' >&2; exit 1;
}
if [[ ! "$version" =~ ^3\.0\.0~pre[0-9]+$ ]]; then
  sysroot update "$version"
else
  sdk_release="${SDK_RELEASE_FILE:-/etc/sdk-release}"
  [[ -f "$sdk_release" ]] || { echo 'Run inside a Modalix SDK container' >&2; exit 1; }
  for tool in setup-sdk-sysroot.sh configure-apt-repos.sh sysroot; do
    command -v "$tool" >/dev/null || { echo "Required installed SDK tool is missing: $tool" >&2; exit 1; }
  done
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  python3 "$ci_dir/platform_policy.py" release "$sdk_release" "$work/sdk-release"
  # Keep installed SDK tools unchanged. Only a temporary wrapper copy omits the
  # retired package request; an already-fixed installed wrapper needs no patch.
  cp -p "$(command -v setup-sdk-sysroot.sh)" "$work/setup-sdk-sysroot.sh"
  python3 "$ci_dir/platform_policy.py" patch-setup "$work/setup-sdk-sysroot.sh"
  export SDK_APT_CHANNEL=pre-release
  configure-apt-repos.sh "$version"
  SDK_RELEASE_FILE="$work/sdk-release" \
    SYSROOT_PLATFORM_SETUP="$work/setup-sdk-sysroot.sh" sysroot update "$version"
fi
python3 "$ci_dir/platform_policy.py" inventory --version "$version" \
  --sysroot "${SYSROOT:-/opt/toolchain/aarch64/modalix}"
sysroot status
