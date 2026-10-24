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
  [[ "${EUID}" == 0 ]] || { echo 'CI sysroot provisioning requires root' >&2; exit 1; }
  [[ -f /etc/sdk-release ]] || { echo 'Run inside a Modalix SDK container' >&2; exit 1; }
  # Current SDK helpers support master Agate, but still request appcomplex-dev.
  # Keep this temporary bundle immutable and patch only that retired request.
  sdk_commit=2c325eeb1f0e4551ac32f2b9ef53a4cc18a07ccf
  sdk_archive_sha256=acb57f5fc634500897689e937de2241acf3589ffbf5e977c600701153ee33ca8
  work="$(mktemp -d)"
  trap 'rm -rf "$work"' EXIT
  # Validate the image receipt before downloading or installing the helper bundle.
  python3 "$ci_dir/platform_policy.py" release /etc/sdk-release "$work/sdk-release"
  curl --fail --location --silent --show-error --retry 3 \
    "https://github.com/sima-neat/sdk/archive/${sdk_commit}.tar.gz" -o "$work/sdk.tar.gz"
  printf '%s  %s\n' "$sdk_archive_sha256" "$work/sdk.tar.gz" | sha256sum --check -
  mkdir "$work/source"
  tar -xf "$work/sdk.tar.gz" --strip-components=1 -C "$work/source"
  python3 "$ci_dir/platform_policy.py" patch-setup "$work/source/scripts/setup-sdk-sysroot.sh"
  for script in configure-apt-repos.sh setup-sdk-sysroot.sh validate-sysroot-package-versions.sh \
    install-sysroot-overlay.sh sysroot-directory.py; do
    install -m 755 "$work/source/scripts/$script" "/usr/local/bin/$script"
  done
  install -m 755 "$work/source/scripts/sysroot.sh" /usr/local/bin/sysroot
  install -m 755 "$work/source/scripts/simaai_setup_sdk.py" /opt/bin/simaai_setup_sdk.py
  install -m 644 "$work/source/config/platform-package-patterns.txt" \
    /usr/local/share/sima-sdk/platform-package-patterns.txt
  export SDK_APT_CHANNEL=pre-release
  configure-apt-repos.sh "$version"
  # Select the new repository without rewriting the image's original receipt.
  SDK_RELEASE_FILE="$work/sdk-release" sysroot update "$version"
fi
python3 "$ci_dir/platform_policy.py" inventory --version "$version" \
  --sysroot "${SYSROOT:-/opt/toolchain/aarch64/modalix}"
sysroot status
