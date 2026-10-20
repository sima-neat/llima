#!/usr/bin/env bash
# Provision only a disposable CI SDK container. Keep shared SDK images unchanged.
set -euo pipefail
version="${1:?Usage: prepare_sysroot.sh EXACT_PLATFORM_VERSION}"
if [[ ! "$version" =~ ^3\.0\.0~pre[0-9]+$ ]]; then
  exec sysroot update "$version"
fi
[[ "${EUID}" == 0 ]] || { echo 'CI sysroot provisioning requires root' >&2; exit 1; }
[[ -f /etc/sdk-release ]] || { echo 'Run inside a Modalix SDK container' >&2; exit 1; }
grep -Eq '^Platform Base *= *3\.0\.0$' /etc/sdk-release || {
  echo 'Master 3.0 provisioning requires an existing Agate 3.0 SDK/toolchain' >&2; exit 1;
}
# The cached SDK predates master Agate support. Use the SDK-owned installer,
# transaction/rollback and validation implementation at a reviewed immutable ref.
sdk_commit=2c325eeb1f0e4551ac32f2b9ef53a4cc18a07ccf
sdk_archive_sha256=acb57f5fc634500897689e937de2241acf3589ffbf5e977c600701153ee33ca8
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
curl --fail --location --silent --show-error --retry 3 \
  "https://github.com/sima-neat/sdk/archive/${sdk_commit}.tar.gz" -o "$work/sdk.tar.gz"
printf '%s  %s\n' "$sdk_archive_sha256" "$work/sdk.tar.gz" | sha256sum --check -
mkdir "$work/source"
tar -xf "$work/sdk.tar.gz" --strip-components=1 -C "$work/source"
# Master retired AppComplex; the pinned SDK still requests its development
# package. Apply this narrowly checked compatibility fix only in this CI image.
python3 - "$work/source/scripts/setup-sdk-sysroot.sh" <<'PYCODE'
from pathlib import Path
import sys
path = Path(sys.argv[1])
text = path.read_text()
assert text.count('a65apps-dev appcomplex-dev simaai-gst-plugins-dev') == 1
path.write_text(text.replace('a65apps-dev appcomplex-dev simaai-gst-plugins-dev',
                             'a65apps-dev simaai-gst-plugins-dev'))
PYCODE
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
# This temporary provisioning receipt selects the new repository while retaining
# the old revision for transaction history. The image receipt is never rewritten.
python3 - /etc/sdk-release "$work/sdk-release" <<'PY'
from pathlib import Path
import re
import sys
text = Path(sys.argv[1]).read_text()
for field, value in {
    'Platform Channel': 'pre-release',
    'Platform Repository': 'https://debian.neat.sima.ai/pre-release',
}.items():
    text, count = re.subn(rf'^{re.escape(field)}\s*=.*$', f'{field} = {value}', text, flags=re.M)
    if count != 1:
        raise SystemExit(f'expected exactly one {field} in SDK receipt')
Path(sys.argv[2]).write_text(text)
PY
SDK_RELEASE_FILE="$work/sdk-release" sysroot update "$version"
sysroot status
# Check the exact platform inventory rather than trusting the image receipt.
python3 - "$version" "${SYSROOT:-/opt/toolchain/aarch64/modalix}" <<'PYCODE'
from pathlib import Path
import sys
root = Path(sys.argv[2]) / 'var/lib/sima-sdk'
versions = [f[2] for line in (root / 'sysroot-packages.tsv').read_text().splitlines()
            if len(f := line.split('\t')) >= 3
            and f[0] == 'simaai-palette-modalix' and f[1] in {'arm64', 'all'}]
if versions != [sys.argv[1]]:
    raise SystemExit(f'Unexpected platform inventory: {versions}')
PYCODE
