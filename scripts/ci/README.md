# LLiMa release SDK admission

`deps/manifest.json` owns firmware B2499 and the explicit official sysroot pin
`3.0.0`. CI uses `resolve-sysroot.yml@main` and its `main` implementation. The
existing Vulcan build workflow also remains on `main`.

`prepare_sysroot.sh` requires a disposable CI SDK and admits official releases
from the existing cache: the SDK release receipt, Palette/base-files inventory,
headers and firmware identity must match, and an overlay is refused. It does not
run `sysroot update` for official releases. Refresh a mismatched SDK snapshot;
do not substitute a daily or pre-release sysroot. Existing pre-release/develop
provisioning remains supported for manifests selecting those versions.

The shared Modalix runner resolver selects compatible device firmware separately
from the SDK firmware identity. Device admission verifies its `devkit_firmware`
output against the actual board, with fallback disabled. Tests still require a
16 GB Modalix; production labels use `vulcan-production`.

Publish this branch before building Core's matching branch so Core can consume
LLiMa through its existing snap policy and verify the complete declared cohort.

CPack writes temporary packaging trees under the build directory. Only completed
Debian archives are copied into `dist/`; `_CPack_Packages` is build scratch and
must never be uploaded as a published resource. The other root, Debian,
compiler and MoLE distributions retain their existing layouts.
