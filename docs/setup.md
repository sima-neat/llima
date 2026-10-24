# System Requirements

For GenAI model compilation and deployment, use **Model Compiler**, provided
with the Neat Development Environment. You do not need the full Neat
Development Environment unless you are also building host or Neat Framework
applications.
Model Compiler needs to be installed on a machine that matches the following
requirements.

| Parameter | Description |
|----|----|
| **Operating System** | Ubuntu 22.04/24.04 LTS or Windows 11 or MacOS 12.0 or later. |
| **Memory** | 128GB or more is recommended. |
| **Storage** | 512GB available space preferred. |

:::note
With 128GB machine, compilation can take several hours to complete depends on the type of model. 64GB may work for models that do not have vision capabilities.
:::

## Prerequisites

- Ensure that the latest [`sima-cli`](/tools/sima-cli/) is installed in Model Compiler.
- Have access to the required SiMa.ai release assets for your SDK environment.
- Have a valid Hugging Face account to download open-source models.
- Some models, such as `google/paligemma`, require accepting a license agreement on Hugging Face. Make sure to review and accept the license before attempting to download these models.
- Authorize the CLI to access Hugging Face using an [user access token](https://huggingface.co/docs/hub/en/security-tokens) and `huggingface-cli`. Note, installing `sima-cli` automatically installs `huggingface-cli`.

### CI sysroot selection

`deps/manifest.json` selects the firmware target `3.0.0_master_B2372`; CI resolves
the matching master sysroot receipt `3.0.0~pre2372`. Develop receipts such as
`3.0.0~git202610140145.e5ece50-2121` remain supported. `sdk.channel` selects the
SDK toolchain image and is independent of the platform build channel.

CI calls `scripts/ci/prepare_sysroot.sh` in its disposable SDK container.
CI validates the shared resolver output against the firmware declaration before
provisioning. Master 3.0 receipts use the SDK-owned transactional installer, pinned
to SDK commit `2c325eeb1f0e4551ac32f2b9ef53a4cc18a07ccf` with an archive checksum,
to provision from the pre-release Agate repository. Current SDK development
helpers support master receipts but still request the retired `appcomplex-dev`
package; the CI helper removes only that request from the pinned installer. This
temporary compatibility step should be retired once the SDK package selection
is fixed. The original SDK image receipt is preserved; the active overlay and
package inventory record the exact selected sysroot. Provisioning fails if that
inventory does not contain exactly the resolved platform version.

This helper is for disposable CI containers, not active developer environments.
LLiMa continues to use platform libraries directly; it does not depend on
Internals packages. Building against this sysroot does not qualify a DevKit or
update its firmware.
