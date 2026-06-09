# System Requirements

For GenAI model compilation and deployment, use the **ModelSDK extension**
provided with Palette SDK. You do not need the full Palette SDK development
workflow unless you are also building host or Neat Framework applications.
The ModelSDK extension needs to be installed on a machine that matches the
following requirements.

| Parameter | Description |
|----|----|
| **Operating System** | Ubuntu 20.04/22.04 LTS or Windows 10/11 or MacOS 12.0 or later. |
| **Memory** | 128GB or more is recommended. |
| **Storage** | 1TB available space preferred. |

> [!NOTE]
> With 128GB machine, compilation can take several hours to complete depends on the type of model. 64GB may work for models that do not have vision capabilities.

## Prerequisites

- Ensure that the latest [`sima-cli`](https://pypi.org/project/sima-cli/#history) is installed in the ModelSDK extension environment.
- Have access to the required SiMa.ai release assets for your SDK environment.
- Have a valid Hugging Face account to download open-source models.
- Some models, such as `google/paligemma`, require accepting a license agreement on Hugging Face. Make sure to review and accept the license before attempting to download these models.
- Authorize the CLI to access Hugging Face using an [user access token](https://huggingface.co/docs/hub/en/security-tokens) and `huggingface-cli`. Note, installing `sima-cli` automatically installs `huggingface-cli`.
