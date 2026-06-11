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

- Ensure that the latest [`sima-cli`](https://pypi.org/project/sima-cli/#history) is installed in Model Compiler.
- Have access to the required SiMa.ai release assets for your SDK environment.
- Have a valid Hugging Face account to download open-source models.
- Some models, such as `google/paligemma`, require accepting a license agreement on Hugging Face. Make sure to review and accept the license before attempting to download these models.
- Authorize the CLI to access Hugging Face using an [user access token](https://huggingface.co/docs/hub/en/security-tokens) and `huggingface-cli`. Note, installing `sima-cli` automatically installs `huggingface-cli`.
