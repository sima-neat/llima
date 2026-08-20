# LLiMa Model Validation

## Quantized Input

For a published pre-quantized checkpoint, record its model repository and
immutable revision. Confirm the checkpoint is a supported compressed-tensors
input and contains the tokenizer and any required processor assets.

For a custom fine-tune, also require:

- the custom source model and revision;
- the exact matching SiMa.ai recipe repository and revision;
- the unmodified repository-specific `quantize.py`, `recipe.yaml`, and
  `versions.txt`;
- successful completion of the script's scale/metadata checks; and
- a compressed-checkpoint load/generation smoke test when documented.

Do not report a generic collection quantization recipe; the recipe provenance
is the individual model repository.

## Compiled Output

Require both `sima_files/devkit/` and `sima_files/mpk/` in every compiled model
tree. Treat missing directories, unexpected partial output, or compiler errors
as failure.

For speculative decoding, validate that the output parent contains one target
tree and one draft tree, each with its own `sima_files/` directory. Keep the two
trees separate and deploy their parent in one command. See
[Model Deployment](../../../docs/deployment.md#speculative-decoding-models).

## Deploy

```bash
llima-deploy \
  <compiled-output-or-speculative-parent> \
  <user@modalix:/media/nvme/llima/models/model-name>
```

Use installed `llima-deploy --help` as the command contract. It validates the
output, extracts runtime ELF files, and synchronizes configuration and optional
LoRA arrays. Confirm expected runtime config and ELFs at the destination.

## LLM Smoke Test

```bash
llima run <model-path-or-name> --mode cli
```

Verify all components load, a short prompt produces tokens, a second turn works
when history matters, and `quit` exits cleanly. This is functional validation,
not a strict quality benchmark.

## VLM Smoke Test

In the same CLI:

```text
add image <image-path>
<ask a prompt grounded in the image>
```

Verify image acceptance, vision/language loading, input-related output, and
clean image clearing or exit. Use non-sensitive media.

## Report

Capture:

- public source model ID/revision, pre-quantized repository/revision, and
  source format;
- custom fine-tune and model-specific quantization-script provenance when
  applicable;
- non-secret options and Model Compiler/LLiMa versions;
- output/deployment paths and exact smoke command;
- LLM/VLM scenario and result; and
- redacted failures or unavailable Modalix validation.

After the smoke test, use `neat-application-builder` for application
integration.
