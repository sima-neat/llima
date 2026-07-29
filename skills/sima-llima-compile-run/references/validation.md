# LLiMa Model Validation

## Deploy

```bash
llima-deploy \
  <compiled-output> \
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

- public model ID/revision and source format;
- non-secret options and Model Compiler/LLiMa versions;
- output/deployment paths and exact smoke command;
- LLM/VLM scenario and result; and
- redacted failures or unavailable Modalix validation.

After the smoke test, use `neat-application-builder` for application
integration.
