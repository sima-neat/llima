# LLiMa Model Validation

## Deploy

Prepare and transfer a successfully compiled model:

```bash
llima-deploy <compiled-output> <user@modalix:/media/nvme/llima/models/model-name>
```

`llima-deploy` validates the compiler output, extracts runtime ELF files from
the MPK archives, and synchronizes the runtime configuration and optional LoRA
arrays. Use `llima-deploy --help` from the installed version as the command
contract.

Confirm the deployed model directory contains the runtime configuration and
ELF files expected by LLiMa.

## LLM Smoke Test

On Modalix:

```bash
llima run <model-path-or-name> --mode cli
```

Verify:

- all model components load;
- a short deterministic prompt is accepted;
- output tokens are produced;
- a second prompt works when chat history is relevant; and
- `quit` exits cleanly.

Do not use output wording as a strict numerical or semantic benchmark. This is
an initial functional smoke test.

## VLM Smoke Test

Start the same CLI mode, then use:

```text
add image <image-path>
<ask a prompt grounded in the image>
```

Verify:

- the image is readable and accepted;
- vision and language components load;
- the prompt produces output related to the supplied input; and
- clearing the image or exiting does not crash the runtime.

Use a non-sensitive representative image. Do not upload customer media into a
report or repository.

## Report

Capture:

- public model ID and immutable revision when available;
- source format;
- non-secret compilation options;
- Model Compiler and LLiMa versions;
- output and deployment paths;
- exact smoke command;
- whether LLM or VLM validation ran; and
- pass/fail outcome with redacted error context.

If Modalix access is unavailable, report compilation and deployment preparation
separately and state that `llima run` remains unvalidated.

After this smoke test succeeds, use `neat-application-builder` for application
integration with public Neat APIs.
