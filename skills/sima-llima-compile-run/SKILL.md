---
name: sima-llima-compile-run
description: Compile supported LLMs/VLMs with LLiMa and deploy or validate them on Modalix. Excludes repository maintenance and ordinary ONNX models.
---

# Compile and Test LLiMa Models

Use the requested model, source format, revision, and precision. If checkpoint
compatibility is uncertain, use `sima-add-llima-model-support` for that assessment.
Fixes to an existing compiler/runtime path and Whisper maintenance use
`sima-contribute-to-llima`. Ordinary ONNX models use the standard Model Compiler
workflow. Application composition is outside model-level compilation and
validation.

## Load Only the Needed Guidance

| Task | Reference |
| --- | --- |
| Select a source, verify its format/assets, or compile a target/draft pair | [Model inputs](references/model-inputs.md) |
| Quantize a custom fine-tune using its exact matching recipe | [Pre-quantized models](references/prequantized-models.md) |
| Set up or diagnose a quantization, compiler, or device environment | Relevant section of [environments](references/environments.md) |
| Select compiler units or configure original FP/BF16 precision | [Configuration files](references/configuration-files.md) |
| Check compiled artifacts, deploy, or smoke-test on Modalix | Relevant section of [validation](references/validation.md) |

References are bundled with the skill. Repository documentation paths refer to
a matching LLiMa checkout; use installed CLI help for the active release's flags.

## Input and Execution Contracts

Honor explicit source or fidelity requirements. Otherwise prefer an exact
compatible SiMa.ai pre-quantized checkpoint, checking the
[collection](https://huggingface.co/collections/simaai/pre-quantized-models)
when choosing a source. For default pre-quantization of a custom fine-tune, use
the exact matching repository's `quantize.py`, `recipe.yaml`, and `versions.txt`;
there is no generic collection recipe. Report a missing match rather than
silently substituting another model, recipe, precision, or format.

Keep GPU quantization separate from the Python 3.12 Model Compiler environment.
A pre-quantized checkpoint's encoded weights and scales are authoritative;
`config.py` can select units but does not requantize them. Prefer local inputs
or approved immutable caches, and keep credentials, private data, and artifacts
out of reports and unauthorized destinations.

## Completion

Complete the requested stage: compilation ends with verified artifacts;
selective debugging ends with the requested units and an explicit partial-build
label. Deployment/runtime requests continue through `llima-deploy` and a
`llima run` smoke test, including an image-grounded prompt for VLMs. Preserve
separate target/draft trees for supported speculative models and validate both.
Compilation or CPU inference alone is not Modalix validation.

Report exact provenance, non-secret options, versions, paths, results, and
unavailable checks. Do not extend a compile-only request into device installation
or application development.
