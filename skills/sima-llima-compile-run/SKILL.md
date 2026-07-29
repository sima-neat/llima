---
name: sima-llima-compile-run
description: Compile a supported LLM or VLM with LLiMa, prioritizing exact SiMa.ai pre-quantized checkpoints or applying the matching checkpoint repository's quantize.py to a custom fine-tune, then deploy it to Modalix and smoke-test it with llima run. Use for Hugging Face safetensors, supported compressed-tensor models, custom fine-tunes, or GGUF LLMs before Neat application development. Do not use for ordinary ONNX compilation, LLiMa repository maintenance, Whisper compilation, or final application development.
---

# Compile and Test LLiMa Models

Stop after model-level validation; application composition belongs to the Neat
application workflow.

## Workflow

1. Read `references/model-inputs.md` and check the
   [SiMa.ai Pre-Quantized Models collection](https://huggingface.co/collections/simaai/pre-quantized-models)
   before recommending or downloading any source model.
2. Prefer an exact compatible pre-quantized checkpoint by default:
   - compile the published checkpoint directly; or
   - for a custom fine-tune, read `references/prequantized-models.md` and run
     the exact matching model repository's own `quantize.py` first.
   Never treat the collection as a generic quantization workflow or reuse a
   script from a merely similar model.
3. Confirm architecture, size, variant, and modality support.
   - A compatible checkpoint on an existing architecture needs no repository
     change.
   - For a new architecture, layout, tokenizer, or prompt contract, stop and
     use `sima-add-llima-model-support`.
   - If no exact pre-quantized checkpoint or matching script exists, report
     that gap. Use original FP/BF16 weights or GGUF for an explicit
     requirement or with user agreement.
4. Read `references/environments.md`. Use the repository-matched GPU
   quantization environment for a custom fine-tune, the installed Python 3.12
   Model Compiler environment for `llima-compile`, and a compatible Modalix
   runtime.
5. Prefer local inputs or an approved cache. Resolve gated access without
   exposing credentials.
6. Use documented defaults. A configuration may select units from a
   pre-quantized input for compiler debugging, but must not claim to override
   its encoded weight quantization. For FP/BF16 precision or any selective
   compilation, read `references/configuration-files.md` and distinguish a
   deployable build from a partial compiler-debug run.
7. Record source model ID/revision, pre-quantized repository/revision,
   quantization script provenance when applicable, and non-secret options.
   Then compile:

   ```bash
   llima-compile <model-path> -o <output-directory>
   ```

   Use only flags shown by installed `llima-compile --help`.
8. Inspect output:
   - complete model: require `sima_files/devkit/` and `sima_files/mpk/`;
   - partial run: verify requested units, then stop unless runtime requirements
     remain complete;
   - unexpected partial output or compiler errors: fail.
9. Read `references/validation.md`; deploy with `llima-deploy`, run
   `llima run <model> --mode cli`, and add an image-grounded VLM prompt where
   applicable.
10. Report provenance, options, paths, target, command, result, and blocked
   checks. After success, hand application work to
   `neat-application-builder`.

## Boundaries

- Never substitute model, revision, precision, or format without agreement.
- Prefer an exact compatible SiMa.ai pre-quantized checkpoint by default. Use
  FP/BF16 or GGUF only for an explicit requirement or with user agreement.
- Never invent a common quantization recipe. Each collection model
  repository's `quantize.py`, `recipe.yaml`, and `versions.txt` are
  model-specific and authoritative for matching custom fine-tunes.
- Do not use this workflow for speculative decoding until issue #128 lands;
  target and draft compilation produce separate artifact trees that the
  current validation does not cover.
- Keep tokens, signed URLs, SSH credentials, customer data, and model
  artifacts out of reports and unauthorized destinations.
- Compilation or CPU inference is not Modalix runtime validation.
- Use `sima-contribute-to-llima` and
  `scripts/gen_models--openai--whisper.py` for existing Whisper maintenance.
- Use the standard Model Compiler workflow for ordinary ONNX vision models.

## References

- `references/environments.md`
- `references/model-inputs.md`
- `references/prequantized-models.md`
- `references/configuration-files.md`
- `references/validation.md`
