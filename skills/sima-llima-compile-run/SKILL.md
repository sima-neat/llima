---
name: sima-llima-compile-run
description: Compile a supported LLM or VLM with LLiMa, deploy it to Modalix, and smoke-test it with llima run. Use for Hugging Face safetensors, supported compressed-tensor models, or GGUF LLMs before Neat application development. Do not use for ordinary ONNX compilation, LLiMa repository maintenance, Whisper compilation, or final application development.
---

# Compile and Test LLiMa Models

Stop after model-level validation; application composition belongs to the Neat
application workflow.

## Workflow

1. Read `references/environments.md`; use the installed Python 3.12 Model
   Compiler environment and a compatible Modalix runtime.
2. Read `references/model-inputs.md`; confirm source format and support before
   downloading or compiling.
   - A compatible checkpoint on an existing architecture needs no repository
     change.
   - For a new architecture, layout, tokenizer, or prompt contract, stop and
     use `sima-add-llima-model-support`.
3. Prefer local inputs or an approved cache. Resolve gated access without
   exposing credentials.
4. Use documented defaults. For precision or selective units, read
   `references/configuration-files.md` and distinguish a deployable build from
   a partial compiler-debug run.
5. Record model ID/revision and non-secret options, then compile:

   ```bash
   llima-compile <model-path> -o <output-directory>
   ```

   Use only flags shown by installed `llima-compile --help`.
6. Inspect output:
   - complete model: require `sima_files/devkit/` and `sima_files/mpk/`;
   - partial run: verify requested units, then stop unless runtime requirements
     remain complete;
   - unexpected partial output or compiler errors: fail.
7. Read `references/validation.md`; deploy with `llima-deploy`, run
   `llima run <model> --mode cli`, and add an image-grounded VLM prompt where
   applicable.
8. Report provenance, options, paths, target, command, result, and blocked
   checks. After success, hand application work to
   `neat-application-builder`.

## Boundaries

- Never substitute model, revision, precision, or format without agreement.
- Keep tokens, signed URLs, SSH credentials, customer data, and model
  artifacts out of reports and unauthorized destinations.
- Compilation or CPU inference is not Modalix runtime validation.
- Use `sima-contribute-to-llima` and
  `scripts/gen_models--openai--whisper.py` for existing Whisper maintenance.
- Use the standard Model Compiler workflow for ordinary ONNX vision models.

## References

- `references/environments.md`
- `references/model-inputs.md`
- `references/configuration-files.md`
- `references/validation.md`
