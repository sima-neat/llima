---
name: sima-llima-compile-run
description: Compile supported LLM and VLM source models with LLiMa, deploy the compiled model to Modalix, and perform initial functional validation with llima run. Use for Hugging Face safetensors, supported compressed-tensor models, or GGUF LLM inputs before Neat application development. Do not use for standard non-GenAI ONNX compilation, LLiMa repository maintenance, or building the final Neat application.
---

# Compile and Test LLiMa Models

Prepare a supported LLM or VLM, compile it in Model Compiler, deploy it to
Modalix, and prove basic behavior with `llima run`. Stop after model-level
validation; application composition belongs to the Neat application workflow.

## Workflow

1. Establish the environment.
   - Read `references/environments.md`.
   - Compile in the installed Python 3.12 Model Compiler environment.
   - Run the compiled model on a reachable Modalix DevKit with a compatible
     LLiMa runtime.
2. Identify and validate the model input.
   - Read `references/model-inputs.md`.
   - Determine whether the input is Hugging Face safetensors, a supported
     compressed-tensor model, or GGUF.
   - Compare the expected runtime-performance and accuracy tradeoffs before
     choosing among available formats.
   - Check the current LLiMa supported-model documentation before downloading
     large artifacts or starting compilation.
3. Resolve access safely.
   - Prefer an existing local model or approved cache.
   - For a gated Hugging Face model, verify access without printing the token.
   - Never place credentials in commands, scripts, logs, configuration files,
     or generated reports.
4. Select the compilation policy.
   - Read `references/configuration-files.md` when choosing precision or
     compiling only selected model parts or layer indices.
   - Start with documented defaults unless the user needs a specific context
     length, precision, vision shape, LoRA mode, or performance tradeoff.
   - Decide whether the goal is a complete deployable model or a deliberately
     partial compiler/debug run.
   - Record the source model identifier, immutable revision when available,
     and non-secret compilation options.
5. Compile with the public LLiMa CLI.

   ```bash
   llima-compile <model-path> -o <output-directory>
   ```

   Add only options supported by `llima-compile --help` and the installed
   documentation. Do not guess flags from memory.
6. Inspect the result before deployment.
   - For a complete model, require `sima_files/devkit/` and `sima_files/mpk/`.
   - For selective compilation, verify only the requested units were produced
     and stop before deployment unless the architecture remains complete.
   - Treat an unexpectedly partial output tree or compiler error as a failed
     compilation.
   - Do not substitute a different model, precision, or source format without
     the user's agreement.
7. Deploy and validate.
   - Read `references/validation.md`.
   - Use `llima-deploy` to prepare and transfer the runtime tree.
   - Run `llima run <model> --mode cli` on Modalix.
   - For a VLM, test a representative image and prompt.
8. Report the handoff.
   - Summarize source provenance, compilation options, output path, deployment
     target, runtime command, and observed result.
   - State any check blocked by model access, Model Compiler availability,
     device access, or hardware/runtime compatibility.
   - After successful smoke validation, hand application work to
     `neat-application-builder`.

## Safety and Boundaries

- Do not expose tokens, signed URLs, SSH credentials, or customer model data.
- Do not upload or publish model artifacts unless the user explicitly requests
  an authorized destination.
- Do not claim runtime success from compilation alone.
- Do not use CPU-only Hugging Face inference as a substitute for Modalix
  `llima run` validation.
- Do not build Neat C++/Python application code in this skill.
- Do not use this workflow for ordinary ONNX vision models; use the standard
  Model Compiler quantize/compile workflow instead.

## References

- `references/environments.md`
- `references/model-inputs.md`
- `references/configuration-files.md`
- `references/validation.md`
