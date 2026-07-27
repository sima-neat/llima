---
name: sima-contribute-to-llima
description: Make safe changes in the sima-neat/llima repository across GenAI compilation, Hugging Face or GGUF ingestion, generated graphs, quantization, Modalix runtime, C++ or Python APIs, packaging, CI, tests, documentation, and skills. Use for repository implementation, diagnosis, refactoring, testing, review fixes, and contributor documentation. Do not use for compiling a customer's model or building a Neat application.
---

# Contribute to LLiMa

Use this skill as the operating checklist for changes to the LLiMa repository.
Keep compiler, runtime, packaging, test, and documentation responsibilities
explicit.

## First Pass

1. Read the repository guardrails:
   - `AGENTS.md`
   - `CONTRIBUTING.md`
   - `docs/contributing.md`
2. Read `references/repository-map.md`.
3. Classify the change:
   - compiler/model ingestion;
   - Modalix runtime;
   - packaging or dependencies;
   - CI, tests, or model caches;
   - documentation or skills.
4. For a new architecture, source layout, tokenizer, or chat template, also use
   `sima-add-llima-model-support`.
5. Search with `rg` for nearby implementations, tests, CLI definitions, and
   documentation before editing.
6. Preserve unrelated user changes and vendored code.

## Invariants

- Keep host-side Model Compiler dependencies out of the lean Modalix runtime.
- Treat installed C++ headers, Python APIs, CLI commands, generated
  configuration, package metadata, and artifact layouts as compatibility
  surfaces.
- Use `deps/manifest.json` as the dependency-version source of truth.
- Keep downloaded models and generated ONNX, NumPy, quantized, MPK, ELF, and
  runtime model outputs outside Git.
- Allow reviewed JSON configuration references when they are the intended
  human-readable contract.
- Use immutable source-model revisions and approved caches in automation.
- Fail explicitly for unsupported models, formats, precisions, missing files,
  and invalid configuration. Do not silently switch inputs or execution paths.
- Never expose credentials, signed URLs, private repositories, or customer
  model data.

## Testing

- Add hermetic coverage for pure configuration, mapping, validation, and
  numerical logic.
- Run model-backed tests with the required Hugging Face/GGUF paths configured;
  unintended skips are not successful validation.
- Generate numerical and ONNX comparison outputs during the run instead of
  adding binary regression references.
- Build affected Debian or wheel profiles for packaging changes.
- Validate MLA/runtime behavior on Modalix when the changed path reaches real
  execution.
- Validate skills structurally and through isolated Playbooks installation.

Use the exact commands and environment variables from
`docs/contributing.md`. Run targeted checks first and then the broader affected
tier.

## Documentation

- Update the relevant official guide for user-visible behavior.
- Keep `README.md` aligned with top-level package, install, build, and runtime
  workflows.
- Keep `CONTRIBUTING.md` concise and use `docs/contributing.md` for detailed
  human guidance.
- Keep `AGENTS.md` focused on enforceable repository rules.
- Keep skills concise and move conditional detail into direct references.

## Done Criteria

- Relevant tests pass without unintended skips.
- Required model- or hardware-dependent validation ran, or the limitation is
  explicit.
- Compatibility and migration impact are documented.
- Generated binary references, model data, and secrets remain outside Git.
- Documentation matches changed behavior.
- The final summary names changed files, checks run, skipped checks, and
  residual risk.

## Reference

- `references/repository-map.md`
