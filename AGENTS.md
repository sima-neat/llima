# Agent Guidance

This file defines repository-specific expectations for automated and human
contributors. Read [Contributor Guide](docs/contributing.md) before changing
LLiMa.

## Classify the Change

Identify the primary surface before editing:

- compilation: `sima_lmm/config`, `hf`, `gguf`, `host`, `model`, `preproc`;
- runtime: `sima_lmm/devkit` and `sima_lmm/devkit/cpp`;
- evaluation: `sima_lmm/mole`;
- packaging: `CMakeLists.txt`, `build*.sh`, `cmake`, `deps`, and installers;
- CI and test infrastructure: `.github/workflows`, `tools/ci`, and `tests`;
- documentation and skills: `README.md`, `docs`, and `skills`.

Preserve the distinction between host-side model preparation and Modalix-side
execution. Do not introduce a runtime dependency on compiler-only Python
packages or make compiler code depend on DevKit runtime state.

## Sources of Truth

- Use `deps/manifest.json` for package and platform dependency versions.
- Use the public CLI entry points declared in `pyproject.toml`.
- Use `docs/setup.md`, `docs/compilation_genai.md`,
  `docs/deployment.md`, and `docs/runtime.md` for user workflows.
- Use `CONTRIBUTING.md` and `docs/contributing.md` for development and
  validation policy.
- Search for nearby implementations and tests before adding a new abstraction.

## Compatibility

- Treat installed C++ headers, Python APIs, CLI commands, generated
  configuration, and artifact layouts as compatibility surfaces.
- Prefer additive changes. Document migration impact when behavior, file
  layout, command syntax, or configuration changes.
- Keep the three runtime package responsibilities intact:
  `sima-lmm-core`, `sima-lmm-dev`, and `sima-lmm-cli`.
- Preserve the compiler, MoLE, and runtime artifact profiles and their
  metadata/checksum contracts.

## Coding Standards

- C++ code must remain compatible with C++20. Python code must remain
  compatible with the versions declared in `pyproject.toml`.
- Follow the formatting and naming of the surrounding code. Keep changes
  focused; do not mix functional changes with broad reformatting or include
  reordering.
- Keep public interfaces intentional and minimal. Prefer backward-compatible
  additions, keep implementation details out of installed headers, and update
  all callers when an internal contract changes.
- Add type annotations to new Python interfaces where practical. Use comments
  and docstrings to explain non-obvious contracts and design decisions, not to
  restate the implementation.
- Keep model compilation, configuration serialization, cache-model selection,
  and artifact naming deterministic for identical inputs.
- Reject invalid or unsupported input with an actionable error. Do not add
  silent fallbacks that select a different architecture, precision, model,
  compilation path, or runtime behavior.
- Keep teardown and worker coordination bounded. Do not block indefinitely,
  and protect state shared across compiler processes or runtime threads.
- Avoid unnecessary abstractions and duplication. Search for an existing
  helper or nearby implementation before introducing a new one.

## Model Inputs and Generated Artifacts

- Never commit downloaded model weights or customer model data.
- Never add generated ONNX, NumPy, quantized, compiled, ELF, MPK, or runtime
  model outputs as persistent regression references.
- Reviewed JSON configuration references are allowed when they are the
  contract under test.
- Use immutable model revisions and the approved Hugging Face/GGUF caches for
  model-backed automation.
- Keep generated outputs in temporary or ignored build directories.
- Do not silently fall back to a different model, precision, revision, or
  source format when a requested input is unavailable.

## Security

- Never expose `HF_TOKEN`, `HUGGINGFACE_TOKEN`, GitHub tokens, SSH credentials,
  private artifact URLs, or signed download URLs in source, logs, reports, or
  documentation.
- Do not ask users to paste tokens or proprietary model contents into chat.
- Prefer redacted logs, public model identifiers, immutable revisions, and
  minimal reproductions.

## Testing

- Pure logic must have hermetic tests that do not require network access or
  model downloads.
- Model-backed compiler changes require the relevant cached Hugging Face or
  GGUF inputs and must not pass merely because fixtures were skipped.
- ONNX or numerical regression outputs must be generated during the test run,
  not read from newly committed binary references.
- Runtime changes that reach MLA execution require Modalix validation when
  practical.
- Packaging changes require the affected Debian and/or wheel build.
- Skill changes require `quick_validate.py` and an isolated
  `sima-cli playbooks install` check.

Use the commands and model-path variables in
[Contributor Guide](docs/contributing.md). Run targeted checks first, then the
broader affected tier.

## Third-Party Code

- Treat `third_party/` as vendored code.
- Avoid edits there unless the task explicitly requires a vendor update or a
  local integration fix.
- Keep submodule revisions intentional and document why they changed.
- Do not mix unrelated generated or vendor changes into a LLiMa contribution.

## Documentation

When behavior changes:

- update the relevant official guide under `docs/`;
- update `README.md` when the top-level install, package, build, or runtime
  workflow changes;
- keep examples and command names synchronized with the actual CLI; and
- update `docs/contributing.md` or this file when contributor policy changes.

Skills must stay concise and use progressive disclosure. Reference live
repository documentation instead of copying contributor policy into
`SKILL.md`.

## Definition of Done

A change is ready when:

- relevant tests pass without unintended skips;
- generated artifacts and secrets remain outside Git;
- compatibility and documentation impact are assessed;
- hardware- or model-dependent checks are completed or explicitly reported as
  unavailable; and
- the final summary lists validation performed and residual risk.
