# Agent Guidance

Read [Contributor Guide](docs/contributing.md) before changing LLiMa.

## Before Editing

1. Classify the primary surface:
   - compiler: `sima_lmm/config`, `hf`, `gguf`, `host`, `model`, `preproc`;
   - runtime: `sima_lmm/devkit` and `sima_lmm/devkit/cpp`;
   - evaluation: `sima_lmm/mole`;
   - packaging: `CMakeLists.txt`, `build*.sh`, `cmake`, `deps`, installers;
   - CI/tests: `.github/workflows`, `tools/ci`, `tests`; or
   - documentation/skills: `README.md`, `docs`, `skills`.
2. Search for nearby implementations, tests, CLI definitions, and docs.
3. Preserve unrelated working-tree changes.

Keep host-side model preparation separate from Modalix execution. Runtime code
must not depend on compiler-only Python packages; compiler code must not depend
on DevKit runtime state.

## Sources of Truth

- Dependencies and platform versions: `deps/manifest.json`
- Public CLI entry points: `pyproject.toml`
- User workflows: `docs/setup.md`, `docs/compilation_genai.md`,
  `docs/deployment.md`, `docs/runtime.md`
- Development and validation: `CONTRIBUTING.md`, `docs/contributing.md`,
  `tests/README.md`

Treat `third_party/` as vendored code. Change it only for an intentional vendor
update or required integration fix, and document submodule revisions.

## Implementation Rules

- Target C++20 and the Python versions declared in `pyproject.toml`.
- Follow local formatting and naming; avoid unrelated reformatting.
- Treat installed C++ headers, CLI commands, serialized configuration, package
  metadata, and artifact layouts as compatibility surfaces.
- Keep public interfaces minimal and prefer backward-compatible additions.
  Document migrations and update callers for intentional breaks.
- Preserve the roles of `sima-lmm-core`, `sima-lmm-dev`, `sima-lmm-cli`, and
  the compiler/MoLE publication profiles.
- Keep compilation, serialization, model selection, and artifact naming
  deterministic for identical inputs.
- Reject invalid or unsupported input with actionable errors; never silently
  switch architecture, model, revision, precision, format, or execution path.
- Bound teardown and worker coordination, protect shared state, and avoid
  unnecessary allocation or synchronization on runtime hot paths.

## Models, Artifacts, and Secrets

- Keep downloaded weights, customer data, and generated ONNX, NumPy,
  quantized, MPK, ELF, or runtime model trees out of Git.
- Reviewed JSON configuration references are allowed when they are the
  human-readable contract under test.
- Use immutable model revisions and approved Hugging Face/GGUF caches in CI.
- Never expose tokens, SSH credentials, private or signed URLs, or proprietary
  model contents. Use public identifiers, redacted logs, and minimal
  reproductions; do not ask users to paste credentials or proprietary assets.

## Validation

- Pure logic: hermetic tests without network or model downloads.
- Model-backed compiler changes: configure required cached inputs; unintended
  skips fail validation.
- ONNX/numerical regression: generate comparison artifacts during the run.
- MLA/runtime behavior: run affected packaged tests on Modalix.
- Packaging: build every affected Debian or wheel profile.
- Skills: run `quick_validate.py` and an isolated
  `sima-cli playbooks install`.

Use exact commands from [Contributor Guide](docs/contributing.md) and
[Test Suites](tests/README.md). Run targeted checks before broader tiers.

## Documentation and Completion

Update the closest official guide for user-visible behavior, `README.md` for
top-level workflows, and contributor policy only where it is authoritative.
Keep skills procedural and move conditional detail into direct references.

A change is ready when affected tests pass, compatibility and documentation
impact are addressed, unavailable model/hardware checks are explicit, and the
final summary reports validation and residual risk.
