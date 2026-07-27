# Contributing to LLiMa

LLiMa spans two different systems: a host-side compiler toolkit for preparing
GenAI models and a C++/Python runtime that executes compiled models on Modalix.
Contributions must keep those environments, dependencies, and validation
requirements explicit.

Use the concise root [CONTRIBUTING.md](../CONTRIBUTING.md) as the GitHub entry
point. This document is the detailed source of truth for human contributors.
Repository-specific agent rules are in [AGENTS.md](../AGENTS.md).

## Repository Map

| Area | Paths | Responsibility |
| --- | --- | --- |
| Configuration | `sima_lmm/config/` | Serializable LLM, VLM, and ASR configuration contracts |
| Model ingestion | `sima_lmm/hf/`, `sima_lmm/gguf/` | Hugging Face and GGUF loading/conversion |
| Model compilation | `sima_lmm/model/`, `sima_lmm/preproc/` | Model-part construction, quantization paths, and preprocessing |
| Host tools | `sima_lmm/host/` | Compile, deploy, LoRA, and benchmark entry points |
| Evaluation | `sima_lmm/mole/` | Modalix Language Model Evaluator workflows |
| Runtime | `sima_lmm/devkit/` | Python runtime orchestration and model management |
| C++ runtime | `sima_lmm/devkit/cpp/` | LLM, VLM, ASR, tokenizer, MLA, CLI, web, and ZMQ runtime |
| Tests | `tests/` | Compiler/configuration tests and shared fixtures |
| Packaging | `CMakeLists.txt`, `cmake/`, `build*.sh`, `tools/install_*.sh` | Debian, wheel, and artifact assembly |
| CI and caches | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` | Vulcan builds, compiler smoke tests, and immutable model caches |
| Documentation | `README.md`, `docs/` | Repository and official user/contributor documentation |
| Skills | `skills/` | Playbooks-compatible customer and contributor skills |

Do not move compiler-only dependencies into `sima_lmm/devkit/`. Runtime
packages must remain usable on Modalix without installing the Model Compiler
Python dependency set.

## Development Environments

### Runtime and packaging

Runtime work requires Linux, CMake 3.24 or newer, a C++20 compiler, and the
native dependencies installed by `build.sh`. Modalix is required for changes
that need MLA execution or real `llima run` validation.

Initialize submodules and install build dependencies:

```bash
git submodule update --init --recursive
./build.sh --install-deps-only
```

Build all runtime packages and publication layouts:

```bash
./build.sh --all --clean
```

Useful narrower modes are:

```bash
./build.sh --clean --core
./build.sh --clean --core --dev
./build.sh --clean --cli
./build.sh --no-dist
```

Build output is generated under `build-deb/`. Installable and downloadable
profiles are staged under `dist/`.

### Compiler development

Use the Python 3.12 environment installed by Model Compiler. Supported
locations, in priority order, are:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

Activate that environment and install the checkout:

```bash
source <model-compiler-venv>/bin/activate
cd /path/to/llima
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

The Model Compiler environment already supplies the internal compiler
packages. Do not create a second virtual environment that shadows them.

Build compiler publication profiles with:

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

These scripts use a Python 3.12 wheel-tools environment under `build/` and
stage verified outputs under `dist/compiler/` and `dist/mole/`.

## Testing

Choose tests by failure surface. A successful build does not replace behavior
validation, and a skipped model-backed test does not prove correctness.

### Hermetic tests

Pure configuration, mapping, serialization, validation, and numerical helper
tests should use small generated inputs and run without network access:

```bash
pytest -q <targeted-test-path>
```

New pure logic should stay separable from model loading so it can run in this
tier.

### Model-backed compiler tests

Compiler tests live under `tests/compilation/`. Select the smallest affected
group using the marker declared in `pytest.ini`. For example, run the
configuration-contract group with:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs

python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

The authoritative group matrix, local commands, expected case counts, and
baseline-generation requirements are maintained in
[`tests/README.md`](../tests/README.md). The CI invocation is maintained in
`.github/workflows/model-compiler-tests.yml`. Several groups require large
model inputs; configure them rather than accepting fixture skips.

#### Model-backed test inputs

| CLI option | Environment variable | Purpose |
| --- | --- | --- |
| `--model-inputs-path` | `LLIMA_HF_MODELS_PATH` | Root containing prepared Hugging Face, configuration-only, and GGUF inputs |

CI prepares approved Hugging Face and GGUF inputs from the manifests under
`tools/hf-safetensors/`. Local runs may point this option or variable at an
equivalent readable cache.

Do not add new persistent ONNX or NumPy baselines. Generate branch-relative
ONNX and numerical comparison inputs during the test run using the same source
model, deterministic inputs, and compatible toolchain.

### Runtime validation

Host builds cannot prove MLA runtime behavior. When a change reaches model
loading, execution, tokenization, multimodal preprocessing, speculative
decoding, CLI/web/ZMQ behavior, or resource lifecycle, deploy a compatible
compiled model to Modalix and run the smallest representative scenario.

Basic validation starts with:

```bash
llima run <model_dir> --mode cli
```

For a VLM, add a representative image and prompt. Confirm the model loads,
accepts the input, produces output, and exits cleanly. Use
[LLiMa CLI](runtime.md) for the supported runtime workflow.

### Packaging validation

Match validation to the package surface:

```bash
./build.sh --all --clean
./build_compiler_wheel.sh
./build_mole_package.sh
```

Check that package names, install manifests, checksums, dependency provenance,
and metadata profiles remain internally consistent.

## Model Input and Reference Policy

Allowed persistent test inputs include:

- reviewed JSON configuration references;
- source-controlled test cases, seeds, tolerances, and comparison policy; and
- manifests that identify approved immutable Hugging Face or GGUF revisions.

Do not commit or publish as regression references:

- downloaded Hugging Face or GGUF weights;
- generated ONNX models;
- NumPy `.npy` or `.npz` outputs;
- quantized intermediate models or tensors;
- compiled MPK, ELF, or runtime model trees; or
- customer model data.

Generated outputs belong in temporary or ignored build directories. CI should
download approved source inputs, generate candidate and baseline outputs in
the same run, compare them, and then discard them.

## Coding Standards

These standards apply to compiler, runtime, evaluation, packaging, and test
code. Follow established local style and keep changes narrowly scoped; do not
combine behavior changes with unrelated formatting.

### Language and formatting

- C++ code targets C++20.
- Python code must remain compatible with the versions declared in
  `pyproject.toml`.
- Follow the formatting, naming, and include grouping of the surrounding code.
  LLiMa does not currently define a repository-wide formatter configuration, so
  avoid broad mechanical reformatting.
- Keep source headers and implementations coherent. Do not expose internal
  implementation details through installed headers.
- Add type annotations to new Python interfaces where practical.
- Use comments and docstrings for non-obvious contracts, numerical assumptions,
  hardware constraints, and design decisions. Do not merely restate the code.
- Avoid duplicate code and unnecessary abstractions. Search for an existing
  helper or nearby implementation before adding a new one.

### API and artifact compatibility

Treat these as compatibility surfaces:

- installed C++ headers and Python APIs;
- public CLI commands and options;
- serialized configuration and metadata;
- generated model-part names and artifact layouts; and
- the responsibilities of runtime, compiler, and MoLE packages.

Prefer backward-compatible additions over breaking changes. When a break is
unavoidable, document the affected interfaces, downstream impact, migration
steps, and release intent in the pull request. Update callers, tests, examples,
and user documentation in the same change.

Keep compiler and runtime dependencies separate. Compiler-only packages
must not become Modalix runtime dependencies, and runtime state must not become
an input to host-side model compilation.

### Determinism

- Identical model inputs and configuration should produce stable model-part
  structure, model selection, serialization, and artifact names.
- Keep cache-model selection and token-boundary behavior reproducible.
- Use deterministic test inputs and record seeds where randomness is required.
- Do not depend on filesystem iteration order, process completion order, or
  mutable remote model state.

### Error handling and diagnostics

- Reject unsupported architectures, formats, precisions, invalid
  configurations, and missing artifacts with actionable context.
- Do not silently select a different model, precision, compilation path, or
  runtime behavior.
- Preserve the original cause when translating exceptions across compiler,
  packaging, or runtime boundaries.
- Include the relevant model part, layer, artifact, or configuration field in
  diagnostics, while keeping credentials and private paths out of logs.

### Concurrency and lifecycle

- Do not block indefinitely during compiler-worker coordination, model loading,
  inference shutdown, or runtime teardown.
- Keep shared state thread-safe and process-safe. Make ownership and lifetime
  explicit for buffers, model handles, temporary files, and worker results.
- Ensure failure paths clean up partial work without deleting valid artifacts
  produced by other workers or previous resumable runs.
- Keep latency-sensitive runtime paths lightweight and avoid unnecessary
  allocation, copies, and synchronization.

### Documentation and review quality

A code contribution should include:

- a clear rationale in the code, commit, and pull request;
- focused tests for new behavior and regressions;
- compatibility and migration analysis for changed contracts;
- updated documentation for user-visible behavior; and
- explicit evidence for model-, compiler-, packaging-, and hardware-dependent
  validation that was performed.

## Dependencies and Vendored Code

- Update `deps/manifest.json` intentionally when package or platform
  dependencies change.
- Treat `third_party/` as vendored code.
- Do not edit vendored sources for ordinary LLiMa behavior changes.
- Document submodule revision changes and keep them isolated from unrelated
  work.
- Avoid adding compiler-only dependencies to runtime Debian packages.

## Credentials and Private Data

- Use organization-managed tokens for automation.
- Never commit or log Hugging Face, GitHub, SSH, artifact, or signing
  credentials.
- Do not include signed URLs or personal filesystem paths in documentation.
- Ask for public model identifiers, immutable revisions, and redacted logs
  rather than customer assets.
- Keep gated-model authorization outside source control.

## Documentation and Skills

Update the relevant official guide with user-visible behavior:

- [System Requirements](setup.md)
- [Model Compilation](compilation_genai.md)
- [Model Deployment](deployment.md)
- [LLiMa CLI](runtime.md)
- [MoLE](mole.md)

Keep root `CONTRIBUTING.md` concise and maintain detailed contributor policy
here. Keep `AGENTS.md` focused on enforceable repository guardrails.

Skills under `skills/` must contain valid `SKILL.md`, `playbook.yml`, and agent
metadata. Keep their main instructions concise, put conditional detail in
`references/`, and avoid copying this guide into a skill.

## Pull Requests

- Branch from the latest `develop`.
- Keep commits focused and use imperative subjects.
- Target normal development pull requests to `develop`.
- Use `.github/PULL_REQUEST_TEMPLATE.md`.
- Include change type, risk, test evidence, documentation impact, and
  compatibility/migration impact.
- Link the issue with `Fixes #<issue>` when the pull request completes it.
- Record model identifiers and package versions used for validation without
  exposing credentials or private assets.
- State any model-, environment-, or hardware-dependent checks that did not
  run.

## Definition of Done

A contribution is ready when:

- relevant hermetic and model-backed tests pass without unintended skips;
- hardware behavior is validated on Modalix when the change requires it;
- affected Debian and wheel profiles build successfully;
- compatibility and migration impact are documented;
- official documentation matches user-visible behavior;
- generated binary references, model data, and secrets remain outside Git; and
- the pull request contains reproducible test evidence and residual risk.
