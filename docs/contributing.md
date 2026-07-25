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
| Graph generation | `sima_lmm/model/`, `sima_lmm/preproc/` | ONNX graph construction, quantization paths, and preprocessing |
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

The current premerge suite is:

```bash
tox -e premerge
```

It runs parallel and serial pytest groups. Several tests require large model
inputs or reference paths. Configure the required inputs rather than accepting
fixture skips.

#### Model-backed test inputs

| CLI option | Environment variable | Purpose |
| --- | --- | --- |
| `--hf-models-path` | `LLIMA_HF_MODELS_PATH` | Root containing prepared Hugging Face model directories |
| `--gguf-models-path` | `LLIMA_GGUF_MODELS_PATH` | Root containing prepared GGUF inputs |
| `--gguf-hf-model-path` | `LLIMA_GGUF_HF_MODEL_PATH` | Hugging Face source used for GGUF comparisons |
| `--reference-onnx-path` | `LLIMA_REFERENCE_ONNX_PATH` | Existing external ONNX input used by legacy regression tests |
| `--reference-draft-onnx-path` | `LLIMA_REFERENCE_DRAFT_ONNX_PATH` | Existing external draft ONNX input used by legacy tests |

CI prepares approved Hugging Face and GGUF inputs from the manifests under
`tools/hf-safetensors/`. Local runs may point these variables at an equivalent
readable cache.

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
- generated ONNX graphs;
- NumPy `.npy` or `.npz` outputs;
- quantized intermediate graphs or tensors;
- compiled MPK, ELF, or runtime model trees; or
- customer model data.

Generated outputs belong in temporary or ignored build directories. CI should
download approved source inputs, generate candidate and baseline outputs in
the same run, compare them, and then discard them.

## Code and Compatibility Standards

- C++ code targets C++20.
- Keep public headers and implementations coherent.
- Treat installed C++ headers, Python APIs, console scripts, generated
  configuration, and artifact layouts as compatibility surfaces.
- Prefer additive changes and document migration impact for breaking behavior.
- Keep compiler and runtime dependency graphs separate.
- Fail clearly for unsupported architectures, formats, precisions, missing
  artifacts, and invalid configuration. Do not silently choose a different
  model or compilation path.
- Keep deterministic inputs, stable configuration serialization, and
  reproducible generated graphs wherever practical.

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
