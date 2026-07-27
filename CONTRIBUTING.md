# Contributing to LLiMa

LLiMa combines host-side GenAI model compilation with a Modalix runtime. Keep
those two environments explicit when developing and validating changes.

The `develop` branch is the integration target. The `main` branch must remain
releasable.

## Prerequisites

- Git with submodule support.
- CMake 3.24 or newer and a C++20 compiler for runtime work.
- Python 3.12 and an installed Model Compiler environment for compiler work.
- A Modalix DevKit for hardware-dependent runtime validation.

Initialize the repository:

```bash
git submodule update --init --recursive
```

See [Contributor Guide](docs/contributing.md) for the supported environment
layout, dependency details, and validation policy.

## Build

Install host build dependencies and build all runtime packages:

```bash
./build.sh --install-deps-only
./build.sh --all --clean
```

Build the compiler and MoLE wheel profiles:

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

Runtime package output is written under `build-deb/` and `dist/`. Compiler
wheel output is staged under `dist/compiler/` and `dist/mole/`.

## Compiler Development

Use the Python 3.12 virtual environment installed by Model Compiler. Do not
create a second environment that shadows its compiler packages.

```bash
source <model-compiler-venv>/bin/activate
cd /path/to/llima
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

Supported Model Compiler locations are:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

## Test Tiers

Run the smallest tier that proves the changed behavior:

- Pure Python configuration or algorithm changes:

  ```bash
  pytest -q <targeted-test-path>
  ```

- Model-backed compiler tests:

  ```bash
  export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
  python -P -m pytest \
    -c pytest.ini \
    tests/compilation/configuration \
    -m compiler_config \
    --strict-markers \
    -vv -ra
  ```

  Choose the affected group and marker from
  [tests/README.md](tests/README.md#running-compiler-tests-locally). Model-backed
  tests use the prepared input root supplied through `LLIMA_HF_MODELS_PATH` or
  `--model-inputs-path`. A skipped model-backed test is not validation of the
  changed behavior.

- Runtime C++ or Python changes:

  ```bash
  ./build.sh --all --clean
  ```

  Validate model loading and inference on Modalix with `llima run` when the
  behavior reaches the hardware runtime.

- Packaging changes:

  ```bash
  ./build.sh --all --clean
  ./build_compiler_wheel.sh
  ./build_mole_package.sh
  ```

Documentation-only changes do not require model compilation, but all changed
links and commands must be checked.

## Model and Artifact Policy

- Do not commit downloaded Hugging Face or GGUF model weights.
- Do not commit generated ONNX, NumPy, quantized, compiled, or runtime model
  outputs as regression references.
- Keep reviewed JSON configuration references in Git when they are the
  intended human-readable contract.
- Use immutable model revisions and approved internal caches in automation.
- Never print or commit Hugging Face tokens, credentials, private URLs, or
  customer model data.

## Coding Standards

- Keep C++ compatible with C++20 and Python compatible with the versions
  declared in `pyproject.toml`.
- Follow surrounding formatting, naming, and include grouping. Keep functional
  changes separate from broad reformatting.
- Treat installed APIs, CLI commands, serialized configuration, and artifact
  layouts as compatibility surfaces. Prefer backward-compatible additions.
- Keep compiler/runtime boundaries and generated output deterministic.
- Fail with actionable context instead of silently changing the model,
  precision, compilation path, or runtime behavior.
- Keep shared state safe and teardown bounded; avoid unnecessary allocation,
  copies, and synchronization in latency-sensitive runtime paths.

See [Coding Standards](docs/contributing.md#coding-standards) for the complete
error-handling, lifecycle, compatibility, documentation, and review rules.

## Commit and Pull Request Standards

- Keep commits focused and use imperative subject lines.
- Target normal development pull requests to `develop`.
- Complete the repository pull request template, including risk, test
  evidence, documentation impact, and compatibility impact.
- Update tests and user-facing documentation in the same pull request as a
  behavior change.
- State which checks could not run and why. Do not present unavailable
  hardware, model inputs, or credentials as successful validation.

## Definition of Done

A contribution is ready when:

- the relevant local checks pass;
- model- or hardware-dependent validation is complete, or its limitation is
  explicit;
- compatibility and generated-artifact impact are assessed;
- documentation is updated for user-visible changes; and
- no credentials, private model assets, or generated binary references are
  included.
