# Contributing to LLiMa

LLiMa combines host-side GenAI compilation with a Modalix runtime. Target
normal changes to `develop`; keep `main` releasable. See the detailed
[Contributor Guide](docs/contributing.md) for repository structure, policies,
and validation requirements.

## Set Up

Requirements:

- The Neat SDK environment for runtime and package builds
- Python 3.12 and Model Compiler for compiler work
- Modalix for hardware-dependent runtime validation

In the Neat SDK, build runtime packages and tests with:

```bash
./build.sh --all --clean
```

The normal build handles its required setup, including submodules; no separate
dependency-bootstrap step is needed.

Build compiler and MoLE wheels:

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

Runtime outputs are staged under `build-deb/` and `dist/`; compiler and MoLE
wheels under `dist/compiler/` and `dist/mole/`.

## Compiler Environment

Activate the Python 3.12 environment installed by Model Compiler, commonly:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

Do not create another environment that shadows the installed compiler
packages. `llima-compile` covers LLMs and VLMs; existing Whisper maintenance
uses the separate
[Whisper/ASR path](docs/contributing.md#whisper-and-asr-development).

## Validate

Run the smallest tier that proves the change.

Pure Python logic:

```bash
pytest -q <targeted-test-path>
```

Model-backed compiler behavior:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

Select the affected group and marker from
[tests/README.md](tests/README.md#running-compiler-tests-locally). Configure
required model inputs; a skipped required case is not validation.

Runtime or packaging behavior:

```bash
./build.sh --all --clean
```

This builds, but does not execute, the packaged C++ runtime and Python
black-box tests. Install the candidate packages on Modalix and follow
[Running runtime tests on a DevKit](tests/README.md#running-runtime-tests-on-a-devkit).
Also run a representative `llima run` scenario when the change reaches model
loading or inference.

If the change affects the installed C++ API/ABI, runtime packages, or behavior
exposed through Neat Core's GenAI API, rebuild Core against the candidate LLiMa
packages and run its affected GenAI tests on Modalix.

For compiler or MoLE packaging changes, also run the corresponding wheel build
shown above. Documentation-only changes require link and command checks, not
model compilation.

## Contribution Rules

- Keep compiler-only dependencies out of the runtime.
- Preserve compatibility of installed APIs, CLI commands, serialized
  configuration, package metadata, and artifact layouts; document intentional
  breaks and migration steps.
- Keep downloaded weights, customer data, secrets, and generated binary model
  artifacts out of Git. Use immutable model revisions and approved caches.
- Follow local style, keep changes focused, and add tests and user
  documentation with behavior changes.
- Fail with actionable context instead of silently changing model, revision,
  precision, format, or execution path.
- Keep shared state safe and teardown bounded.

See [Coding Standards](docs/contributing.md#coding-standards) for details.

## Pull Requests

Use `.github/PULL_REQUEST_TEMPLATE.md`, link completed issues with
`Fixes #<issue>`, and include:

- change type and risk;
- commands and hardware/model evidence;
- compatibility and migration impact;
- documentation impact; and
- checks that could not run, with residual risk.

Keep commits focused and use imperative subjects.
