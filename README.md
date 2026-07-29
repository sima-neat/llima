<img src="docs/images/simaai_logo.png" alt="SiMa.ai" width="240" />

# LLiMa

[![Vulcan CI](https://github.com/sima-neat/llima/actions/workflows/vulcan-ci.yml/badge.svg)](https://github.com/sima-neat/llima/actions/workflows/vulcan-ci.yml)
![Runtime](https://img.shields.io/badge/runtime-LLM%20%7C%20VLM%20%7C%20ASR-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)
![Python](https://img.shields.io/badge/Python-3.11-informational)

LLiMa is SiMa.ai's runtime and Model Compiler toolkit for generative AI models
on Modalix DevKits, including LLMs, VLMs, and ASR models.

It provides a C++ runtime, Python-backed command-line orchestration, and Model
Compiler tooling for running compiled GenAI model directories on SiMa
hardware. The Python layer supports the CLI; it is not a separate public
runtime API.

## Packages

The Modalix DevKit runtime is delivered as three Debian packages:

- `sima-lmm-core`: C++ runtime library, including `libsima_lmm_runtime.so`.
- `sima-lmm-dev`: public C++ headers and `SimaLMM` CMake package metadata.
- `sima-lmm-cli`: Python CLI modules, internal nanobind bridge, and `llima`
  command-line entry point.

The Model Compiler tooling is delivered as a Python wheel:

- `sima-lmm[sdk]`: compiler SDK dependencies, including internal SiMa packages.
- `sima-lmm[sdk_ext]`: external model, MoLE, benchmark, and evaluation dependencies for use
  alongside an installed Model Compiler environment.

## Install Runtime Packages

Install the latest LLiMa runtime and its exact Internals dependencies on a
Modalix DevKit with `sima-cli`:

```bash
sima-cli neat install llima
```

To install a specific release, branch, or artifact reference, include it in the
target:

```bash
sima-cli neat install llima@<version-or-ref>
```

The root artifact contains `install_llima.sh`, an explicit install manifest, the
three LLiMa Debian packages, and every Debian package from the exact resolved
Internals artifact used for the build.

Build consumers that only need to download the three LLiMa Debian packages can
use the download-only subpackage:

```bash
sima-cli neat install llima/debs@<version-or-ref>
```

This preserves the previous LLiMa artifact behavior and does not install the
downloaded packages. Use `sima-cli neat install --help` for the full target
syntax and environment options.

## Build LLiMa

For source builds, use `build.sh`:

```bash
./build.sh --all --clean
```

Common build modes:

```bash
./build.sh --install-deps-only   # install host build dependencies
./build.sh --all --clean         # build all runtime debs and artifact layouts
./build.sh --clean --core        # package only sima-lmm-core
./build.sh --clean --core --dev  # package core and development files
./build.sh --no-dist             # build debs without publication layouts
```

Build output is generated under `build-deb/`. The installable bundle is written
to `dist/`; its download-only three-package profile and archive are written to
`dist/debs/`.

On a fresh DevKit, install the native build requirements first:

```bash
sudo apt update
sudo apt install -y git cmake pkg-config python3-pip python3-venv python3-dev \
  libfftw3-dev libyaml-cpp-dev cppzmq-dev libreadline-dev simaai-memory-lib-dev
```

LLiMa also builds third-party tokenizer code that uses Rust. Install Rust if
`cargo` is not already available:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

## Run

LLiMa expects a compiled runtime model directory produced by Model Compiler. The
directory must contain runtime configuration and compiled ELF files required by
the DevKit runtime.

The LLiMa model zoo is hosted on Hugging Face:

https://huggingface.co/simaai

Run the interactive CLI:

```bash
llima run <model_dir> --mode cli
```

Run the web server:

```bash
llima run <model_dir> --mode web
```

Model files load in parallel by default. When starting LLiMa alongside an
active MLA workload, set `SIMA_LLIMA_RUN_DISABLE_PARALLEL_LOAD=1` to load models
one at a time so other inference requests can run between model loads.

For EAGLE3 speculative decoding, pass the parent compiled-model directory that
contains both the target and draft model subdirectories. `llima run` reads each
subdirectory's `sima_files/devkit/vlm_config.json` and automatically selects the
target and draft.

Model resolution order:

1. Use the provided path if it exists.
2. Check `/media/nvme/llima/models/<model>`.
3. Check `$LLIMA_MODELS_PATH/<model>` if set.

Useful model-management commands:

```bash
llima search <term>
llima pull <model>
llima list
llima rm <model>
```

## Interactive CLI

Inside `llima run --mode cli`:

- `add image <file>`: add an image.
- `clear image`: clear all images.
- `set system <prompt>`: set the system prompt.
- `clear system`: clear system prompt, chat history, and images.
- `clear history`: clear chat history and images.
- `print history`: print chat history.
- `set audio <file>`: set an audio file for transcription.
- `set language <lang>`: set transcription language.
- `set lora <name>`: load LoRA weights if present in the model package.
- `unset lora`: clear LoRA weights.
- `help`: print available commands.

Set `SIMA_LLIMA_ENABLE_DRAFT_HIGHLIGHT=1` to highlight tokens accepted from the
draft model when running EAGLE3 speculative decoding in CLI mode.

## Model Compiler Python Package

### Install the Model Compiler tooling in editable mode

For active compiler development, use the Python 3.12 virtual environment
installed by Model Compiler; do not create a separate LLiMa virtual
environment. The Model Compiler installer creates a `model-compiler` virtual
environment and installation directory in the first writable location:

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler` (legacy fallback)
3. `$HOME/sdk-extensions/model-compiler`

Activate the installation used by your system, then install LLiMa with the
`sdk_ext` dependencies. The installed Model Compiler environment already
provides the internal SiMa compiler packages.

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext]'
llima-compile --help
```

Alternatively, use a self-managed Python 3.12 development environment that
already has a compatible Model Compiler package set installed.

### Build wheel artifacts

Build and validate the pure-Python compiler wheel and the independently
installable MoLE package:

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

The commands are independent; `build_mole_package.sh` does not require
`build_compiler_wheel.sh` to run first. Both packaging scripts create or reuse the shared
`build/wheel-tools-venv` Python 3.12 environment. The helper installs the
`build` package there when needed and does not modify the system Python.

Wheel versions follow the Debian package policy. Exact release tags produce the
tag version; branch builds use the base version from `deps/manifest.json` plus
the normalized branch and 12-character commit, for example
`0.3.0+develop.0123456789ab`. Set `LLIMA_WHEEL_VERSION` only when an explicit
version override is required.

The root of `dist/` is the installable DevKit bundle. The compiler wheel,
guarded Model Compiler installer, and package metadata are written to
`dist/compiler/`. The same verified wheel is staged with the MoLE installer and
package metadata in `dist/mole/`. As in Core packages, artifact checksums are
recorded in each profile's `metadata.json`; the legacy Debian archive keeps its
separate checksum file under `dist/debs/`.

`dist/compiler/metadata-wheel.json` is a download-only profile for consumers
that need the compiler wheel as an input to another package bundle. It contains
only the wheel and its checksum. Its installation command is the POSIX no-op
`:`—the same download-only convention as Core. For example:

```bash
sima-cli neat install --type wheel --install-dir <wheel-dir> llima/compiler@develop
```

```text
dist/
├── sima-lmm-<version>-Linux-{core,dev,cli}.deb
├── <resolved Internals packages>.deb
├── install_llima.sh
├── llima-install-manifest.txt
├── resolved-deps-manifest.json
├── metadata.json
├── debs/
│   ├── sima-lmm-<version>-Linux-{core,dev,cli}.deb
│   ├── sima-llima-<ref>.tar.gz
│   ├── sima-llima-<ref>.tar.gz.sha256
│   └── metadata.json
├── compiler/
│   ├── sima_lmm-<version>-py3-none-any.whl
│   ├── install_compiler.sh
│   ├── metadata.json
│   └── metadata-wheel.json
└── mole/
    ├── sima_lmm-<version>-py3-none-any.whl
    ├── install_mole.sh
    └── metadata.json
```

### Install a published Model Compiler artifact

Activate an existing Model Compiler environment, then install the latest
published compiler artifact from the default branch:

```bash
source <model-compiler-venv>/bin/activate
sima-cli neat install llima/compiler
```

To install the latest artifact published for a specific branch:

```bash
source <model-compiler-venv>/bin/activate
sima-cli neat install llima/compiler@<branch>
```

### Install MoLE

For a persistent MoLE installation with a dedicated virtual environment, run:

```bash
sima-cli neat install llima/mole
```

## Documentation

LLiMa documentation is maintained in this repository:

| Topic | Description |
| --- | --- |
| [LLiMa](docs/index.md) | Overview of LLiMa, supported models, model manager commands, and capabilities. |
| [System Requirements](docs/setup.md) | Prerequisites and system requirements for model compilation. |
| [Model Compilation](docs/compilation_genai.md) | Complete guide to compiling LLM and VLM models for Modalix. |
| [Model Deployment](docs/deployment.md) | Deploy compiled models to Modalix devices. |
| [LLiMa CLI](docs/runtime.md) | Manage, run, and serve GenAI models on Modalix. |
| [MoLE - Modalix Language Model Evaluator](docs/mole.md) | Benchmark LLM accuracy and performance on Modalix. |
| [Contributor Guide](docs/contributing.md) | Develop, test, document, and review changes to LLiMa. |
