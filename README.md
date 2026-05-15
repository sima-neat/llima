# LLiMa

[![Build and Publish](https://github.com/sima-neat/llima/actions/workflows/build-and-publish-tar.yml/badge.svg)](https://github.com/sima-neat/llima/actions/workflows/build-and-publish-tar.yml)
![Runtime](https://img.shields.io/badge/runtime-LLM%20%7C%20VLM%20%7C%20ASR-green)
![Language](https://img.shields.io/badge/C%2B%2B-20-informational)
![Python](https://img.shields.io/badge/Python-3.11-informational)

<img src="docs/images/simaai_logo.png" alt="SiMa.ai" width="240" />

LLiMa is SiMa.ai's runtime and SDK-facing toolkit for generative AI models on
Modalix DevKits, including LLMs, VLMs, and ASR models.

It provides a C++ runtime, Python bindings, a command-line experience, and
SDK-side compile/deploy tooling for running compiled GenAI model directories on
SiMa hardware.

## Packages

The Modalix DevKit runtime is delivered as three Debian packages:

- `sima-lmm-core`: C++ runtime library, including `libsima_lmm_runtime.so`.
- `sima-lmm-dev`: public C++ headers and `SimaLMM` CMake package metadata.
- `sima-lmm-cli`: lean Python runtime package, nanobind extension, and `llima`
  command-line entry point.

The NEAT SDK compile/deploy tooling is delivered as a Python wheel:

- `sima-lmm[sdk]`: compiler SDK dependencies, including internal SiMa packages.
- `sima-lmm[sdk_ext]`: external MoLE, benchmark, and evaluation dependencies.

## Install

Download the latest `main` branch runtime bundle:

```bash
curl -fLO https://artifacts.sima-neat.com/llima/sima-llima-main-latest.tar.gz
curl -fLO https://artifacts.sima-neat.com/llima/sima-llima-main-latest.tar.gz.sha256
echo "$(cat sima-llima-main-latest.tar.gz.sha256)  sima-llima-main-latest.tar.gz" | sha256sum -c -
tar -xzf sima-llima-main-latest.tar.gz
```

Install the runtime packages on the DevKit:

```bash
sudo apt install ./sima-lmm-*-Linux-core.deb ./sima-lmm-*-Linux-cli.deb
```

Install the development package only on systems that compile C++ consumers
against LLiMa:

```bash
sudo apt install ./sima-lmm-*-Linux-dev.deb
```

## Build LLiMa

For source builds, use `build.sh`:

```bash
./build.sh --all --clean
```

Common build modes:

```bash
./build.sh --install-deps-only   # install host build dependencies
./build.sh --all --clean         # build all runtime debs and dist archive
./build.sh --clean --core        # package only sima-lmm-core
./build.sh --clean --core --dev  # package core and development files
./build.sh --no-dist             # build debs without a dist tarball
```

Build output is generated under `build-deb/`. Debian packages are written to the
repository root, and the release archive is written to `dist/`.

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

LLiMa depends on the matching NEAT internals development package. In SDK
containers, the release helper can install the published internals artifacts into
the SDK sysroot:

```bash
./scripts/release/install_neat_internals_dev.sh
```

`neat-internals-dev` provides dispatcher/config headers and the `NeatInternals`
CMake package used by LLiMa's dispatcher backend.

## Run

LLiMa expects a compiled runtime model directory produced by the NEAT SDK
ModelSDK plugin. The directory must contain runtime configuration and compiled
ELF files required by the DevKit runtime.

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

## Python Wheel

The Python wheel path is used for SDK/build-facing workflows and optional
dependency sets.

Build a pure Python SDK wheel:

```bash
python3 -m build --wheel -Cbuild-dir="build/{wheel_tag}" -Cwheel.cmake=false
```

Install the wheel with an optional dependency set:

```bash
python3 -m pip install "dist/sima_lmm-*.whl[sdk]"
python3 -m pip install "dist/sima_lmm-*.whl[sdk_ext]"
```

## Documentation

Full GenAI documentation is published here:

https://docs.sima.ai/pages/genai/main.html
