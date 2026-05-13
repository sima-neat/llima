# LLiMa

LLiMa is SiMa.ai's framework for compiling, deploying, and running GenAI models,
including LLMs, VLMs, and ASR models.

This repository provides both runtime code and SDK-side compile/deploy tooling.
It can be installed on a Modalix DevKit to run compiled model directories, and
in the NEAT SDK for model compilation workflows. The Debian package sections
below focus on the DevKit runtime installation.

## Packages

The Modalix DevKit runtime is delivered as three Debian packages:

- `sima-lmm-core` - C++ runtime library: `libsima_lmm_runtime.so`.
- `sima-lmm-dev` - public C++ headers and `SimaLMM` CMake package metadata.
- `sima-lmm-cli` - lean Python runtime package, nanobind extension, and `LLiMa`
  command-line entry point.

The NEAT SDK compile/deploy tooling is delivered as a pure Python wheel:

- `sima-lmm[sdk]` - compiler SDK dependencies, including internal SiMa packages.
- `sima-lmm[sdk_ext]` - external MoLE, benchmark, and evaluation dependencies.

## Runtime Requirements

Install basic build dependencies:

```bash
sudo apt update
sudo apt install -y git cmake pkg-config python3-pip python3-venv python3-dev libfftw3-dev libyaml-cpp-dev cppzmq-dev libreadline-dev simaai-memory-lib-dev
python3 -m pip install --user build
```

Install the matching NEAT internals development package before configuring
LLiMa:

```bash
sudo apt install ./neat-internals-dev_*_arm64.deb
```

`neat-internals-dev` provides dispatcher/config headers and the `NeatInternals`
CMake package used by LLiMa's dispatcher backend.

Install Rust, then restart the shell or source the generated cargo environment:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Populate public third-party dependencies:

```bash
git submodule update --init --recursive
```

## Build On DevKit

### Debian Runtime Packages

Build all runtime packages:

```bash
./scripts/release/build_LLiMa_deb.sh --clean
```

The packages are written to the LLiMa repo root, matching NEAT core's main CPack
behavior. The script uses CPack and creates a local `build/.deb-build-venv` for
CMake's Python build requirements, including nanobind.

Build only selected packages:

```bash
./scripts/release/build_LLiMa_deb.sh --clean --core
./scripts/release/build_LLiMa_deb.sh --clean --core --dev
./scripts/release/build_LLiMa_deb.sh --clean --package sima-lmm-cli
```

### Install Debian Packages

Install the generated runtime packages on the DevKit:

```bash
sudo apt install ./sima-lmm-*-core.deb ./sima-lmm-*-cli.deb
```

Install `sima-lmm-dev` only on systems that compile C++ consumers against
LLiMa:

```bash
sudo apt install ./sima-lmm-*-dev.deb
```

### Python Wheels For SDK Work

The Python wheel path is still used for SDK/build-facing workflows and optional
dependency sets:

- `sima-lmm[sdk]` - compiler SDK dependencies, including internal SiMa packages.
- `sima-lmm[sdk_ext]` - external MoLE, benchmark, and evaluation dependencies.

Build a pure Python SDK wheel:

```bash
python3 -m build --wheel -Cbuild-dir="build/{wheel_tag}" -Cwheel.cmake=false
```

Install the wheel with an optional dependency set when needed:

```bash
python3 -m pip install "dist/sima_lmm-*.whl[sdk]"
python3 -m pip install "dist/sima_lmm-*.whl[sdk_ext]"
```

## Run

LLiMa expects a compiled runtime model directory produced by the NEAT SDK
ModelSDK plugin. The directory should contain the runtime configuration and
compiled ELF files required by the DevKit runtime.

Run the CLI:

```bash
LLiMa run <model_dir> --mode cli
```

Run the web server:

```bash
LLiMa run <model_dir> --mode web
```

Model resolution order:

1. Use the provided path if it exists.
2. Check `/media/nvme/LLiMa/models/<model>`.
3. Check `$LLiMa_MODELS_PATH/<model>` if set.

## Useful Runtime Environment Variables

```bash
SIMA_LLiMa_RUN_PRINT_INOUTS=1    # log model input/output tensor summaries
SIMA_LLiMa_RUN_PROFILE=1         # log per-model runtime latency
SIMA_LLiMa_RUN_DISABLE_QUEUE=1   # run models immediately instead of queueing
SIMA_LLiMa_RUN_DISABLE_VISION=1  # language-only debugging
```
## Interactive CLI Commands

Inside `LLiMa run --mode cli`:

- `add image <file>` - add an image.
- `clear image` - clear all images.
- `set system <prompt>` - set the system prompt.
- `clear system` - clear system prompt, chat history, and images.
- `clear history` - clear chat history and images.
- `print history` - print chat history.
- `set audio <file>` - set an audio file for transcription.
- `set language <lang>` - set transcription language.
- `set lora <name>` - load LoRA weights if present in the model package.
- `unset lora` - clear LoRA weights.
- `help` - print available commands.
