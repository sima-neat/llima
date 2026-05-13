# LLIMA

LLIMA is the on-device runtime for SiMa.ai generative models. It provides the
Python and C++ runtime used to run LLM, VLM, and ASR model packages on a Modalix
DevKit.

Model compilation is not documented in this repository. Compile models in the
NEAT SDK using the ModelSDK plugin, then copy the generated runtime model
directory to the DevKit.

## Debian Packages

The DevKit runtime is delivered as three Debian packages:

- `sima-lmm-core` - C++ runtime library: `libsima_lmm_runtime.so`.
- `sima-lmm-dev` - public C++ headers and `SimaLMM` CMake package metadata.
- `sima-lmm-cli` - lean Python runtime package, nanobind extension, and `llima`
  command-line entry point.

## Runtime Requirements

Install basic build dependencies:

```bash
sudo apt update
sudo apt install -y git cmake pkg-config python3-pip python3-venv python3-dev libfftw3-dev libyaml-cpp-dev cppzmq-dev libreadline-dev simaai-memory-lib-dev
python3 -m pip install --user build
```

Install the matching NEAT internals development package before configuring
LLIMA:

```bash
sudo apt install ./neat-internals-dev_*_arm64.deb
```

`neat-internals-dev` provides dispatcher/config headers and the `NeatInternals`
CMake package used by LLIMA's dispatcher backend.

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
./scripts/release/build_llima_deb.sh --clean
```

The packages are written to the LLIMA repo root, matching NEAT core's main CPack
behavior. The script uses CPack and creates a local `build/.deb-build-venv` for
CMake's Python build requirements, including nanobind.

Build only selected packages:

```bash
./scripts/release/build_llima_deb.sh --clean --core
./scripts/release/build_llima_deb.sh --clean --core --dev
./scripts/release/build_llima_deb.sh --clean --package sima-lmm-cli
```

### Install Debian Packages

Install the generated runtime packages on the DevKit:

```bash
sudo apt install ./sima-lmm-*-core.deb ./sima-lmm-*-cli.deb
```

Install `sima-lmm-dev` only on systems that compile C++ consumers against
LLIMA:

```bash
sudo apt install ./sima-lmm-*-dev.deb
```

### Python Wheels For SDK Work

The Python wheel path is still used for SDK/build-facing workflows and optional
dependency sets:

- `sima-lmm[sdk]` - compiler SDK dependencies, including internal SiMa packages.
- `sima-lmm[sdk_ext]` - external MoLE, benchmark, and evaluation dependencies.

Build a wheel:

```bash
python3 -m build --wheel -Cbuild-dir="build/{wheel_tag}"
```

Install the wheel with an optional dependency set when needed:

```bash
python3 -m pip install "dist/sima_lmm-*.whl[sdk]"
python3 -m pip install "dist/sima_lmm-*.whl[sdk_ext]"
```

If the source tree is on an NFS share and the build cannot write generated
files, make the workspace writable from the DevKit:

```bash
sudo chmod -R u+rwX,go+rwX /workspace/neat/llima
```

## Run

LLIMA expects a compiled runtime model directory produced by the NEAT SDK
ModelSDK plugin. The directory should contain the runtime configuration and
compiled ELF files required by the DevKit runtime.

Run the CLI:

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

## Useful Runtime Environment Variables

```bash
SIMA_LLIMA_RUN_PRINT_INOUTS=1    # log model input/output tensor summaries
SIMA_LLIMA_RUN_PROFILE=1         # log per-model runtime latency
SIMA_LLIMA_RUN_DISABLE_QUEUE=1   # run models immediately instead of queueing
SIMA_LLIMA_RUN_DISABLE_VISION=1  # language-only debugging
```
## Interactive CLI Commands

Inside `llima run --mode cli`:

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
