# LLIMA

LLIMA is the on-device runtime for SiMa.ai generative models. It provides the
Python and C++ runtime used to run LLM, VLM, and ASR model packages on a Modalix
DevKit.

Model compilation is not documented in this repository. Compile models in the
NEAT SDK using the ModelSDK plugin, then copy the generated runtime model
directory to the DevKit.

## Runtime Layout

- `sima_lmm/devkit` - Python runtime entry points and model loading helpers.
- `sima_lmm/devkit/cpp` - C++ runtime, Python bindings, CLI, web, and ZMQ server.
- `sima_lmm/config` - Runtime configuration dataclasses.
- `sima_lmm/model` - Python model representation used by tooling.
- `sima_lmm/assets` - Sample media used for runtime warm-up.

## Runtime Requirements

Build and run on a Modalix DevKit with the NEAT runtime packages installed.
The MLA path uses the system MLASHM dispatcher:

- `libneatdispatchercore.so`
- `libsimaaimem.so`
- a running `mlashmcomplex` service

If using the workspace appcomplex service, the LLIMA shell must use the same
MLASHM endpoints as the service:

```bash
export MLASHM_CTRL_PATH=/tmp/mlactrl_workspace
export MLASHM_DATA_PATH=/mlashmdata_workspace
export MLASHM_RUNTIME_SHM_PATH=/dev/shm/mlashmdata_workspace
```

## Build On DevKit

Install basic build dependencies:

```bash
sudo apt update
sudo apt install -y git python3-pip libfftw3-dev libyaml-cpp-dev cppzmq-dev simaai-memory-lib-dev
python3 -m pip install --user build
```

Install Rust, then restart the shell or source the generated cargo environment:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

Populate public third-party dependencies:

```bash
git submodule update --init --recursive
```

Build a wheel:

```bash
GIT_HASH=local python3 -m build --wheel -Cbuild-dir="build/{wheel_tag}"
```

Or install editable for active development:

```bash
GIT_HASH=local python3 -m pip install -e . -Cbuild-dir="build/{wheel_tag}"
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

Dispatcher diagnostics:

```bash
SIMA_RPCMLASHM_PROFILE=1
SIMA_MLASHM_MULTI_IO_STATS=1
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
