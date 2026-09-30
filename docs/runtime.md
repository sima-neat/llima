# LLiMa CLI

Use the `llima` CLI on Modalix to manage precompiled models and do simple
runtime testing. It is useful for checking that a model loads, accepts prompts,
and produces output before you integrate it with Neat Framework direct APIs or
the Neat GenAI server endpoints.

## Model Manager

LLiMa includes a model manager through the `llima` CLI. It lets you search,
download, list, remove, and run precompiled models directly from the command
line. Models are stored under `/media/nvme/llima/models` by default. Set
`LLIMA_MODELS_PATH` to use a different models directory.

Browse available models:

``` console
modalix:~$ llima search
modalix:~$ llima search qwen
```

Download a model by name, without the `simaai/` organization prefix:

``` console
modalix:~$ llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

Model artifacts download concurrently, with the largest artifacts scheduled
first. Transient HTTP failures are retried automatically. Pulls for the same
model are serialized, and cancelling a pull retains every artifact that was
already downloaded and verified.

List and remove locally installed models:

``` console
modalix:~$ llima list
modalix:~$ llima rm Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## Running LLiMa

Use `llima run` as a simple runtime for initial model validation on Modalix.

In CLI mode, chat history is enabled by default. Each prompt and response is
kept as context for the next turn until you clear it with `clear history`. Images
submitted with a prompt are also retained as part of that history. `clear history`
removes the submitted prompts, responses, and all images. The configured system
prompt remains active until you use `clear system` or replace it with `set system`.

``` console
modalix:~$ llima run <model> [options]
```

| Argument | Description |
|----|----|
| `model` | Model ID or path (e.g., `Qwen3-VL-8B-Instruct-a16w4`). |
| `--stt_model_path` | Path to the elf files for a Speech-to-Text model (optional). |

For all available options, run `llima run -h`.

To disable automatic embedding offloading and keep the tables in DRAM, run
`SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off llima run <model>`.

**Examples**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## Running a Model over PCIe

On a Modalix PCIe card, the model can stay on the x86 host. With `--pcie`,
`llima run` on the card pulls each model file from the host over PCIe when it
needs it. It loads the ELF files one at a time and deletes each one after it is
loaded, so the card needs space for only the largest single file.

``` console
modalix:~$ llima run <model> --pcie [--pcie-serve-root models] [--pcie-recv-root <dir>]
```

| Argument | Description |
|----|----|
| `model` | The model folder name under the host serve root (e.g., `Llama-3.2-3B-Instruct-a16w4`). |
| `--pcie` | Pull the model files over PCIe instead of reading the local disk. |
| `--pcie-serve-root` | Name of the host serve root that holds the models (default: `models`). |
| `--pcie-recv-root` | Card folder where pulled files land. Default: the pep daemon's `default-recv` folder. |

Both sides need the PCIe service daemons running and configured:

- **Host** (`simaai-mla-daemon`, `/etc/simaai/simaai-mla-daemon.conf`): a
  `[serve]` root that holds the model folders. The model is read from
  `<serve root>/<model>/`.

  ``` ini
  [serve]
  models = /scratch/simaai/models
  ```

- **Card** (`simaai-pep-daemon`, `/etc/simaai/simaai-pep-daemon.conf`): a
  receive folder, set as the default. It must have room for the largest model
  file (for example, a large `embeddings.bin`).

  ``` ini
  default-recv = recv5g

  [recv]
  recv5g = /tmp/pcie-recv
  ```

  `--pcie-recv-root` must be the same folder as `default-recv`, because the
  daemon writes the pulled files there.

Limits: `--pcie` works only in CLI mode (`--mode cli`), no draft model for
speculative decoding is used with `--pcie`, `--stt_model_path` must be a
local folder on the card, and `set lora` is not supported yet.

Only one PCIe model user runs on a card at a time, because all of them use the
same receive folder. `llima run --pcie` stops with an error while
`pcie-genai-backend` (or another `llima run --pcie`) is running, and the other
way round. The lock is `/run/sima-neat/pcie/recv-root.pid.lock`.

To chat with the card from the host instead, use the `pcie-genai` host CLI. It
starts the card program `pcie-genai-backend` (installed in `/usr/bin` by the
`sima-lmm-core` package) over SSH. See
[HOW-TO-RUN-PCIE-GENAI.md](https://github.com/sima-neat/core/blob/release-3.0.0-prep/pcie_host/HOW-TO-RUN-PCIE-GENAI.md)
in the Neat core repository.

## Interactive Commands

Once `llima run` starts in CLI mode, use these commands at the prompt:

| Command | Description |
|----|----|
| `add image <file>` | Add an image to the current prompt context. |
| `set system <prompt>` | Set the system prompt. |
| `clear system` | Clear the system prompt, chat history, and images. |
| `clear history` | Clear submitted prompts, responses, and all images while preserving the system prompt. |
| `print history` | Print chat history. |
| `set audio <file>` | Set the audio file to transcribe as the query. |
| `set language <lang>` | Set the language string used for transcription. |
| `set lora <name>` | Use LoRA weights from a `npy_files` folder. |
| `unset lora` | Revert the LoRA model to the baseline model. |
| `enable-thinking` | Enable thinking mode and clear chat history. |
| `disable-thinking` | Disable thinking mode and clear chat history. |
| `quit` | Quit. |
| `help` | Print available commands. |


## Build an Application with Neat

After validating your model with `llima run`, see
[GenAI Model](/develop-apps/development-workflow/genai-model/) to serve it
through common API endpoints or use it directly from a C++ or Python
application.
