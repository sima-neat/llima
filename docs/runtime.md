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

List and remove locally installed models:

``` console
modalix:~$ llima list
modalix:~$ llima rm Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## Running LLiMa

Use `llima run` as a simple runtime for initial model validation on Modalix.

In CLI mode, chat history is enabled by default. Each prompt and response is
kept as context for the next turn until you clear it with `clear history`.

``` console
modalix:~$ llima run <model> [options]
```

| Argument | Description |
|----|----|
| `model` | Model ID or path (e.g., `Qwen3-VL-8B-Instruct-a16w4`). |
| `--stt_model_path` | Path to the elf files for a Speech-to-Text model (optional). |

For all available options, run `llima run -h`.

**Examples**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## Interactive Commands

Once `llima run` starts in CLI mode, use these commands at the prompt:

| Command | Description |
|----|----|
| `add image <file>` | Add an image to the current prompt context. |
| `clear image` | Clear all images. |
| `set system <prompt>` | Set the system prompt. |
| `clear system` | Clear the system prompt, chat history, and images. |
| `clear history` | Clear chat history and images. |
| `print history` | Print chat history. |
| `set audio <file>` | Set the audio file to transcribe as the query. |
| `set language <lang>` | Set the language string used for transcription. |
| `set lora <name>` | Use LoRA weights from a `npy_files` folder. |
| `unset lora` | Revert the LoRA model to the baseline model. |
| `quit` | Quit. |
| `help` | Print available commands. |

## Build an Application with Neat

After validating your model with `llima run`, see
[GenAI Model](/develop-apps/development-workflow/genai-model/) to serve it
through common API endpoints or use it directly from a C++ or Python
application.
