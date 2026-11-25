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

Embedding tables are offloaded by default (`auto`) when raw `.bin` files are
available in the model's `devkit/` directory; otherwise they remain in DRAM.
Non-NVMe storage produces a warning because slower reads may reduce performance.
Use `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off llima run <model>` to keep the tables in
DRAM, or `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=on llima run <model>` to require
offloading and fail if the raw files are missing or invalid.

**Examples**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

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

## Reuse independent prompt contexts

C++ applications can load a bounded pool with `ModelOptions.max_kv_cache_slots`
(default one) and select a context with `GenerationOptions.cache_id`:

```cpp
#include <sima_lmm/vision_language_model.hpp>

simaai::llima::VisionLanguageModel model(model_dir, simaai::llima::ModelOptions{4});
auto chat = model.create_chat();
chat.add_query("Hello");
simaai::llima::GenerationOptions options;
options.cache_id = "assistant";
auto result = model.generate(chat, options);
model.remove_kv_cache("assistant");
```

Send the complete prompt/history every time. IDs select reusable prefix state;
they do not store application messages. The unnamed legacy context also consumes
a slot. Device cache storage is allocated at construction; weights and execution
workspace are shared. `kv_cache_bytes_per_slot()` reports device bytes per slot
and `kv_cache_count()` counts assigned IDs, including the legacy context.
`remove_kv_cache()` releases a named association, and `clear_kv_caches()` releases
all associations while retaining physical allocations. New IDs at capacity raise
`KVCacheCapacityError`; there is no automatic eviction. Lifecycle calls require
an idle model and reject calls made during generation. Cancellation invalidates
the selected context; changing LoRA invalidates all contexts. A speculative
target and draft must have equal slot limits. Rebuild consumers against matching
LLiMa headers and library; the added options and metrics change value-type layouts.
