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
| `--max-kv-cache-slots` | Maximum reusable KV-cache sessions for this model instance (default: `1`). The alias `--max_kv_cache_slots` is also accepted. |

For all available options, run `llima run -h`.

**Examples**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4 --max-kv-cache-slots 4
```

## Interactive Commands

Once `llima run` starts in CLI mode, use these commands at the prompt:

| Command | Description |
|----|----|
| `add image <file>` | Add an image to the current prompt context. |
| `set system <prompt>` | Set the system prompt for the active cache session. |
| `clear system` | Clear the system prompt, chat history, and images for the active session. |
| `clear history` | Clear submitted prompts, responses, and all images for the active session while preserving the system prompt. |
| `print history` | Print chat history for the active session. |
| `use cache <id>` | Select or create a named cache session with independent chat history. |
| `use default cache` | Select the legacy unnamed cache session. |
| `remove cache <id>` | Remove a named cache session and make its slot available. |
| `clear caches` | Remove all KV caches and reset all session histories. |
| `print caches` | Print the active session, allocated-slot count, and bytes per allocated slot. |
| `set audio <file>` | Set the audio file to transcribe as the query. |
| `set language <lang>` | Set the language string used for transcription. |
| `set lora <name>` | Use LoRA weights from a `npy_files` folder. |
| `unset lora` | Revert the LoRA model to the baseline model. |
| `enable-thinking` | Enable thinking mode and clear chat history. |
| `disable-thinking` | Disable thinking mode and clear chat history. |
| `quit` | Quit. |
| `help` | Print available commands. |



## Reusable KV Caches in Web Mode

Start the web server with the number of simultaneous prompt contexts the
application needs:

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4 \
  --mode web \
  --max-kv-cache-slots 4
```

Pass a stable, non-empty `cache_id` at the top level of every inference request.
This extension is accepted by `/v1/chat/completions`, `/v1/completions`,
`/api/chat`, and `/api/generate`:

``` console
modalix:~$ curl http://localhost:9998/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ivi-assistant",
    "cache_id": "driver-profile",
    "messages": [
      {"role": "system", "content": "<large system and tool prompt>"},
      {"role": "user", "content": "Navigate home"}
    ]
  }'
```

The HTTP API remains stateless with respect to messages: the application must
send the complete conversation required for each request. `cache_id` selects
reusable device-side prompt state; it does not replace `messages` or `prompt`.
Omitting `cache_id`, or setting it to `null`, selects the legacy default cache.
That default cache consumes one slot when first used.

Slots are assigned lazily and there is no automatic eviction. A request for a
new ID after the configured limit is reached returns HTTP `429` with error type
`cache_capacity_error` for a non-streaming request. A streaming response that
has already started reports the same typed error in its final SSE or NDJSON
event.

Remove a completed named session or clear the complete pool:

``` console
modalix:~$ curl http://localhost:9998/remove_cache \
  -H 'Content-Type: application/json' \
  -d '{"model": "ivi-assistant", "cache_id": "driver-profile"}'

modalix:~$ curl http://localhost:9998/clear_caches \
  -H 'Content-Type: application/json' \
  -d '{"model": "ivi-assistant"}'
```

Removing a session releases its logical slot. Its physical device allocation is
retained for reuse until the model instance is destroyed. Responses expose
`cache_created` and `cached_prompt_tokens`; OpenAI-compatible non-streaming and
final streaming responses place the token count at
`usage.cached_prompt_tokens`, while Ollama-compatible responses use a top-level
`cached_prompt_tokens` field.

Choose the slot limit from the number of live sessions and available device
DRAM. The upper-bound cache allocation is approximately the per-slot cache
footprint multiplied by `--max-kv-cache-slots`; model weights, activations, and
other runtime buffers also consume device memory.


## Build an Application with Neat

After validating your model with `llima run`, see
[GenAI Model](/develop-apps/development-workflow/genai-model/) to serve it
through common API endpoints or use it directly from a C++ or Python
application.

Neat applications can configure a bounded number of reusable prompt contexts
per model instance and provide a stable cache ID with each inference request.
The default remains one context. Applications must explicitly remove completed
contexts; the runtime reports capacity exhaustion instead of evicting another
context automatically. Prefix-hit and cache-creation metrics let applications
verify that large system and tool prompts are actually being reused.
