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
| `--stt_prompt`, `--stt_prompt_file` | Default transcription context, such as domain terms, given as text or a file (optional; requires `--stt_model_path`). |

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
| `set prompt <text>` | Set the transcription context; omit `<text>` to clear it. |
| `set lora <name>` | Use LoRA weights from a `npy_files` folder. |
| `unset lora` | Revert the LoRA model to the baseline model. |
| `enable-thinking` | Enable thinking mode and clear chat history. |
| `disable-thinking` | Disable thinking mode and clear chat history. |
| `quit` | Quit. |
| `help` | Print available commands. |

## Transcription Context

Whisper accepts preceding text as context, as OpenAI Whisper does with `prompt`
(`initial_prompt` in the open-source package). Use it to steer spelling and
vocabulary, such as product names and acronyms. Context makes these words more
likely but does not guarantee them. Keep it to the terms that matter: long or
repetitive context can make Whisper repeat it or skip the speech, as upstream
Whisper does.

In web mode, `/v1/audio/transcriptions` and `/v1/audio/translations` accept
`prompt` or `initial_prompt`:

``` console
$ curl http://<modalix-ip>:9998/v1/audio/transcriptions \
    -F "file=@meeting.wav" \
    -F "prompt=Modalix, SiMa.ai, PCIe, CSI-2, NPU"
```

A request prompt, including an empty one, replaces the `--stt_prompt` default.
The runtime keeps the last 223 prompt tokens, and each one adds a decoder step
before the first transcribed token of each 30-second window. Prompts must not contain Whisper special
tokens such as `<|en|>`; an invalid `--stt_prompt` stops `llima run` at startup
and an invalid request prompt returns HTTP 400. Options that cannot be honored
also return HTTP 400:
`response_format` other than `json`, nonzero `temperature`,
`timestamp_granularities[]`, `chunking_strategy` and `include[]`.

## Long Audio

Audio longer than 30 seconds is transcribed in consecutive 30-second windows,
and the text of all windows is returned together. Streaming sends text as each
window is decoded. Windows are fixed, so a word that spans a window boundary can
be split or dropped. A final remainder shorter than 100 ms, such as a few extra
samples on a 30-second recording, is not transcribed, because Whisper tends to
produce invented text for an almost empty window. Audio up to 2 hours is
accepted; longer audio returns HTTP 400, or an error event when streaming.

As in OpenAI Whisper, each window also receives the most recent text of the
preceding windows as context, after the prompt and within the same 223-token
limit. The prompt applies to every window, so a prompt near the limit leaves
little room for preceding text. If a window stops at the decoder token limit,
usually because the model is repeating itself, the preceding text is dropped and
later windows start again from the prompt alone. To always use only the prompt
as context, send `condition_on_previous_text=false`.


## Build an Application with Neat

After validating your model with `llima run`, see
[GenAI Model](/develop-apps/development-workflow/genai-model/) to serve it
through common API endpoints or use it directly from a C++ or Python
application.
