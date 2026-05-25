# Runtime & Orchestration

This chapter details the structure of the LLiMa application on the Modalix device, how to run models using the `llima` CLI or the GenAI demo script, and how to utilize the OpenAI-compatible API endpoint.

## LLiMa Structure on Modalix

By default, the LLiMa application and its associated models are installed in the `llima` directory. The structure is organized to separate the compiled model artifacts from the demo running logic.

The following tree illustrates the directory layout and the purpose of key files:

``` text
/media/nvme/llima/
├── models/                                # Pre-compiled model artifacts
│   ├── gemma3-siglip448-a16w4/            # Vision-Language Model (standard)
│   ├── gte-small-local/                   # Local embedding model for RAG (Retrieval Augmented Generation)
│   └── whisper-small-a16w8/               # Speech-to-Text model artifacts
├── simaai-genai-demo/                     # Main LLiMa Application Orchestration Directory
│   ├── app.py                             # Main Python entry point for the frontend
│   ├── milvus.db                          # Vector database storing embeddings for RAG
│   ├── vectordb/                          # Storage configuration for Vector DB
│   └── *.log                              # Application logs (app.log, server.log)
├── run.sh                                 # Shell script to launch the application with multiple modes
└── install.sh                             # Setup script to install dependencies and environment
```

## Running LLiMa

There are two ways to run a model on the Modalix device:

- **llima run** — lightweight inference via the `llima` CLI. Starts the model in CLI or web mode without the demo frontend or Piper TTS. Use this for direct API access or interactive terminal sessions.
- **GenAI Demo (run.sh)** — full demo stack. Wraps `llima run` and additionally starts the web frontend application and Piper TTS. Use this for the complete demo experience.

### llima run

``` console
modalix:~$ llima run <model> [options]
```

| Argument | Description |
|----|----|
| `model` | Model ID or path (e.g., `Qwen3-VL-8B-Instruct-a16w4`). |
| `--mode` | Run mode: `cli` (default) for interactive terminal, `web` for OpenAI-compatible API server. |
| `--stt_model_path` | Path to the elf files for a Speech-to-Text model (optional). |

For all available options, run `llima run -h`.

**Examples**

Run a model in interactive CLI mode:

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-a16w4
```

Run a model and whisper in web mode (exposes OpenAI-compatible API on port 9998):

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-a16w4 --stt_model_path /media/nvme/llima/models/whisper-small-a16w8 --mode web
```

### GenAI Demo (run.sh)

The `run.sh` script launches the full GenAI demo stack, including the web frontend and Piper TTS. It is located in the `/media/nvme/llima/` directory and internally uses `llima run`.

``` console
modalix:/media/nvme/llima$ ./run.sh [options]
```

| Argument | Description |
|----|----|
| `--ragfps <IP>` | Connects the application to an external RAG (Retrieval Augmented Generation) file processing server at the specified IP address. The script assumes the server is listening on port 7860. |
| `--httponly` | Starts the application server in HTTP-only mode, disabling Web UI specific features. |
| `--api-only` | Starts the OpenAI-compatible API endpoints without enabling the Web UI or TTS services. Ideal for headless integrations. |
| `--system-prompt-file <path>` | Allows the user to provide a text file containing a custom system prompt to override the default behavior. |
| `-h, --help` | Displays the help message and exits. |

> [!NOTE]
> The `run.sh` script automatically scans the `../models/` directory for available model folders. If multiple valid models are detected, the script will prompt you to interactively select which model to load.

**Examples**

**1. Default Mode (Secure Web App)**

Running the script without arguments launches the full LLiMa experience. This includes the backend API, the web-based User Interface (accessible via HTTPS), and loads the Speech-to-Text (STT) and Text-to-Speech (TTS) models.

``` console
modalix:/media/nvme/llima$ ./run.sh
```

**2. HTTP Only Mode (Insecure but Faster)**

Use this flag to launch the full application (Endpoints + UI) over HTTP instead of HTTPS. This is often useful for local testing or if you are behind a proxy handling SSL termination.

``` console
modalix:/media/nvme/llima$ ./run.sh --httponly
```

**3. Advanced Combinations**

You can combine flags to tailor the runtime environment. For example, the following command launches the API endpoints over HTTP without loading the Web UI or TTS/STT services (lightweight inference server).

``` console
modalix:/media/nvme/llima$ ./run.sh --httponly --api-only
```

## LoRA Switching

When a model has been compiled with LoRA support, adapters can be dynamically applied or removed at runtime without restarting the model. Switching a LoRA adapter clears the chat history.

**CLI mode** (`llima run <model>`)

Type the following commands directly into the interactive prompt:

``` text
>>> set lora <adapter_name>
>>> unset lora
```

**Web mode** (`llima run <model> --mode web`)

Use the following HTTP endpoints on port 9998:

| Endpoint | Description |
|----|----|
| `POST /set_lora` | Activates a LoRA adapter. Request body: `{"name": "<adapter_name>"}` |
| `POST /unset_lora` | Deactivates the current LoRA adapter and reverts to the base model. No request body required. |

Example:

``` bash
$ curl -X POST "http://<modalix-ip>:9998/set_lora" \
  -H "Content-Type: application/json" \
  -d '{"name": "my_adapter"}'

$ curl -X POST "http://<modalix-ip>:9998/unset_lora"
```

## API Endpoint Reference

The LLiMa application exposes a RESTful API for chat generation, audio processing, and system control. The following sections detail the available endpoints and their required parameters.

> [!NOTE]
> **Port Usage:**
>
> - **Port 9998** (HTTP) — inference server endpoints: chat/image generation, speech-to-text (`/v1/audio/transcriptions`), and LoRA switching. Available with both `llima run --mode web` and the GenAI Demo (`run.sh`).
> - **Port 5000** (HTTPS) — GenAI Demo application layer: text-to-speech, voice selection, system control, and RAG endpoints. **Only available when running the full demo via** `run.sh`.

**Chat & Generation (Port 9998)**

The application supports two endpoints for text generation, allowing integration with clients expecting either OpenAI or Ollama formats. Both endpoints trigger the inference backend and support streaming.

**Endpoints** (served on port 9998)

- **OpenAI Compatible:** `POST /v1/chat/completions`
- **Ollama Compatible:** `POST /v1/chat`
- **LoRA Activate:** `POST /set_lora`
- **LoRA Deactivate:** `POST /unset_lora`

**Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `messages` | Array | **Required**. A list of message objects representing the conversation history. For text-only requests, use a simple string content such as `{"role": "user", "content": "Hello"}`. For vision requests, use an array of content parts containing text and base64-encoded image data URIs. |
| `stream` | Boolean | If `true`, response is streamed. Default is `true` for OpenAI and `false` for Ollama. |
| `options` | Object | *(Ollama only)* Optional configuration parameters passed to the backend. |

Vision request example:

```json
{
  "role": "user",
  "content": [
    {"type": "image", "image": "data:image/jpeg;base64,<base64_string>"},
    {"type": "text", "text": "Describe this image"}
  ]
}
```

> [!NOTE]
> **Using Custom System Prompts**
>
> To define a custom system behavior (System Prompt), include a message with `"role": "system"` as the *first* element in your `messages` array. To ensure the model retains this persona across interactions, you must include this system message at the start of every request. On the first request it is cached, then the cached tokens are passed automatically without need for re-processing the system prompt tokens.
>
> **Example Structure:**
>
> ``` python
> # Initial Request
> messages = [
>     {"role": "system", "content": "You are a helpful assistant that speaks in rhymes."}, 
>     {"role": "user", "content": "Hi there!"}
> ]
>
> # Subsequent Inference (Append new user query after system prompt)
> messages = [
>     {"role": "system", "content": "You are a helpful assistant that speaks in rhymes."}, 
>     {"role": "user", "content": "What is the weather?"}
> ]
> ```

---

**Speech-to-Text (Port 9998)**

**POST** `/v1/audio/transcriptions`

Transcribes an audio file into text (Speech-to-Text).

| Parameter | Type | Description |
|----|----|----|
| `file` | File | **Required**. The audio file to transcribe (sent as `multipart/form-data`). |
| `language` | String | Language code of the audio content. Default is "en". |

---

**Text-to-Speech & Voice (Port 5000 — GenAI Demo only)**

**POST** `/v1/audio/speech`

Generates audio from input text (Text-to-Speech).

| Parameter | Type | Description |
|----|----|----|
| `input` | String | **Required**. The text to generate audio for. |
| `voice` | String | The ID of the specific voice model to use. |
| `language` | String | Language code (e.g., "en", "fr"). Default is "en". |
| `response_format` | String | Audio format (e.g., "wav"). Default is "wav". |

**GET** `/voices`

Retrieves a list of available TTS voices.

| Parameter | Type | Description |
|----|----|----|
| `lang` | String | Filter voices by language code (e.g., `?lang=en`). Default is "en". |

**POST** `/voices/select`

Switches the active voice model for the system.

| Parameter | Type | Description |
|----|----|----|
| `voiceId` | String | **Required**. The unique identifier of the voice to select (obtained from `/voices`). |
| `lang` | String | The language context for the voice. Default is "en". |

---

**System & RAG Operations (Port 5000 — GenAI Demo only)**

**POST** `/stop`

Immediately halts any ongoing inference or processing tasks.

- **Parameters:** None.

**GET** `/raghealth`

Checks the status of the Retrieval Augmented Generation (RAG) services.

- **Parameters:** None.
- **Returns:** JSON object with status of `rag_db` and `rag_fps` services.

**POST** `/upload-to-rag`

Uploads a document to the vector database for RAG context.

| Parameter | Type | Description |
|----|----|----|
| `file` | File | **Required**. The document to upload. Supported formats: PDF, TXT, MD, XML. Sent as `multipart/form-data`. |

**POST** `/import-rag-db`

Replaces the existing vector database with a provided DB file.

| Parameter | Type | Description                              |
|-----------|------|------------------------------------------|
| `dbfile`  | File | **Required**. The `.db` file to restore. |

## Endpoint Examples

This section provides practical examples of how to interact with the LLiMa Endpoint provided APIs using both command-line tools and Python. Examples will include the `/v1/chat/completions` and the `/stop` endpoints.

### Chat Completions

The following examples demonstrate how to send a request to the OpenAI-compatible `/v1/chat/completions` endpoint.

**cURL (Terminal)**

Use the `-N` flag to enable immediate streaming output. Replace `<modalix-ip>` with your device's IP address.

**1. Text Generation**

``` bash
$ curl -N -X POST "http://<modalix-ip>:9998/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      { "role": "user", "content": "Why is the sky blue?" }
    ],
    "stream": true
  }'
```

**2. Visual Language Model (Text + Image)**

To send an image, encode it as a base64 data URI and pass it within the `content` array.

``` bash
$ curl -N -X POST "http://<modalix-ip>:9998/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "image": "data:image/jpeg;base64,<YOUR_BASE64_STRING_HERE>"
          },
          {
            "type": "text",
            "text": "Describe this image"
          }
        ]
      }
    ]
  }'
```

**Python**

The following Python request structure handles both text-only and multimodal (image + text) requests.

``` python
#... INITIALIZATIONS AND IMPORTS

## Message fromat for image + text
messages = [{"role": "user", "content": USER_PROMPT}]                        

## Message fromat for image + text
messages = [{
  "role": "user",
  "content": [
    {"type": "image", "image": f"data:image/jpeg;base64,{encoded_img}"},
    {"type": "text", "text": USER_PROMPT}
  ]
}]                                                                          

payload = {"messages": messages, "stream": True}

response = requests.post("http://<modalix-ip>:9998/v1/chat/completions", json=payload, stream=True, timeout=60)

## ... STREAMED RESPONSE HANDLING
```

### Stopping Inference

If a generation is running too long or needs to be cancelled, use the `/stop` endpoint on the inference server.

**cURL (Terminal)**

``` bash
$ curl -X POST http://<modalix-ip>:9998/stop
```

**Python**

``` python
import requests

MODALIX_IP = "192.168.1.20" # Replace with your device IP
URL = f"http://{MODALIX_IP}:9998/stop"

try:
    response = requests.post(URL, timeout=10)

    if response.status_code == 200:
        print("Inference stopped successfully.")
    else:
        print(f"Failed to stop. Status: {response.status_code}")

except Exception as e:
    print(f"Error: {e}")
```

> [!NOTE]
> To stop the full application (including audio and RAG services), use the application-layer endpoint instead: `POST https://<modalix-ip>:5000/stop`
