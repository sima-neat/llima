# Introduction to LLiMa

## Overview

The GenAI Model Compilation feature streamlines the process of compiling GenAI models from three input formats: HuggingFace safetensors, GGUF files, and pre-quantized compressed tensor models (e.g. GPTQ/AWQ models created with llm-compressor). For a wide set of different models like `Llama`, `Gemma`, `Phi`, `Qwen`, `Mistral` or `LFM` from Hugging Face, the SDK automatically generates all required binary/elf files along with the Python orchestration script, enabling direct execution on the Sima.ai Modalix platform.

SiMa has precompiled several popular models and published them on [Hugging Face](https://huggingface.co/simaai). LLiMa is not installed by default. To get started, create the LLiMa directory structure and install it globally on your Modalix device:

``` console
modalix:~$ cd /media/nvme && mkdir llima && cd llima
modalix:~$ sima-cli install -v 2.1.0 tools/llima -t full
```

This creates the required directories (including `/media/nvme/llima/models` for model storage) and makes the `llima` CLI available system-wide.

### Model Manager

LLiMa includes a model manager accessible via the `llima` CLI. It lets you search, download, and run precompiled models directly from the command line. Models are stored under `/media/nvme/llima/models` by default; this path can be overridden by setting the `LLIMA_MODELS_PATH` environment variable.

Browse available models:

``` console
modalix:~$ llima search
modalix:~$ llima search qwen
```

Download a model by name (without the `simaai/` org prefix):

``` console
modalix:~$ llima pull Qwen3-VL-8B-Instruct-a16w4
```

List and remove locally installed models:

``` console
modalix:~$ llima list
modalix:~$ llima rm Qwen3-VL-8B-Instruct-a16w4
```

Run a model directly in CLI or web mode:

``` console
modalix:~$ llima run Qwen3-VL-8B-Instruct-a16w4
modalix:~$ llima run Qwen3-VL-8B-Instruct-a16w4 --mode web
```

More details on the full set of `llima run` options can be found in [Runtime & Orchestration](runtime.md).

### GenAI Demo

For the full GenAI demo experience — including the web frontend and speech-to-text/text-to-speech support — use the `run.sh` script instead:

``` console
modalix:/media/nvme/llima$ ./run.sh
```

This prompts you to select a precompiled model and launches the complete demo application.

## Supported Models

The following table shows the supported model architectures and their capabilities:

| Model Architecture | Type | Supported Sizes |
|----|----|----|
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family) | LLM | [7b](https://huggingface.co/simaai/Llama-2-7b-chat-hf-a16w4) |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31) | LLM | [8b](https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4) |
| [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32) | LLM | 1b, [3b](https://huggingface.co/simaai/Llama-3.2-3B-Instruct-a16w4) |
| [Gemma 1](https://huggingface.co/collections/google/gemma-release) | LLM | 2b, 7b |
| [Gemma 2](https://huggingface.co/collections/google/gemma-2-release) | LLM | 2b, 9b |
| [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) | LLM | [1b](https://huggingface.co/simaai/gemma-3-1b-it-a16w4), [4b](https://huggingface.co/simaai/gemma-3-4b-it-a16w4) |
| [Phi 3.5 mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-3.5-mini-instruct-a16w4) |
| [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25) | LLM | [0.5b](https://huggingface.co/simaai/Qwen2.5-0.5B-instruct), [1.5b](https://huggingface.co/simaai/Qwen2.5-1.5B-instruct), 3b, [7b](https://huggingface.co/simaai/Qwen2.5-7B-instruct) |
| [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) | LLM | [0.6b](https://huggingface.co/simaai/Qwen3-0.6B), [1.7b](https://huggingface.co/simaai/Qwen3-1.7B), [4b](https://huggingface.co/simaai/Qwen3-4B-Instruct-2507), [8b](https://huggingface.co/simaai/Qwen3-8B) |
| [Mistral 1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | LLM | [7b](https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2) | LLM | 350m, 1.2b, 2.6b |
| [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | VLM | [7b](https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4) |
| [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | VLM | [3b](https://huggingface.co/simaai/paligemma-3b-pt-224-a16w8) |
| [Gemma 3](https://huggingface.co/simaai/gemma3-siglip448) | VLM | [4b](https://huggingface.co/simaai/gemma3-siglip448-a16w4) |
| [Qwen 2.5 VL](https://huggingface.co/collections/Qwen/qwen25-vl) | VLM | [3b](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [7b](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| [Qwen 3 VL](https://huggingface.co/collections/Qwen/qwen3-vl) | VLM | 2b, [4b](https://huggingface.co/simaai/Qwen3-VL-4B-Instruct-a16w4), [8b](https://huggingface.co/simaai/Qwen3-VL-8B-Instruct-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2-vl) | VLM | [450m](https://huggingface.co/simaai/LFM2-VL-450M-a16w4), [1.6b](https://huggingface.co/simaai/LFM2-VL-1.6B-a16w4), [3b](https://huggingface.co/simaai/LFM2-VL-3B-a16w4) |

## Limitations

| Limitation Type | Description |
|----|----|
| Model Architecture | Only models based on the architectures listed above are supported. |
| Model Parameters | Only models with parameter count less than 10B are supported. |
| HF Models | Models must be downloaded from Hugging Face and contain: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` and weights in safetensors format |
| GGUF Models | GGUF format is supported for LLMs only. VLMs must be compiled from the Hugging Face safetensors format. Note that performance may decrease compared to HuggingFace safetensor compilation. |
| Compressed Tensor Models | Pre-quantized safetensor models (GPTQ/AWQ) created with llm-compressor are supported for LLMs only. The model must use symmetric quantization. These models can achieve better accuracy than standard INT4 quantization while maintaining high performance. |
| Gemma3 VLM | Supported with modified SigLip 448 vision encoder |
| LLAMA 3.2 Vision | Vision models are not supported |
