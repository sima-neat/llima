# GenAI with LLiMa

LLiMa is the GenAI toolkit in Model Compiler for compiling, testing,
benchmarking, deploying, and running LLM, VLM, and ASR models on Modalix.

LLiMa supports three input formats:

- **Hugging Face safetensors** — standard LLM and VLM model directories
- **GGUF files** — LLM models packaged in GGUF format
- **Compressed tensor models** — pre-quantized GPTQ/AWQ-style safetensor models

SiMa.ai also publishes precompiled GenAI models on
[Hugging Face](https://huggingface.co/simaai). Start there when a suitable model
already exists.

For concrete GenAI demos, see the [examples](https://developer.sima.ai/examples).

## LLiMa Availability

LLiMa compilation tools are installed by default in Model Compiler.
The LLiMa runtime is installed natively on Modalix as part of the Neat runtime.
See [Neat Framework installation](/getting-started/neat-library/)
for the runtime installation flow.

## Contributing

Contributors changing LLiMa itself should follow the
[LLiMa Contributor Guide](contributing.md) for repository structure,
development environments, test tiers, model-input policy, and pull-request
requirements.

## Supported Models

The following table shows the supported model architectures and their
capabilities:

| Model Architecture | Type | Supported Sizes |
|----|----|----|
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family) | LLM | [7b](https://huggingface.co/simaai/Llama-2-7b-chat-hf-a16w4) |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31) | LLM | [8b](https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4) |
| [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32) | LLM | [1b](https://huggingface.co/simaai/Llama-3.2-1B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Llama-3.2-3B-Instruct-a16w4) |
| [Gemma 1](https://huggingface.co/collections/google/gemma-release) | LLM | 2b, 7b |
| [Gemma 2](https://huggingface.co/collections/google/gemma-2-release) | LLM | 2b, 9b |
| [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) | LLM | [1b](https://huggingface.co/simaai/gemma-3-1b-it-a16w4), [4b](https://huggingface.co/simaai/gemma-3-4b-it-a16w4) |
| [Phi 3.5 mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-3.5-mini-instruct-a16w4) |
| [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25) | LLM | [0.5b](https://huggingface.co/simaai/Qwen2.5-0.5B-Instruct-GPTQ-a16w4), [1.5b](https://huggingface.co/simaai/Qwen2.5-1.5B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Qwen2.5-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-7B-Instruct-GPTQ-a16w4) |
| [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) | LLM | [0.6b](https://huggingface.co/simaai/Qwen3-0.6B-GPTQ-a16w4), [1.7b](https://huggingface.co/simaai/Qwen3-1.7B-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-4B-Instruct-2507-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-8B-GPTQ-a16w4) |
| [Mistral 1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | LLM | [7b](https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2) | LLM | [350m](https://huggingface.co/simaai/LFM2-350M-a16w4), [1.2b](https://huggingface.co/simaai/LFM2-1.2B-a16w4), [2.6b](https://huggingface.co/simaai/LFM2-2.6B-a16w4) |
| [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | VLM | [7b](https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4) |
| [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | VLM | [3b](https://huggingface.co/simaai/paligemma-3b-pt-224-a16w8) |
| [Gemma 3](https://huggingface.co/simaai/gemma3-siglip448-a16w4) | VLM | [4b](https://huggingface.co/simaai/gemma3-siglip448-a16w4) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4) | VLM | [E2B](https://huggingface.co/simaai/gemma4-E2B-it), [E4B](https://huggingface.co/simaai/gemma4-E4B-it) |
| [Qwen 2.5 VL](https://huggingface.co/collections/Qwen/qwen25-vl) | VLM | [3b](https://huggingface.co/simaai/Qwen2.5-VL-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4) |
| [Qwen 3 VL](https://huggingface.co/collections/Qwen/qwen3-vl) | VLM | [2b](https://huggingface.co/simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-VL-8B-Instruct-GPTQ-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2-vl) | VLM | [450m](https://huggingface.co/simaai/LFM2-VL-450M-a16w4), [1.6b](https://huggingface.co/simaai/LFM2-VL-1.6B-a16w4), [3b](https://huggingface.co/simaai/LFM2-VL-3B-a16w4) |
| [Whisper](https://huggingface.co/openai/whisper-small) | ASR | [small](https://huggingface.co/simaai/whisper-small-a16w8) |

## Limitations

| Limitation Type | Description |
|----|----|
| Model Architecture | Only models based on the architectures listed above are supported. |
| Model Parameters | Only models with parameter count less than 10B are supported. |
| HF Models | Models must be available as a local Hugging Face directory containing `config.json`, safetensor weights, and the tokenizer and processor files required by the architecture. `generation_config.json` is optional. |
| GGUF Models | GGUF format is supported for LLMs only. VLMs must be compiled from the Hugging Face safetensors format. Note that performance may decrease compared to Hugging Face safetensor compilation. |
| Compressed Tensor Models | Supported LLMs and VLMs can use pre-quantized safetensor models (GPTQ/AWQ) created with llm-compressor. The model must use symmetric quantization and a supported compressed-tensors layout. These models can achieve better accuracy than standard INT4 quantization while maintaining high performance. |
| Gemma3 VLM | Supported with modified SigLip 448 vision encoder |
| LLAMA 3.2 Vision | Vision models are not supported |
