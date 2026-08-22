# GenAI搭配LLiMa

LLiMa 是 GenAI 工具組，用於在 Model Compiler 中編譯、測試、進行基準測試、部署和執行 LLM、VLM 和 ASR 模型，並在 Modalix 上運行。

LLiMa 支援三種輸入格式：

- **Hugging Face safetensors** — 標準的 LLM 和 VLM 模型目錄
- **GGUF 檔案** — 以 GGUF 格式封裝的 LLM 模型。
- **壓縮張量模型**——預先量化過的 GPTQ/AWQ 樣式的 safetensor 模型。

SiMa.ai 同時也會發布預先編譯的 GenAI 模型位於
[Hugging Face](https://huggingface.co/simaai)。當已存在合適的模型時，從這裡開始。

對於具體情況 GenAI 範例，請參閱 [範例](https://developer.sima.ai/examples).

## LLiMa 可用性

LLiMa 編譯工具預設會安裝在 Model Compiler 中。
LLiMa 執行階段會原生安裝在 Modalix 上，作為 Neat 執行階段的一部分。
請參閱 [Neat Framework 安裝](/getting-started/neat-library/)，以了解執行階段的安裝流程。

## 貢獻

修改 LLiMa 本身的貢獻者應遵循
[《LLiMa 貢獻者指南》](contributing.md)，其中涵蓋了程式碼庫結構、開發環境、測試層級、模型輸入政策以及提交請求的要求。

## 支援的模型

下表顯示支援的模型架構及其功能：

| 模型架構 | 類型 | 支援的尺寸 |
|----|----|----|
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family) | LLM | [7b](https://huggingface.co/simaai/Llama-2-7b-chat-hf-a16w4) |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31) | LLM | [8b](https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4) |
| [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32) | LLM | [1b](https://huggingface.co/simaai/Llama-3.2-1B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Llama-3.2-3B-Instruct-a16w4) |
| [Gemma 1](https://huggingface.co/collections/google/gemma-release) | LLM | 2b、7b |
| [Gemma 2](https://huggingface.co/collections/google/gemma-2-release) | LLM | 2b、9b |
| [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) | LLM | [1b](https://huggingface.co/simaai/gemma-3-1b-it-a16w4), [4b](https://huggingface.co/simaai/gemma-3-4b-it-a16w4) |
| [ Phi 3.5 mini ](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-3.5-mini-instruct-a16w4) |
| [ Phi 4 mini ](https://huggingface.co/microsoft/Phi-4-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-4-mini-instruct-Autoround-a16w4) |
| [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25) | LLM | [0.5b](https://huggingface.co/simaai/Qwen2.5-0.5B-Instruct-Autoround-a16w4)、[1.5b](https://huggingface.co/simaai/Qwen2.5-1.5B-Instruct-GPTQ-a16w4)、[3b](https://huggingface.co/simaai/Qwen2.5-3B-Instruct-GPTQ-a16w4)、[7b](https://huggingface.co/simaai/Qwen2.5-7B-Instruct-GPTQ-a16w4) |
| [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) | LLM | [0.6b](https://huggingface.co/simaai/Qwen3-0.6B-Autoround-a16w4)、[1.7b](https://huggingface.co/simaai/Qwen3-1.7B-GPTQ-a16w4)、[4b](https://huggingface.co/simaai/Qwen3-4B-Instruct-2507-GPTQ-a16w4)、[8b](https://huggingface.co/simaai/Qwen3-8B-GPTQ-a16w4) |
| [Mistral 1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | LLM | [7b](https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2) | LLM | [350 公尺](https://huggingface.co/simaai/LFM2-350M-a16w4)、[1.2 億](https://huggingface.co/simaai/LFM2-1.2B-a16w4)、[2.6 億](https://huggingface.co/simaai/LFM2-2.6B-a16w4)。 |
| [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | VLM | [7b](https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4) |
| [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | VLM | [3b](https://huggingface.co/simaai/paligemma-3b-pt-224-a16w8) |
| [Gemma 3](https://huggingface.co/simaai/gemma3-siglip448-a16w4) | VLM | [4b](https://huggingface.co/simaai/gemma3-siglip448-a16w4) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4) | VLM | [E2B](https://huggingface.co/simaai/gemma4-E2B-it), [E4B](https://huggingface.co/simaai/gemma4-E4B-it) |
| [ Qwen 2.5 VL ](https://huggingface.co/collections/Qwen/qwen25-vl) | VLM | [3b](https://huggingface.co/simaai/Qwen2.5-VL-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4) |
| [ Qwen 3 VL ](https://huggingface.co/collections/Qwen/qwen3-vl) | VLM | [2b](https://huggingface.co/simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-VL-8B-Instruct-GPTQ-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2-vl) | VLM | [450 公尺](https://huggingface.co/simaai/LFM2-VL-450M-a16w4)，[16 億](https://huggingface.co/simaai/LFM2-VL-1.6B-a16w4)，[30 億](https://huggingface.co/simaai/LFM2-VL-3B-a16w4)。 |
| [Whisper](https://huggingface.co/openai/whisper-small) | ASR | [small](https://huggingface.co/simaai/whisper-small-a16w8) |

## 限制事項

| 限制類型 | 描述 |
|----|----|
| 模型架構 | 僅支援基於上述架構的模型。 |
| 模型參數 | 僅支援參數數量少於 100 億的模型。 |
| HF 模型 | 模型必須以本機 Hugging Face 目錄的形式提供，該目錄應包含 `config.json`、safetensor 權重，以及架構所需的權杖化器和處理器檔案。`generation_config.json` 則是可選的。 |
| GGUF 模型 | GGUF 格式僅適用於 LLM。VLM 必須從 Hugging Face 的 safetensors 格式進行編譯。請注意，與 Hugging Face safetensor 編譯相比，效能可能會降低。 |
| 壓縮張量模型 | 支援的 LLM 和 VLM 可以使用預先量化的 safetensor 模型（GPTQ/AWQ），這些模型是使用 llm-compressor 建立的。模型必須使用對稱量化，並且使用支援的 compressed-tensors 佈局。這些模型可以在維持高效能的同時，達到比標準 INT4 量化更好的準確度。 |
| Gemma3 VLM | 支援經過修改的 SigLip 448 視覺編碼器。 |
| LLAMA 3.2 視覺模型 | 視覺模型目前不支援。 |
