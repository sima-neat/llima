# GenAI와 LLiMa

LLiMa는 Model Compiler 내의 GenAI 도구 모음으로, LLM, VLM 및 ASR 모델을 컴파일, 테스트, 벤치마크, 배포 및 실행하는 데 사용됩니다. 이 모든 작업은 Modalix에서 수행됩니다.

LLiMa는 세 가지 입력 형식을 지원합니다.

- **Hugging Face safetensors** — 표준 LLM 및 VLM 모델 디렉터리
- **GGUF 파일** — GGUF 형식으로 패키징된 LLM 모델
- **압축된 텐서 모델** — 미리 양자화된 GPTQ/AWQ 스타일의 세이프텐서 모델

SiMa.ai는 또한 GenAI 모델을 미리 컴파일하여 [ Hugging Face ](https://huggingface.co/simaai)에 게시합니다. 적절한 모델이 이미 존재하는 경우 해당 위치에서 시작하세요.

구체적인 GenAI 데모는 [ 예제 ](https://developer.sima.ai/examples)를 참조하세요.

## LLiMa 사용 가능 여부

LLiMa 컴파일 도구는 기본적으로 Model Compiler에 설치됩니다.
LLiMa 런타임은 Neat 런타임의 일부로 Modalix에 기본적으로 설치됩니다.
런타임 설치 흐름에 대해서는 [Neat Framework 설치](/getting-started/neat-library/)를 참조하십시오.

## 기여

LLiMa 자체를 수정하는 기여자는 저장소 구조, 개발 환경, 테스트 단계, 모델 입력 정책 및 풀 리퀘스트 요구 사항에 대해 [ LLiMa 기여자 가이드 ](contributing.md)를 따라야 합니다.

## 지원되는 모델

다음 표는 지원되는 모델 아키텍처와 해당 기능 목록을 보여줍니다.

| 모델 아키텍처 | 유형 | 지원되는 크기 |
|----|----|----|
| [Llama 2](https://huggingface.co/collections/meta-llama/llama-2-family) | LLM | [7b](https://huggingface.co/simaai/Llama-2-7b-chat-hf-a16w4) |
| [Llama 3.1](https://huggingface.co/collections/meta-llama/llama-31) | LLM | [8b](https://huggingface.co/simaai/Llama-3.1-8B-Instruct-a16w4) |
| [Llama 3.2](https://huggingface.co/collections/meta-llama/llama-32) | LLM | [1b](https://huggingface.co/simaai/Llama-3.2-1B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Llama-3.2-3B-Instruct-a16w4) |
| [Gemma 1](https://huggingface.co/collections/google/gemma-release) | LLM | 2b, 7b |
| [Gemma 2](https://huggingface.co/collections/google/gemma-2-release) | LLM | 2b, 9b |
| [Gemma 3](https://huggingface.co/collections/google/gemma-3-release) | LLM | [1b](https://huggingface.co/simaai/gemma-3-1b-it-a16w4), [4b](https://huggingface.co/simaai/gemma-3-4b-it-a16w4) |
| [ Phi 3.5 mini ](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | LLM | [3.8b](https://huggingface.co/simaai/Phi-3.5-mini-instruct-a16w4) |
| [Qwen 2.5](https://huggingface.co/collections/Qwen/qwen25) | LLM | [0.5b](https://huggingface.co/simaai/Qwen2.5-0.5B-Instruct-GPTQ-a16w4), [1.5b](https://huggingface.co/simaai/Qwen2.5-1.5B-Instruct-GPTQ-a16w4), [3b](https://huggingface.co/simaai/Qwen2.5-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-7B-Instruct-GPTQ-a16w4) |
| [Qwen 3](https://huggingface.co/collections/Qwen/qwen3) | LLM | [0.6b](https://huggingface.co/simaai/Qwen3-0.6B-GPTQ-a16w4), [1.7b](https://huggingface.co/simaai/Qwen3-1.7B-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-4B-Instruct-2507-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-8B-GPTQ-a16w4) |
| [Mistral 1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | LLM | [7b](https://huggingface.co/simaai/Mistral-7B-Instruct-v0.3-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2) | LLM | [350m](https://huggingface.co/simaai/LFM2-350M-a16w4), [1.2b](https://huggingface.co/simaai/LFM2-1.2B-a16w4), [2.6b](https://huggingface.co/simaai/LFM2-2.6B-a16w4) |
| [Llava 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | VLM | [7b](https://huggingface.co/simaai/llava-1.5-7b-hf-a16w4) |
| [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | VLM | [3b](https://huggingface.co/simaai/paligemma-3b-pt-224-a16w8) |
| [Gemma 3](https://huggingface.co/simaai/gemma3-siglip448-a16w4) | VLM | [4b](https://huggingface.co/simaai/gemma3-siglip448-a16w4) |
| [Gemma 4](https://huggingface.co/collections/google/gemma-4) | VLM | [E2B](https://huggingface.co/simaai/gemma4-E2B-it), [E4B](https://huggingface.co/simaai/gemma4-E4B-it) |
| [ Qwen 2.5 VL ](https://huggingface.co/collections/Qwen/qwen25-vl) | VLM | [3b](https://huggingface.co/simaai/Qwen2.5-VL-3B-Instruct-GPTQ-a16w4), [7b](https://huggingface.co/simaai/Qwen2.5-VL-7B-Instruct-GPTQ-a16w4) |
| [ Qwen 3 VL ](https://huggingface.co/collections/Qwen/qwen3-vl) | VLM | [2b](https://huggingface.co/simaai/Qwen3-VL-2B-Instruct-GPTQ-a16w4), [4b](https://huggingface.co/simaai/Qwen3-VL-4B-Instruct-GPTQ-a16w4), [8b](https://huggingface.co/simaai/Qwen3-VL-8B-Instruct-GPTQ-a16w4) |
| [LFM 2](https://huggingface.co/collections/LiquidAI/lfm2-vl) | VLM | [450m](https://huggingface.co/simaai/LFM2-VL-450M-a16w4), [1.6b](https://huggingface.co/simaai/LFM2-VL-1.6B-a16w4), [3b](https://huggingface.co/simaai/LFM2-VL-3B-a16w4) |
| [Whisper](https://huggingface.co/openai/whisper-small) | ASR | [small](https://huggingface.co/simaai/whisper-small-a16w8) |

## 제한 사항

| 제한 유형 | 설명 |
|----|----|
| 모델 아키텍처 | 위에 나열된 아키텍처를 기반으로 하는 모델만 지원됩니다. |
| 모델 파라미터 | 100억 개 미만의 파라미터 수를 가진 모델만 지원됩니다. |
| HF 모델 | 모델은 `config.json`, 안전 텐서 가중치, 그리고 해당 아키텍처에서 요구하는 토크나이저 및 프로세서 파일을 포함하는 로컬 Hugging Face 디렉터리로 제공되어야 합니다. `generation_config.json`은 선택 사항입니다. |
| GGUF 모델 | GGUF 형식은 LLM에만 적용됩니다. VLM은 Hugging Face의 safetensors 형식으로 컴파일해야 합니다. 성능은 Hugging Face safetensor 컴파일에 비해 저하될 수 있다는 점에 유의하십시오. |
| 압축된 텐서 모델 | 지원되는 LLM 및 VLM은 llm-compressor를 사용하여 생성된 사전 양자화된 safetensor 모델(GPTQ/AWQ)을 사용할 수 있습니다. 모델은 대칭 양자화를 사용하고 지원되는 compressed-tensors 레이아웃을 사용해야 합니다. 이러한 모델은 높은 성능을 유지하면서 표준 INT4 양자화보다 더 나은 정확도를 달성할 수 있습니다. |
| Gemma3 VLM | 수정된 SigLip 448 비전 인코더를 지원합니다. |
| 라마 3.2 비전 | 비전 모델은 지원되지 않습니다. |
