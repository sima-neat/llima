# 모델 컴파일

## 개요

**Model Compiler**는 LLiMa 명령줄 도구 `llima-compile`를 제공하여,
Hugging Face의 safetensors 파일, GGUF 파일 또는 미리 양자화된
compressed-tensors 모델(GPTQ/AutoRound)에서 모델을 컴파일합니다.

``` console
llima-compile [options] <model_path>
```

### 모델 입력 형식

LLiMa는 세 개의 모델 입력 경로를 허용합니다. 체크포인트 가용성, 정확도 요구 사항, 그리고 해당 모델이 LLM인지 VLM인지에 따라 하나를 선택하십시오.

| 입력 | 설명 | 사용 시기 |
| --- | --- | --- |
| 원래 Hugging Face safetensors | FP/BF16 체크포인트를 사용하여 LLiMa 컴파일 과정 중에 양자화됩니다. | 정확히 일치하는 사전 양자화된 값이 없거나, 원래 가중치를 사용해야 합니다. |
| 사전 양자화된 Hugging Face safetensors (GPTQ/AutoRound) | 양자화된 가중치를 재사용하는 체크포인트이며, 이는 LLiMa에 의해 수행됩니다. | 컬렉션에서 정확히 일치하는 항목을 찾을 때 사용하는 것이 좋습니다. |
| GGUF | 기존의 양자화된 LLM 체크포인트입니다. | 편리한 LLM 대체 방법입니다. VLM에는 적용되지 않습니다. |

입력 형식만으로는 호환성을 보장할 수 없습니다. 모델 아키텍처, 크기, 토크나이저, 그리고 모든 다중 모달 구성 요소도 함께 지원되어야 합니다.

### SiMa.ai 사전 양자화된 모델부터 시작하세요.

:::tip 권장 입력 값
Hugging Face 또는 GGUF 형식의 원본 모델 가중치를 다운로드하기 전에 [SiMa.ai 사전 양자화 모델 컬렉션](https://huggingface.co/collections/simaai/pre-quantized-models)을 확인하세요.
요청한 아키텍처, 파라미터 크기, 변형, 모달리티에 대해 정확히 일치하는 모델이 있는 경우 해당 모델을 사용하세요.
:::

컬렉션 체크포인트는 특정 모델에 대한 사전 LLiMa compressed-tensors 아티팩트이며, 이를 `llima-compile`에 직접 전달할 수 있습니다. 이를 통해 추가적인 부동 소수점-양자화 컴파일러 단계를 거치지 않고, 정확도와 레이아웃을 이해하는 데 필요한 양자화 출처를 포함할 수 있습니다. 이는 컴파일된 Modalix 모델이 아닌, 컴파일러 입력입니다.

기존에 지원되는 모델을 사용자 지정으로 미세 조정하는 경우, 정확히 일치하는 컬렉션 저장소가 해당 모델에 대한 `quantize.py`, `recipe.yaml` 및 `versions.txt`를 제공할 수도 있습니다. 해당 저장소의 모델 카드를 읽고 문서화된 스크립트를 사용하십시오. 단순히 유사한 모델에서 레시피를 재사용하지 마십시오.

``` console
hf download simaai/<model-repository> \
    --revision <immutable-revision> \
    --local-dir <prequantized-model-directory>
llima-compile <prequantized-model-directory> -o <output-directory>
```

### 모델에 대해 설명합니다. LLiMa 기술을 갖춘 에이전트가 이를 작성합니다.

SiMa.ai와 LLiMa는 Neat 개발 환경(Neat SDK)에 포함된 기능을 통해 에이전트 기반 모델 컴파일을 즉시 지원합니다. 이러한 기능은 코딩 에이전트에게 LLM 및 VLM 호환성을 평가하고, 사용 가능한 경우 정확한 사전 양자화된 입력을 선택하고, 설치된 LLiMa CLI를 사용하고, Modalix 배포 및 검증 워크플로를 따르는 데 필요한 컨텍스트를 제공합니다.

권장되는 에이전트 기반 경로는 모델을 컴파일하고, 접근 가능한 Modalix DevKit에 배포하고, 결과를 검사하고, 진단하고, 컴파일을 개선할 수 있습니다. 기존 CLI 컴파일은 동일한 도구를 통해 직접 제어할 수 있는 병렬 경로로 유지됩니다. 둘 다 표준 검사 가능한 LLiMa 아티팩트를 생성하므로 선택한 모델, 명령, 옵션 및 출력을 검토하거나 요구 사항이 변경됨에 따라 두 워크플로 간에 전환할 수 있습니다. 에이전트 기반 컴파일을 활성화하려면 [Set up the Neat SDK](https://developer.sima.ai/software/getting-started/dev-environment/)를 참조하십시오.

예를 들어 자연어로 전체 워크플로를 요청하십시오.

``` text
Compile <model ID or local path> with LLiMa, deploy it to my Modalix at
<user@host>, and smoke-test it. Prefer an exact SiMa.ai pre-quantized
checkpoint when available.
```

에이전트는 모델 및 레시피의 출처를 기록하고, 설치된 버전의 CLI 계약을 준수하며, 지원되지 않는 모델 범위 또는 사용할 수 없는 하드웨어 검증이 감지될 경우 다른 모델이나 형식으로 조용히 대체하는 대신 해당 내용을 보고합니다.

### 컴파일 결과

기본 완전 파이프라인은 다음 디렉터리 구조를 생성합니다.

``` text
output_directory/
└── sima_files/                # Compiled model files
    ├── devkit/                # Runtime configuration and model data
    │   ├── tokenizer.json
    │   ├── vlm_config.json
    │   └── ...
    ├── mpk/                   # MPK archives with compiled binaries
    │   ├── layer_0.tar.gz
    │   └── ...
    ├── npy_files/             # LoRA adapter weights (only when compiled with LoRA)
    │   ├── <adapter_name>/
    │   │   └── *.npy
    │   └── ...
    └── ...
```

## 명령줄 인수

`llima-compile` 도구는 컴파일 프로세스를 사용자 정의하기 위한 다양한 인수를 허용합니다. 다음 표에는 사용 가능한 옵션이 설명되어 있습니다.

| 논쟁 | 설명 |
|----|----|
| `model_path` | 모델 경로를 입력하세요(HuggingFace 디렉터리, GGUF 파일 또는 미리 양자화된 압축된 텐서 디렉터리). |
| `-o, --output` | 컴파일된 파일이 저장될 출력 디렉터리입니다. 기본값은 모델 이름입니다. |
| `-c, --configuration_file` | 각 레이어별 정밀도를 구성하기 위한 Python 스크립트(예: 혼합 정밀도 사용 시). |
| `--max_num_tokens` | 최대 컨텍스트 길이입니다. 1024의 배수여야 합니다. 기본값: 4096. |
| `--resume` | 기존 파일을 건너뛰어 빌드를 재개하여 중단된 빌드를 이어갑니다. |
| `-j, --jobs` | 동시에 실행되는 컴파일 작업의 수입니다. 기본값: 실제 CPU 코어의 수입니다. |
| `--log_level` | 로그 기록 수준(DEBUG, INFO, WARNING, ERROR). 기본값: WARNING. |
| `--input_height` | 픽셀 단위의 입력 이미지 높이입니다. `--input_width`와 함께 제공해야 합니다. Qwen 2 VL, Qwen 3 VL, Gemma 4 모델에 필수이며, SigLIP2 모델의 설정된 크기를 변경할 때 선택적으로 사용할 수 있습니다. |
| `--input_width` | 픽셀 단위의 입력 이미지 너비입니다. `--input_height`와 함께 제공해야 합니다. Qwen 2 VL, Qwen 3 VL, Gemma 4 모델에 필수이며, SigLIP2 모델의 설정된 크기를 변경할 때 선택적으로 사용할 수 있습니다. |
| `--system_prompt` | CLI 모드와 모델 초기화를 위해 저장할 시스템 프롬프트입니다. |
| `--system_prompt_file` | 시스템 프롬프트를 포함하는 텍스트 파일의 경로입니다. |
| `--chat_template` | 컴파일된 모델에 저장할 채팅 템플릿 문자열입니다. 시스템 프롬프트 및 채팅 템플릿 파일 옵션과 상호 배타적입니다. |
| `--chat_template_file` | 채팅 템플릿이 포함된 파일의 경로입니다. 시스템 프롬프트 옵션과 `--chat_template`는 상호 배타적입니다. |

:::note
대부분의 모델은 최대 8192개의 토큰까지의 컨텍스트 길이를 지원합니다. 8K 컨텍스트 길이를 사용하려면 `--max_num_tokens 8192`를 사용하세요.
:::

| 고급 논증 | 설명 |
|----|----|
| `--language_group_size` | 사전 학습 단계에서 병렬 토큰 처리를 위한 배치 크기입니다. 더 큰 값(예: 256)을 사용하면 큰 입력 프롬프트에 대해 TTFT(Time To First Token)를 개선할 수 있지만, 작은 입력 프롬프트에 대해서는 TTFT를 저하시킬 수 있습니다. 기본값: 128. |
| `--future_token_mask_size` | 토큰 위치 간에 컴파일된 모델을 재사용하기 위한 마스크 크기입니다. 값이 클수록 컴파일된 바이너리 파일의 수가 줄어들지만, 초당 토큰 수(TPS)가 감소할 수 있습니다. 기본값: 128. |
| `--enable_filter_sharing` | 그룹 모델과 단일 모델 간에 필터 공유를 활성화하여 DRAM 사용량을 줄이되, TTFT(첫 번째 토큰 생성까지의 시간)는 증가하고 TPS(초당 토큰 수)는 감소합니다. 이는 두 모델 유형 모두 동일한 정밀도를 사용할 때만 효과가 있으며, LoRA를 사용하여 컴파일할 때 필수적으로 적용해야 합니다. |
| `--no-quantize_embeddings` | 기본적으로 지원되는 LLM 및 VLM에 대해 활성화되어 있는 임베딩 테이블 양자화를 비활성화합니다. |
| `--no-quantize_kv_cache` | 기본적으로 활성화되어 있는 KV 캐시 양자화를 비활성화합니다. |
| `--return_logits` | 마지막 레이어의 출력에서 로짓 값을 반환합니다(모델 평가에 필요). |
| `--draft_model_path` | 추론 디코딩을 위한 EAGLE3 초안 모델의 경로입니다. |
| `--lora_name` | 기본 모델과 함께 컴파일되는 LoRA 어댑터의 이름입니다. |
| `--lora_path` | 기본 모델과 함께 컴파일할 때 사용할 LoRA 어댑터 디렉터리의 경로입니다. |
| `--compile_lora`, `--no-compile_lora` | LoRA 경로가 제공될 때 어댑터 가중치 컴파일을 활성화하거나 비활성화합니다. 기본적으로 활성화되어 있습니다. |

## 시스템 프롬프트

컴파일된 모델 구성에 시스템 프롬프트를 저장하려면 `--system_prompt` 또는 `--system_prompt_file`을 사용하세요. 인수는 서로 배타적입니다.

``` console
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
    --system_prompt "You are a concise technical assistant." \
    -o Llama-3.2-3B-Instruct_out
```

CLI 모드에서는 이 설정이 기본 시스템 프롬프트가 됩니다. 세션 중에 `set system <prompt>`로 대체하거나 `clear system`을 사용하여 제거할 수 있습니다.

웹 모드/GenAIServer에서는 저장된 프롬프트가 모델 초기화 과정에서 사용되며, 첫 번째 요청에 대해 캐시될 수 있습니다. 이 프롬프트는 API 요청에 자동으로 추가되지 않습니다. 클라이언트는 첫 번째 요청과 후속 요청의 `messages` 배열에 시스템 프롬프트를 포함해야 합니다.

## 설정 파일

구성 파일은 각 컴파일러 단위에 대해 컴파일을 사용자 정의하여 혼합 정밀도 및 선택적 컴파일을 가능하게 합니다.

LLM 추론은 두 개의 뚜렷한 단계로 구성되며, 컴파일러는 각 단계에 대해 최적화된 모델을 생성합니다.

- **사전 처리(그룹 모델)**: 입력 프롬프트를 `language_group_size` 단위(예: 한 번에 128개의 토큰)로 일괄 처리합니다. 이 단계에서는 TTFT(첫 번째 토큰 생성까지의 시간)를 결정하며 처리량을 최적화하는 데 중점을 둡니다.
- **디코딩(단일 토큰 모델)**: 출력 토큰을 한 번에 하나씩 자동 회귀 방식으로 생성합니다. 이 단계에서는 초당 토큰 수(TPS)를 결정하며, 낮은 지연 시간으로 출력을 생성하도록 최적화됩니다.

이러한 단계는 서로 다른 성능 특성을 가지므로, 구성 함수에서 `is_group` 플래그를 사용하여 각 단계에 다른 양자화 전략을 적용할 수 있습니다.

**입력 매개변수**

`get_layer_configuration` 함수는 각 컴파일러 단위에 대해 호출되며, 다음 값을 받습니다.

- `model_properties`: `{"num_hidden_layers": int}`를 포함하는 딕셔너리

- `layer`: 다음 항목이 포함된 사전:
  - `"part"`: `"PRE"`, `"CACHE"`, `"POST"`와 같은 논리적 구성 요소
    `"VISION"`, `"DRAFT_FC"` 또는 `"PER_LAYER"`
  - `"is_group"`: 다중 토큰/그룹 변형의 경우 `True`이고, 그렇지 않은 경우 `False`입니다.
  - `"index"`: 해당 컴파일러 단위의 인덱스입니다. `"PRE"` 및 `"POST"`의 경우
    일반적으로 트랜스포머 레이어에 해당합니다. `"CACHE"`의 경우 트랜스포머 레이어가 아닌 캐시 또는 토큰 위치 변형을 식별합니다.

**반환 값**

이 함수는 다음과 같은 키-값 쌍을 포함하는 딕셔너리를 반환합니다.

- `"precision"`: 양자화 수준(선택 사항, 기본값: `"BF16"`).
  - `"BF16"`: 최고 정밀도 - 최고의 품질, 가장 큰 크기, 가장 느린 속도
  - `"A_BF16_W_INT8"`: 중간 수준의 양자화 - 양호한 품질, 적당한 크기
  - `"A_BF16_W_INT4"`: 높은 양자화 수준 - 허용 가능한 품질, 가장 작은 크기, 가장 빠른 속도

- `"compile"`: 이 레이어의 컴파일을 건너뛰려면 `False`로 설정합니다(선택 사항, 기본값: `True`).

- `"lora"`: 이 레이어에 대한 LoRA 모드(선택 사항, 기본값: `"LORA_DISABLED"`)
  - `"LORA_DISABLED"`: 이 레이어에서는 LoRA를 지원하지 않습니다. 구성 파일이 제공되지 않을 때 기본적으로 적용되며, 그 결과 어댑터 오버헤드가 없는 표준 모델이 생성됩니다.
  - `"LORA_BRANCH"`: 기본 모델과 함께 가중치가 0인 병렬 LoRA 분기를 컴파일합니다. 어댑터 가중치는 런타임에 `.npy` 파일에서 로드되므로, 모델을 다시 시작하지 않고도 어댑터 간의 동적 전환이 가능합니다. 필요에 따라 어댑터를 즉시 전환해야 할 때 이 모드를 사용하십시오.
  - `"LORA_MERGED"`: LoRA 가중치는 런타임에 기본 모델 가중치에 병합됩니다. 어댑터는 세션 동안 영구적으로 활성화되며, 전환하거나 제거할 수 없습니다. 어댑터를 항상 적용하고 동적으로 전환할 필요가 없을 때 이 모드를 사용하십시오.

:::note
**권장 사항:** 사전 학습 단계에서 품질을 유지하기 위해 그룹 레이어에는 INT8(`A_BF16_W_INT8`)을 사용하고, 빠른 생성을 위해 단일 토큰 레이어에는 INT4(`A_BF16_W_INT4`)을 사용하며, 이미지 이해 품질을 유지하기 위해 비전 인코더에는 BF16을 사용합니다. 대부분의 모델에서 이 구성은 모델 정확도, 처리량 및 메모리 사용량 간의 최적의 균형을 제공합니다.
:::

## 예시

**예시 1: 간단한 LLM 모델 컴파일**

Hugging Face에서 다운로드한 Llama 모델을 기본 설정으로 컴파일합니다.

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

이를 통해 다음과 같은 효과를 얻을 수 있습니다.

- 모든 레이어에 기본 BF16 정밀도를 사용합니다.
- 컨텍스트 길이를 4096 토큰으로 설정합니다.
- `Llama-3.2-3B-Instruct_out` 디렉터리에 출력합니다.

**예제 2: 사용자 지정 컨텍스트 길이를 사용하여 컴파일**

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile --max_num_tokens 4096 Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

이를 통해 다음과 같은 효과를 얻을 수 있습니다.

- 모든 레이어에 기본 BF16 정밀도를 사용합니다.
- 컨텍스트 길이를 4096 토큰으로 설정합니다.
- `Llama-3.2-3B-Instruct_out` 디렉터리에 출력합니다.

**예제 3: 혼합 정밀도를 사용하여 Gemma 3 VLM 모델 컴파일**

Gemma 3 VLM과 같이 복잡한 모델의 경우, 서로 다른 레이어에 대해 다른 정밀도를 지정해야 할 수 있습니다(예: 비전 인코더를 BF16으로 유지).

1.  **모델 다운로드**:

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download simaai/gemma3-siglip448 --local-dir gemma-3-model
    ```

2.  **구성 파일을 생성합니다.** (예: `config.py`)

    ``` python
    def get_layer_configuration(model_properties, layer):
        # Keep vision encoder in full precision
        if layer["part"] == "VISION":
            precision = "BF16"
        # Use INT8 for batch processing layers (better quality)
        elif layer["is_group"]:
            precision = "A_BF16_W_INT8"
        # Use INT4 for single-token layers (smaller size)
        else:
            precision = "A_BF16_W_INT4"
        return {"precision": precision}
    ```

3.  **컴파일러를 실행합니다**:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile -c config.py --max_num_tokens 2048 gemma-3-model -o gemma-3-model_out
    ```

**예시 4: 고급 구성**

트랜스포머 레이어별로 제어하는 혼합 정밀도:

``` python
def get_layer_configuration(model_properties, layer):
    # PRE and POST indices normally identify transformer layers.
    if layer["part"] in {"PRE", "POST"} and layer["index"] < 4:
        return {"precision": "BF16"}

    # Keep every required compiler unit and use INT8 elsewhere.
    return {"precision": "A_BF16_W_INT8"}
```

`"CACHE"` 인덱스를 트랜스포머 레이어 인덱스로 해석하지 마십시오. 캐시 변형을 생략하면 컴파일된 결과물이 불완전해지고 런타임에 사용할 수 없게 될 수 있습니다.

**예제 5: LoRA를 사용하여 LLM 컴파일**

LoRA (저랭크 적응)를 사용하면 기본 모델을 미세 조정하고 어댑터를 동적으로 적용하거나 런타임에 기본 모델을 다시 컴파일하지 않고 제거할 수 있습니다. 기본 모델은 병렬 LoRA 분기와 함께 컴파일됩니다(0으로 초기화됨). 어댑터 가중치는 별도로 `.npy` 파일로 컴파일되어 필요에 따라 로드됩니다.

:::note
LoRA를 사용하여 컴파일할 때는 필터 공유가 필요합니다. `--enable_filter_sharing`를 사용하여 활성화하세요. 더 나은 정확도를 위해 INT4가 지정되었더라도 LoRA 브랜치는 항상 INT8으로 컴파일됩니다.
:::

1.  **기본 모델과 LoRA 어댑터를 다운로드하세요**:

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B-Instruct --local-dir Llama-3.2-3B-Instruct
    sima-user@docker-image-id:/home/docker$ hf download <org>/<lora-adapter> --local-dir my-lora
    ```

2.  **구성 파일을 생성합니다.** (예: `lora_config.py`)

    `lora` 키는 각 레이어별로 LoRA 모드를 제어합니다. `"LORA_BRANCH"`를 사용하여 런타임에 동적 전환을 활성화합니다.

    ``` python
    def get_layer_configuration(model_properties, layer):
        if layer["is_group"]:
            return {"precision": "A_BF16_W_INT8", "compile": True, "lora": "LORA_BRANCH"}
        else:
            return {"precision": "A_BF16_W_INT4", "compile": True, "lora": "LORA_BRANCH"}
    ```

3.  **LoRA 어댑터를 사용하여 기본 모델을 컴파일합니다**:

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter \
        --lora_path my-lora \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

    이 작업은 기본 모델을 하나의 LoRA 분기와 함께 컴파일하고 어댑터 가중치를 `Llama-3.2-3B-lora-out/sima_files/npy_files/my_adapter/`에 자동으로 컴파일합니다.

`--lora_name` 및 `--lora_path`를 반복하여 동일한 단계에서 여러 어댑터를 컴파일할 수 있습니다.

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter_A --lora_path my-lora_A \
        --lora_name my_adapter_B --lora_path my-lora_B \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

4.  기본 모델을 다시 컴파일하지 않고 **더 많은 어댑터를 추가하려면** 각 추가 어댑터에 대해 `llima-compile-lora`를 사용하세요.

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile-lora Llama-3.2-3B-Instruct ./lora-c \
        -w Llama-3.2-3B-lora-out/sima_files/mpk \
        -o Llama-3.2-3B-lora-out/sima_files/npy_files/adapter_c
    ```

    **llima-compile-lora 명령어 인자**

    | 논쟁 | 설명 |
    |----|----|
    | `base_path` | 원본 기본 모델 디렉터리의 경로(HuggingFace 형식). |
    | `lora_path` | 컴파일할 LoRA 어댑터 디렉터리의 경로입니다. |
    | `-w, --weight_map_path` | **필수.** 기본 모델 컴파일부터 시작하여 `mpk/` 폴더까지의 경로입니다. 어댑터를 컴파일하는 데 필요한 가중치 맵이 포함되어 있습니다. |
    | `-o, --output` | 컴파일된 어댑터의 출력 디렉터리 `.npy` 파일. 기본적으로 어댑터 디렉터리 이름을 사용합니다. |
