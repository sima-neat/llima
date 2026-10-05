# LLiMa 명령줄 인터페이스

`llima` CLI를 사용하여 Modalix에서 사전 컴파일된 모델을 관리하고 간단한 런타임 테스트를 수행합니다. 이는 모델이 로드되고, 프롬프트를 수신하고, 출력을 생성하는지 확인하는 데 유용합니다. 그런 다음 Neat Framework의 직접 API 또는 Neat GenAI 서버 엔드포인트와 통합하기 전에 이를 확인하는 데 도움이 됩니다.

## 모델 관리자

LLiMa는 `llima` CLI를 통해 모델 관리자를 제공합니다. 이를 통해 미리 컴파일된 모델을 검색, 다운로드, 목록으로 표시, 제거하고 명령줄에서 직접 실행할 수 있습니다. 모델은 기본적으로 `/media/nvme/llima/models`에 저장됩니다. 다른 모델 디렉터리를 사용하려면 `LLIMA_MODELS_PATH`를 설정하십시오.

사용 가능한 모델을 찾아보세요:

``` console
modalix:~$ llima search
modalix:~$ llima search qwen
```

이름으로 모델을 다운로드하되, `simaai/` 조직 접두사는 사용하지 마세요.

``` console
modalix:~$ llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

모델 아티팩트를 동시에 다운로드하며, 가장 큰 아티팩트부터 먼저 다운로드하도록 예약합니다. 일시적인 HTTP 오류는 자동으로 재시도됩니다. 동일한 모델에 대한 다운로드는 순차적으로 진행되며, 다운로드를 취소해도 이미 다운로드되고 검증된 모든 아티팩트는 유지됩니다.

로컬에 설치된 모델 목록을 확인하고 제거합니다.

``` console
modalix:~$ llima list
modalix:~$ llima rm Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## LLiMa를 실행 중입니다.

Modalix에서 초기 모델 검증을 위해 간단한 런타임으로 `llima run`을 사용합니다.

CLI 모드에서는 기본적으로 채팅 기록이 활성화됩니다. 각 프롬프트와 응답은 사용자가 `clear history`를 사용하여 기록을 지울 때까지 다음 대화의 맥락으로 유지됩니다. 프롬프트와 함께 제출한 이미지도 기록의 일부로 유지됩니다. `clear history`는 제출된 프롬프트, 응답 및 모든 이미지를 삭제합니다. 설정된 시스템 프롬프트는 `clear system`으로 삭제하거나 `set system`으로 교체할 때까지 활성 상태로 유지됩니다.

``` console
modalix:~$ llima run <model> [options]
```

| 논쟁 | 설명 |
|----|----|
| `model` | 모델 ID 또는 경로(예: `Qwen3-VL-8B-Instruct-a16w4`). |
| `--stt_model_path` | 음성-텍스트 변환 모델에 대한 ELF 파일 경로(선택 사항). |

사용 가능한 모든 옵션에 대해 `llima run -h`를 실행합니다.

자동 임베딩 오프로드를 비활성화하고 테이블을 DRAM에 유지하려면 `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off llima run <model>`을 실행합니다.

**예시**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## 대화형 명령어

CLI 모드에서 `llima run`을 실행하면 프롬프트에서 다음 명령어를 사용하십시오.

| 명령 | 설명 |
|----|----|
| `add image <file>` | 현재 프롬프트 컨텍스트에 이미지를 추가합니다. |
| `set system <prompt>` | 시스템 프롬프트를 설정합니다. |
| `clear system` | 시스템 프롬프트, 채팅 기록, 이미지를 모두 삭제합니다. |
| `clear history` | 시스템 프롬프트는 유지하면서 제출된 프롬프트, 응답 및 모든 이미지를 삭제합니다. |
| `print history` | 채팅 기록을 인쇄합니다. |
| `set audio <file>` | 음성 파일을 텍스트로 변환할 파일로 설정합니다. |
| `set language <lang>` | 음성 인식에 사용되는 언어 설정을 지정합니다. |
| `set lora <name>` | `npy_files` 폴더에서 LoRA 가중치를 사용하세요. |
| `unset lora` | LoRA 모델을 기본 모델로 되돌립니다. |
| `enable-thinking` | 사고 모드를 활성화하고 채팅 기록을 삭제합니다. |
| `disable-thinking` | 사고 모드를 비활성화하고 채팅 기록을 삭제합니다. |
| `quit` | 그만두세요. |
| `help` | 사용 가능한 명령어를 출력합니다. |


## KV 캐시 축출

기본적으로 컴파일된 컨텍스트(`max_num_tokens`)는 엄격한 한도입니다. 이보다 긴 프롬프트는 거부되고, 생성은 "Cache full"과 함께 중단됩니다. KV 캐시 축출을 사용하면 런타임이 대신 캐시된 토큰을 버리므로, 대화를 모델의 학습된 컨텍스트 길이(최대 약 65,000 토큰)까지 이어갈 수 있습니다. 축출된 토큰은 사라지므로 모델은 더 이상 해당 토큰에 관한 질문에 답할 수 없습니다. 대화가 컴파일된 컨텍스트에 들어가는 동안에는 출력이 바뀌지 않습니다.

활성화하려면 배포된 `devkit/vlm_config.json`의 `pipeline_cfg`에 `kv_eviction`을 추가하거나 `VisionLanguageModel::set_kv_eviction()`을 호출하세요.

``` json
"pipeline_cfg": {
    "max_num_tokens": 2048,
    "kv_eviction": { "policy": "keydiff" }
}
```

| 필드 | 설명 |
|----|----|
| `policy` | `off`(기본값), `sink_window`(최신 토큰 유지) 또는 `keydiff`(키가 평균 키와 가장 다른 토큰 유지). |
| `budget_tokens` | 선택 사항. 각 축출에서 유지되는 토큰 수로, 모델의 컴파일된 프리필 그룹 오프셋 중 하나여야 합니다(오류 메시지에 목록이 표시됩니다). 기본값은 캐시의 7/8이며, 각 축출은 캐시의 8분의 1(최소 프리필 그룹 하나)을 확보합니다. |

축출은 항상 시스템 프롬프트와 도구 정의, 처음 토큰, 최신 토큰을 유지합니다. 시스템 프롬프트와 도구 정의는 예산 안에 프리필 그룹 하나(최소 128 토큰)를 위한 공간을 남겨야 하며, 그렇지 않으면 모든 요청이 실패합니다.

이전 메시지가 변경 없이 다시 전송되는 한 새 채팅 턴은 캐시에서 이어집니다. 축출 후 이전 메시지가 바뀌면, 예를 들어 클라이언트가 기록을 잘라내거나, 채팅 템플릿이 이전 Qwen3 사고 블록을 제거하거나, 템플릿이 시스템 프롬프트를 최신 사용자 메시지로 옮기는 경우(Mistral v0.3), 런타임은 전체 대화를 다시 처리합니다.

모든 레이어가 전체 어텐션을 사용하고 RoPE 스케일링이 기본, 선형 또는 Llama 3인 텍스트 전용 모델만 지원됩니다. 그 외의 모델은 지원되지 않는 기능을 알려 주는 오류와 함께 로드에 실패합니다. GGUF 모델은 배포된 모델에 학습된 컨텍스트 길이가 기록되지 않으므로 아직 지원되지 않습니다. 긴 프롬프트는 여전히 길이에 비례하는 시간이 걸립니다. 축출을 지원하지 않는 런타임은 `kv_eviction` 필드를 무시합니다.

## Neat를 사용하여 애플리케이션을 구축하세요.

`llima run`을 사용하여 모델을 검증한 후,
[GenAI 모델](/develop-apps/development-workflow/genai-model/)을 사용하여 일반적인 API 엔드포인트를 통해 서비스를 제공하거나, C++ 또는 Python 애플리케이션에서 직접 사용하십시오.
