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

CLI 모드에서는 기본적으로 채팅 기록이 활성화됩니다. 각 프롬프트와 응답은 사용자가 `clear history`를 사용하여 기록을 지울 때까지 다음 대화의 맥락으로 유지됩니다.

``` console
modalix:~$ llima run <model> [options]
```

| 논쟁 | 설명 |
|----|----|
| `model` | 모델 ID 또는 경로(예: `Qwen3-VL-8B-Instruct-a16w4`). |
| `--stt_model_path` | 음성-텍스트 변환 모델에 대한 ELF 파일 경로(선택 사항). |

사용 가능한 모든 옵션에 대해 `llima run -h`를 실행합니다.

**예시**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## 대화형 명령어

CLI 모드에서 `llima run`을 실행하면 프롬프트에서 다음 명령어를 사용하십시오.

| 명령 | 설명 |
|----|----|
| `add image <file>` | 현재 프롬프트 컨텍스트에 이미지를 추가합니다. |
| `clear image` | 모든 이미지를 삭제합니다. |
| `set system <prompt>` | 시스템 프롬프트를 설정합니다. |
| `clear system` | 시스템 프롬프트, 채팅 기록, 이미지를 모두 삭제합니다. |
| `clear history` | 채팅 기록과 이미지를 모두 삭제합니다. |
| `print history` | 채팅 기록을 인쇄합니다. |
| `set audio <file>` | 음성 파일을 텍스트로 변환할 파일로 설정합니다. |
| `set language <lang>` | 음성 인식에 사용되는 언어 설정을 지정합니다. |
| `set lora <name>` | `npy_files` 폴더에서 LoRA 가중치를 사용하세요. |
| `unset lora` | LoRA 모델을 기본 모델로 되돌립니다. |
| `enable-thinking` | 사고 모드를 활성화하고 채팅 기록을 삭제합니다. |
| `disable-thinking` | 사고 모드를 비활성화하고 채팅 기록을 삭제합니다. |
| `quit` | 그만두세요. |
| `help` | 사용 가능한 명령어를 출력합니다. |


## Neat를 사용하여 애플리케이션을 구축하세요.

`llima run`을 사용하여 모델을 검증한 후,
[GenAI 모델](/develop-apps/development-workflow/genai-model/)을 사용하여 일반적인 API 엔드포인트를 통해 서비스를 제공하거나, C++ 또는 Python 애플리케이션에서 직접 사용하십시오.
