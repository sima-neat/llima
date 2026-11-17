# MoLE - Modalix 언어 모델 평가 도구

## 개요

MoLE(Modalix Language Model Evaluator)는 Modalix 플랫폼에서 실행되는 LLM과 VLM의 정확도와 성능을 평가하는 벤치마크 도구입니다.

텍스트 정확도 평가는 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)를 사용합니다. Modalix의 이미지와 텍스트를 함께 사용하는 정확도 평가는 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)을 사용하며 업스트림 태스크, 답변 추출 및 채점 방식을 재사용합니다. 두 평가 모두 `llima-benchmark accuracy`를 사용합니다. MoLE는 두 가지 백엔드를 지원합니다.

- **hf** — 호스트에서 HuggingFace transformers를 사용하여 텍스트 정확도를 평가합니다(참조 기준).
- **modalix** — `llima benchmark-server`를 통해 Modalix 보드에서 평가를 실행합니다.

## 설치

SDK Docker 컨테이너 외부의 호스트 또는 Modalix DevKit에 별도의 가상 환경을 만들어 MoLE를 설치합니다. DevKit에서 실행하려면 LLiMa 런타임도 설치해야 합니다. 런타임 설치 절차는 [Neat Framework 설치](/getting-started/neat-library/)를 참조하세요.

`sima-cli`로 MoLE를 설치합니다.

``` console
host:~$ sima-cli neat install llima/mole
```

이 명령은 `~/sima-mole-venv`의 가상 환경에 MoLE를 설치합니다.

### 수동 설치

`sima-cli`를 사용하지 않는 경우, 다운로드한 MoLE 패키지 디렉터리에서 포함된 설치 프로그램을 실행합니다.

```bash
bash ./install_mole.sh --venv ~/sima-mole-venv
```

wheel을 직접 설치하려면 Python 3.11–3.13 가상 환경을 활성화한 다음, wheel 파일 이름을 다운로드한 버전으로 바꾸어 다음 명령을 실행합니다.

```bash
python -m pip install './sima_lmm-<version>-py3-none-any.whl[mole]'
python -m pip install --no-deps "lmms-eval==0.7.3" "tenacity>=8.3.0"
```

두 번째 명령은 MoLE의 공통 의존성을 재사용하면서 이미지 벤치마크를 활성화합니다.

## 사용법

먼저 MoLE 가상 환경을 활성화합니다.

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

호스트에서 Modalix 평가를 실행할 때는 `--board_ip`가 필요합니다. Modalix DevKit에서는 이 옵션을 생략하여 로컬 런타임을 사용합니다. 서버를 시작할 때는 `--board_model`을 지정합니다. 로컬 서버 로그는 `<output_dir>/server.log`에 저장됩니다. 기존 서버를 시작하거나 중지하지 않고 연결하려면 `--no-board_start_server`를 사용합니다.

MoLE는 `llima-benchmark accuracy`와 `llima-benchmark perf` 하위 명령을 제공합니다. 텍스트 워크로드는 배포된 모델과 일치하는 HuggingFace 모델 ID 또는 로컬 토크나이저/설정 디렉터리를 사용합니다. 이미지 워크로드는 배포된 VLM의 리소스를 사용하며, `<model_id>`는 실행의 이름표로 사용됩니다.

### 정확도 벤치마크

#### 텍스트 정확도

표준 태스크로 모델 품질을 평가합니다.

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| 인수 | 설명 |
|----|----|
| `model_id` | 텍스트 정확도 평가에 사용할 HuggingFace 모델 ID 또는 로컬 토크나이저/설정 디렉터리입니다. 이미지 정확도 평가에서는 실행을 설명하는 이름표입니다. |
| `-b` | 사용할 백엔드: `modalix`(보드에서 실행) 또는 `hf`(호스트에서 참조 기준으로 실행). |
| `-t` | **필수.** 하나 이상의 평가 태스크입니다. 예: `hellaswag`, `triviaqa`, `piqa`, `winogrande`, `wikitext`. 전체 태스크는 [태스크 목록](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md)을 참조하세요. |
| `-o` | 벤치마크 결과를 저장할 출력 디렉터리입니다. |
| `--board_ip` | 원격 보드 IP 또는 호스트 이름입니다. 호스트에서 실행할 때는 필수이며, 로컬 Modalix 실행에서는 생략합니다. |
| `--board_model` | DevKit에 있는 컴파일된 모델 디렉터리입니다. 서버를 시작할 때 필요합니다. |
| `--max_num_tokens` | 최대 컨텍스트 길이입니다. 컴파일 시 사용한 값 이하여야 합니다. |
| `-n, --num_samples` | 평가할 샘플 수입니다. 지정하지 않으면 전체 태스크 집합을 실행합니다. |
| `--board_ssh_user` | Modalix 보드의 SSH 사용자 이름입니다. 선택 사항이며 기본값은 `sima`입니다. \# |
| `--board_ssh_pass` | Modalix 보드의 SSH 비밀번호입니다. 선택 사항이며, 비대화형 자동 벤치마크를 활성화하려면 설정합니다. |

:::important
로짓이나 로그우도를 사용하는 태스크는 컴파일 시 `--return_logits`가 필요합니다. 이미지와 텍스트 평가를 포함한 생성 기반 태스크에는 필요하지 않습니다. [모델 컴파일](compilation_genai.md)을 참조하세요.
:::

`-b modalix` 모드에서는 결과 표에 Modalix 백엔드 결과라는 표시와 대상 보드가 포함됩니다. MoLE가 토큰화와 태스크 메타데이터에 HuggingFace의 `model_id`를 사용하므로 해당 ID도 계속 표시됩니다.

HuggingFace 백엔드를 참조 기준으로 사용하려면 다음과 같이 실행합니다.

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

사용 가능한 모든 옵션은 `llima-benchmark accuracy -h`로 확인합니다.

#### 이미지와 텍스트 정확도

평가 도구는 MoLE와 함께 설치되며, 배포된 VLM의 기본 채팅 템플릿, 이미지 전처리 및 업스트림 태스크 채점 방식을 사용합니다. 다음 태스크는 Modalix에서 Qwen3.5-2B로 테스트되었습니다.

| 태스크 | 평가 대상 | 검증 |
|----|----|----|
| `realworldqa` | 실제 장면 및 공간 추론 | 전체 데이터셋 |
| `docvqa_val` | 문서 읽기 및 OCR | 스모크 테스트 |
| `chartqa` | 차트 읽기 및 수치 추론 | 스모크 테스트 |
| `mmstar` | 시각 인지 및 멀티모달 추론 | 10개 샘플 스모크 테스트 |
| `ai2d` | 과학 도해 이해 | 10개 샘플 스모크 테스트 |
| `ocrbench` | 텍스트 인식 및 이해 | 10개 샘플 스모크 테스트 |
| `textvqa_val` | 자연 이미지 속 텍스트에 관한 질문 | 10개 샘플 스모크 테스트 |

아래 제한 사항을 준수하는 다른 개별 `lmms-eval` 이미지 입력 생성 태스크도 `-t`로 선택할 수 있지만, 모두 테스트된 것은 아닙니다.

```bash
llima-benchmark accuracy qwen3.5-2b -b modalix \
    -t realworldqa docvqa_val chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-on-devkit> \
    --max_num_tokens 4096 -o results/vlm
```

DevKit에서는 `--board_ip`를 생략합니다. 호스트의 모델 리소스는 필요하지 않습니다. 서로 일치하는 런타임과 MoLE 릴리스를 사용하세요.

| 이미지 정확도 옵션 | 설명 |
|----|----|
| `--max_new_tokens` | 태스크의 출력 토큰 한도를 덮어씁니다. 지정하지 않으면 업스트림 기본값을 유지합니다. |
| `--enable_thinking` | 모델의 기본 사고 템플릿을 활성화합니다. 기본적으로 비활성화되어 있습니다. |
| `--accuracy_manifest` | 아래와 같이 고정된 데이터셋 리비전이나 준비된 로컬 데이터셋을 사용합니다. |

텍스트 태스크와 이미지 태스크는 개별 태스크 이름을 지정하여 따로 실행합니다. 이미지 정확도 평가는 탐욕적 디코딩을 사용하는 일반 VLM을 지원합니다. 오디오/비디오, 멀티턴, 시각적 퓨샷 및 반복 요청 태스크는 지원하지 않습니다. 입력과 출력 토큰 한도의 합은 요청한 컨텍스트와 컴파일된 컨텍스트에 들어가야 하며, 입력은 잘리지 않습니다. 여러 이미지는 태스크에서 정의한 순서를 유지합니다. 모델의 기본 EOS가 생성을 종료하고, 태스크의 문자열 중지 조건이 반환된 답변을 잘라냅니다.

#### 준비된 이미지 데이터셋

데이터셋은 자동으로 다운로드됩니다. 리비전을 고정하거나 오프라인으로 평가하려면 `--accuracy_manifest manifest.json`을 전달합니다. 예를 들어 `-t chartqa`를 사용하는 경우:

```json
{
  "schema_version": 1,
  "tasks": {
    "chartqa": {
      "revision": "<40-character-dataset-commit>",
      "split": "test",
      "local_path": "datasets/chartqa"
    }
  }
}
```

HuggingFace의 `DatasetDict.save_to_disk()`로 로컬 데이터셋을 준비하고, 업스트림의 열, 이미지 및 채점용 분할을 유지합니다. 경로는 매니페스트를 기준으로 하며, 매니페스트의 태스크 이름은 `-t`와 일치해야 합니다. 고정된 리비전을 다운로드하려면 `local_path`를 생략합니다. 오프라인 실행에서는 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`을 설정합니다.

#### 이미지 정확도 결과

출력 디렉터리에는 다음 파일이 포함됩니다.

- `accuracy.json`: 집계 지표, 샘플 수, 데이터셋 리비전, 배포 경로, 런타임 버전 및 설정.
- `accuracy-answers.jsonl`: 생성된 답변, 이미지/프롬프트 식별 정보, 토큰 수, 종료 사유 및 오류.
- `accuracy-samples.jsonl`: 업스트림의 샘플별 채점 결과.

`-n`을 생략하면 전체 데이터셋을 실행합니다. 오류가 발생하면 평가를 중지하고 일부 답변을 보존하며 집계 지표는 생성하지 않습니다. 출력 토큰 한도에 따른 잘림은 집계되며 해당 답변도 채점됩니다.

### 성능 벤치마크

Modalix 보드에서 다양한 입력 길이에 대한 첫 토큰까지의 시간(TTFT)과 초당 토큰 수(TPS)를 측정합니다.

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| 인수 | 설명 |
|----|----|
| `model_id` | 텍스트 워크로드에서는 배포된 모델과 일치하는 HuggingFace 모델 ID 또는 로컬 토크나이저/설정 디렉터리입니다. 이미지 워크로드에서는 실행을 설명하는 이름표입니다. |
| `-o` | 벤치마크 결과를 저장할 출력 디렉터리입니다. |
| `--board_ip` | 원격 보드 IP 또는 호스트 이름입니다. 호스트에서 실행할 때는 필수이며, 로컬 Modalix 실행에서는 생략합니다. |
| `--board_model` | DevKit에 있는 컴파일된 모델 디렉터리입니다. 서버를 시작할 때 필요합니다. |
| `--max_num_tokens` | 최대 컨텍스트 길이입니다. 컴파일 시 사용한 값 이하여야 합니다. |
| `--max_new_tokens` | 출력에서 생성할 최대 토큰 수입니다. |
| `--input_lengths` | 벤치마크할 정확한 입력 토큰 길이를 지정하는 선택 옵션입니다. 값은 중복되지 않아야 하며, 각 값에 `--max_new_tokens`를 더한 값이 `--max_num_tokens` 이내여야 합니다. 생략하면 MoLE가 2의 거듭제곱 길이 버킷을 자동으로 생성합니다. |
| `--board_ssh_user` | Modalix 보드의 SSH 사용자 이름입니다. 선택 사항이며 기본값은 `sima`입니다. |
| `--board_ssh_pass` | Modalix 보드의 SSH 비밀번호입니다. 선택 사항이며, 비대화형 자동 벤치마크를 활성화하려면 설정합니다. |

사용 가능한 모든 옵션은 `llima-benchmark perf -h`로 확인합니다.

토큰 길이 버킷 대신 기본 채팅/도구 프롬프트를 사용하려면 `--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum`을 지정합니다. `-n`을 생략하면 선택한 전체 데이터셋을 실행하고, `-n 20`은 처음 20개 샘플을 실행합니다. 데이터셋은 자동으로 다운로드되어 캐시됩니다. BFCL은 단순/다중 호출 사례를 다루며, 도구는 실행하지 않습니다.

`--board_model`이 일반 디코딩 또는 추측 디코딩을 결정합니다. 같은 데이터셋과 설정으로 별도 실행하되 `--board_model`과 `-o`만 바꾸어 모델을 비교합니다. 각 실행은 워밍업 1회를 측정에서 제외하고, 요청마다 KV 상태를 초기화합니다. 모델의 기본 채팅 템플릿과 EOS 종료 조건이 적용됩니다. 데이터셋의 출력 토큰 한도에는 첫 토큰이 포함되며, 기존 버킷은 첫 토큰과 `--max_new_tokens`회의 디코딩 단계를 측정합니다.

#### 이미지와 텍스트 성능

`--dataset realworldqa|docvqa_val|chartqa`를 사용하여 정확도 평가와 동일한 업스트림 이미지/텍스트 프롬프트로 일반 또는 추측 디코딩 VLM을 측정합니다.

```bash
llima-benchmark perf qwen3.5-vlm --dataset chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-or-speculative-bundle> \
    --max_num_tokens 4096 --max_new_tokens 256 -o results/vlm-perf
```

`-n`을 생략하면 워밍업 1회를 측정에서 제외한 후 전체 데이터셋을 한 번 실행합니다. `--dataset_manifest`는 `--accuracy_manifest`와 동일한 고정 리비전/로컬 데이터셋 형식을 사용하며, 태스크 이름은 `--dataset`과 일치해야 합니다. 이미지는 데이터셋의 해상도를 유지하고 기본 런타임 전처리를 사용합니다. 호스트의 모델 리소스는 필요하지 않습니다. 정확도 평가는 여전히 일반 디코딩으로 제한됩니다.

`-o`는 요청별 시간, 토큰 수, 집계 지표, 드래프트 기여도, 실패 내역, 벤치마크 설정 및 데이터셋 출처를 `performance.json`에 저장합니다. 디코딩 TPS는 TTFT를 제외합니다. 이미지 TTFT에는 기본 이미지 전처리와 비전 인코딩이 포함되고, 텍스트 TTFT는 언어 모델의 프리필부터 시작합니다. 두 값 모두 네트워크 전송과 호스트의 데이터셋 준비 시간은 제외합니다. 드래프트 기여도는 채택된 드래프트 토큰 수를 전체 생성 토큰 수로 나눈 값입니다. 이 워크로드는 정확도가 아닌 성능을 측정합니다.
