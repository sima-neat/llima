# MoLE - Modalix 언어 모델 평가 도구

## 개요

MoLE(Modalix 언어 모델 평가기)는 Modalix 플랫폼에서 실행되는 LLM의 정확성과 성능을 평가하기 위한 벤치마크 도구입니다.

이는 [ EleutherAI의 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)를 확장하며, 두 가지 백엔드를 지원합니다.

- **hf** — 호스트에서 HuggingFace 트랜스포머를 사용하여 평가를 실행합니다(기준).
- **modalix** — `llima benchmark-server`를 통해 Modalix 보드에서 벤치마크를 실행합니다.

## 설치 방법

SDK Docker 컨테이너 외부의 호스트 또는 Modalix DevKit에서 별도의 가상 환경에 MoLE을 설치하십시오. DevKit에서 실행하려면 LLiMa 런타임도 설치되어 있어야 합니다. 런타임 설치 절차는 [Neat Framework 설치](/getting-started/neat-library/)를 참조하십시오.

`sima-cli`를 사용하여 MoLE을 설치합니다.

``` console
host:~$ sima-cli neat install llima/mole
```

이 명령은 MoLE을 `~/sima-mole-venv`의 가상 환경에 설치합니다.

## 사용법

먼저, MoLE 가상 환경을 활성화합니다.

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

호스트에서 Modalix 평가를 실행할 때는 `--board_ip`가 필수입니다. Modalix DevKit에서는 이 옵션을 생략하면 로컬 런타임을 사용합니다. 서버를 시작할 때는 `--board_model`을 지정하십시오. 로컬 서버 로그는 `<output_dir>/server.log`에 저장됩니다. 기존 서버를 시작하거나 중지하지 않고 연결하려면 `--no-board_start_server`를 사용하십시오.

MoLE은 `llima-benchmark accuracy`와 `llima-benchmark perf`를 제공합니다. Modalix에서는 `<model_id>`에 HuggingFace 모델 ID 또는 배포 모델과 일치하는 로컬 토크나이저/설정 디렉터리를 지정할 수 있습니다. MoLE은 해당 토크나이저로 요청을 준비합니다.

### 정확도 벤치마크

표준 작업에 대한 모델의 품질을 평가합니다.

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| 논쟁 | 설명 |
|----|----|
| `model_id` | HuggingFace 모델 ID(예: `meta-llama/Llama-3.2-3B-Instruct`). `-b modalix`의 경우, 이는 배포된 모델의 토크나이저/구성 파일과 일치해야 합니다. |
| `-b` | 사용할 백엔드: `modalix` (보드에서 실행) 또는 `hf` (참조 기준선으로 호스트에서 실행). |
| `-t` | **필수.** 하나 이상의 평가 작업이 필요합니다. 예시 작업: `hellaswag`, `triviaqa`, `piqa`, `winogrande`, `wikitext`. 사용 가능한 모든 작업은 [ 작업 목록 ](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md)에서 확인하십시오. |
| `-o` | 벤치마크 결과가 저장될 출력 디렉터리입니다. |
| `--board_ip` | 원격 보드의 IP 주소 또는 호스트 이름. 호스트에서 실행할 때 필수이며, Modalix에서 로컬로 실행할 때는 생략합니다. |
| `--board_model` | DevKit의 컴파일된 모델 디렉터리. 서버를 시작할 때 필수입니다. |
| `--max_num_tokens` | 최대 컨텍스트 길이입니다. 컴파일 과정에서 사용된 값과 같거나 더 작아야 합니다. |
| `-n, --num_samples` | 평가할 샘플의 개수입니다. 지정하지 않으면 전체 작업 세트를 실행합니다. |
| `--board_ssh_user` | Modalix 보드에 사용할 SSH 사용자 이름입니다. 선택 사항이며, 기본값은 `sima`입니다. |
| `--board_ssh_pass` | Modalix 보드의 SSH 비밀번호입니다. 선택 사항입니다. 비대화형 자동 벤치마크를 활성화하려면 설정하세요. |

:::important
`-b modalix`를 사용하여 정확도 및 로그 우도 벤치마크를 수행하려면 배포된 모델이 `--return_logits`로 컴파일되어야 합니다. 이 플래그는 기본적으로 비활성화되어 있습니다. [모델 컴파일](compilation_genai.md)을 참조하십시오. 모델이 이 플래그 없이 컴파일된 경우, 벤치마크가 실패하고 다음과 같은 오류가 발생합니다: `model not compiled with --return_logits; accuracy/loglikelihood tasks are unsupported`.
:::

`-b modalix` 모드에서는 결과 테이블이 Modalix 백엔드 결과로 표시되며, 대상 보드가 포함됩니다. MoLE이 토큰화 및 작업 메타데이터에 사용하기 때문에 HuggingFace `model_id`가 여전히 나타납니다.

참조 기준선으로 HuggingFace 백엔드를 사용하려면:

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

사용 가능한 모든 옵션에 대해 `llima-benchmark accuracy -h` 벤치마크를 실행합니다.

### 성능 벤치마크

다양한 입력 길이에서 Modalix 보드에서 첫 번째 토큰이 생성되는 데 걸리는 시간(TTFT)과 초당 토큰 수(TPS)를 측정합니다.

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| 논쟁 | 설명 |
|----|----|
| `model_id` | HuggingFace 모델 ID(예: `meta-llama/Llama-3.2-3B-Instruct`). Modalix 성능 테스트를 수행할 때는 배포된 모델의 토크나이저/구성 파일과 일치해야 합니다. |
| `-o` | 벤치마크 결과가 저장될 출력 디렉터리입니다. |
| `--board_ip` | 원격 보드의 IP 주소 또는 호스트 이름. 호스트에서 실행할 때 필수이며, Modalix에서 로컬로 실행할 때는 생략합니다. |
| `--board_model` | DevKit의 컴파일된 모델 디렉터리. 서버를 시작할 때 필수입니다. |
| `--max_num_tokens` | 최대 컨텍스트 길이입니다. 컴파일 과정에서 사용된 값과 같거나 더 작아야 합니다. |
| `--max_new_tokens` | 출력 결과로 생성할 수 있는 최대 토큰 수입니다. |
| `--input_lengths` | 벤치마크를 위해 사용할 수 있는 선택적인 정확한 입력 토큰 길이입니다. 값은 고유해야 하며, 각 값에 `--max_new_tokens`를 더한 값이 `--max_num_tokens` 내에 있어야 합니다. 생략하면 MoLE가 자동으로 2의 거듭제곱을 기반으로 하는 버킷을 생성합니다. |
| `--board_ssh_user` | Modalix 보드에 사용할 SSH 사용자 이름입니다. 선택 사항이며, 기본값은 `sima`입니다. |
| `--board_ssh_pass` | Modalix 보드의 SSH 비밀번호입니다. 선택 사항입니다. 비대화형 자동 벤치마크를 활성화하려면 설정하세요. |

사용 가능한 모든 옵션에 대해 `llima-benchmark perf -h`를 실행합니다.

토큰 길이 버킷 대신 `--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum`으로 네이티브 채팅/도구 프롬프트를 사용합니다. `-n`을 생략하면 선택한 데이터셋 전체를 실행하고 `-n 20`은 처음 20개 샘플을 실행합니다. 데이터셋은 자동으로 다운로드되어 캐시됩니다. BFCL은 simple/multiple 사례를 포함하며 실제 도구는 실행하지 않습니다.

`--board_model`에 따라 일반 또는 투기적 디코딩이 결정됩니다. 모델을 비교하려면 같은 데이터셋과 설정으로 별도 실행하되 `--board_model`과 `-o`만 변경합니다. 워밍업 1회는 측정에서 제외하며 매 요청 전 KV 상태를 초기화합니다. 모델의 네이티브 채팅 템플릿과 EOS 종료를 사용합니다. 데이터셋 출력 한도에는 첫 토큰이 포함되며 기존 버킷은 첫 토큰과 `--max_new_tokens` 디코드 단계를 측정합니다.

`-o`는 시간, 토큰 수, 집계 지표, 드래프트 기여도, 실패, 벤치마크 설정 및 데이터셋 출처를 `performance.json`에 저장합니다. 디코드 TPS는 TTFT를 제외하며 생성 시간은 입력 준비와 네트워크 전송을 제외합니다. 이 워크로드는 정확도가 아닌 성능을 측정합니다.
