# MoLE - Modalix 언어 모델 평가 도구

## 개요

MoLE(Modalix 언어 모델 평가기)는 Modalix 플랫폼에서 실행되는 LLM의 정확성과 성능을 평가하기 위한 벤치마크 도구입니다.

이는 [ EleutherAI의 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)를 확장하며, 두 가지 백엔드를 지원합니다.

- **hf** — 호스트에서 HuggingFace 트랜스포머를 사용하여 평가를 실행합니다(기준).
- **modalix** — `llima benchmark-server`를 통해 Modalix 보드에서 벤치마크를 실행합니다.

## 설치 방법

MoLE은 호스트 측 벤치마크 도구입니다. SDK Docker 컨테이너가 아닌 호스트 시스템에 설치하고 실행하며, Modalix 장치 내에서 실행하지 마십시오. Modalix 장치에는 LLiMa 런타임과 `llima benchmark-server` 프로세스만 있으면 됩니다. 런타임 설치 절차는 [Neat Framework 설치](/getting-started/neat-library/)를 참조하십시오.

`sima-cli`를 사용하여 호스트에 MoLE을 설치합니다.

``` console
host:~$ sima-cli neat install llima/mole
```

이 명령은 MoLE을 `~/sima-mole-venv`에 있는 호스트 가상 환경에 설치합니다.

## 사용법

먼저, MoLE 가상 환경을 활성화합니다.

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

MoLE은 두 개의 하위 명령을 사용하여 `llima-benchmark` CLI를 통해 호출됩니다. `<model_id>` 인수는 항상 HuggingFace 모델 ID(예: `meta-llama/Llama-3.2-3B-Instruct`)입니다. `-b modalix` 모드에서는 단순한 표시 레이블이 아니라, 배포된 보드 모델을 컴파일하는 데 사용된 토크나이저 및 구성과 일치해야 합니다. 왜냐하면 보드는 토큰 점수만 반환하고 토크나이저 메타데이터는 제공하지 않기 때문입니다.

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
| `--board_ip` | Modalix 보드의 IP 주소입니다. `-b modalix`에 필요합니다. |
| `--board_model` | Modalix 장치에서 컴파일된 모델 디렉터리의 경로입니다(예: `/media/nvme/llima/models/Llama-3.2-3B-Instruct-a16w4`). `-b modalix`에 필요합니다. |
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
| `--board_ip` | Modalix 보드의 IP 주소입니다. |
| `--board_model` | Modalix 장치에서 컴파일된 모델 디렉터리의 경로입니다(예: `/media/nvme/llima/models/Llama-3.2-3B-Instruct-a16w4`). |
| `--max_num_tokens` | 최대 컨텍스트 길이입니다. 컴파일 과정에서 사용된 값과 같거나 더 작아야 합니다. |
| `--max_new_tokens` | 출력 결과로 생성할 수 있는 최대 토큰 수입니다. |
| `--input_lengths` | 벤치마크를 위해 사용할 수 있는 선택적인 정확한 입력 토큰 길이입니다. 값은 고유해야 하며, 각 값에 `--max_new_tokens`를 더한 값이 `--max_num_tokens` 내에 있어야 합니다. 생략하면 MoLE가 자동으로 2의 거듭제곱을 기반으로 하는 버킷을 생성합니다. |
| `--board_ssh_user` | Modalix 보드에 사용할 SSH 사용자 이름입니다. 선택 사항이며, 기본값은 `sima`입니다. |
| `--board_ssh_pass` | Modalix 보드의 SSH 비밀번호입니다. 선택 사항입니다. 비대화형 자동 벤치마크를 활성화하려면 설정하세요. |

사용 가능한 모든 옵션에 대해 `llima-benchmark perf -h`를 실행합니다.
