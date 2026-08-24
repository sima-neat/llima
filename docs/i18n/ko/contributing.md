# LLiMa에 기여하기

LLiMa에는 호스트 측 GenAI 컴파일러와 Modalix용 C++ 런타임이 포함되어 있습니다. 런타임은 패키징된 CLI/HTTP/ZMQ 진입점을 통해 운영됩니다. Python은 CLI 오케스트레이션이며, 별도의 공개 런타임 API가 아닙니다. 컴파일러와 런타임 환경 및 종속성을 분리하십시오. 저장소 체크아웃 시, `CONTRIBUTING.md`는 빠른 시작 방법을 제공하고, `AGENTS.md`는 에이전트별 규칙을 정의합니다. 이 가이드는 상세한 기여 정책입니다.

## 코딩 에이전트 기술

표준 기여자 설정의 일부로 두 가지 기여자 기술인 LLiMa를 모두 설치합니다.
이 기술들은 기본 Neat SDK 플레이북 인덱스에 의도적으로 설치되지 않습니다.

```bash
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-contribute-to-llima
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-add-llima-model-support
```

일반 기여 기술은 저장소 전체의 컴파일러, 런타임, 패키징, 테스트, 문서, 그리고 기술 변경 사항을 포괄합니다. 모델 지원 기술은 LLM 및 VLM 아키텍처, 체크포인트, 텐서 레이아웃, 토크나이저 및 프롬프트 계약에 대한 호환성과 구현 워크플로를 추가합니다. 두 가지 모두 설치하여 기여 내용이 해당 범위를 넘나들 때 적절한 지침을 제공할 수 있도록 합니다.


## 저장소 지도

| 면적 | 경로 | 책임 |
| --- | --- | --- |
| 구성 | `sima_lmm/config/` | LLM, VLM 및 ASR 구성 계약 |
| 섭취 | `sima_lmm/hf/`, `sima_lmm/gguf/` | Hugging Face 및 GGUF 로드 및 변환 |
| 컴파일 | `sima_lmm/model/`, `sima_lmm/preproc/` | 모델 구성 요소, 양자화, 그래프, 전처리 |
| 호스트 도구 | `sima_lmm/host/` | 컴파일하고, 배포하고, LoRA를 적용하고, 벤치마크 진입 지점을 설정합니다. |
| 평가 | `sima_lmm/mole/` | MoLE 워크플로우 |
| 런타임 CLI | `sima_lmm/devkit/` | Python CLI를 사용한 오케스트레이션 및 모델 관리 |
| C++ 런타임 | `sima_lmm/devkit/cpp/` | 모델, 토크나이저, MLA, CLI/HTTP/ZMQ 구현, 그리고 내부 CLI 바인딩 |
| 테스트 | `tests/` | 컴파일러 및 Modalix 런타임 테스트 |
| 포장 | `CMakeLists.txt`, `cmake/`, `build*.sh`, `tools/install_*.sh` | 데비안, 휠, 그리고 아티팩트 패키징 |
| 지속적 통합/캐시 | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` | 빌드, 테스트, 모델 캐시 생성 |
| 문서/기술 | `README.md`, `docs/`, `skills/` | 사용자, 기여자, 그리고 플레이북 안내 |

컴파일러 전용 종속성은 `sima_lmm/devkit/` 또는 Modalix 런타임 패키지에 포함되어서는 안 됩니다.

## 개발 환경

### 런타임 및 패키징

지원되는 빌드 환경으로 Neat SDK를 사용하세요. 모든 런타임 패키지와 패키지화된 테스트를 다음 명령어로 빌드합니다.

```bash
./build.sh --all --clean
```

일반 빌드에서는 필요한 설정을 처리하며, 여기에는 하위 모듈이 포함됩니다. 표준 워크플로에서는 별도의 종속성 부트스트랩 단계를 실행하지 마십시오.

유용한 세분화된 빌드:

```bash
./build.sh --clean --core
./build.sh --clean --core --dev
./build.sh --clean --cli
./build.sh --no-dist
```

결과는 `build-deb/`에서 생성되고 `dist/`에 저장됩니다. MLA 실행과 실제 `llima run` 검증을 위해서는 Modalix가 필요합니다.

### 컴파일러 개발

Model Compiler에서 설치한 Python 3.12 환경을 사용하세요. 다음 순서대로 검색합니다.

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

설치된 컴파일러 패키지를 가리는 또 다른 환경을 만들지 마십시오. 다음을 사용하여 게시 프로필을 생성하십시오.

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

그들은 `build/`에서 휠 도구를 사용하고, `dist/compiler/` 및 `dist/mole/`에서 결과물을 생성합니다.

### Whisper 및 ASR 개발

공개된 `llima-compile` 워크플로는 LLM 및 VLM을 포함합니다. 기존 Whisper 컴파일은 대신 기여자 유틸리티인 `scripts/gen_models--openai--whisper.py`를 사용합니다.

```bash
python scripts/gen_models--openai--whisper.py \
  --model_path /path/to/openai/whisper-small \
  --output /path/to/whisper-output \
  --part all
```

명시적인 모델 경로를 사용하여 Model Compiler 환경에서 실행합니다.

`--part`는 `all`, `encoder`, `language_detect`, `init`, `single_pre`, `single_post` 및 `single_cache`를 허용합니다. 로그 프로브가 활성화된 디코더 출력을 컴파일하려면 `--enable_log_probe`를 추가합니다. 완전한 로그 프로브 빌드를 위해서는 `--part all --enable_log_probe`를 사용합니다.

컴파일러 변경 사항은 일반적으로 `sima_lmm/config/whisper_config.py`, `sima_lmm/model/whisper_*.py` 및 스크립트에 영향을 미치고, 런타임 변경 사항은 `sima_lmm/devkit/cpp/whisper_*`에 영향을 미칩니다. `tests/README.md`에 설명된 패키지된 C++ ASR 런타임 테스트와 Modalix의 대표 오디오를 사용하여 유효성을 검사합니다. 이것은 일반적인 ASR 아키텍처 프레임워크가 아닌 Whisper에 특정한 경로입니다.

## 테스트

실패 지표에 따라 테스트를 선택합니다. 빌드는 동작 검증을 대체하지 않으며, 건너뛴 필수 테스트 케이스는 통과로 간주되지 않습니다.

### 기밀성 테스트

순수 구성, 매핑, 직렬화, 유효성 검사 및 숫자 관련 로직을 모델 다운로드와 독립적으로 유지합니다.

```bash
pytest -q <targeted-test-path>
```

### 모델 기반 컴파일러 테스트

컴파일러 테스트는 `tests/compilation/` 디렉터리에 있습니다. 영향을 받는 그룹과 `tests/README.md`에 설명된 마커를 선택하세요. 예를 들어:

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

`--model-inputs-path` 및 `LLIMA_HF_MODELS_PATH`는 준비된 Hugging Face/GGUF 입력 루트를 선택합니다. CI는 `tools/hf-safetensors/` 아래의 매니페스트를 사용합니다.
테스트 생략을 허용하는 대신 필요한 입력을 구성합니다.

테스트 매트릭스, 예상 값, 기본 정책은 `tests/README.md`에, CI 호출은 `.github/workflows/model-compiler-tests.yml`에 있습니다. 실행 중에 ONNX 및 숫자 비교 아티팩트를 생성하고, 이진 기본값을 커밋하는 대신 생성합니다.

### 런타임 유효성 검사

후보 패키지와 런타임 테스트용 추가 기능을 빌드합니다.

```bash
./build.sh --all --clean
```

이 작업은 빌드를 수행하지만 테스트는 실행하지 않습니다. 호환되는 LLiMa 및 내부 패키지를 Modalix에 설치하고, 추가 아카이브를 추출한 다음, `tests/README.md`에 있는 DevKit 런타임 테스트 지침에 따라 패키지화된 CTest 및 pytest를 실행합니다.

모델 로딩, 추론, 토큰화, 다중 모달 전처리, 추론 디코딩, CLI/HTTP/ZMQ 또는 리소스 수명 주기에 변경 사항이 적용될 때 관련 하드웨어 테스트를 실행합니다. 필요한 경우 대표적인 간단한 테스트를 추가합니다.

```bash
llima run <model_dir> --mode cli
```

VLM 변경 사항의 경우, 이미지 기반 프롬프트를 포함합니다. 수동 스모크 테스트는 영향을 받는 패키지 적용 범위를 보완하지만 대체하지는 않습니다.

Neat Core는 LLiMa의 설치된 C++ API 및 런타임 패키지를 사용합니다. 해당 요소 중 하나 또는 Core의 GenAI API를 통해 노출되는 동작이 변경되면, 게시되거나 캐시된 LLiMa 빌드 대신 후보 `sima-lmm-core` 및 `sima-lmm-dev` 패키지를 사용하여 Core를 빌드합니다. 영향을 받는 Core GenAI C++ 테스트를 Modalix에서 실행합니다. 이 하위 단계 검증은 독립적인 컴파일러, 문서 또는 테스트 전용 변경 사항에는 필요하지 않습니다.

### 포장 검증

변경된 각 프로필을 빌드합니다.

```bash
./build.sh --all --clean
./build_compiler_wheel.sh
./build_mole_package.sh
```

패키지 이름, 파일 소유권, 설치 매니페스트, 종속성, 체크섬 및 메타데이터를 확인합니다.

## 코딩 표준

### 호환성과 한계

설치된 C++ 헤더, CLI 명령어, 직렬화된 구성, 패키지 메타데이터 및 생성된 아티팩트 레이아웃을 호환성 영역으로 취급합니다. 점진적인 변경을 선호합니다. 호환성이 깨지는 변경 사항이 있는 경우, 영향을 받는 사용자와 마이그레이션 방법, 릴리스 의도를 문서화하고, 호출자, 테스트, 예제 및 사용자 문서를 업데이트합니다.

컴파일러, 런타임 및 MoLE 종속성을 분리합니다. 런타임 상태는 호스트 컴파일의 입력으로 사용될 수 없습니다. 런타임 패키지 경계에 대한 변경 사항은 `sima-lmm-core`, `sima-lmm-dev` 및 `sima-lmm-cli`의 역할을 유지해야 하며, 적절한 API/ABI 검증을 포함해야 합니다.

### 구현 품질

- `pyproject.toml`에 명시된 대상 C++ 20 및 Python 버전입니다.
- 주변 형식, 명명 규칙, 그룹화 방식을 따르고, 포괄적인 내용을 피하십시오.
  기계적인 재구성.
- 설치된 인터페이스의 수를 최소화하고 구현 세부 사항은 비공개로 유지하십시오.
- 가능한 경우 Python 유형 주석을 추가하세요.
- 명확하지 않은 계약 조건, 수치적 가정, 그리고 하드웨어에 대해 설명하십시오.
  제약 조건: 코드에 대한 설명을 덧붙이지 마십시오.
- 추상화를 추가하기 전에 먼저 근처에 있는 기존 함수나 코드를 재사용하세요.
- 결정론적 모델 선택, 그래프 구조, 직렬화 등을 유지합니다.
  아티팩트 이름. 시드 값을 기록하고 파일 시스템/프로세스 순서 지정은 피합니다.
- 지원되지 않거나 유효하지 않은 입력은 관련 정보와 함께 거부하고, 원래 원인은 보존합니다.
  여러 계층을 거치면서도 다른 실행 경로를 조용히 선택하지 않습니다.
- 바운드된 작업자 조정을 수행하고, 필요한 구성 요소를 분해합니다. 버퍼, 핸들, 스레드 등을 생성합니다.
  임시 파일의 소유권을 명시하고, 부분적으로 완료된 작업을 안전하게 정리합니다.
- 성능에 중요한 영향을 미치는 코드 영역에서는 불필요한 메모리 할당, 복사 및 동기화를 피하십시오.

### 모델, 아티팩트, 종속성, 그리고 비밀 정보

지속적으로 사용할 수 있는 테스트 자료:

- JSON 구성 계약을 검토했습니다.
- 소스 코드로 관리되는 사례, 초기 데이터, 허용 오차, 비교 정책 등을 포함합니다.
- 승인된 변경 불가능한 Hugging Face/GGUF 버전의 매니페스트 파일입니다.

다운로드한 가중치, 고객 데이터 또는 생성된 ONNX, NumPy, 양자화된 파일, MPK, ELF 또는 런타임 모델 트리를 저장소에 커밋하지 마세요. 생성된 출력은 무시하거나 임시 디렉터리에 보관하세요.

패키지/플랫폼 버전을 관리하기 위해 `deps/manifest.json`을 사용하세요. `third_party/`는 외부 라이브러리 코드로 취급하고, 의도적인 서브모듈 업데이트를 분리하고 문서화하세요. 런타임 Debian 패키지에 컴파일러 종속성을 추가하지 마세요.

토큰, SSH 자격 증명, 비공개 저장소, 서명된 URL 또는 개인 경로를 절대로 저장소에 커밋하거나 로그에 기록하지 마세요. 게이트된 모델 권한 부여는 소스 제어 외부로 유지하고, 보고서에는 공개 모델 ID, 변경 불가능한 리비전 및 삭제된 로그를 사용하세요.

## 문서, 기술, 그리고 코드 변경 요청

가장 최신 버전의 사용자 가이드로 업데이트하세요.

- [시스템 요구 사항](setup.md)
- [모델 컴파일](compilation_genai.md)
- [모델 배포](deployment.md)
- [ LLiMa CLI ](runtime.md)
- [MoLE](mole.md)

루트 디렉터리에 있는 `CONTRIBUTING.md` 파일을 빠른 시작 가이드로, 이 파일을 상세 정책으로, 그리고 `AGENTS.md` 파일을 적용 가능한 에이전트 규칙으로 유지합니다. 스킬에는 유효한 `SKILL.md`, `playbook.yml` 및 에이전트 메타데이터가 포함되어야 합니다. 주요 워크플로를 간결하게 유지하고 조건부 세부 사항은 직접 참조로 이동합니다.

Neat SDK에서 저장소 루트의 모든 스킬 페이로드를 검증하되, 설치된 에이전트의 상태를 변경하지 마십시오.

```bash
playbooks_validation_dir="$(mktemp -d)"
CODEX_HOME="${playbooks_validation_dir}/codex" \
CLAUDE_HOME="${playbooks_validation_dir}/claude" \
SIMA_CLI_HOME="${playbooks_validation_dir}/sima-cli" \
sima-cli playbooks install ./skills
```

설치 요약에는 `detected: 3`, `valid: 3` 및 `discarded: 0`가 반드시 포함되어야 합니다.
각 `playbook.yml`에서 `sima-cli` 버전은 `min_cli_version`을 충족해야 합니다.

풀 리퀘스트의 경우:

- 현재 `develop` 브랜치에서 분기하여 해당 브랜치를 대상으로 함
- 명령형 주어를 사용하여 커밋 내용을 명확하고 간결하게 작성하세요.
- `.github/PULL_REQUEST_TEMPLATE.md`를 사용합니다.
- 해결된 문제와 `Fixes #<issue>`를 연결합니다.
- 위험 요소 보고, 호환성/마이그레이션, 문서에 미치는 영향, 재현 가능한 명령어
  모델/패키지 버전, 하드웨어 증거, 건너뛴 검사, 잔여 위험; 그리고
- 인증 정보와 개인 자산은 제외합니다.

예상치 않은 건너뛰기 없이 관련 테스트가 통과하고, 필수 패키지 및 Modalix
검사가 완료되거나 사용할 수 없음이 명시되며, 호환성과 문서 문제가 해결되고,
PR에 재현 가능한 증거가 포함되면 기여할 준비가 된 것입니다.
