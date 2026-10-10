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

Whisper 모델 리포지토리에는 인코더 계층마다 하나의 ELF가 포함됩니다. 런타임은 인코더 ELF가 하나로 통합된 기존 리포지토리를 지원하지 않습니다. 계층화된 모델을 다운로드하거나 현재 LLiMa 버전으로 체크포인트를 다시 컴파일하십시오.

컴파일러 변경 사항은 일반적으로 `sima_lmm/config/whisper_config.py`, `sima_lmm/model/whisper_*.py` 및 스크립트에 영향을 미치고, 런타임 변경 사항은 `sima_lmm/devkit/cpp/whisper_*`에 영향을 미칩니다. `tests/README.md`에 설명된 패키지된 C++ ASR 런타임 테스트와 Modalix의 대표 오디오를 사용하여 유효성을 검사합니다. 이것은 일반적인 ASR 아키텍처 프레임워크가 아닌 Whisper에 특정한 경로입니다.

### 네이티브 그래프 컴포넌트

`sima_lmm/model/model_graph.py`의 `ModelGraph`를 사용합니다. 컴포넌트의 소스 가중치, 정밀도, 출력 경로를 한 번 설정합니다. 그래프 노드 구성은 클래스 안에 두고, 독립적인 배열 레이아웃 및 이름 지정 유틸리티는 모듈 함수로 유지합니다. 모델 컴포넌트는 그래프 메서드를 사용하므로 구현에서는 토폴로지에 집중할 수 있습니다:

```python
from sima_lmm.model.model_graph import ModelGraph

def generate_graph(self, layer_cfg, quantizable):
    graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.cfg.d_model)}, quantizable)
    hidden = graph.layer_norm("model.norm", graph.inputs["hidden"])
    output = graph.mlp("model.mlp", hidden, "gelu", residual=graph.inputs["hidden"])
    graph.save([output])
```

독립적인 컴포넌트의 입력, 최상위 토폴로지, `graph.save()`를 `generate_graph()` 안에서 정의합니다. Whisper의 통합 디코더 그래프에 사용되는 pre/cache/post 부분처럼 공유하는 그래프 구성에는 `_build_nodes(graph, inputs)`를 유지합니다.

로그 범위는 `BaseModel.gen_files()`에서 설정하므로 그래프 구성에는 별도의 로그 인수가 필요하지 않습니다.

형상에서 추론하는 입력 유형은 `quantizable=True`일 때 FP32(나중에 양자화할 부동소수점 그래프), `False`일 때 BF16(소스 가중치 정밀도를 사용하는 직접 그래프)입니다. 캐시 같은 정수 입력에는 `input_dtypes={"cache": np.int8}`를 사용하고 이름을 입력 사양과 일치시킵니다. 저수준 호출에는 기존의 명시적인 AFE 텐서 사양도 지원됩니다. 입력과 출력 순서는 제공한 사양을 따르며, 형상과 출력 유형은 AFE가 추론합니다. `constant()`는 부동소수점 데이터를 활성화 정밀도로 변환합니다. 정수 상수에 특정 비트 폭이 필요하면 예를 들어 `dtype=np.int32`를 지정합니다. 노드 이름은 AFE의 결정적인 생성 카운터를 따릅니다.

그래프 노드 유형 주석에는 `model_graph.py`에서 `Node`를 가져옵니다. 헬퍼에는 별도의 `quantizable` 플래그 대신 그래프를 전달합니다. `graph.constant()`는 부동소수점 정밀도를 자동으로 선택합니다. 호스트 측 배열 계산에 해당 정밀도가 필요하면 NumPy `graph.dtype`를 사용합니다. 단계 플래그는 `generate_graph()`와 그래프 생성에만 유지합니다.

`save()`는 MLA 서브넷을 종료하고 외부 그래프의 출력 튜플을 생성합니다. 정수 출력을 유지하고, EV에서 BF16 출력을 FP32로 변환하며, 표준 아티팩트 이름으로 저장합니다(부동소수점 그래프는 `.fp32`). 완성된 네트워크를 얻으려면 `finish()`를 사용합니다. 둘 다 `transform_subnet`를 받아 외부 출력을 추출하기 전에 모델별 재작성을 수행할 수 있습니다. 컴포넌트 헬퍼에는 그래프 자체를 전달하며 입력, 정밀도, 소스 가중치를 같은 객체에 유지합니다.

이름으로 지정한 NumPy 입력을 사용해 완성된 그래프를 실행합니다:

```python
graph.finish([output])
outputs = graph.run(hidden=x)
outputs_jax = graph.run(hidden=x, use_jax=True)
graph.save()
```

입력 이름, 형상, 데이터 유형은 선언한 입력과 일치해야 하며 암시적 유형 변환은 적용하지 않습니다. 출력은 `finish()`에 전달한 순서를 따릅니다. NumPy 실행은 AFE fast mode를 사용하므로 MLA 기준 연산과 결과가 다를 수 있습니다. `use_jax=True`는 JAX 기준 실행을 선택하며 fast mode는 영향을 주지 않습니다. JAX 연산은 설정된 백엔드를 사용하며 호환되는 JAX 설치 환경에서는 GPU를 사용할 수 있습니다. 이 API는 양자화, 컴파일 또는 Modalix 실행을 수행하지 않습니다. 그래프를 한 번만 완료한 뒤 `run()`과 `save()`를 반복 사용합니다. `save(outputs)`로 완료와 저장을 한 번에 수행할 수도 있습니다.

일반적인 연산에는 `add`, `sub`, `mul`, `matmul`, `concat`, `slice`, `transpose`, `reshape`, `softmax`, `topk`, `sum_channels`, `argmax`, `linear`, `conv`, `layer_norm`, `rms_norm`, `activation`, `softcap`, `mlp`, `rope`, `rope2d`, `split_heads`, `merge_heads`, `split_concat`, `clip`, `avgpool2d`, `space_to_depth`, `quant`, `dequant`가 있습니다. 게이트 MLP는 `projections=("gate_proj", "up_proj", "down_proj")`를 사용하며 기본값은 `("fc1", "fc2")`입니다. `rms_norm("model.norm", input)`은 언어 모델 설정의 epsilon과 가중치 오프셋을 사용합니다. vision 및 GDN 정규화에서 `epsilon`을 명시하면 `weight_offset`도 지정하지 않는 한 가중치 오프셋은 0입니다. `rms_norm(None, input, epsilon=...)`는 가중치 없는 채널을 추론합니다. RoPE는 전체, 부분 및 비율 지정 split-half 회전을 지원합니다. `rope2d(input, cos_x, sin_x, cos_y, sin_y)`는 채널의 네 분할을 `[x-real, x-imag, y-real, y-imag]` 순서로 회전합니다. `split_heads(input, heads, repeat=...)`는 grouped-query 패턴을 위해 각 헤드를 반복합니다. `split_concat()`는 reshape만으로 표현할 수 없는 공간/토큰 레이아웃에 AFE의 재그룹화 연산을 제공합니다. `space_to_depth(input, blocksize)`는 완전한 공간 블록을 채널로 병합합니다. `quant(input)`는 `(int8_values, scale)`을 반환하고 `dequant(int8_values, scale)`는 그래프의 활성화 정밀도를 복원합니다. `argmax(input)`는 INT32 채널 인덱스를 반환합니다. `slice(input, begin, end, stride, axis)`는 순위 4 FP32/BF16 텐서의 정렬되지 않은 연속 단일 축 채널 슬라이스에 선택자 컨볼루션을 자동으로 사용합니다. 다른 슬라이스는 AFE의 네이티브 동작을 유지합니다.

`linear("model.proj", input)`는 `model.proj.weight`와 선택적 바이어스를 찾아 OI 소스 가중치를 SiMa 레이아웃으로 변환합니다. 패킹된 가중치 값, 스케일, 정렬되지 않은 그룹 크기, 재배치 메타데이터를 유지합니다. `conv()`는 OIW 또는 OIHW 소스 레이아웃을 추론합니다. 명시적인 변환으로 Qwen의 5차원 패치 가중치 같은 다른 레이아웃도 변환할 수 있습니다. `WeightOptions`는 소스 이름, 레이아웃, 가중치/스케일/바이어스 변환 및 재배치 재정의 옵션을 설명합니다. 그룹 가중치를 슬라이싱할 때는 해당 `scale_process_func`를 제공하고 체크포인트의 실제 그룹 크기를 유지해야 합니다. LoRA 순위와 병합 어댑터 동작은 `linear()` 및 `mlp()`의 명시적 인수입니다.

프로젝션과 헤드 레이아웃 연산으로 attention을 구성합니다. 기존 프로젝션 반올림을 유지하도록 쿼리 스케일링을 `linear()`에 포함합니다:

```python
queries = graph.split_heads(graph.linear("attn.q_proj", hidden, scale=head_dim ** -0.5), heads)
keys = graph.split_heads(graph.linear("attn.k_proj", hidden), heads)
values = graph.split_heads(graph.linear("attn.v_proj", hidden), heads)
context = graph.attention(queries, keys, values)
output = graph.linear("attn.out_proj", graph.merge_heads(context))
```

`split_heads()`는 정렬되지 않은 헤드 채널을 선택자 컨볼루션으로 처리합니다. 표준 vision attention은 이러한 컨볼루션을 피하도록 정렬되지 않은 헤드의 프로젝션 가중치를 패딩합니다. 그룹 출력 가중치는 원래 레이아웃을 유지합니다. `attention()`는 쿼리/키 길이를 사용해 cross-attention을 포함한 큰 attention 텐서에 대해 별도의 헤드 분기를 선택합니다. 가산 `mask`를 받고, `score_scale`를 지정하지 않으면 쿼리가 이미 스케일링되었다고 가정합니다. 이 스케일은 Q×K 이후, 마스크 이전에 적용하여 Qwen vision 같은 모델의 BF16 연산 순서를 유지합니다. 마스크는 순위 4 텐서, 벡터 또는 스칼라이고 `[N,H,T_query,T_key]`로 브로드캐스트할 수 있어야 합니다. 지원하지 않는 형상은 `ValueError`를 발생시킵니다.

`graph.matmul(lhs, rhs)`는 MLA 배치 행렬 곱을 생성합니다. 두 전치 플래그의 기본값은 `False`입니다. Kᵀ×V에는 `transpose_a=True`, Q×Kᵀ에는 `transpose_b=True`를 사용합니다. 입력은 배치와 축약 차원이 일치하는 순위 4 FP32/BF16 텐서여야 합니다. 헤드 수가 나누어떨어지면 grouped-query attention에 암시적 반복을 사용합니다. 잘못된 형상이나 유형은 `ValueError`를 발생시킵니다.

`graph.softmax(x)`는 기본적으로 마지막 축을 사용하며 명시적인 `axis`도 지원합니다. 연속 단일 축 슬라이스에는 `graph.slice(x, start=0, stop=128, axis=-1)`를 사용하고 다중 축에는 기존 `begin`/`end`/`stride`/`axis` 목록을 사용합니다. 단일 축 형식의 시작 기본값은 0이며, 축을 명시하고 범위 안의 비어 있지 않은 경계를 지정해야 합니다. 정렬되지 않은 채널 처리도 유지합니다. 두 형식을 혼합하지 마세요.

`ModelGraph`는 AFE의 `SimaBuilder`를 확장하므로 같은 객체에서 공통 헬퍼와 네이티브 연산을 사용할 수 있습니다:

```python
projected = graph.linear("model.proj", graph.inputs["hidden"])
output = graph.add(projected, graph.inputs["hidden"])
graph.save([output])
```

간결한 기본 연산은 AFE 메서드의 직접 별칭이며 시그니처, 유형 추론, 결정적인 이름 지정을 유지합니다. 일반적이지 않은 연산에는 상속된 `create_*` 메서드를 같은 그래프에서 사용할 수 있습니다. 사용자 정의 다중 서브넷 생명주기가 필요한 그래프는 AFE의 `SimaBuilder`와 저수준 연산 헬퍼를 직접 사용할 수 있습니다.

테셀레이션은 `sima_analysis.get_tessellate_parameters()`에서 중앙 관리하여 추론합니다. HWC16 레이아웃, 자동 타일 크기, 결정적인 영구 버퍼 이름을 사용합니다. 일반 컴포넌트는 `BaseModel`에서 빈 재정의 설정을 상속합니다. 스트라이드 KV 캐시 같은 특수 레이아웃은 기존 `get_mla_input_tessellate_params()` / `get_mla_output_tessellate_params()` 메서드를 재정의합니다. 키는 텐서 인덱스이며 음수 인덱스는 끝에서부터 셉니다.

Whisper 컴포넌트, 모든 네이티브 언어 그래프 부분, 표준 vision tower에서 이 API를 사용합니다. attention, MLP, 회전 헬퍼는 같은 구현을 공유합니다.

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

테스트 매트릭스, 예상 개수, 베이스라인 정책은 `tests/README.md`에 있습니다. CI 호출은 `.github/workflows/model-compiler-tests.yml`에 있습니다. 바이너리 베이스라인을 커밋하는 대신 실행 중에 네이티브 SDK 그래프와 수치 비교 아티팩트를 생성하세요.

### 런타임 유효성 검사

후보 패키지와 런타임 테스트용 추가 기능을 빌드합니다.

```bash
./build.sh --all --clean
```

이 작업은 빌드만 수행하며 테스트는 실행하지 않습니다. Modalix에 일치하는 후보 LLiMa 패키지를 설치하고, 추가 아카이브를 추출한 다음, `tests/README.md`의 DevKit 런타임 테스트 지침에 따라 패키지화된 CTest와 pytest를 실행합니다.

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
