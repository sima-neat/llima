# 모델 배포

## 개요

컴파일 후에는 모델을 Modalix 장치에 배포하여 실행해야 합니다. Model Compiler는 이 과정을 간소화하기 위해 `llima-deploy` 유틸리티를 제공합니다.

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy <source_directory> <destination_directory>
```

위치:

- `source_directory` - 컴파일된 모델 디렉터리의 경로입니다(`devkit/`, `mpk/`와 함께 `sima_files/`가 포함되어 있으며, 선택적으로 `npy_files/` 하위 디렉터리가 포함될 수 있음).
- `destination_directory` - Modalix 장치(또는 rsync 배포를 위한 로컬 경로)의 대상 디렉터리

이 명령을 실행하면 배포 도구가 다음 세 가지 주요 단계를 수행합니다.

1.  소스 디렉터리에 필요한 파일(`sima_files/devkit/` 및 `sima_files/mpk/`)이 포함되어 있는지 확인합니다.
2.  MPK 아카이브에서 ELF 파일을 추출합니다(`*.tar.gz`).
3.  다음 항목을 `rsync`를 사용하여 대상 위치에 **동기화**합니다.
    - `devkit/` - 런타임 오케스트레이션 파일
    - `elf_files/` - 추출된 바이너리 파일
    - `npy_files/` - LoRA 어댑터 가중치(해당 파일이 있는 경우 자동으로 포함됨)

이 도구는 효율적인 파일 전송을 위해 내부적으로 `rsync`를 사용하며, 이미 최신 버전인 파일은 건너뜁니다.

## 배포 워크플로우

`llima-compile`을 사용하여 모델을 컴파일한 후 다음과 같은 디렉터리 구조를 갖게 됩니다.

``` text
Llama-3.2-3B-Instruct_out/
├── onnx_files/
└── sima_files/
    ├── devkit/
    └── mpk/
```

이것을 Modalix 장치에 배포하려면 다음 두 가지 옵션이 있습니다.

**옵션 A: Modalix 장치에 직접 배포**

호스트 장치가 Modalix 장치에 네트워크를 통해 연결되어 있는 경우:

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out sima@192.168.1.20:/media/nvme/llima/llama3_2
```

**옵션 B: 로컬 디렉터리에 배포하여 수동으로 전송**

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out llama3_2
sima-user@docker-image-id:/home/docker$ scp -r llama3_2 sima@192.168.1.20:/media/nvme/llima/
```

:::note
`192.168.1.20`은 Modalix IP 주소의 예입니다. 장치의 IP 주소를 사용하십시오.
:::

배포가 완료되면 Modalix 장치에 SSH로 접속하여 모델을 실행합니다.

``` console
modalix:~$ ssh sima@192.168.1.20
```

그런 다음 `llima` CLI를 사용하여 모델을 실행합니다. 자세한 내용은 [LLiMa CLI](runtime.md)를 참조하십시오.

``` console
modalix:~$ llima run <model_name>
```

## 추론적 디코딩 모델

`llima-compile`에 `--draft_model_path`를 입력하면, 결과에는 대상 및 초안 컴파일러의 출력이 하나의 상위 디렉터리에 포함됩니다. 상위 디렉터리를 한 번의 명령으로 배포합니다.

``` console
llima-deploy compiled-eagle3 spec-decoding-output
```

배포된 패키지에는 일반적인 런타임 모델 디렉터리 두 개가 포함되어 있습니다.

``` text
spec-decoding-output/
├── <target-model>/
│   ├── devkit/
│   └── elf_files/
└── <draft-model>/
    ├── devkit/
    └── elf_files/
```

상위 디렉터리를 실행하여 LLiMa가 해당 디렉터리에서 두 개의 모델을 식별하고 로드할 수 있도록 합니다. 이때, 직렬화된 추론 디코딩 구성이 사용됩니다.

``` console
llima run spec-decoding-output
```

## 문제 해결

**오류: "devkit 디렉터리를 찾을 수 없습니다."**

소스 디렉터리가 `llima-compile`의 출력 디렉터리인지 확인하세요. 해당 디렉터리에는 `sima_files` 하위 디렉터리가 포함되어야 합니다.

**오류: "mpk 디렉터리를 찾을 수 없습니다."**

컴파일이 성공적으로 완료되었는지 확인하세요. `sima_files/mpk/` 디렉터리에는 `.tar.gz` 파일이 포함되어야 합니다.

**배포 속도 저하**

- 압축 기능을 사용하여 `rsync`를 사용합니다. 기본적으로 이 도구는 `rsync -aP`를 사용합니다.
- 더 빠른 모델 로딩을 위해 Modalix에 NVMe 스토리지를 배포합니다.
- 변경된 파일만 배포하는 방법을 고려해 보세요. `--resume` 컴파일하는 동안
