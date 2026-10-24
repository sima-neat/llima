# 시스템 요구 사항

GenAI 모델의 컴파일 및 배포를 위해 Model Compiler를 사용하십시오. 이 도구는 Neat 개발 환경과 함께 제공됩니다. 호스트 또는 Neat Framework 애플리케이션을 함께 개발하는 경우가 아니라면 전체 Neat 개발 환경이 필요하지 않습니다.

Model Compiler는 다음 요구 사항을 충족하는 시스템에 설치해야 합니다.

| 매개변수 | 설명 |
|----|----|
| **운영 체제** | Ubuntu 22.04/24.04 LTS 또는 Windows 11 또는 MacOS 12.0 이상. |
| **메모리** | 128GB 이상의 RAM을 권장합니다. |
| **저장** | 512GB의 여유 공간이 있는 제품을 선호합니다. |

:::note
RAM이 128GB인 시스템에서는 모델 유형에 따라 컴파일을 완료하는 데 몇 시간이 걸릴 수 있습니다. 시각 기능이 없는 모델에는 64GB RAM으로도 충분할 수 있습니다.
:::

## 필수 조건

- 최신 [`sima-cli`](/tools/sima-cli/)가 Model Compiler에 설치되어 있는지 확인하십시오.
- SDK 환경에 필요한 SiMa.ai 배포 자산에 접근할 수 있는지 확인하세요.
- 오픈 소스 모델을 다운로드하려면 유효한 Hugging Face 계정이 필요합니다.
- 일부 모델, 예를 들어 `google/paligemma`와 같은 모델은 Hugging Face에서 라이선스 계약에 동의해야 합니다. 해당 모델을 다운로드하기 전에 라이선스 내용을 검토하고 동의했는지 확인하십시오.
- CLI가 [사용자 액세스 토큰](https://huggingface.co/docs/hub/en/security-tokens) 및 `huggingface-cli`를 사용하여 Hugging Face에 액세스하도록 승인합니다. 참고로 `sima-cli`를 설치하면 `huggingface-cli`가 자동으로 설치됩니다.

### CI sysroot 선택

`deps/manifest.json`은 펌웨어 대상 `3.0.0_master_B2372`를 선택하며, CI는 일치하는
master sysroot 리비전 `3.0.0~pre2372`를 찾습니다.
`3.0.0~git202610140145.e5ece50-2121` 같은 develop 리비전도 지원합니다.
`sdk.channel`은 SDK 툴체인 이미지를 선택하며 플랫폼 빌드 채널과는 독립적입니다.

CI는 일회용 SDK 컨테이너에서 `scripts/ci/prepare_sysroot.sh`를 호출합니다.
준비 전에 공유 리졸버 출력이 펌웨어 선언과 일치하는지 검증합니다. master 3.0 리비전은
아카이브 체크섬으로 검증한 SDK 커밋 `2c325eeb1f0e4551ac32f2b9ef53a4cc18a07ccf`의
SDK 관리 트랜잭션 설치 프로그램을 사용해 pre-release Agate 저장소에서 준비합니다.
현재 SDK 개발 헬퍼는 master 리비전을 지원하지만 폐기된 `appcomplex-dev` 패키지도
요청하므로, CI 헬퍼는 고정된 설치 프로그램에서 해당 요청만 제거합니다. SDK 패키지
선택이 수정되면 이 임시 호환성 처리를 제거해야 합니다. 원본 SDK 이미지 기록은 보존되며,
활성 오버레이와 패키지 목록에 선택된 정확한 sysroot가 기록됩니다. 이 목록에
리졸버가 선택한 정확한 플랫폼 버전이 없으면 준비가 실패합니다.

이 헬퍼는 일회용 CI 컨테이너용이며 사용 중인 개발 환경을 위한 도구가 아닙니다.
LLiMa는 계속 플랫폼 라이브러리를 직접 사용하며 Internals 패키지에 의존하지 않습니다.
이 sysroot로 빌드하는 것은 DevKit 검증이나 펌웨어 업데이트를 의미하지 않습니다.
