# 시스템 요구 사항

GenAI 모델의 컴파일 및 배포를 위해 Model Compiler를 사용하십시오. 이 도구는 Neat 개발 환경과 함께 제공됩니다. 호스트 또는 Neat Framework 애플리케이션을 함께 개발하는 경우가 아니라면 전체 Neat 개발 환경이 필요하지 않습니다.

Model Compiler는 다음 요구 사항을 충족하는 시스템에 설치해야 합니다.

| 매개변수 | 설명 |
|----|----|
| **운영 체제** | Ubuntu 22.04/24.04 LTS 또는 Windows 11 또는 MacOS 12.0 이상. |
| **기억** | 128GB 이상을 권장합니다. |
| **저장** | 512GB의 여유 공간이 있는 제품을 선호합니다. |

:::note
128GB 용량의 장치에서는 컴파일이 완료되는 데 몇 시간이 걸릴 수 있으며, 이는 모델의 유형에 따라 달라집니다. 64GB 용량은 시각 기능을 갖추지 않은 모델에는 적합할 수 있습니다.
:::

## 필수 조건

- 최신 [`sima-cli`](/tools/sima-cli/)가 Model Compiler에 설치되어 있는지 확인하십시오.
- SDK 환경에 필요한 SiMa.ai 배포 자산에 접근할 수 있는지 확인하세요.
- 오픈 소스 모델을 다운로드하려면 유효한 Hugging Face 계정이 필요합니다.
- 일부 모델, 예를 들어 `google/paligemma`와 같은 모델은 Hugging Face에서 라이선스 계약에 동의해야 합니다. 해당 모델을 다운로드하기 전에 라이선스 내용을 검토하고 동의했는지 확인하십시오.
- CLI가 [사용자 액세스 토큰](https://huggingface.co/docs/hub/en/security-tokens) 및 `huggingface-cli`를 사용하여 Hugging Face에 액세스하도록 승인합니다. 참고로 `sima-cli`를 설치하면 `huggingface-cli`가 자동으로 설치됩니다.
