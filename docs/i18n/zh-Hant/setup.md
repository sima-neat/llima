# 系統需求

對於 GenAI 模型進行編譯和部署時，請使用 **Model Compiler**，該工具與 Neat 開發環境一起提供。除非您還在開發主機應用程式或 Neat Framework 應用程式，否則您不需要完整的 Neat 開發環境。

Model Compiler 需要安裝在一台符合以下要求的機器上。

| 參數 | 描述 |
|----|----|
| **作業系統** | Ubuntu 22.04/24.04 LTS，或 Windows 11，或 MacOS 12.0 或更新版本。 |
| **記憶** | 建議使用 128GB 或更大的儲存空間。 |
| **儲存** | 建議使用 512GB 的可用儲存空間。 |

:::note
在 128GB 的機器上，編譯完成可能需要數小時，具體時間取決於模型的類型。對於沒有視覺功能的模型，64GB 的機器或許也能正常運作。
:::

## 先決條件

- 請確認已在 Model Compiler 中安裝最新的 [`sima-cli`](/tools/sima-cli/)。
- 請取得您 SDK 環境所需的相關資源，以便進行 SiMa.ai 版本發布。
- 請擁有有效的 Hugging Face 帳戶，以便下載開源模型。
- 某些模型，例如 `google/paligemma`，需要您在 Hugging Face 上接受授權協議。請務必在嘗試下載這些模型之前，先檢閱並接受授權協議。
- 授權 CLI 存取 Hugging Face 使用一個 [使用者存取權杖](https://huggingface.co/docs/hub/en/security-tokens) 以及 `huggingface-cli`請注意，正在安裝。 `sima-cli` 自動安裝 `huggingface-cli`.
