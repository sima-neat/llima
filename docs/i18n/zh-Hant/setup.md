# 系統需求

對於 GenAI 模型進行編譯和部署時，請使用 **Model Compiler**，該工具與 Neat 開發環境一起提供。除非您還在開發主機應用程式或 Neat Framework 應用程式，否則您不需要完整的 Neat 開發環境。

Model Compiler 需要安裝在一台符合以下要求的機器上。

| 參數 | 描述 |
|----|----|
| **作業系統** | Ubuntu 22.04/24.04 LTS，或 Windows 11，或 MacOS 12.0 或更新版本。 |
| **記憶體** | 建議配備 128GB 或更多的系統記憶體。 |
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

### CI sysroot 選擇

`deps/manifest.json` 選擇韌體目標 `3.0.0_master_B2372`，CI 會解析相符的
master sysroot 版本記錄 `3.0.0~pre2372`。
`3.0.0~git202610140145.e5ece50-2121` 等 develop 版本記錄仍受支援。
`sdk.channel` 選擇 SDK 工具鏈映像，與平台建置通道無關。

CI 在拋棄式 SDK 容器中呼叫 `scripts/ci/prepare_sysroot.sh`。
準備前會驗證共用解析器輸出是否符合韌體宣告。master 3.0 版本使用 SDK 管理的
交易式安裝程式，固定至 SDK 提交 `2c325eeb1f0e4551ac32f2b9ef53a4cc18a07ccf` 並以封存檔
檢查碼驗證，從 pre-release Agate 儲存庫準備環境。目前 SDK 開發版輔助程式支援
master 版本，但仍會要求已停用的 `appcomplex-dev` 套件；CI 輔助程式只會從固定版本的
安裝程式移除該要求。SDK 套件選擇修正後，應移除此暫時性的相容處理。
原始 SDK 映像記錄會保留；使用中的覆疊與套件清單會記錄選定的確切 sysroot。
若該清單未包含解析器選定的確切平台版本，準備作業便會失敗。

此輔助程式僅用於拋棄式 CI 容器，不適用於使用中的開發環境。
LLiMa 仍直接使用平台函式庫，不依賴 Internals 套件。使用此 sysroot 建置並不代表
已完成 DevKit 驗證，也不會更新其韌體。
