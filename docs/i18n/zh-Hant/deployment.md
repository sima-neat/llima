# 模型部署

## 總覽

編譯完成後，需要將模型部署到 Modalix 裝置上以進行執行。Model Compiler 提供了 `llima-deploy` 工具，以簡化此流程：

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy <source_directory> <destination_directory>
```

地點：

- `source_directory` - 編譯後的模型目錄路徑（包含 `sima_files/`，以及 `devkit/`、`mpk/` 子目錄，並且可選擇性地包含 `npy_files/` 子目錄）。
- `destination_directory` - Modalix 裝置上的目標目錄（或用於 rsync 部署時的本機路徑）。

當您執行此指令時，部署工具會執行三個關鍵步驟：

1.  **驗證**原始目錄是否包含必要的檔案（`sima_files/devkit/` 和 `sima_files/mpk/`）。
2.  **從** MPK 壓縮檔中提取 ELF 檔案 (`*.tar.gz`)。
3.  使用 `rsync` 將以下內容同步到目標位置：
    - `devkit/` - 執行階段協調檔案
    - `elf_files/` - 提取的二進位檔案
    - `npy_files/` - LoRA 配接器權重（如果存在，則會自動包含）。

這個工具在內部使用 `rsync`，以實現高效的檔案傳輸，並且會跳過那些已經是最新的檔案。

## 部署流程

在用 `llima-compile` 編譯您的模型之後，您將會得到如下的目錄結構：

``` text
Llama-3.2-3B-Instruct_out/
├── onnx_files/
└── sima_files/
    ├── devkit/
    └── mpk/
```

若要將此軟體部署到 Modalix 裝置，您有兩種選擇：

**選項 A：直接部署到 Modalix 裝置**

如果您的主機機器可以透過網路連線到 Modalix 裝置：

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out sima@192.168.1.20:/media/nvme/llima/llama3_2
```

**選項 B：部署到本機目錄，以便手動傳輸**

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out llama3_2
sima-user@docker-image-id:/home/docker$ scp -r llama3_2 sima@192.168.1.20:/media/nvme/llima/
```

:::note
`192.168.1.20` 是一個 Modalix IP 位址的範例。請使用您裝置的 IP 位址。
:::

部署完成後，透過 SSH 連接到 Modalix 裝置，然後執行模型：

``` console
modalix:~$ ssh sima@192.168.1.20
```

然後使用 `llima` 命令列介面來執行模型。請參閱 [LLiMa 命令列介面 ](runtime.md)，以取得詳細資訊。

``` console
modalix:~$ llima run <model_name>
```

## 推測式解碼模型

當提供 `llima-compile` 參數 `--draft_model_path` 時，其輸出內容將包含目標和草稿編譯器的輸出，且這些輸出都位於同一個父目錄下。使用單一指令部署該父目錄：

``` console
llima-deploy compiled-eagle3 spec-decoding-output
```

已部署的套件包含兩個普通的執行階段模型目錄：

``` text
spec-decoding-output/
├── <target-model>/
│   ├── devkit/
│   └── elf_files/
└── <draft-model>/
    ├── devkit/
    └── elf_files/
```

執行父目錄，以便 LLiMa 可以識別並載入兩個模型，這些模型都來自其序列化的推測式解碼設定：

``` console
llima run spec-decoding-output
```

## 疑難排解

**錯誤：「devkit directory cannot be found」**

請確認原始目錄是 `llima-compile` 的輸出目錄，該目錄應包含 `sima_files` 子目錄。

**錯誤：「mpk directory cannot be found」**

請確認編譯已成功完成。`sima_files/mpk/` 目錄應包含 `.tar.gz` 檔案。

**部署速度慢**

- 使用壓縮功能搭配 `rsync`：此工具預設使用 `rsync -aP`。
- 部署到 NVMe 儲存裝置上，以加快 Modalix 的模型載入速度。
- 在編譯期間，請考慮僅部署已變更的檔案，並使用 `--resume`。
