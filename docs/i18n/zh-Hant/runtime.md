# LLiMa 指令列介面

在 Modalix 上使用 `llima` CLI 來管理預先編譯的模型，並進行簡單的執行階段測試。這對於檢查模型是否能正確載入、接受提示，以及在您將其與 Neat Framework 的直接 API 或 Neat GenAI 伺服器端點整合之前，是否能產生輸出結果，都非常有用。

## 模型管理員

LLiMa 包含一個模型管理工具，透過 `llima` CLI 進行操作。您可以透過它來搜尋、下載、列出、移除，以及直接從命令列執行預先編譯的模型。模型預設儲存在 `/media/nvme/llima/models` 目錄下。設定 `LLIMA_MODELS_PATH` 以使用不同的模型目錄。

瀏覽可用的模型：

``` console
modalix:~$ llima search
modalix:~$ llima search qwen
```

透過名稱下載模型，但不要包含 `simaai/` 組織的字首：

``` console
modalix:~$ llima pull Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

模型成品會同時下載，其中最大的成品會優先排程。暫時性的 HTTP 錯誤會自動重試。對於同一個模型，下載請求會依序處理，且取消下載請求會保留所有已下載並驗證過的成品。

列出並移除本機安裝的模型：

``` console
modalix:~$ llima list
modalix:~$ llima rm Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## 執行 LLiMa

使用 `llima run` 作為一個簡單的執行階段，用於在 Modalix 上進行初始模型驗證。

在 CLI 模式下，聊天記錄預設會啟用。每個提示和回應都會被保留，作為下一次對話的上下文，直到您使用 `clear history` 清除它。與提示一起提交的圖片也會保留為該記錄的一部分。`clear history` 會移除已提交的提示、回應和所有圖片。已設定的系統提示會維持有效，直到您使用 `clear system` 將其清除，或使用 `set system` 將其取代。

``` console
modalix:~$ llima run <model> [options]
```

| 論點；爭論 | 描述 |
|----|----|
| `model` | 模型 ID 或路徑（例如：`Qwen3-VL-8B-Instruct-a16w4`）。 |
| `--stt_model_path` | 語音轉文字模型的 ELF 檔案路徑（選用）。 |

若要查看所有可用的選項，請執行 `llima run -h`。

若要停用自動嵌入卸載並將表格保留在 DRAM 中，請執行 `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off llima run <model>`。

**範例**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## 互動式指令

一旦在命令列介面 (CLI) 模式下啟動 `llima run`，請在提示字元處使用以下指令：

| 指令 | 描述 |
|----|----|
| `add image <file>` | 將圖片新增到目前的提示詞內容中。 |
| `set system <prompt>` | 設定系統提示。 |
| `clear system` | 清除系統提示、聊天記錄和圖片。 |
| `clear history` | 清除已提交的提示、回應和所有圖片，同時保留系統提示。 |
| `print history` | 列印聊天記錄。 |
| `set audio <file>` | 將音訊檔案設定為要進行轉錄的查詢內容。 |
| `set language <lang>` | 設定用於轉錄的語言字串。 |
| `set lora <name>` | 使用來自 `npy_files` 資料夾的 LoRA 權重。 |
| `unset lora` | 將 LoRA 模型還原至基準模型。 |
| `enable-thinking` | 啟用思考模式，並清除聊天記錄。 |
| `disable-thinking` | 關閉思考模式並清除聊天記錄。 |
| `quit` | 退出。 |
| `help` | 列印可用的指令。 |


## KV 快取逐出

預設情況下，編譯後的上下文（`max_num_tokens`）是硬性上限：較長的提示會被拒絕，而生成會在出現「Cache full」時停止。啟用 KV 快取逐出後，執行階段會改為捨棄已快取的權杖，讓對話可以持續到模型訓練時的上下文長度（最多約 65,000 個權杖）。被逐出的權杖將會遺失：模型無法再回答與其相關的問題。只要對話仍能放入編譯後的上下文中，輸出就不會改變。

若要啟用，請在已部署的 `devkit/vlm_config.json` 中將 `kv_eviction` 加入 `pipeline_cfg`，或呼叫 `VisionLanguageModel::set_kv_eviction()`：

``` json
"pipeline_cfg": {
    "max_num_tokens": 2048,
    "kv_eviction": { "policy": "keydiff" }
}
```

| 欄位 | 描述 |
|----|----|
| `policy` | `off`（預設）、`sink_window`（保留最新的權杖）或 `keydiff`（保留其鍵與平均鍵差異最大的權杖）。 |
| `budget_tokens` | 選填。每次逐出保留的權杖數；必須是模型編譯時的預填群組偏移量之一（錯誤訊息會列出這些值）。預設為快取的 7/8：每次逐出會釋放快取的八分之一，至少一個預填群組。 |

逐出時一律會保留系統提示與工具定義、最前面的權杖，以及最新的權杖。系統提示與工具定義必須在預算中為一個預填群組（至少 128 個權杖）保留空間，否則每個請求都會失敗。

只要先前的訊息原封不動地再次傳送，新的對話輪次就會從快取繼續。如果在逐出之後先前的訊息有所變更，例如客戶端截斷了歷史記錄、聊天範本移除了先前的 Qwen3 思考區塊，或範本將系統提示移到最新的使用者訊息中（Mistral v0.3），執行階段會重新處理整段對話。

僅支援所有層都使用完整注意力（full attention），且 RoPE 縮放為預設、線性或 Llama 3 的純文字模型；其他模型會在載入時失敗，並顯示指出不支援功能的錯誤。GGUF 模型目前尚不支援，因為已部署的模型沒有記錄其訓練時的上下文長度。處理長提示所需的時間仍與其長度成正比。不支援逐出的執行階段會忽略 `kv_eviction` 欄位。

## 使用 Neat 建立應用程式

在用 `llima run` 驗證您的模型後，請參閱
[GenAI 模型 ](/develop-apps/development-workflow/genai-model/)，以便透過通用的 API 端點來部署它，或直接從 C++ 或 Python 應用程式中使用它。
