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

在 CLI 模式下，聊天記錄預設會啟用。每個提示和回應都會被保留，作為下一次對話的上下文，直到您使用 `clear history` 清除它。與提示一起提交的圖片也會保留為該記錄的一部分。`clear image` 只會捨棄為下一個提示排入佇列的圖片，而 `clear history` 會移除已提交的提示、回應和所有圖片。已設定的系統提示會維持有效，直到您使用 `clear system` 將其清除，或使用 `set system` 將其取代。

``` console
modalix:~$ llima run <model> [options]
```

| 論點；爭論 | 描述 |
|----|----|
| `model` | 模型 ID 或路徑（例如：`Qwen3-VL-8B-Instruct-a16w4`）。 |
| `--stt_model_path` | 語音轉文字模型的 ELF 檔案路徑（選用）。 |

若要查看所有可用的選項，請執行 `llima run -h`。

**範例**

``` console
modalix:~$ llima run Qwen3-VL-4B-Instruct-GPTQ-a16w4
```

## 互動式指令

一旦在命令列介面 (CLI) 模式下啟動 `llima run`，請在提示字元處使用以下指令：

| 指令 | 描述 |
|----|----|
| `add image <file>` | 將圖片新增到目前的提示詞內容中。 |
| `clear image` | 捨棄自上次提交提示後排入佇列的圖片；先前對話中的圖片會保留在記錄中。 |
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


## 使用 Neat 建立應用程式

在用 `llima run` 驗證您的模型後，請參閱
[GenAI 模型 ](/develop-apps/development-workflow/genai-model/)，以便透過通用的 API 端點來部署它，或直接從 C++ 或 Python 應用程式中使用它。
