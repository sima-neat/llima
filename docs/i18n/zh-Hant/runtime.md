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

在預設模式（`auto`）下，若原始 `.bin` 檔案位於模型的 `devkit/` 目錄中，
就會卸載嵌入表格。若沒有這些檔案，表格會保留在 DRAM 中。
使用非 NVMe 儲存裝置時會顯示警告，因為讀取速度較慢可能會降低效能。
若要將表格保留在 DRAM 中，請執行 `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=off llima run <model>`。
若要強制卸載，請執行 `SIMA_LLIMA_RUN_EMBEDDING_OFFLOAD=on llima run <model>`。
在此模式下，若原始檔案不存在或無效，指令就會失敗。

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


## 使用 Neat 建立應用程式

在用 `llima run` 驗證您的模型後，請參閱
[GenAI 模型 ](/develop-apps/development-workflow/genai-model/)，以便透過通用的 API 端點來部署它，或直接從 C++ 或 Python 應用程式中使用它。

## 重複使用獨立的提示詞上下文

C++ 應用程式可透過 `ModelOptions.max_kv_cache_slots`（預設為 1）載入有容量上限的集區，並以 `GenerationOptions.cache_id` 選擇上下文。

```cpp
#include <sima_lmm/vision_language_model.hpp>

simaai::llima::VisionLanguageModel model(model_dir, simaai::llima::ModelOptions{4});
auto chat = model.create_chat();
chat.add_query("Hello");
simaai::llima::GenerationOptions options;
options.cache_id = "assistant";
auto result = model.generate(chat, options);
model.remove_kv_cache("assistant");
```

每次都應傳送完整提示詞或歷史記錄。ID 用於選擇可重複使用的前綴狀態，不會儲存應用程式訊息。省略 ID 的既有上下文也會使用一個槽位。裝置快取空間在建構時配置，權重和執行工作區則共用。`kv_cache_bytes_per_slot()` 回報每個槽位的裝置位元組數，`kv_cache_count()` 回報包含既有上下文的已指派數量。`remove_kv_cache()` 解除具名關聯，`clear_kv_caches()` 解除所有關聯，但保留實體配置。容量已滿時要求新 ID 會引發 `KVCacheCapacityError`，不會自動淘汰。生命週期呼叫需要閒置模型，生成期間的呼叫會遭拒絕。取消會使所選上下文失效，變更 LoRA 會使所有上下文失效。推測解碼的目標與草稿必須具有相同槽位上限。新增選項與指標會改變值型別的配置，因此請使用相符的 LLiMa 標頭與程式庫重新建置使用端。
