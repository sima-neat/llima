# MoLE - Modalix 語言模型評估工具

## 概述

MoLE（Modalix Language Model Evaluator）是用來評估在 Modalix 平台上執行的 LLM 與 VLM 之準確度與效能的基準測試工具。

文字準確度評估使用 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)。Modalix 的影像加文字準確度評估使用 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)，沿用上游的任務、答案擷取與評分方式。兩者皆使用 `llima-benchmark accuracy`。MoLE 支援兩種後端：

- **hf** — 在主機上使用 HuggingFace transformers 執行文字準確度評估（參考基準）
- **modalix** — 透過 `llima benchmark-server` 在 Modalix 開發板上執行評估

## 安裝

請在 SDK Docker 容器外，於主機或 Modalix DevKit 上的獨立虛擬環境中安裝 MoLE。在 DevKit 上執行還需要安裝 LLiMa 執行階段。執行階段的安裝流程請參閱 [Neat Framework 安裝](/getting-started/neat-library/)。

使用 `sima-cli` 安裝 MoLE：

``` console
host:~$ sima-cli neat install llima/mole
```

此命令會將 MoLE 安裝至 `~/sima-mole-venv` 的虛擬環境中。

### 手動安裝

若不使用 `sima-cli`，請在下載的 MoLE 套件目錄中執行隨附的安裝程式：

```bash
bash ./install_mole.sh --venv ~/sima-mole-venv
```

若要直接安裝 wheel，請啟用 Python 3.11–3.13 虛擬環境，將 wheel 檔名替換為下載的版本後，執行以下命令：

```bash
python -m pip install './sima_lmm-<version>-py3-none-any.whl[mole]'
python -m pip install --no-deps "lmms-eval==0.7.3" "tenacity>=8.3.0"
```

第二個命令會沿用 MoLE 的共用相依套件，啟用影像基準測試。

## 使用方式

首先，啟用 MoLE 虛擬環境：

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

從主機執行 Modalix 評估時，必須提供 `--board_ip`。在 Modalix DevKit 上可省略此選項，以使用本機執行階段。啟動伺服器時請提供 `--board_model`；本機伺服器記錄會儲存至 `<output_dir>/server.log`。使用 `--no-board_start_server` 可連接既有伺服器，且不會啟動或停止該伺服器。

MoLE 提供 `llima-benchmark accuracy` 與 `llima-benchmark perf` 子命令。文字工作負載使用與已部署模型相符的 HuggingFace 模型 ID 或本機權杖化器/設定目錄。影像工作負載使用已部署 VLM 的資源；`<model_id>` 是該次執行的標籤。

### 準確度基準測試

#### 文字準確度

透過標準任務評估模型品質：

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| 引數 | 說明 |
|----|----|
| `model_id` | 文字準確度評估使用的 HuggingFace 模型 ID 或本機權杖化器/設定目錄；影像準確度評估則使用描述性標籤。 |
| `-b` | 使用的後端：`modalix`（在開發板上執行）或 `hf`（在主機上執行，作為參考基準）。 |
| `-t` | **必要。** 一或多個評估任務。例如：`hellaswag`、`triviaqa`、`piqa`、`winogrande`、`wikitext`。所有可用任務請參閱[任務清單](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md)。 |
| `-o` | 基準測試結果的輸出目錄。 |
| `--board_ip` | 遠端開發板的 IP 或主機名稱。從主機執行時為必要選項；在 Modalix 本機執行時請省略。 |
| `--board_model` | DevKit 上已編譯模型的目錄。啟動伺服器時為必要選項。 |
| `--max_num_tokens` | 最大上下文長度。必須小於或等於編譯時使用的值。 |
| `-n, --num_samples` | 要評估的樣本數。若未指定，會執行完整任務集。 |
| `--board_ssh_user` | Modalix 開發板的 SSH 使用者名稱。選填，預設值為 `sima`。\# |
| `--board_ssh_pass` | Modalix 開發板的 SSH 密碼。選填。設定後可啟用非互動式自動基準測試。 |

:::important
使用 logits 或對數概似值的任務，必須在編譯時指定 `--return_logits`。以生成為基礎的任務（包含影像加文字評估）則不需要。請參閱[模型編譯](compilation_genai.md)。
:::

在 `-b modalix` 模式下，結果表格會標示為 Modalix 後端結果，並包含目標開發板。由於 MoLE 會使用 HuggingFace 的 `model_id` 進行權杖化並取得任務中繼資料，因此仍會顯示該 ID。

若要使用 HuggingFace 後端作為參考基準：

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

若要查看所有可用選項，請執行 `llima-benchmark accuracy -h`。

#### 影像加文字準確度

評估工具隨 MoLE 安裝，並使用已部署 VLM 的原生聊天範本、影像前處理與上游任務評分方式。以下任務已在 Modalix 上使用 Qwen3.5-2B 測試：

| 任務 | 評估重點 | 驗證 |
|----|----|----|
| `realworldqa` | 真實場景與空間推理 | 完整資料集 |
| `docvqa_val` | 文件閱讀與 OCR | 冒煙測試 |
| `chartqa` | 圖表閱讀與數值推理 | 冒煙測試 |
| `mmstar` | 視覺感知與多模態推理 | 10 筆樣本的冒煙測試 |
| `ai2d` | 科學圖解理解 | 10 筆樣本的冒煙測試 |
| `ocrbench` | 文字辨識與理解 | 10 筆樣本的冒煙測試 |
| `textvqa_val` | 自然影像中的文字問答 | 10 筆樣本的冒煙測試 |

也可透過 `-t` 選擇其他個別的 `lmms-eval` 影像輸入生成任務，但須符合下列限制，且並非所有任務都已測試。

```bash
llima-benchmark accuracy qwen3.5-2b -b modalix \
    -t realworldqa docvqa_val chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-on-devkit> \
    --max_num_tokens 4096 -o results/vlm
```

在 DevKit 上請省略 `--board_ip`。不需要主機端模型資源。請使用相互匹配的執行階段與 MoLE 發行版本。

| 影像準確度選項 | 說明 |
|----|----|
| `--max_new_tokens` | 覆寫任務的輸出權杖上限；否則保留上游預設值。 |
| `--enable_thinking` | 啟用模型的原生思考範本；預設為停用。 |
| `--accuracy_manifest` | 使用固定的資料集修訂版本或預先準備的本機資料集，如下所示。 |

請使用個別任務名稱，分開執行文字與影像任務。影像準確度評估支援使用貪婪解碼的一般 VLM。不支援音訊/影片、多輪對話、視覺少樣本與重複請求任務。輸入加上輸出權杖上限必須符合指定與編譯的上下文長度；輸入不會被截斷。多張影像會保留任務定義的順序。原生 EOS 會停止生成；任務的停止字串會截去回傳答案中的後續內容。

#### 預先準備的影像資料集

資料集會自動下載。若要固定修訂版本或離線評估，請傳入 `--accuracy_manifest manifest.json`。例如搭配 `-t chartqa`：

```json
{
  "schema_version": 1,
  "tasks": {
    "chartqa": {
      "revision": "<40-character-dataset-commit>",
      "split": "test",
      "local_path": "datasets/chartqa"
    }
  }
}
```

使用 HuggingFace 的 `DatasetDict.save_to_disk()` 準備本機資料集，並保留上游的欄位、影像與評分用資料分割。路徑以資訊清單所在位置為基準，且任務名稱必須與 `-t` 相符。省略 `local_path` 可下載固定的修訂版本。離線執行時，請設定 `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`。

#### 影像準確度結果

輸出目錄包含：

- `accuracy.json`：彙總指標、樣本數、資料集修訂版本、部署路徑、執行階段版本與設定。
- `accuracy-answers.jsonl`：生成的答案、影像/提示識別資訊、權杖數、結束原因與錯誤。
- `accuracy-samples.jsonl`：上游的逐樣本評分結果。

省略 `-n` 會執行完整資料集。失敗時會停止評估、保留部分答案，且不會產生彙總指標。因輸出權杖上限而遭截斷的情況會被計數，且仍會評分。

### 效能基準測試

在 Modalix 開發板上，針對不同輸入長度測量首個權杖時間（TTFT）與每秒權杖數（TPS）：

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| 引數 | 說明 |
|----|----|
| `model_id` | 文字工作負載使用與已部署模型相符的 HuggingFace 模型 ID 或本機權杖化器/設定目錄；影像工作負載則使用描述性的執行標籤。 |
| `-o` | 基準測試結果的輸出目錄。 |
| `--board_ip` | 遠端開發板的 IP 或主機名稱。從主機執行時為必要選項；在 Modalix 本機執行時請省略。 |
| `--board_model` | DevKit 上已編譯模型的目錄。啟動伺服器時為必要選項。 |
| `--max_num_tokens` | 最大上下文長度。必須小於或等於編譯時使用的值。 |
| `--max_new_tokens` | 輸出中可生成的最大權杖數。 |
| `--input_lengths` | 選填，指定要測試的確切輸入權杖長度。數值不得重複，且每個值加上 `--max_new_tokens` 都必須在 `--max_num_tokens` 以內。省略時，MoLE 會自動產生長度為 2 的冪次的分組。 |
| `--board_ssh_user` | Modalix 開發板的 SSH 使用者名稱。選填，預設值為 `sima`。 |
| `--board_ssh_pass` | Modalix 開發板的 SSH 密碼。選填。設定後可啟用非互動式自動基準測試。 |

若要查看所有可用選項，請執行 `llima-benchmark perf -h`。

使用 `--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum`，可採用原生聊天/工具提示，取代依權杖長度分組的測試。省略 `-n` 會執行所選的完整資料集；`-n 20` 則執行前 20 筆樣本。資料集會自動下載並快取。BFCL 涵蓋簡單/多重呼叫案例；不會執行工具。

`--board_model` 決定使用一般解碼或推測解碼。比較模型時，請以相同資料集與設定分別執行，僅變更 `--board_model` 與 `-o`。每次執行都會排除一次暖機，並在每個請求前重設 KV 狀態。會套用模型的原生聊天範本與 EOS 停止機制。資料集的輸出權杖上限包含第一個權杖；既有的長度分組則測量第一個權杖加上 `--max_new_tokens` 個解碼步驟。

#### 影像加文字效能

使用 `--dataset realworldqa|docvqa_val|chartqa`，可採用與準確度評估相同的上游影像/文字提示，測量一般或推測解碼 VLM：

```bash
llima-benchmark perf qwen3.5-vlm --dataset chartqa \
    --board_ip <devkit-address> --board_model <compiled-model-or-speculative-bundle> \
    --max_num_tokens 4096 --max_new_tokens 256 -o results/vlm-perf
```

省略 `-n` 時，會先排除一次暖機，再執行完整資料集一次。`--dataset_manifest` 接受與 `--accuracy_manifest` 相同的固定修訂版本/本機資料集格式；任務名稱必須與 `--dataset` 相符。影像會保留資料集中的解析度，並使用原生執行階段的前處理。不需要主機端模型資源。準確度評估仍僅限一般解碼。

`-o` 會將每個請求的時間、權杖數、彙總指標、草稿貢獻率、失敗紀錄、基準測試設定與資料集來源儲存至 `performance.json`。解碼 TPS 不包含 TTFT。影像 TTFT 包含原生影像前處理與視覺編碼；文字 TTFT 從語言模型的預填開始計算。兩者都不包含網路傳輸與主機端的資料集準備時間。草稿貢獻率為接受的草稿權杖數除以所有生成的權杖數。這些工作負載測量的是效能，而非準確度。
