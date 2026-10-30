# MoLE - Modalix 語言模型評估工具

## 總覽

MoLE（Modalix 語言模型評估器）是一個基準測試工具，用於評估在 Modalix 平台上運行的 LLM 的準確性和效能。

它擴展了 [ EleutherAI 的 lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)，並支援兩種後端：

- **hf** — 在主機上執行評估，使用 HuggingFace 的轉換器模型（作為基準參考）。
- **modalix** — 在 `llima benchmark-server` 上，透過 Modalix 板卡執行基準測試。

## 安裝

請在 SDK Docker 容器外的主機或 Modalix DevKit 上，以獨立虛擬環境安裝 MoLE。在 DevKit 上執行時，也必須安裝 LLiMa 執行階段。執行階段安裝流程請參閱 [Neat Framework 安裝](/getting-started/neat-library/)。

使用 `sima-cli` 安裝 MoLE：

``` console
host:~$ sima-cli neat install llima/mole
```

這會將 MoLE 安裝到 `~/sima-mole-venv` 的虛擬環境中。

## 使用方式

首先，啟動 MoLE 虛擬環境：

``` console
host:~$ source ~/sima-mole-venv/bin/activate
```

從主機執行 Modalix 評估時，必須提供 `--board_ip`。在 Modalix DevKit 上省略此選項即可使用本機執行階段。啟動伺服器時請提供 `--board_model`；本機伺服器日誌會儲存至 `<output_dir>/server.log`。使用 `--no-board_start_server` 可連線至現有伺服器，而不啟動或停止該伺服器。

MoLE 提供 `llima-benchmark accuracy` 和 `llima-benchmark perf` 子命令。在 Modalix 模式下，`<model_id>` 可指定 HuggingFace 模型 ID，或與部署模型相符的本機 tokenizer／設定目錄。MoLE 使用該 tokenizer 準備請求。

### 精確度基準測試

評估模型在標準任務中的品質：

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b modalix \
    -t <task> \
    -o <output_dir> \
    --max_num_tokens <max_num_tokens> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board>
```

| 論點；爭論 | 描述 |
|----|----|
| `model_id` | HuggingFace 模型 ID（例如：`meta-llama/Llama-3.2-3B-Instruct`）。對於 `-b modalix`，這必須與已部署模型的權杖化器/設定檔相符。 |
| `-b` | 後端程式可選擇使用：`modalix`（在板子上執行）或 `hf`（在主機上執行，作為基準）。 |
| `-t` | **必要。** 一個或多個評估任務。範例任務：`hellaswag`、`triviaqa`、`piqa`、`winogrande`、`wikitext`。請參閱[任務清單](https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.11/lm_eval/tasks/README.md)，以查看所有可用的任務。 |
| `-o` | 基準測試結果的輸出目錄。 |
| `--board_ip` | 遠端板卡的 IP 位址或主機名稱。從主機執行時為必填；在 Modalix 上本機執行時請省略。 |
| `--board_model` | DevKit 上的已編譯模型目錄。啟動伺服器時為必填。 |
| `--max_num_tokens` | 最大上下文長度。必須等於或小於編譯時使用的值。 |
| `-n, --num_samples` | 要評估的樣本數量。如果未指定，則執行整個任務集。 |
| `--board_ssh_user` | 用於 Modalix 裝置的 SSH 使用者名稱。為選用設定，預設值為：`sima`。 |
| `--board_ssh_pass` | 用於 Modalix 裝置的 SSH 密碼。為選用設定。設定後可啟用非互動式的自動基準測試。 |

:::important
使用 `-b modalix` 進行準確度和對數似然基準測試，需要部署的模型必須使用 `--return_logits` 進行編譯。此選項預設為關閉狀態。請參閱 [模型編譯](compilation_genai.md)。如果模型在沒有此選項的情況下進行編譯，基準測試將失敗，並顯示：`model not compiled with --return_logits; accuracy/loglikelihood tasks are unsupported`。
:::

在 `-b modalix` 模式下，結果表格會標記為 Modalix 後端結果，並包含目標板。 HuggingFace `model_id` 仍然會出現，因為 MoLE 會將其用於分詞和任務元資料。

若要將 HuggingFace 後端作為參考基準：

``` console
(sima-mole-venv) host:~$ llima-benchmark accuracy <model_id> -b hf -t <task> -o <output_dir>
```

對於所有可用的選項，請執行 `llima-benchmark accuracy -h` 基準測試。

### 效能基準測試

測量在 Modalix 晶片上，針對不同輸入長度時，從開始到產生第一個 Token 的時間（TTFT），以及每秒產生的 Token 數量（TPS）。

``` console
(sima-mole-venv) host:~$ llima-benchmark perf <model_id> \
    -o <output_dir> \
    --board_ip <board_ip> \
    --board_model <model_path_on_board> \
    --max_num_tokens <max_num_tokens> --max_new_tokens <max_new_tokens> \
    --input_lengths 1024 2048 3072 4096
```

| 論點；爭論 | 描述 |
|----|----|
| `model_id` | HuggingFace 模型 ID（例如：`meta-llama/Llama-3.2-3B-Instruct`）。對於 Modalix 效能測試，這應與已部署模型的權杖化器/設定檔相符。 |
| `-o` | 基準測試結果的輸出目錄。 |
| `--board_ip` | 遠端板卡的 IP 位址或主機名稱。從主機執行時為必填；在 Modalix 上本機執行時請省略。 |
| `--board_model` | DevKit 上的已編譯模型目錄。啟動伺服器時為必填。 |
| `--max_num_tokens` | 最大上下文長度。必須等於或小於編譯時使用的值。 |
| `--max_new_tokens` | 輸出的最大權杖數量。 |
| `--input_lengths` | 可選的精確輸入詞元長度，用於基準測試。數值必須是唯一的，且每個數值加上 `--max_new_tokens` 的總和必須小於或等於 `--max_num_tokens`。如果未指定，MoLE 將產生自動的 2 的冪次方分桶。 |
| `--board_ssh_user` | 用於 Modalix 裝置的 SSH 使用者名稱。為選用設定，預設值為：`sima`。 |
| `--board_ssh_pass` | 用於 Modalix 裝置的 SSH 密碼。為選用設定。設定後可啟用非互動式的自動基準測試。 |

對於所有可用的選項，請執行 `llima-benchmark perf -h` 基準測試。

使用 `--dataset bfcl-v3|ifeval|alpacaeval|gsm8k|dialogsum`，以原生聊天／工具提示取代 token 長度分組。省略 `-n` 會執行整個所選資料集；`-n 20` 則執行前 20 個樣本。資料集會自動下載並快取。BFCL 涵蓋 simple/multiple 案例，不會執行工具。

`--board_model` 決定一般或推測式解碼。比較模型時，以相同資料集與設定分別執行，只變更 `--board_model` 和 `-o`。每次執行會排除一次暖機，並在每個請求前重設 KV 狀態。使用模型的原生聊天範本與 EOS 停止。資料集輸出上限包含第一個 token；既有分組則測量第一個 token 加上 `--max_new_tokens` 次解碼。

`-o` 將時間、token 數量、彙整指標、草稿貢獻、失敗、基準測試設定及資料集來源儲存至 `performance.json`。解碼 TPS 不包含 TTFT；生成時間不包含輸入準備和網路傳輸。這些工作負載測量效能而非準確度。
