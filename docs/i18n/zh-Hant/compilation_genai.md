# 模型編譯

## 總覽

**Model Compiler** 提供 LLiMa 命令列工具 `llima-compile`，用於編譯來自 Hugging Face safetensors、GGUF 檔案，或預先量化的 compressed-tensors 模型（GPTQ/AutoRound）的模型。

``` console
llima-compile [options] <model_path>
```

### 模型輸入格式

LLiMa 接受三個模型輸入路徑。請根據檢查點的可用性、所需的精確度，以及模型是 LLM 還是 VLM 來選擇其中一個。

| 輸入 | 描述 | 何時使用 |
| --- | --- | --- |
| 原始 Hugging Face safetensors | 一個 FP/BF16 檢查點，它 LLiMa 在編譯期間進行量化。 | 沒有完全符合的預先量化匹配結果，或者需要原始權重。 |
| 預先量化過的 Hugging Face safetensors（GPTQ/ AutoRound） | 一個檢查點，其量化後的權重會被 LLiMa 重新使用。 | 當資料集提供完全符合的結果時，建議使用此方法。 |
| GGUF | 現有的量化模型 LLM 檢查點。 | 一個方便的備用方案 LLM；不支援用於 VLM。 |

僅僅是輸入格式本身並不能保證相容性。模型架構、大小、權杖化器，以及任何多模態組件也必須受到支援。

### 從一個開始 SiMa.ai 預先量化過的模型

:::tip 建議輸入
在下載原始的 Hugging Face 或 GGUF 模型權重之前，請檢查
[SiMa.ai 預量化模型集合](https://huggingface.co/collections/simaai/pre-quantized-models)。
如果有的話，請針對所需的架構、參數大小、變體和模態使用完全匹配。
:::

資料收集的檢查點是特定於模型的，預先設定的。LLiMa compressed-tensors
可以直接傳遞給的成品 `llima-compile`它們避免了額外的浮點數到量化的編譯階段，並包含了解它們的準確性和佈局所需之量化來源資訊。它們是編譯器的輸入，而非已編譯的程式碼。 Modalix 模型。

對於現有支援的模型進行客製化的微調時，精確匹配的資料庫也可能提供其模型特定的 `quantize.py`,
`recipe.yaml`以及 `versions.txt`請閱讀該程式碼庫的「模型卡」，並使用其中記載的指令碼；請勿重複使用僅僅是相似模型的指令碼。

``` console
hf download simaai/<model-repository> \
    --revision <immutable-revision> \
    --local-dir <prequantized-model-directory>
llima-compile <prequantized-model-directory> -o <output-directory>
```

### 描述這個模型；具備 LLiMa 技能的代理人會編譯它

SiMa.ai LLiMa 透過技能，可立即支援代理模型編譯。這些技能包含在 Neat 開發環境（Neat SDK）中。這些技能可讓程式碼代理擁有評估 LLM 和 VLM 相容性的上下文，選擇精確的預先量化輸入（如果有的話），使用已安裝的 LLiMa CLI，並遵循 Modalix 部署和驗證工作流程。

建議的代理路徑可以編譯模型，將其部署到可存取的 Modalix DevKit，檢查結果和診斷資訊，並優化編譯。傳統 CLI 編譯仍然是一種平行路徑，可透過相同的工具進行直接控制。兩者都會產生標準的可檢查 LLiMa 成品，因此您可以檢閱選定的模型、指令、選項和輸出，或根據需求變化在兩種工作流程之間切換。請參閱 [設定 Neat SDK](https://developer.sima.ai/software/getting-started/dev-environment/)，以啟用代理編譯。

以自然語言要求完整的流程，例如：

``` text
Compile <model ID or local path> with LLiMa, deploy it to my Modalix at
<user@host>, and smoke-test it. Prefer an exact SiMa.ai pre-quantized
checkpoint when available.
```

這個代理程式會記錄模型和配方的使用來源，遵循已安裝版本的 CLI 規範，並回報任何不支援的模型範圍或無法使用的硬體驗證，而不是在不發出任何通知的情況下替換為另一個模型或格式。

### 編譯輸出

預設的完整流程會產生以下目錄結構：

``` text
output_directory/
└── sima_files/                # Compiled model files
    ├── devkit/                # Runtime configuration and model data
    │   ├── tokenizer.json
    │   ├── vlm_config.json
    │   └── ...
    ├── mpk/                   # MPK archives with compiled binaries
    │   ├── layer_0.tar.gz
    │   └── ...
    ├── npy_files/             # LoRA adapter weights (only when compiled with LoRA)
    │   ├── <adapter_name>/
    │   │   └── *.npy
    │   └── ...
    └── ...
```

## 命令列參數

`llima-compile` 工具接受各種參數，以便自訂編譯程序。以下表格描述了可用的選項：

| 論點；爭論 | 描述 |
|----|----|
| `model_path` | 輸入模型路徑（可以是 HuggingFace 目錄、GGUF 檔案，或是預先量化壓縮的張量目錄）。 |
| `-o, --output` | 編譯後檔案的輸出目錄。預設為模型名稱。 |
| `-c, --configuration_file` | Python 腳本，用於設定每個層級的精確度（例如，用於混合精確度）。 |
| `--max_num_tokens` | 最大上下文長度。必須是 1024 的倍數。預設值：4096。 |
| `--resume` | 透過略過現有檔案來恢復中斷的建置程序。 |
| `-j, --jobs` | 並行編譯作業的數量。預設值：實體 CPU 核心的數量。 |
| `--log_level` | 日誌記錄層級（DEBUG、INFO、WARNING、ERROR）。預設值：WARNING。 |
| `--input_height` | 輸入圖片的高度，以像素為單位。必須與 `--input_width` 一起提供。對於 Qwen 2 VL、Qwen 3 VL 和 Gemma 4 而言，這是必需的；對於覆寫 SigLIP2 模型已設定的大小而言，則是可選的。 |
| `--input_width` | 輸入圖片的寬度，單位為像素。必須與 `--input_height` 一起提供。對於 Qwen 2 VL、Qwen 3 VL 和 Gemma 4 而言，這是必需的；對於覆寫 SigLIP2 模型已設定的大小而言，則是可選的。 |
| `--system_prompt` | 系統提示，用於儲存以供 CLI 模式使用，以及用於模型預熱。 |
| `--system_prompt_file` | 指向包含系統提示的文字檔案的路徑。 |
| `--chat_template` | 聊天範本字串，用於儲存在編譯後的模型中。與「系統提示」和「聊天範本檔案」選項為互斥關係。 |
| `--chat_template_file` | 檔案路徑，指向包含對話範本的檔案。與「系統提示」選項互斥，且與 `--chat_template` 互斥。 |

:::note
大多數模型支援最多 8192 個 token 的上下文長度。請使用 `--max_num_tokens 8192` 以啟用 8K 的上下文長度。
:::

| 進階論證 | 描述 |
|----|----|
| `--language_group_size` | 在預填階段，用於平行處理分詞的批次大小。較大的值（例如，256）可以改善大型輸入提示的首次產生詞元所需時間（TTFT），但可能會使小型輸入提示的 TTFT 變差。預設值：128。 |
| `--future_token_mask_size` | 用於跨分詞位置重複使用已編譯模型的遮罩大小。較大的值會減少已編譯的二進位檔案數量，但可能會降低每秒詞元數（TPS）。預設值：128。 |
| `--enable_filter_sharing` | 啟用群組模型和單一模型之間的篩選器共享功能，以減少 DRAM 使用量，但會增加 TTFT（首次產生詞元所需時間）並降低 TPS（每秒詞元數）。此功能僅在兩種模型類型都使用相同的精確度時才有效，並且在使用 LoRA 進行編譯時是必需的。 |
| `--no-quantize_embeddings` | 停用嵌入表格量化功能。預設情況下，此功能會針對支援的 LLM 和 VLM 啟用。 |
| `--no-quantize_kv_cache` | 停用 KV 快取量化功能，因為預設情況下已啟用此功能。 |
| `--return_logits` | 傳回最後一層輸出的 logits 值（模型評估器需要此值）。 |
| `--draft_model_path` | 用於推測性解碼的 EAGLE3 草稿模型的路徑。 |
| `--lora_name` | 與基礎模型一同編譯的 LoRA 配接器的名稱。 |
| `--lora_path` | 用於與基礎模型一起編譯的 LoRA 轉接器目錄的路徑。 |
| `--compile_lora`, `--no-compile_lora` | 當提供 LoRA 路徑時，啟用或停用配接器權重編譯。預設為啟用。 |

## 系統提示

使用 `--system_prompt` 或 `--system_prompt_file` 將系統提示儲存在已編譯的模型設定中。這些參數是互斥的。

``` console
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
    --system_prompt "You are a concise technical assistant." \
    -o Llama-3.2-3B-Instruct_out
```

在 CLI 模式下，這會變成預設的系統提示。在會話期間，可以使用 `set system <prompt>` 來取代它，或使用 `clear system` 來移除它。

在網頁模式/GenAIServer 中，儲存的提示會在模型預熱期間使用，並且可以快取以用於第一次請求。它不會自動新增到 API 請求中。使用者端必須在第一次請求和後續每個請求的 `messages` 陣列中包含系統提示。

## 設定檔

組態檔案會針對每個編譯單元自訂編譯設定，以啟用混合精度和選擇性編譯。

LLM 推論包含兩個不同的階段，編譯器會為每個階段產生最佳化的模型。

- **預先填入（群組模型）**：以批次方式處理輸入提示，使用 `language_group_size`（例如，一次處理 128 個詞元）。此階段決定了 TTFT（首次產生詞元所需時間），並針對處理量進行優化。
- **解碼（單詞元模型）**：以自迴歸方式逐一產生輸出詞元。這個階段決定了每秒產生的詞元數（TPS），並針對低延遲生成進行優化。

由於這些階段具有不同的效能特性，因此您可以透過在設定函式中使用 `is_group` 旗標，為每個階段套用不同的量化策略。

**輸入參數**

`get_layer_configuration` 函式會針對每個編譯單元進行呼叫，並接收：

- `model_properties`：包含 `{"num_hidden_layers": int}` 的字典。

- `layer`：包含以下詞彙的字典：
  - `"part"`：邏輯元件，例如 `"PRE"`、`"CACHE"`、`"POST"`。
    `"VISION"`、`"DRAFT_FC"` 或 `"PER_LAYER"`。
  - `"is_group"`：對於多個詞彙/群組的變體，其值為`True`，否則為`False`。
  - `"index"`：該編譯單元的索引。對於 `"PRE"` 和 `"POST"`，
    通常對應於一個轉換器層。對於 `"CACHE"`，它會識別
一個快取或權杖位置變體，而不是一個轉換器層。

**傳回值**

該函式會傳回一個包含以下內容的字典：

- `"precision"`：量化層級（可選，預設值：`"BF16"`）
  - `"BF16"`：完全精確度——品質最佳、檔案大小最大、速度最慢。
  - `"A_BF16_W_INT8"`：中等量化——品質良好，大小適中。
  - `"A_BF16_W_INT4"`：高量化程度——品質可接受，檔案大小最小，速度最快。

- `"compile"`：設定為 `False`，以跳過編譯此層級（可選，預設值：`True`）。

- `"lora"`：此層的 LoRA 模式（可選，預設值：`"LORA_DISABLED"`）
  - `"LORA_DISABLED"`：此層不支援 LoRA。當未提供任何設定檔時，預設會使用此設定，結果是產生一個標準模型，且沒有額外的配接器開銷。
  - `"LORA_BRANCH"`：編譯並行 LoRA 分支，並將其權重設為零，與基礎模型一起使用。在執行階段，會從 `.npy` 檔案中載入適配器權重，從而可以在不重新啟動模型的情況下，動態切換不同的適配器。當您需要即時切換適配器時，請使用此模式。
  - `"LORA_MERGED"`：LoRA權重會在執行階段與基礎模型權重合併。此時，該適配器會在整個會話期間保持啟用狀態，且無法切換或移除。當您希望始終應用該適配器，且不需要動態切換時，請使用此模式。

:::note
**最佳實務：** 對於群組層，請使用 INT8 (`A_BF16_W_INT8`)，以在預填充期間維持品質；對於單個標記層，請使用 INT4 (`A_BF16_W_INT4`)，以加快生成速度；對於視覺編碼器，請使用 BF16，以保留圖像理解的品質。對於大多數模型，此設定可在模型準確性、吞吐量和記憶體使用量之間提供最佳平衡。
:::

## 範例

**範例 1：編譯一個簡單的 LLM**

編譯一個從 Hugging Face 下載的 Llama 模型，並使用預設設定：

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

這將會：

- 對所有層級使用預設的 BF16 精度。
- 將上下文長度設定為 4096 個詞元。
- 將輸出內容儲存到 `Llama-3.2-3B-Instruct_out` 目錄。

**範例 2：使用自訂上下文長度進行編譯**

``` console
sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B --local-dir Llama-3.2-3B-Instruct
sima-user@docker-image-id:/home/docker$ llima-compile --max_num_tokens 4096 Llama-3.2-3B-Instruct -o Llama-3.2-3B-Instruct_out
```

這將會：

- 對所有層級使用預設的 BF16 精度。
- 將上下文長度設定為 4096 個詞元。
- 將輸出內容儲存到 `Llama-3.2-3B-Instruct_out` 目錄。

**範例 3：使用混合精度編譯 Gemma 3 VLM**

對於像 Gemma 3 VLM 這樣複雜的模型，您可能需要為不同的層指定不同的精度（例如，將視覺編碼器保持在 BF16 狀態）。

1.  **下載模型**：

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download simaai/gemma3-siglip448 --local-dir gemma-3-model
    ```

2.  **建立一個設定檔**（例如：`config.py`）：

    ``` python
    def get_layer_configuration(model_properties, layer):
        # Keep vision encoder in full precision
        if layer["part"] == "VISION":
            precision = "BF16"
        # Use INT8 for batch processing layers (better quality)
        elif layer["is_group"]:
            precision = "A_BF16_W_INT8"
        # Use INT4 for single-token layers (smaller size)
        else:
            precision = "A_BF16_W_INT4"
        return {"precision": precision}
    ```

3.  **執行編譯器**：

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile -c config.py --max_num_tokens 2048 gemma-3-model -o gemma-3-model_out
    ```

**範例 4：進階設定**

針對變換器層進行特定控制的混合精度：

``` python
def get_layer_configuration(model_properties, layer):
    # PRE and POST indices normally identify transformer layers.
    if layer["part"] in {"PRE", "POST"} and layer["index"] < 4:
        return {"precision": "BF16"}

    # Keep every required compiler unit and use INT8 elsewhere.
    return {"precision": "A_BF16_W_INT8"}
```

請勿將 `"CACHE"` 索引解讀為變壓器層索引。省略快取變體可能會導致編譯後的輸出不完整，並且在執行階段無法使用。

**範例 5：編譯具有 LoRA 的 LLM**

LoRA（低階適應）允許對基礎模型進行微調，並且可以在執行階段動態地應用或移除適配器，而無需重新編譯基礎模型。基礎模型使用並行的 LoRA 分支進行編譯（初始化為零），並且適配器權重會單獨編譯到 `.npy` 檔案中，這些檔案會在需要時載入。

:::note
在搭配 LoRA 進行編譯時，需要啟用篩選器共享功能。請透過 `--enable_filter_sharing` 來啟用。即使指定了 INT4，LoRA 分支總是會以 INT8 進行編譯，以提高準確度。
:::

1.  **下載基礎模型和 LoRA 轉接器**：

    ``` console
    sima-user@docker-image-id:/home/docker$ hf download meta-llama/Llama-3.2-3B-Instruct --local-dir Llama-3.2-3B-Instruct
    sima-user@docker-image-id:/home/docker$ hf download <org>/<lora-adapter> --local-dir my-lora
    ```

2.  **建立一個設定檔**（例如：`lora_config.py`）：

    `lora` 參數控制每個層級的 LoRA 模式。使用 `"LORA_BRANCH"` 以在執行階段啟用動態切換。

    ``` python
    def get_layer_configuration(model_properties, layer):
        if layer["is_group"]:
            return {"precision": "A_BF16_W_INT8", "compile": True, "lora": "LORA_BRANCH"}
        else:
            return {"precision": "A_BF16_W_INT4", "compile": True, "lora": "LORA_BRANCH"}
    ```

3.  **使用 LoRA 轉接器來編譯基礎模型**：

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter \
        --lora_path my-lora \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

    這會將基礎模型與一個 LoRA 分支合併，並自動將配接器權重合併到 `Llama-3.2-3B-lora-out/sima_files/npy_files/my_adapter/`。

透過重複使用 `--lora_name` 和 `--lora_path`，可以在同一個步驟中合併多個配接器：

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile Llama-3.2-3B-Instruct \
        --enable_filter_sharing \
        --lora_name my_adapter_A --lora_path my-lora_A \
        --lora_name my_adapter_B --lora_path my-lora_B \
        -c lora_config.py \
        -o Llama-3.2-3B-lora-out
    ```

4.  **若要新增更多配接器**，而無需重新編譯基礎模型，請針對每個新增的配接器使用 `llima-compile-lora`：

    ``` console
    sima-user@docker-image-id:/home/docker$ llima-compile-lora Llama-3.2-3B-Instruct ./lora-c \
        -w Llama-3.2-3B-lora-out/sima_files/mpk \
        -o Llama-3.2-3B-lora-out/sima_files/npy_files/adapter_c
    ```

    **llima-compile-lora 命令列參數**

    | 論點；爭論 | 描述 |
    |----|----|
    | `base_path` | 原始基礎模型目錄的路徑（HuggingFace 格式）。 |
    | `lora_path` | 用於編譯的 LoRA 轉接器目錄路徑。 |
    | `-w, --weight_map_path` | **必要。** 從基礎模型編譯開始，指定到 `mpk/` 資料夾的路徑。其中包含編譯配接器所需權重地圖。 |
    | `-o, --output` | 編譯後的轉接器的輸出目錄。 `.npy` 檔案。預設為轉接器目錄名稱。 |
