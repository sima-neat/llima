# 為 LLiMa 做出貢獻

LLiMa 包含一個主機端 GenAI 編譯器和一個用於 Modalix 的 C++ 執行階段。該執行階段透過封裝後的 CLI/HTTP/ZMQ 入口點進行操作；Python 是 CLI 編排工具，而不是一個獨立的公開執行階段 API。請將編譯器和執行階段環境及其依賴項分開。在程式碼庫的檢出過程中，`CONTRIBUTING.md` 提供快速入門指南，而 `AGENTS.md` 定義了特定代理程式的規則；本指南是詳細的貢獻者政策。

## 程式設計代理人的技能

將兩個 LLiMa 開發者技能作為標準開發者設定的一部分進行安裝。
它們不會被預設的 Neat SDK 腳本索引安裝，這是故意的：

```bash
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-contribute-to-llima
sima-cli playbooks install \
  gh:sima-neat/llima/skills/sima-add-llima-model-support
```

一般貢獻者技能涵蓋整個程式碼庫的編譯器、執行階段、封裝、測試、檔案和技能變更。模型支援技能新增了對 LLM 和 VLM 架構、檢查點、張量佈局、權杖化器和提示合約的相容性和實作流程。請確保同時安裝這兩者，以便在貢獻跨越這些界線時，提供適當的指導。


## 儲存庫地圖

| 區域 | 路徑 | 責任 |
| --- | --- | --- |
| 設定 | `sima_lmm/config/` | LLM、VLM 和 ASR 的設定合約。 |
| 攝取 | `sima_lmm/hf/`, `sima_lmm/gguf/` | Hugging Face 和 GGUF 的載入和轉換。 |
| 編譯 | `sima_lmm/model/`, `sima_lmm/preproc/` | 模型組件、量化、圖表和預處理。 |
| 主機工具 | `sima_lmm/host/` | 編譯、部署、LoRA，以及進行基準測試，以評估程式的進入點。 |
| 評估 | `sima_lmm/mole/` | MoLE 工作流程 |
| 執行階段 CLI | `sima_lmm/devkit/` | Python CLI 流程協調與模型管理 |
| C++ 執行階段 | `sima_lmm/devkit/cpp/` | 模型、權杖化器、MLA、CLI/HTTP/ZMQ 的實作，以及內部 CLI 繫結。 |
| 測試 | `tests/` | 編譯器和 Modalix 執行階段測試 |
| 包裝 | `CMakeLists.txt`, `cmake/`, `build*.sh`, `tools/install_*.sh` | Debian、wheel 格式，以及成品組裝。 |
| 持續整合/快取 | `.github/workflows/`, `tools/ci/`, `tools/hf-safetensors/` | 建立、測試和快取模型。 |
| 檔案/技能 | `README.md`, `docs/`, `skills/` | 使用者、貢獻者和《Playbooks》指南 |

僅編譯時所需的相依性不應納入 `sima_lmm/devkit/` 或 Modalix 執行階段套件。

## 開發環境

### 執行階段和封裝

請使用 Neat SDK 作為支援的建置環境。建置所有執行階段套件和封裝的測試：

```bash
./build.sh --all --clean
```

標準的建置程序會處理其所需的設定，包括子模組；請勿為標準工作流程執行單獨的依賴項啟動步驟。

有用的更精簡的建置程序：

```bash
./build.sh --clean --core
./build.sh --clean --core --dev
./build.sh --clean --cli
./build.sh --no-dist
```

輸出內容會在 `build-deb/` 中產生，並在 `dist/` 中進行分階段處理。Modalix 是執行 MLA 以及進行實際 `llima run` 驗證所必需的。

### 編譯器開發

請使用由 Model Compiler 安裝的 Python 3.12 環境。依序搜尋：

1. `/sdk-extensions/model-compiler`
2. `/sdk-add-on/model-compiler`
3. `$HOME/sdk-extensions/model-compiler`

```bash
source <model-compiler-venv>/bin/activate
python -m pip install -e '.[sdk_ext,tests]'
llima-compile --help
```

請勿建立另一個環境，以免遮蔽已安裝的編譯器套件。請使用以下方式建立發布設定檔：

```bash
./build_compiler_wheel.sh
./build_mole_package.sh
```

他們在 `build/` 中使用輪式工具，並在 `dist/compiler/` 和 `dist/mole/` 中進行階段性輸出。

### Whisper 和 ASR 的開發

公開的 `llima-compile` 工作流程涵蓋 LLM 模型和 VLM 模型。現有的 Whisper 編譯流程改用貢獻者工具：`scripts/gen_models--openai--whisper.py`。

```bash
python scripts/gen_models--openai--whisper.py \
  --model_path /path/to/openai/whisper-small \
  --output /path/to/whisper-output \
  --part all
```

在 Model Compiler 環境中執行，並明確指定模型路徑。
`--part` 接受 `all`、`encoder`、`language_detect`、`init`、`single_pre`、`single_post`，以及 `single_cache`。新增 `--enable_log_probe` 以編譯啟用日誌探測功能的解碼器輸出；使用 `--part all --enable_log_probe` 以進行完整的日誌探測建置。

Whisper 模型儲存庫中，每個編碼器層各有一個 ELF。執行階段不支援編碼器 ELF 為單一整體的舊版儲存庫；請下載分層模型，或使用目前的 LLiMa 版本重新編譯檢查點。

編譯器變更通常會影響 `sima_lmm/config/whisper_config.py`、`sima_lmm/model/whisper_*.py` 和腳本；執行階段變更會影響 `sima_lmm/devkit/cpp/whisper_*`。使用封裝的 C++ ASR 執行階段測試進行驗證，該測試的相關檔案位於 `tests/README.md` 中，並使用 Modalix 上的代表性音訊進行測試。這是一個 Whisper 專用的路徑，而不是一個通用的 ASR 架構框架。

### 原生圖元件

使用 `sima_lmm/model/model_graph.py` 中的 `ModelGraph`，一次綁定元件的來源權重、精度與輸出路徑。圖節點的建構放在類別內，獨立的陣列布局與命名工具則保留為模組函式。模型元件使用圖的方法，讓元件實作專注於拓撲：

```python
from sima_lmm.model.model_graph import ModelGraph

def generate_graph(self, layer_cfg, quantizable):
    graph = ModelGraph(self, {"hidden": (1, 1, self.num_tokens, self.cfg.d_model)}, quantizable)
    hidden = graph.layer_norm("model.norm", graph.inputs["hidden"])
    output = graph.mlp("model.mlp", hidden, "gelu", residual=graph.inputs["hidden"])
    graph.save([output])
```

在 `generate_graph()` 中定義獨立元件的輸入、頂層拓撲與 `graph.save()`。對於共用的圖建構，例如 Whisper 整合解碼器圖使用的 pre/cache/post 部分，保留 `_build_nodes(graph, inputs)`。

日誌範圍由 `BaseModel.gen_files()` 設定，圖建構不需要額外的日誌引數。

形狀會推論輸入型別：`quantizable=True` 使用 FP32（稍後量化的浮點圖），`False` 使用 BF16（依來源權重精度直接建構的圖）。快取等整數輸入使用 `input_dtypes={"cache": np.int8}`，名稱必須符合輸入規格。低階呼叫仍可使用現有的明確 AFE 張量規格。輸入與輸出順序遵循指定規格，形狀與輸出型別由 AFE 推論。`constant()` 將浮點資料轉為啟用值的精度；若整數常數需要特定位元寬度，例如使用 `dtype=np.int32`。節點名稱遵循 AFE 可確定的建立計數器。

圖節點的型別註記從 `model_graph.py` 匯入 `Node`。輔助函式接收圖，而非額外的 `quantizable` 旗標。`graph.constant()` 自動選擇浮點精度；主機端陣列計算需要該精度時，使用 NumPy 的 `graph.dtype`。階段旗標只保留在 `generate_graph()` 與圖建構處。

`save()` 完成 MLA 子網路並建立外層圖的輸出 tuple，保留整數輸出，在 EV 上將 BF16 輸出轉為 FP32，並使用標準成品名稱寫入檔案（浮點圖使用 `.fp32`）。若要取得完成的網路，使用 `finish()`。兩者都接受 `transform_subnet`，可在擷取外層輸出前進行模型專屬改寫。將圖本身傳給元件輔助函式，輸入、精度與來源權重仍綁定於同一物件。

以具名的 NumPy 輸入執行已完成的圖：

```python
graph.finish([output])
outputs = graph.run(hidden=x)
outputs_jax = graph.run(hidden=x, use_jax=True)
graph.save()
```

輸入名稱、形狀與資料型別必須符合宣告的輸入，不會套用隱含型別轉換。輸出遵循傳給 `finish()` 的順序。NumPy 執行使用 AFE fast mode，結果可能與 MLA 參考運算不同。`use_jax=True` 選擇 JAX 參考執行，此時 fast mode 沒有作用。JAX 運算使用所設定的後端，並可在相容的 JAX 安裝環境中使用 GPU。此 API 不會量化、編譯或在 Modalix 上執行。只完成圖一次，之後重複使用 `run()` 與 `save()`；仍可用 `save(outputs)` 一次完成並儲存圖。

常用操作包含 `add`, `sub`, `mul`, `matmul`, `concat`, `slice`, `transpose`, `reshape`, `softmax`, `topk`, `sum_channels`, `argmax`, `linear`, `conv`, `layer_norm`, `rms_norm`, `activation`, `softcap`, `mlp`, `rope`, `rope2d`, `split_heads`, `merge_heads`, `split_concat`, `clip`, `avgpool2d`, `space_to_depth`, `quant`, `dequant`。閘控 MLP 使用 `projections=("gate_proj", "up_proj", "down_proj")`，預設為 `("fc1", "fc2")`。`rms_norm("model.norm", input)` 使用語言模型設定的 epsilon 與權重偏移。vision 和 GDN 的正規化若明確指定 `epsilon`，除非也提供 `weight_offset`，否則權重偏移維持零。`rms_norm(None, input, epsilon=...)` 推論無權重的通道。RoPE 支援完整、部分與按比例的 split-half 旋轉。`rope2d(input, cos_x, sin_x, cos_y, sin_y)` 依 `[x-real, x-imag, y-real, y-imag]` 順序旋轉四等分的通道。`split_heads(input, heads, repeat=...)` 為 grouped-query 模式重複各個 head。`split_concat()` 提供 AFE 的重新分組操作，處理僅靠 reshape 無法表達的空間／權杖布局。`space_to_depth(input, blocksize)` 將完整的空間區塊合併至通道。`quant(input)` 回傳 `(int8_values, scale)`；`dequant(int8_values, scale)` 還原圖的啟用值精度。`argmax(input)` 回傳 INT32 通道索引。`slice(input, begin, end, stride, axis)` 對四維 FP32/BF16 張量中未對齊、連續的單軸通道切片，自動使用選擇器卷積；其他切片維持 AFE 的原生行為。

`linear("model.proj", input)` 解析 `model.proj.weight` 與可選的偏置，將 OI 來源權重轉為 SiMa 布局，並保留封裝的權重值、縮放係數、未對齊的群組大小與重定位中繼資料。`conv()` 推論 OIW 或 OIHW 來源布局；明確的轉換也能處理其他布局，例如 Qwen 的五維 patch 權重。`WeightOptions` 說明來源名稱、布局、權重／縮放係數／偏置轉換與重定位覆寫選項。切片群組權重時，提供對應的 `scale_process_func`；群組必須保留檢查點的實際大小。LoRA 秩與合併 adapter 行為是 `linear()` 和 `mlp()` 的明確引數。

以投影與 head 布局操作建構 attention。將 query 縮放併入 `linear()`，以保留現有投影的捨入行為：

```python
queries = graph.split_heads(graph.linear("attn.q_proj", hidden, scale=head_dim ** -0.5), heads)
keys = graph.split_heads(graph.linear("attn.k_proj", hidden), heads)
values = graph.split_heads(graph.linear("attn.v_proj", hidden), heads)
context = graph.attention(queries, keys, values)
output = graph.linear("attn.out_proj", graph.merge_heads(context))
```

`split_heads()` 透過選擇器卷積處理未對齊的 head 通道。標準 vision attention 會填補未對齊 head 的投影權重，以避免這些卷積；群組輸出權重保留原始布局。`attention()` 依 query/key 長度，為包含 cross-attention 的大型 attention 張量選擇各 head 的獨立分支。它接受加法 `mask`，且除非提供 `score_scale`，否則假設 query 已縮放。該縮放在 Q×K 之後、遮罩之前套用，保留 Qwen vision 等模型的 BF16 運算順序。遮罩必須是四維張量、向量或純量，且可廣播至 `[N,H,T_query,T_key]`；不支援的形狀會引發 `ValueError`。

`graph.matmul(lhs, rhs)` 建立 MLA 批次矩陣乘法。兩個轉置旗標都預設為 `False`；Kᵀ×V 使用 `transpose_a=True`，Q×Kᵀ 使用 `transpose_b=True`。輸入必須是四維 FP32/BF16 張量，且批次與縮約維度相符。head 數可整除時，grouped-query attention 使用隱含重複；無效的形狀或型別會引發 `ValueError`。

`graph.softmax(x)` 預設使用最後一個軸，也支援明確的 `axis`。連續單軸切片使用 `graph.slice(x, start=0, stop=128, axis=-1)`；多軸切片使用現有的 `begin`/`end`/`stride`/`axis` 列表。單軸形式的起點預設為零，必須明確指定軸與範圍內的非空邊界，並保留未對齊通道的處理。請勿混用兩種形式。

`ModelGraph` 擴充 AFE 的 `SimaBuilder`，因此可在同一物件使用原生操作與共用輔助函式：

```python
projected = graph.linear("model.proj", graph.inputs["hidden"])
output = graph.add(projected, graph.inputs["hidden"])
graph.save([output])
```

簡潔的基礎操作是 AFE 方法的直接別名，保留其簽章、型別推論與可確定的命名。較少使用的操作仍可透過同一個圖繼承的 `create_*` 方法取得。需要自訂多子網路生命週期的圖，仍可直接使用 AFE 的 `SimaBuilder` 與低階操作輔助函式。

細分布局集中由 `sima_analysis.get_tessellate_parameters()` 推論：HWC16 布局、自動 tile 大小與可確定的持久緩衝區名稱。一般元件從 `BaseModel` 繼承空的覆寫設定。具步幅的 KV 快取等特殊布局，覆寫現有的 `get_mla_input_tessellate_params()` / `get_mla_output_tessellate_params()` 方法；鍵是張量索引，負索引從尾端計算。

Whisper 元件、所有原生語言圖部分與標準 vision tower 都使用此 API；其 attention、MLP 與旋轉輔助函式共用相同實作。

## 測試

依據失效面選擇測試。建置作業並不能取代行為驗證，而且跳過的必要測試案例不視為通過。

### 氣密性測試

將純粹的設定、映射、序列化、驗證和數值邏輯與模型下載分開：

```bash
pytest -q <targeted-test-path>
```

### 以模型為基礎的編譯器測試

編譯器測試位於 `tests/compilation/`。請選擇受影響的群組，以及 `tests/README.md` 中描述的標記。例如：

```bash
export LLIMA_HF_MODELS_PATH=/path/to/llima-model-inputs
python -P -m pytest \
  -c pytest.ini \
  tests/compilation/configuration \
  -m compiler_config \
  --strict-markers \
  -vv -ra
```

`--model-inputs-path` 和 `LLIMA_HF_MODELS_PATH` 選取已準備好的 Hugging Face/GGUF 輸入根目錄。CI 使用位於 `tools/hf-safetensors/` 中的資訊檔。

設定所需的輸入，而不是接受跳過測試。

測試矩陣、預期數量與基準政策位於 `tests/README.md`；CI 呼叫位於 `.github/workflows/model-compiler-tests.yml`。請在執行期間產生原生 SDK 圖與數值比較成品，而非提交二進位基準檔。

### 執行階段驗證

建立候選套件和執行階段測試的額外元件：

```bash
./build.sh --all --clean
```

這會建置程式碼，但不會執行測試。請在 Modalix 上安裝相符的候選 LLiMa 套件、解壓縮額外的封存檔，並按照 `tests/README.md` 中的 DevKit 執行階段測試說明，執行封裝的 CTest 和 pytest。

當程式碼變更影響到模型載入、推論、分詞、多模態預處理、推測式解碼、CLI/HTTP/ZMQ 或資源生命週期時，請執行相關的硬體測試。必要時，新增一個具有代表性的簡短測試。

```bash
llima run <model_dir> --mode cli
```

對於 VLM 的變更，請包含一個基於圖像的提示。手動煙霧測試是補充，但不會取代受影響的套件覆蓋範圍。

Neat Core 會使用已安裝的 LLiMa 的 C++ API 和執行階段套件。當上述任何一個部分（或透過 Core 的 GenAI API 公開的行為）發生變更時，請使用候選的 `sima-lmm-core` 和 `sima-lmm-dev` 套件來建置 Core，而不是使用已發布或快取的 LLiMa 建置版本。在 Modalix 上執行受影響的 Core GenAI C++ 測試。此下游驗證對於獨立的編譯器、檔案或僅測試的變更而言，並非必要。

### 包裝驗證

建立每個已變更的設定檔：

```bash
./build.sh --all --clean
./build_compiler_wheel.sh
./build_mole_package.sh
```

驗證套件名稱、檔案擁有者、安裝資訊檔、相依性、校驗總和以及中繼資料。

## 程式碼規範

### 相容性與界限

將已安裝的 C++ 標頭檔、CLI 指令、序列化的組態、套件中繼資料，以及產生的成品佈局視為相容性介面。 優先採用增量變更。 如果需要進行重大變更，請記錄受影響的元件、移轉方式和發布意圖；更新呼叫者、測試、範例和使用者檔案。

將編譯器、執行階段和 MoLE 依賴項分開。 執行階段狀態不得作為主機編譯的輸入。 對執行階段套件邊界的變更必須保留 `sima-lmm-core`、`sima-lmm-dev` 和 `sima-lmm-cli` 的角色，並包含適當的 API/ABI 驗證。

### 實施品質

- 目標 C++20 歲 Python 在下列位置宣告的版本： `pyproject.toml`.
- 遵循周圍的格式、命名方式，並包含群組；避免過於寬泛。
  機械格式重置。
- 盡可能減少已安裝介面的數量，並將實作細節保持私密。
- 在適當的地方新增 Python 類型註解。
- 解釋那些不易理解的合約條款、數值假設，以及硬體規格。
  限制；請勿逐行解釋程式碼。
- 在新增抽象概念之前，先重複使用附近的輔助函式。
- 保留確定性模型選擇、圖形結構和序列化功能。
  成品名稱。記錄種子，並避免檔案系統/程序排序。
- 拒絕不支援或無效的輸入，並提供相關資訊；保留原始原因。
  跨越各層級，且永遠不會靜默地選擇另一個執行路徑。
- 協調並拆解綁定工作流程。建立緩衝區、處理程序、執行緒，以及
  明確指定暫存檔案的所有權；安全地清理部分已完成的工作。
- 在熱門程式碼路徑中，避免不必要的資源設定、複製和同步操作。

### 模型、成品、依賴項和機密資訊

允許保留的測試材料：

- 已審閱 JSON 設定合約；
- 受版本控制的案例、原始資料、容差值和比較原則；以及
- 用於已批准且不可變更的 Hugging Face/GGUF 版本。

請勿提交已下載的權重、客戶資料或產生的 ONNX、NumPy、量化模型、MPK、ELF 或執行階段模型樹。將產生的輸出儲存在已忽略或暫存目錄中。

使用 `deps/manifest.json` 來管理套件/平台的版本。將 `third_party/` 視為已包含的程式碼；隔離並記錄有意的子模組更新。請勿將編譯器依賴項新增到執行階段 Debian 套件中。

切勿提交或記錄任何 Token、SSH 憑證、私人儲存庫、已簽署的 URL 或個人路徑。將受控模型的授權權限置於版本控制系統之外，並在報告中使用公開的模型 ID、不可變的修訂版本和已刪除的日誌。

## 檔案、技能和程式碼變更請求

更新最接近的使用者指南：

- [系統需求](setup.md)
- [模型編譯](compilation_genai.md)
- [模型部署](deployment.md)
- [ LLiMa CLI ](runtime.md)
- [MoLE](mole.md)

將根目錄的 `CONTRIBUTING.md` 檔案保留為快速入門指南，將此檔案作為詳細的策略檔案，並將 `AGENTS.md` 檔案作為可執行的代理規則。技能必須包含有效的 `SKILL.md`、`playbook.yml` 和代理中繼資料；保持其主要工作流程簡潔，並將條件細節移至直接引用中。

在 Neat SDK 中，驗證來自儲存庫根目錄的所有技能負載，且不得更改已安裝的代理狀態：

```bash
playbooks_validation_dir="$(mktemp -d)"
CODEX_HOME="${playbooks_validation_dir}/codex" \
CLAUDE_HOME="${playbooks_validation_dir}/claude" \
SIMA_CLI_HOME="${playbooks_validation_dir}/sima-cli" \
sima-cli playbooks install ./skills
```

安裝摘要必須報告 `detected: 3`、`valid: 3` 和 `discarded: 0`。
每個 `playbook.yml` 中的 `sima-cli` 版本必須滿足 `min_cli_version`。

對於提交的程式碼變更：

- 從目前的分支切換到並以目前的分支作為目標分支：`develop`；
- 讓提交訊息更具重點，使用祈使語氣的動詞開頭；
- 使用 `.github/PULL_REQUEST_TEMPLATE.md`。
- 將已完成的問題與「`Fixes #<issue>`」連結；
- 回報風險、相容性／移轉問題、檔案影響、可重現的指令
  模型/套件版本、硬體證據、已跳過的檢查項目，以及剩餘風險；
- 排除憑證和私人資產。

當相關測試在沒有意外跳過的情況下通過，所需套件和 Modalix 檢查完成或明確指出無法使用，且已處理相容性和檔案問題，並且 PR 包含可重複驗證的證據時，即表示貢獻已準備就緒。
